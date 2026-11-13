"""
update_cash.py - keep data/positions.json (cash + holdings) in sync with the broker.

positions.json is the single source of truth for cash_available and open_positions.
daily_pipeline.py never records buys/sells itself; run this after each broker fill.

Usage (from stock-trading-system/):
    python update_cash.py --show
    python update_cash.py --set 2250.00
    python update_cash.py --buy  MAR --shares 2 --price 428.39 [--stop 411.75]
    python update_cash.py --sell MAR --shares 2 --price 440.00
    python update_cash.py --sell MAR --all --price 440.00      # close whole position
    python update_cash.py --import fills.csv                   # batch of buy fills (CSV/TSV)
    python update_cash.py --set-stop TRSP --stop 109.10         # record a stop AFTER you place it

Fill files need a header row with a ticker column (Stock/Symbol/Ticker), a shares column
(Shares/Quantity/Qty) and a price column (Price/Fill Price/Avg Price). Optional: Date.
$ and commas are fine. A Stop column is ignored on import: record stops separately with
--set-stop once the broker has finalized the purchase and you have placed the order.

Add --dry-run to preview without writing, --force to allow cash to go negative,
--file PATH to operate on a different positions file.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "data" / "positions.json"


def load(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = {}
    d.setdefault("cash_available", 0.0)
    d.setdefault("open_positions", {})
    d.setdefault("stop_loss_orders", {})
    return d


def save(path: Path, d: dict) -> None:
    d["as_of"] = date.today().isoformat()
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)  # atomic: never leaves a half-written file


def show(d: dict) -> None:
    print(f"Cash available: ${d['cash_available']:,.2f}")
    pos = d["open_positions"]
    if not pos:
        print("Open positions: none")
        return
    print("Open positions:")
    for t, p in pos.items():
        print(f"  {t}: shares={p.get('shares')}  cost_basis={p.get('cost_basis')}  "
              f"stop={p.get('stop')}  added={p.get('added_date')}")


# -- Reusable operations (used by the CLI and by the dashboard) -----------------

_ALIASES = {
    "ticker": ("stock", "symbol", "ticker", "security"),
    "shares": ("shares", "quantity", "qty", "filled", "filled qty"),
    "price":  ("price", "fill price", "avg price", "average price", "executed price"),
    "date":   ("date", "trade date", "fill date", "executed"),
}


def _num(x) -> float:
    return float(str(x).replace("$", "").replace(",", "").strip())


def parse_fills(text: str) -> list:
    """Parse pasted/uploaded broker fills into [{ticker, shares, price, date?}, ...].

    Accepts tab-, comma- or space-separated text with a header row. Raises ValueError
    with a row-specific message on anything it cannot read, so nothing is half-imported.
    """
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Need a header row and at least one fill.")

    def split(ln):
        if "\t" in ln:
            return [c.strip() for c in ln.split("\t")]
        if "," in ln:
            import csv
            return [c.strip() for c in next(csv.reader([ln]))]
        return ln.split()

    header = [h.strip().lower() for h in split(lines[0])]
    col = {}
    for key, names in _ALIASES.items():
        for i, h in enumerate(header):
            if h in names:
                col[key] = i
                break
    missing = [k for k in ("ticker", "shares", "price") if k not in col]
    if missing:
        raise ValueError(f"Header is missing column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(header)}")
    fills = []
    for n, ln in enumerate(lines[1:], start=2):
        cells = split(ln)
        try:
            f = {"ticker": cells[col["ticker"]].upper().strip(),
                 "shares": _num(cells[col["shares"]]),
                 "price": _num(cells[col["price"]])}
            if "date" in col and col["date"] < len(cells) and cells[col["date"]]:
                f["date"] = cells[col["date"]].strip()
        except (IndexError, ValueError):
            raise ValueError(f"Row {n} could not be read: {ln!r}")
        if not f["ticker"].isalpha() or f["shares"] <= 0 or f["price"] <= 0:
            raise ValueError(f"Row {n} has an invalid ticker, share count or price: {ln!r}")
        fills.append(f)
    return fills


def apply_buy(d: dict, t: str, shares: float, price: float,
              stop: float | None = None, force: bool = False) -> float:
    """Record one buy fill into d (mutates). Returns the cost. Raises ValueError."""
    if shares <= 0 or price <= 0:
        raise ValueError(f"{t}: shares and price must be > 0")
    cost = shares * price
    if cost > d["cash_available"] + 1e-9 and not force:
        raise ValueError(f"{t}: cost ${cost:,.2f} exceeds cash ${d['cash_available']:,.2f}")
    pos = d["open_positions"]
    cur = pos.get(t)
    if cur and cur.get("shares"):
        total = cur["shares"] + shares
        avg = (cur["shares"] * cur["cost_basis"] + cost) / total
        cur.update(shares=total, cost_basis=round(avg, 4))
    else:
        pos[t] = {"shares": shares, "cost_basis": round(price, 4),
                  "stop": None, "added_date": date.today().isoformat()}
    if stop is not None:
        pos[t]["stop"] = stop
    if pos[t].get("stop") is not None:
        d["stop_loss_orders"][t] = pos[t]["stop"]
    d["cash_available"] = round(d["cash_available"] - cost, 2)
    return cost


def import_fills(d: dict, fills: list, force: bool = False,
                 allow_duplicates: bool = False) -> list:
    """Apply a batch of buy fills to d ALL-OR-NOTHING. Returns report lines.

    Duplicate protection: a fill identical (ticker/shares/price/date) to one already
    imported is skipped, so pasting the same download twice cannot double-count.
    On any error d is left untouched and ValueError is raised.
    """
    import copy
    work = copy.deepcopy(d)
    log = work.setdefault("fills_log", [])
    report = []
    for f in fills:
        key = f"{f['ticker']}|{f['shares']:g}|{f['price']:.4f}|{f.get('date', '')}"
        if key in log and not allow_duplicates:
            report.append(f"SKIPPED duplicate: {f['ticker']} {f['shares']:g} @ ${f['price']:,.2f}")
            continue
        cost = apply_buy(work, f["ticker"], f["shares"], f["price"], force=force)
        log.append(key)
        report.append(f"BOUGHT {f['ticker']}: {f['shares']:g} @ ${f['price']:,.2f} = ${cost:,.2f}")
    d.clear()
    d.update(work)
    return report


def set_stop(d: dict, t: str, stop: float) -> None:
    """Record a stop for an open position (call after the stop order is placed)."""
    if t not in d["open_positions"]:
        raise ValueError(f"{t} is not an open position.")
    if stop <= 0:
        raise ValueError("stop must be > 0")
    d["open_positions"][t]["stop"] = stop
    d["stop_loss_orders"][t] = stop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--show", action="store_true", help="Print cash and holdings")
    g.add_argument("--set", type=float, metavar="AMOUNT", help="Set cash_available directly")
    g.add_argument("--buy", metavar="TICKER", help="Record a buy fill")
    g.add_argument("--sell", metavar="TICKER", help="Record a sell fill")
    g.add_argument("--import", dest="import_file", type=Path, metavar="FILE",
                   help="Batch-record buy fills from a CSV/TSV file")
    g.add_argument("--set-stop", metavar="TICKER",
                   help="Record a stop you have placed (use with --stop)")
    ap.add_argument("--shares", type=float, help="Shares filled")
    ap.add_argument("--all", action="store_true", help="With --sell: sell the whole position")
    ap.add_argument("--price", type=float, help="Fill price per share")
    ap.add_argument("--stop", type=float, help="With --buy: trailing stop price")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Allow negative cash / oversell")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    a = ap.parse_args()

    d = load(a.file)

    if a.show:
        show(d)
        return 0

    if a.set is not None:
        if a.set < 0 and not a.force:
            print("Refusing negative cash (use --force).")
            return 1
        d["cash_available"] = round(a.set, 2)

    elif a.import_file:
        try:
            text = a.import_file.read_text(encoding="utf-8-sig")
            print("\n".join(import_fills(d, parse_fills(text), force=a.force)))
        except (ValueError, OSError) as e:
            print(f"Import failed, nothing recorded: {e}")
            return 1

    elif a.set_stop:
        if a.stop is None:
            print("--stop PRICE is required with --set-stop.")
            return 1
        try:
            set_stop(d, a.set_stop.upper(), a.stop)
        except ValueError as e:
            print(e)
            return 1

    elif a.buy or a.sell:
        t = (a.buy or a.sell).upper()
        pos = d["open_positions"]
        if a.price is None or a.price <= 0:
            print("--price is required (> 0).")
            return 1

        if a.buy:
            if not a.shares or a.shares <= 0:
                print("--shares is required (> 0) for --buy.")
                return 1
            try:
                apply_buy(d, t, a.shares, a.price, a.stop, a.force)
            except ValueError as e:
                print(f"{e} (use --force to override the cash check).")
                return 1

        else:  # sell
            cur = pos.get(t)
            if not cur:
                print(f"{t} is not an open position.")
                return 1
            held = cur.get("shares") or 0
            qty = held if a.all else a.shares
            if not qty or qty <= 0:
                print("Give --shares N or --all for --sell.")
                return 1
            if qty > held and not a.force:
                print(f"Selling {qty} but only {held} held (use --force).")
                return 1
            d["cash_available"] = round(d["cash_available"] + qty * a.price, 2)
            remaining = held - qty
            if remaining <= 0:
                pos.pop(t, None)
                d["stop_loss_orders"].pop(t, None)
            else:
                cur["shares"] = remaining

    show(d)
    if a.dry_run:
        print("(dry run - nothing written)")
        return 0
    save(a.file, d)
    print(f"Saved {a.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
