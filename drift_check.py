"""
drift_check.py - price-drift allowance for the app's buy / stop advice.

The pipeline prices its advice at the last close. Orders are placed later, at a
different price, and the protective stop-limit goes in only after the broker has
finalized (settled) the purchase. Two pure functions cover the two orders:

    buy_band()    Order 1: range of prices where the recommended buy is still valid,
                  plus the buy-limit to place (the band ceiling).
    stop_check()  Order 2: given the actual fill and the current price, whether and
                  where to place the stop-limit.

No Streamlit / pandas dependency, so both are unit-testable. All money is dollars.
Dates are estimates only (weekdays, no holiday calendar): the broker confirming the
purchase is final is the real gate, and the UI says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

# Defaults (agreed design). Allowance is a fraction of the per-stock stop distance,
# so volatile stocks (wide stops) tolerate more drift than calm ones.
DEFAULT_ALLOWANCE_FRAC = 0.25   # of stop distance, each side
DEFAULT_MIN_RR = 1.25           # minimum reward/risk at the buy price
DEFAULT_LIMIT_BUFFER_PCT = 0.5  # stop-limit's limit sits this % below the stop trigger
DEFAULT_CUSHION_FRAC = 0.25     # warn if live price is within this fraction of the
                                # stop distance above the stop


def rr_blocker(price: float, target_price: Optional[float],
               stop_distance: Optional[float],
               min_rr: float = DEFAULT_MIN_RR) -> Optional[str]:
    """Reason a buy fails the reward/risk rule, or None if it passes.

    The pipeline's buy rule uses this so it cannot recommend a stock whose own
    90-day target is at or below the price, or whose upside is small relative to
    the backtested stop distance. Missing data never blocks a buy here: the other
    rules already cover missing forecasts, and without a stop distance only the
    target-above-price test is possible.
    """
    if not price or price <= 0 or not target_price:
        return None
    if target_price <= price:
        return (f"90-day forecast target ${target_price:,.2f} is not above the "
                f"price ${price:,.2f}")
    if stop_distance and stop_distance > 0:
        rr = (target_price - price) / stop_distance
        if rr < min_rr:
            return (f"reward/risk {rr:.2f} is below the {min_rr:g} minimum (target "
                    f"${target_price:,.2f}, price ${price:,.2f}, stop distance "
                    f"${stop_distance:,.2f})")
    return None


def next_business_day(d: date) -> date:
    """Next weekday after d (ESTIMATE: ignores market holidays)."""
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def unprotected_days(buy_date: date) -> int:
    """Calendar days the position is held with no stop (estimate, no holidays)."""
    return (next_business_day(buy_date) - buy_date).days


@dataclass
class BuyBand:
    valid: bool                 # False: even the analysis price fails the rules
    floor: float                # below this at the open -> skip and re-run
    ceiling: float              # above this -> skip (chasing); also the buy-limit price
    buy_limit: float            # price for the buy limit order (== ceiling)
    shares: int                 # shares affordable at the ceiling within cap / cash
    max_cost: float             # shares * buy_limit
    rr_at_price: float          # reward/risk at the analysis price
    unprotected_days: int       # days until the stop can be placed (estimate)
    stop_after_fill: str        # human description of Order 2
    notes: list = field(default_factory=list)


def buy_band(price: float, stop_distance: float, target_price: float,
             max_cost: float, cash_left: Optional[float] = None,
             buy_date: Optional[date] = None,
             allowance_frac: float = DEFAULT_ALLOWANCE_FRAC,
             min_rr: float = DEFAULT_MIN_RR) -> BuyBand:
    """Validity band for Order 1 (the buy).

    price          analysis-time price the advice was built on
    stop_distance  per-share dollar distance of the backtested stop
    target_price   90-day forecast target
    max_cost       position cap in dollars (shares were sized to this)
    cash_left      cash actually available (optional extra cap)
    """
    buy_date = buy_date or date.today()
    if price <= 0 or stop_distance <= 0:
        raise ValueError("price and stop_distance must be positive")
    allowance_frac = min(max(allowance_frac, 0.0), 0.9)
    notes: list = []

    down = allowance_frac * stop_distance
    up = allowance_frac * stop_distance
    days = unprotected_days(buy_date)
    if days > 1:
        down *= 0.5   # longer unprotected window -> tighter floor
        notes.append(f"{buy_date:%A} buy: no stop can go in for about {days} days "
                     f"(until {next_business_day(buy_date):%a %m/%d}, estimate). "
                     f"Floor tightened to half the usual allowance.")

    floor = price - down
    # Reward/risk with the stop distance held constant: (target - p) / d >= min_rr
    rr_ceiling = target_price - min_rr * stop_distance
    ceiling = min(price + up, rr_ceiling)
    rr_at_price = (target_price - price) / stop_distance

    valid = True
    if rr_ceiling < price:
        valid = False
        ceiling = price
        notes.append(f"Reward/risk at the analysis price is {rr_at_price:.2f}, below the "
                     f"{min_rr:g} floor. Treat this buy as not valid.")
    elif ceiling < price + up:
        notes.append("Ceiling is capped by the reward/risk floor, not by the drift allowance.")

    cap = max_cost if cash_left is None else min(max_cost, cash_left)
    shares = int(cap // ceiling) if ceiling > 0 else 0
    if shares < 1:
        valid = False
        notes.append(f"One share at the ceiling (${ceiling:,.2f}) exceeds the ${cap:,.2f} cap.")

    return BuyBand(
        valid=valid, floor=round(floor, 2), ceiling=round(ceiling, 2),
        buy_limit=round(ceiling, 2), shares=shares,
        max_cost=round(shares * ceiling, 2), rr_at_price=round(rr_at_price, 2),
        unprotected_days=days,
        stop_after_fill=("After the broker confirms the purchase is final: "
                         f"stop = fill price - ${stop_distance:,.2f} "
                         "(raised if the stock has risen)."),
        notes=notes)


@dataclass
class StopCheck:
    verdict: str                # "PLACE" | "PLACE_WARN" | "STOP_BREACHED"
    stop_trigger: Optional[float]
    limit_price: Optional[float]
    cushion: float              # live price - stop trigger (dollars)
    message: str


def stop_check(fill_price: float, current_price: float, stop_distance: float,
               limit_buffer_pct: float = DEFAULT_LIMIT_BUFFER_PCT,
               cushion_frac: float = DEFAULT_CUSHION_FRAC) -> StopCheck:
    """Order 2: the protective stop-limit, checked against the actual fill and the
    current price. Anchored to the fill price and trailed up if the stock has risen.

    The stop is never silently placed at a worse level: if the current price is at
    or below the intended stop the verdict is STOP_BREACHED and you must decide.
    """
    if fill_price <= 0 or current_price <= 0 or stop_distance <= 0:
        raise ValueError("fill_price, current_price and stop_distance must be positive")

    intended = fill_price - stop_distance
    if current_price <= intended:
        return StopCheck(
            "STOP_BREACHED", None, None, round(current_price - intended, 2),
            f"Current ${current_price:,.2f} is at or below the intended stop "
            f"${intended:,.2f}. A sell-stop must sit below the market, so it cannot be "
            f"placed as planned. Decide: sell now, or re-run the analysis.")

    anchor = max(fill_price, current_price)          # trail up, never down
    stop = round(anchor - stop_distance, 2)
    limit = round(stop * (1 - limit_buffer_pct / 100.0), 2)
    cushion = round(current_price - stop, 2)

    msg = (f"Place stop-limit: trigger ${stop:,.2f}, limit ${limit:,.2f}. "
           f"Cushion above the stop is ${cushion:,.2f}.")
    if anchor > fill_price:
        msg += f" Stop raised from ${intended:,.2f} to lock in part of the gain."
    verdict = "PLACE"
    if cushion < cushion_frac * stop_distance:
        verdict = "PLACE_WARN"
        msg += " Cushion is thin; the stop may trigger on ordinary noise."
    msg += (" If the stock gaps below the trigger, a stop-limit may not fill "
            "(the price can skip past the limit).")
    return StopCheck(verdict, stop, limit, cushion, msg)
