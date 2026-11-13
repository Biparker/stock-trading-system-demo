#!/usr/bin/env python3
"""
generate_demo_data.py - build the SAMPLE data the demo dashboard reads.

Everything here is synthetic: fictional companies and tickers, seeded random
price paths, and a simplified version of the real pipeline's rules. Nothing is
real market data and nothing is a recommendation.

Writes (same layout the real system uses):
    data/pipeline_output.json
    data/positions.json
    forecasts/<TICKER>_forecast_report.json
    Analysis_Outcomes/<TICKER>_stage2_backtest.json

Usage:
    python generate_demo_data.py                  # as of today
    python generate_demo_data.py --as-of 2026-11-13
Same --as-of + same SEED => identical output.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FORECASTS_DIR = ROOT / "forecasts"
OUTCOMES_DIR = ROOT / "Analysis_Outcomes"

SEED = 2026
BUDGET = 10_000.0          # sample account size
MIN_RR = 1.25              # minimum reward/risk to buy
HORIZON = 90               # forecast / Monte Carlo days
HIST_DAYS = 502            # ~2 years of trading days
N_SIM = 1000
MULTS = [1.5, 2.0, 2.5, 3.0]   # ATR multiples tested in the backtest

# Fictional universe. role is only a design target; the rules below decide the
# real outcome and main() asserts the two agree.
UNIVERSE = [
    dict(ticker="NVLX", name="Novalux Semiconductor Corp.", sector="Technology",
         price=138.40, vol=0.38, drift=0.22, ret=18.5, sent=66.2, analyst="buy",
         risk="medium", best=2.5, q=1.45, role="buy",
         risks=["Frequent negative terms: supply constraints, impairment"]),
    dict(ticker="HLXT", name="Helixa Therapeutics Inc.", sector="Healthcare",
         price=84.20, vol=0.27, drift=0.12, ret=12.4, sent=61.5, analyst="buy",
         risk="low", best=2.5, q=1.20, role="buy",
         risks=["Frequent negative terms: litigation, recall"]),
    dict(ticker="TRSP", name="Transpora Logistics Group", sector="Industrials",
         price=57.85, vol=0.24, drift=0.09, ret=9.2, sent=57.0, analyst="buy",
         risk="low", best=2.0, q=1.05, role="buy",
         risks=["Frequent negative terms: decreased, delays"]),
    dict(ticker="CRSN", name="Cresson Home Brands", sector="Consumer Discretionary",
         price=46.75, vol=0.33, drift=0.10, ret=14.1, sent=54.3, analyst="buy",
         risk="medium", best=3.0, q=0.95, role="buy",
         risks=["Frequent negative terms: restructuring, markdowns", "Elevated inventory levels"]),
    dict(ticker="PRMR", name="Primera Energy Partners", sector="Energy",
         price=41.15, vol=0.31, drift=0.07, ret=6.0, sent=52.0, analyst="hold",
         risk="medium", best=2.5, q=0.85, role="hold", held=(30, 38.90),
         risks=["Frequent negative terms: write-down, volatility"]),
    dict(ticker="FNBK", name="Fennimore Bancorp", sector="Financials",
         price=49.60, vol=0.29, drift=-0.06, ret=-9.8, sent=38.4, analyst="sell",
         risk="high", best=3.0, q=0.18, role="sell", held=(25, 52.10),
         risks=["Frequent negative terms: loan losses, impairment, downgrade",
                "Rising credit provisions"]),
    dict(ticker="VRNT", name="Verant Networks Inc.", sector="Technology",
         price=96.10, vol=0.34, drift=0.05, ret=4.1, sent=55.9, analyst="buy",
         risk="medium", best=3.0, q=0.90, role="monitor",
         risks=["Frequent negative terms: competition, churn"]),
    dict(ticker="STFD", name="Stellfield Foods Co.", sector="Consumer Staples",
         price=72.30, vol=0.18, drift=0.02, ret=-3.4, sent=49.2, analyst="hold",
         risk="low", best=2.0, q=0.80, role="monitor",
         risks=["Frequent negative terms: input costs, decreased"]),
]


# ── helpers ─────────────────────────────────────────────────────────────────
def r2(x, n=2):
    return round(float(x), n)


def trend_label(ret: float) -> tuple[str, str]:
    if ret >= 15:  return "strongly_bullish", "strong"
    if ret >= 5:   return "bullish", "moderate"
    if ret > -5:   return "neutral", "weak"
    if ret > -15:  return "bearish", "moderate"
    return "strongly_bearish", "strong"


def ny_iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts.date(), tz="America/New_York").isoformat()


def sentiment_label(score: float) -> str:
    if score >= 65: return "bullish"
    if score >= 55: return "slightly_bullish"
    if score >= 45: return "neutral"
    return "bearish"


# ── per-ticker builders ─────────────────────────────────────────────────────
def build_history(u: dict, rng: np.random.Generator, last_bday: pd.Timestamp):
    sig_d = u["vol"] / math.sqrt(252)
    rets = rng.normal(u["drift"] / 252 - 0.5 * sig_d ** 2, sig_d, HIST_DAYS)
    path = np.exp(np.cumsum(rets))
    path = path / path[-1] * u["price"]          # end exactly at the current price
    dates = pd.bdate_range(end=last_bday, periods=HIST_DAYS)
    split_at = int(HIST_DAYS * 0.8)
    hist = [{"date": d.strftime("%Y-%m-%d"), "close": r2(p),
             "split": "train" if i < split_at else "val"}
            for i, (d, p) in enumerate(zip(dates, path))]
    return hist, path, dates, sig_d


def build_forecast(u: dict, rng, last_bday: pd.Timestamp, sig_d: float):
    p0, ret = u["price"], u["ret"] / 100
    fdates = pd.bdate_range(start=last_bday + pd.offsets.BDay(1), periods=HORIZON)
    t = np.arange(1, HORIZON + 1)
    s = t / HORIZON
    phase = rng.uniform(0, 6.28)
    mid = p0 * (1 + ret * s + 0.015 * np.sin(t / 6.0 + phase) * (1 - s))
    half = p0 * 1.96 * sig_d * np.sqrt(t) * 0.45
    return fdates, mid, mid - half, mid + half


def build_forecast_report(u, rng, last_bday):
    hist, path, dates, sig_d = build_history(u, rng, last_bday)
    fdates, mid, lo, hi = build_forecast(u, rng, last_bday, sig_d)
    p0 = u["price"]

    x = np.arange(HIST_DAYS)
    slope, _ = np.polyfit(x, path, 1)
    r_sq = float(np.corrcoef(x, path)[0, 1] ** 2)
    strength = "Weak" if r_sq < 0.1 else "Moderate" if r_sq < 0.4 else "Strong"
    vol_level = "Low" if sig_d < 0.016 else "Medium" if sig_d < 0.024 else "High"

    rmse = p0 * sig_d * 1.45
    mae = rmse * 0.75
    report = {
        "metadata": {
            "ticker": u["ticker"], "company_name": u["name"],
            "analysis_start_date": ny_iso(dates[0]), "analysis_end_date": ny_iso(dates[-1]),
            "forecast_start_date": ny_iso(fdates[0]), "forecast_end_date": ny_iso(fdates[-1]),
            "historical_records": HIST_DAYS, "forecast_periods": HORIZON,
            "note": "SAMPLE DATA - fictional company, synthetic prices",
        },
        "current_metrics": {
            "current_price": u["price"], "start_price": r2(path[0]),
            "min_price": r2(path.min()), "max_price": r2(path.max()),
            "avg_price": r2(path.mean()),
            "price_change_pct": r2((path[-1] / path[0] - 1) * 100),
            "volatility_pct": r2(sig_d * 100),
        },
        "data_characteristics": {
            "stationarity": {"is_stationary": "False", "adf_statistic": -2.1, "p_value": 0.24,
                             "critical_values": {"1%": -3.44, "5%": -2.87, "10%": -2.57},
                             "interpretation": "Non-stationary (differencing needed)"},
            "trend": {"slope": float(slope), "r_squared": r_sq, "strength": strength,
                      "direction": "Uptrend" if slope > 0 else "Downtrend"},
            "seasonality": {"has_seasonality": False, "seasonal_strength": 0, "period": None,
                            "interpretation": "No significant seasonality detected"},
            "volatility": {"volatility": r2(sig_d * 100), "coefficient_of_variation": 0.0002,
                           "level": vol_level},
        },
        "selected_method": "prophet",
        "price_history": hist,
        "forecast_results": [
            {"date": ny_iso(d), "forecast": r2(m), "lower_bound": r2(l), "upper_bound": r2(h)}
            for d, m, l, h in zip(fdates, mid, lo, hi)
        ],
        "forecast_summary": {"mean": r2(mid.mean()), "min": r2(mid.min()), "max": r2(mid.max()),
                             "change_pct": r2((mid[-1] / p0 - 1) * 100)},
        "model_metrics": {"rmse": r2(rmse, 4), "mae": r2(mae, 4),
                          "mape": r2(mae / p0 * 100, 4),
                          "residual_std": r2(rmse, 4),
                          "direction_accuracy": r2(rng.uniform(0.52, 0.62), 3)},
        "stale_forecast": False,
    }
    return report, float(mid[-1])


def build_backtest(u, rng):
    sig_d = u["vol"] / math.sqrt(252)
    atr_pct = sig_d * 100 * 1.1
    price, q, best = u["price"], u["q"], u["best"]
    results, pcts = {}, {}
    for m in MULTS:
        pct = float(np.clip(m * atr_pct, 2.0, 12.0))
        label = f"{m:.1f}xATR ({pct:.2f}%)"
        if m == best:
            sh = q
        else:
            sh = q * (1 - 0.18 * abs(m - best) / 0.5) - 0.05 - abs(rng.normal(0, 0.02))
        mean = sh * 11 + rng.normal(0, 0.4)
        results[label] = {
            "sharpe": r2(sh, 4), "mean_ret_pct": r2(mean), "median_ret_pct": r2(mean * 0.92),
            "p_loss_gt5": r2(max(0.1, (14 - 4 * sh) * (1 - 0.12 * (m - 1.5)) / 4), 2),
            "p_profit": r2(min(85, 50 + 13 * sh), 1),
            "avg_stops_yr": r2(u["vol"] * 30 / m, 1),
        }
        pcts[label] = pct
    rec_label = next(k for k in results if k.startswith(f"{best:.1f}x"))
    rec = results[rec_label]
    stop_pct = pcts[rec_label]
    passed = rec["sharpe"] >= 0.5
    raw = {
        "ticker": u["ticker"], "methodology": "ATR-based trailing stop + Prophet drift signal (SAMPLE DATA)",
        "atr14": r2(atr_pct / 100 * price), "atr_pct_of_price": r2(atr_pct),
        "parkinson_sigma_pct": r2(sig_d * 100), "stage2_pass": passed,
        "recommended_stop": rec_label, "results": results,
    }
    stop_dist = stop_pct / 100 * price
    max_pos = min(40.0, 100.0 / stop_pct)
    signal = {
        "ticker": u["ticker"], "stage2_pass": passed,
        "failure_reason": None if passed else
            f"Sharpe at best stop ({rec['sharpe']:.2f}) is below the 0.50 minimum",
        "recommended_stop_label": rec_label, "recommended_stop_pct": r2(stop_pct, 4),
        "sharpe_at_recommended": rec["sharpe"], "mean_annual_return_pct": rec["mean_ret_pct"],
        "p_loss_gt5_pct": rec["p_loss_gt5"], "avg_stops_per_year": rec["avg_stops_yr"],
        "is_high_volatility": u["vol"] > 0.45, "stop_dollar_amount": r2(stop_dist, 4),
        "max_position_pct": r2(max_pos),
        "portfolio_action": "include_with_stop" if passed else "exclude",
        "position_size_advice": (
            f"Stop distance is ${stop_dist:.2f}/share (risk per share). With a 1% risk budget on a "
            f"${BUDGET:,.0f} account (${BUDGET * 0.01:,.0f} at risk), position size is capped at "
            f"{max_pos:.1f}% of the account (${max_pos / 100 * BUDGET:,.0f})."),
    }
    return raw, signal, stop_pct, stop_dist, max_pos


# ── Monte Carlo ─────────────────────────────────────────────────────────────
def _stats(final_ret_pct: np.ndarray, dd_pct: np.ndarray) -> dict:
    mean, std = final_ret_pct.mean(), final_ret_pct.std()
    return {
        "mean_return_pct": float(mean), "median_return_pct": float(np.median(final_ret_pct)),
        "std_return_pct": float(std), "min_return_pct": float(final_ret_pct.min()),
        "max_return_pct": float(final_ret_pct.max()),
        "percentile_5_pct": float(np.percentile(final_ret_pct, 5)),
        "percentile_25_pct": float(np.percentile(final_ret_pct, 25)),
        "percentile_75_pct": float(np.percentile(final_ret_pct, 75)),
        "percentile_95_pct": float(np.percentile(final_ret_pct, 95)),
        "win_rate_pct": r2((final_ret_pct > 0).mean() * 100, 1),
        "max_drawdown_pct": float(dd_pct.mean()),
        "sharpe_ratio": float(mean / std * math.sqrt(252 / HORIZON)) if std else 0.0,
    }


def _confidence(win_rate: float, sharpe: float) -> int:
    return int(np.clip(round(win_rate * 0.8 + sharpe * 8), 5, 95))


def _paths(ret_pct: float, sig_d: float, z: np.ndarray) -> np.ndarray:
    """Cumulative value paths (N x HORIZON), starting at 1.0. Drift is set so the
    mean final value equals the forecast."""
    m = (math.log(1 + ret_pct / 100) - 0.5 * sig_d ** 2 * HORIZON) / HORIZON
    return np.exp(np.cumsum(m + sig_d * z, axis=1))


def _drawdown(cum: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(np.concatenate([np.ones((cum.shape[0], 1)), cum], axis=1), axis=1)[:, 1:]
    return (cum / peak - 1).min(axis=1) * 100


def _stop_hit(cum: np.ndarray, stop_pct: float) -> np.ndarray:
    peak = np.maximum.accumulate(np.maximum(cum, 1.0), axis=1)
    return (cum <= peak * (1 - stop_pct / 100)).any(axis=1)


def mc_single(u, stop_pct, rng) -> dict:
    sig_d = u["vol"] / math.sqrt(252)
    cum = _paths(u["ret"], sig_d, rng.standard_normal((N_SIM, HORIZON)))
    final = (cum[:, -1] - 1) * 100
    res = _stats(final, _drawdown(cum))
    res["portfolio_stop_hits_pct"] = r2(_stop_hit(cum, stop_pct).mean() * 100, 1)
    res["prob_loss_gt_5pct"] = r2((final < -5).mean() * 100, 1)
    conf = _confidence(res["win_rate_pct"], res["sharpe_ratio"])
    res["returns_pct"] = [r2(x) for x in final]
    return {
        "forecast": {"expected_change_pct": u["ret"]},
        "volatility": {"daily_sigma_pct": sig_d, "annualized_sigma_pct": u["vol"]},
        "strategy": {"atr_stop_pct": r2(stop_pct)},
        "results": res, "confidence_score": conf,
        "recommendation": "bullish" if conf >= 60 else "neutral" if conf >= 40 else "bearish",
    }


def mc_portfolio(members, rng):
    """members: list of dicts with u, weight, stop_pct. Correlated GBM via Cholesky."""
    n = len(members)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            same = members[i]["u"]["sector"] == members[j]["u"]["sector"]
            corr[i, j] = corr[j, i] = 0.55 if same else 0.15 + 0.02 * ((i + j) % 3)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((N_SIM, HORIZON, n)) @ chol.T
    w = np.array([m["weight"] for m in members])
    cums = [_paths(m["u"]["ret"], m["u"]["vol"] / math.sqrt(252), z[:, :, k])
            for k, m in enumerate(members)]
    port = sum(w[k] * cums[k] for k in range(n))
    final = (port[:, -1] - 1) * 100
    res = _stats(final, _drawdown(port))
    any_stop = np.zeros(N_SIM, dtype=bool)
    for k, m in enumerate(members):
        any_stop |= _stop_hit(cums[k], m["stop_pct"])
    res["portfolio_stop_hits_pct"] = r2(any_stop.mean() * 100, 1)
    res["returns_pct"] = [r2(x) for x in final]
    conf = _confidence(res["win_rate_pct"], res["sharpe_ratio"])
    rec = ("STRONG HOLD — Well-balanced portfolio" if conf >= 70 else
           "HOLD — Acceptable risk profile" if conf >= 45 else
           "CAUTION — Elevated risk; review position sizes")
    iu = np.triu_indices(n, 1)
    pairs = [(members[i]["u"]["ticker"], members[j]["u"]["ticker"], r2(corr[i, j]))
             for i, j in zip(*iu) if corr[i, j] > 0.7]
    return res, conf, rec, corr[iu], pairs


# ── main ────────────────────────────────────────────────────────────────────
def main(as_of: date) -> None:
    for d in (DATA_DIR, FORECASTS_DIR, OUTCOMES_DIR):
        d.mkdir(exist_ok=True)
    last_bday = pd.offsets.BDay().rollback(pd.Timestamp(as_of))
    as_of_s = as_of.isoformat()

    held = {u["ticker"]: u["held"] for u in UNIVERSE if "held" in u}
    cash = BUDGET - sum(sh * cb for sh, cb in held.values())

    info = {}
    for i, u in enumerate(UNIVERSE):
        rng = np.random.default_rng(SEED + i)
        report, target = build_forecast_report(u, rng, last_bday)
        raw, bt, stop_pct, stop_dist, max_pos = build_backtest(u, rng)
        t = u["ticker"]
        (FORECASTS_DIR / f"{t}_forecast_report.json").write_text(json.dumps(report), encoding="utf-8")
        (OUTCOMES_DIR / f"{t}_stage2_backtest.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")

        trend, strength = trend_label(u["ret"])
        target = r2(target)
        fc = {"ticker": t, "sector": u["sector"], "forecast_trend": trend,
              "current_price": u["price"], "target_price": target,
              "predicted_return_pct": u["ret"], "model_confidence": 0.6,
              "trend_strength": strength,
              "recommendation": "exclude" if u["ret"] <= -5 else "include",
              "reasoning": f"Forecast return {u['ret']:+.1f}% is {trend.replace('_', ' ')} "
                           f"with model confidence 0.60."}
        sc = u["sent"]
        se = {"ticker": t, "sentiment_score": sc, "sentiment_label": sentiment_label(sc),
              "analyst_rating": u["analyst"], "confidence": 1.0, "risk_level": u["risk"],
              "position_multiplier": 0.5 if u["risk"] == "high" else 1.0,
              "key_risks": u["risks"],
              "recommendation": "exclude" if sc < 40 else "include"}
        f_comp = float(np.clip(50 + u["ret"] * 1.6, 0, 100))
        b_comp = float(np.clip(bt["sharpe_at_recommended"] * 55 + 10, 0, 100))
        score = r2(0.4 * f_comp + 0.3 * sc + 0.3 * b_comp)
        info[t] = dict(u=u, fc=fc, se=se, bt=bt, stop_pct=stop_pct, stop_dist=stop_dist,
                       max_pos=max_pos, score=score, target=target, idx=i)

    # decisions
    order = sorted(info, key=lambda k: -info[k]["score"])
    actions, reasons, buy_plan = {}, {}, {}
    spend_cash = cash
    for t in order:
        x = info[t]; u = x["u"]
        rr = (x["target"] - u["price"]) / x["stop_dist"]
        x["rr"] = rr
        if t in held:
            actions[t] = "sell" if (u["ret"] <= -5 or not x["bt"]["stage2_pass"]) else "hold"
        elif x["target"] <= u["price"]:
            actions[t] = "monitor"
            reasons[t] = f"90-day forecast target ${x['target']:.2f} is not above the price ${u['price']:.2f}"
        elif rr < MIN_RR:
            actions[t] = "monitor"
            reasons[t] = (f"reward/risk {rr:.2f} is below the {MIN_RR} minimum "
                          f"(target ${x['target']:.2f}, price ${u['price']:.2f}, "
                          f"stop distance ${x['stop_dist']:.2f})")
        elif not x["bt"]["stage2_pass"] or x["fc"]["recommendation"] != "include":
            actions[t] = "monitor"; reasons[t] = "forecast or backtest rule not met"
        else:
            shares = int(min(x["max_pos"] / 100 * BUDGET, spend_cash) // u["price"])
            if shares < 1:
                actions[t] = "monitor"; reasons[t] = "not enough cash for one share"
                continue
            cost = shares * u["price"]
            spend_cash -= cost
            actions[t] = "buy"
            buy_plan[t] = {"shares": shares, "price": u["price"], "cost": r2(cost),
                           "stop_price": r2(u["price"] - x["stop_dist"]),
                           "stop_distance": r2(x["stop_dist"]), "target_price": x["target"],
                           "max_cost": r2(x["max_pos"] / 100 * BUDGET)}
    for t, x in info.items():
        assert actions[t] == x["u"]["role"], f"{t}: designed role {x['u']['role']} but rules gave {actions[t]}"

    buys = [t for t in order if actions[t] == "buy"]
    holds = [t for t in order if actions[t] == "hold"]
    sells = [t for t in order if actions[t] == "sell"]
    mons = [t for t in order if actions[t] == "monitor"]

    # Monte Carlo
    mc = {t: mc_single(info[t]["u"], info[t]["stop_pct"], np.random.default_rng(SEED + 100 + info[t]["idx"]))
          for t in order}

    # Portfolio (planned buys + holds)
    members_t = buys + holds
    dollars = {t: (buy_plan[t]["cost"] if t in buy_plan else held[t][0] * info[t]["u"]["price"])
               for t in members_t}
    total = sum(dollars.values())
    weights = {t: dollars[t] / total for t in members_t}
    members = [dict(u=info[t]["u"], weight=weights[t], stop_pct=info[t]["stop_pct"]) for t in members_t]
    pres, pconf, prec, corr_vals, hi_pairs = mc_portfolio(members, np.random.default_rng(SEED + 999))

    sectors: dict[str, int] = {}
    for t in members_t:
        sectors[info[t]["u"]["sector"]] = sectors.get(info[t]["u"]["sector"], 0) + 1
    n = len(members_t)
    max_sec = max(sectors.values()) / n * 100
    sector_issues = [f"{s} is {c / n * 100:.0f}% of the portfolio (limit 40%)"
                     for s, c in sectors.items() if c / n * 100 > 40]
    corr_issues = [f"{a} and {b} are highly correlated ({c})" for a, b, c in hi_pairs]
    bal_issues = [f"{t} is {w * 100:.0f}% of the portfolio (limit 40%)" for t, w in weights.items() if w > 0.40]
    issues = sector_issues + corr_issues + bal_issues
    rating = "WELL_DIVERSIFIED" if not issues else "ACCEPTABLE" if len(issues) == 1 else "POORLY_DIVERSIFIED"
    div = {
        "analysis_date": datetime.now().isoformat(), "portfolio_size": n, "overall_rating": rating,
        "sector_diversity": {"sectors": sectors, "n_sectors": len(sectors),
                             "max_sector_pct": r2(max_sec), "is_diverse": not sector_issues,
                             "issues": sector_issues},
        "correlation_analysis": {"avg_correlation": float(corr_vals.mean()),
                                 "max_correlation": float(corr_vals.max()),
                                 "high_correlation_pairs": hi_pairs, "is_diverse": not corr_issues,
                                 "issues": corr_issues},
        "position_balance": {"weights_pct": {t: r2(w * 100, 1) for t, w in weights.items()},
                             "max_weight_pct": r2(max(weights.values()) * 100, 1),
                             "issues": bal_issues},
        "portfolio_size_check": {"portfolio_size": n,
                                 "status": "OPTIMAL" if 4 <= n <= 7 else "OUTSIDE_RANGE", "issues": []},
        "all_issues": issues,
        "recommendation": ("Excellent diversification — portfolio is well-balanced" if not issues
                           else "Review the issues above and consider rebalancing"),
    }

    # event alerts (relative to as_of)
    bd = pd.offsets.BDay()
    ex_d = (last_bday + 2 * bd); er_d = (last_bday + 3 * bd)
    alerts = [
        {"ticker": "TRSP", "event_type": "ex_dividend", "event_date": ex_d.strftime("%Y-%m-%d"),
         "days_until_event": 2, "position_context": "buy", "severity": "critical",
         "recommendation": "Ex-dividend in 2 days: the price usually drops by the dividend amount. "
                           "Consider buying after that date."},
        {"ticker": "HLXT", "event_type": "earnings", "event_date": er_d.strftime("%Y-%m-%d"),
         "days_until_event": 3, "position_context": "buy", "severity": "warning",
         "recommendation": "Earnings in 3 days: expect a large move. Consider waiting for the report."},
    ]

    planned = sum(p["cost"] for p in buy_plan.values())
    held_txt = ", ".join(f"{t} ({held[t][0]} sh)" for t in held)
    buy_txt = "; ".join(f"{t} ({p['shares']} sh @ ${p['price']:.2f} = ${p['cost']:,.2f}, stop ${p['stop_price']:.2f})"
                        for t, p in buy_plan.items())
    prompt = (
        f"[SAMPLE DATA] Recommended buys, in score order: {buy_txt}.\n"
        f"Planned spend ${planned:,.2f} of ${cash:,.2f} cash; ${cash - planned:,.2f} would remain. "
        f"Nothing is bought until you place the orders at your broker and record them under Record Fills.\n"
        f"You currently hold: {held_txt}. Sell signal: {', '.join(sells) or 'none'}.\n"
        f"Not bought:\n" + "\n".join(f"  - {t} (score {info[t]['score']:.1f}): {reasons[t]}." for t in mons)
        + "\n\n**Data Currency:**\n\n| Ticker | Prophet 90-day Forecast | Sentiment Report | Backtest |\n"
          "|--------|------------------------|------------------|----------|\n"
        + "\n".join(f"| {t} | 0 days | {1 if info[t]['idx'] % 3 == 0 else 0} day(s) | 0 days |" for t in order)
        + "\n\n⚠️ EVENT CALENDAR ALERTS (next 3 days):\n"
        + "\n".join(f"  • {a['ticker']} [{a['severity'].upper()}]: {a['event_type'].replace('_', ' ')} "
                    f"on {a['event_date']} ({a['days_until_event']}d)" for a in alerts)
    )

    out = {
        "date": as_of_s, "mode": "demo", "budget": BUDGET, "all_tickers": order,
        "forecast": {t: info[t]["fc"] for t in order},
        "sentiment": {t: info[t]["se"] for t in order},
        "backtest": {t: info[t]["bt"] for t in order},
        "monte_carlo": mc,
        "portfolio_analysis": {
            "monte_carlo": {"tickers": members_t, "weights": weights, "portfolio_size": n,
                            "n_simulations": N_SIM, "days": HORIZON,
                            "analysis_date": datetime.now().isoformat(),
                            "results": pres, "confidence_score": pconf, "recommendation": prec},
            "diversification": div},
        "actions_by_ticker": actions,
        "signal_dates": {t: {"forecast_date": as_of_s, "sentiment_date": as_of_s, "backtest_date": as_of_s}
                         for t in order},
        "data_ages": {t: {"forecast_days": 0, "sentiment_days": 1 if info[t]["idx"] % 3 == 0 else 0,
                          "backtest_days": 0} for t in order},
        "buy_plan": buy_plan, "monitor_reasons": reasons, "min_positions_warning": None,
        "cash_remaining": r2(cash),
        "tickers_to_buy": buys, "tickers_to_sell": sells, "tickers_to_hold": holds,
        "tickers_to_monitor": mons,
        "stop_loss_orders": {t: p["stop_price"] for t, p in buy_plan.items()},
        "combined_scores": {t: info[t]["score"] for t in order},
        "buy_trigger_prices": {t: info[t]["target"] for t in mons},
        "user_prompt": prompt, "event_alerts": alerts, "event_guard_window": 3,
    }
    (DATA_DIR / "pipeline_output.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    positions = {
        "as_of": as_of_s, "cash_available": r2(cash),
        "open_positions": {t: {"shares": float(sh), "cost_basis": cb,
                               "stop": r2(cb * (1 - info[t]["stop_pct"] / 100)),
                               "added_date": (last_bday - 20 * bd).strftime("%Y-%m-%d")}
                           for t, (sh, cb) in held.items()},
        "stop_loss_orders": {t: r2(cb * (1 - info[t]["stop_pct"] / 100)) for t, (sh, cb) in held.items()},
        "fills_log": [],
    }
    (DATA_DIR / "positions.json").write_text(json.dumps(positions, indent=2), encoding="utf-8")

    print(f"Wrote sample data as of {as_of_s} for {len(order)} fictional tickers")
    print(f"  buy {buys} | hold {holds} | sell {sells} | monitor {mons}")
    print(f"  cash ${cash:,.2f}, planned spend ${planned:,.2f}")
    print(f"  portfolio: {rating}, confidence {pconf}, mean {pres['mean_return_pct']:+.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                    help="pipeline date, YYYY-MM-DD (default: today)")
    main(ap.parse_args().as_of)
