"""
Streamlit dashboard — Stock Trading System DEMO (sample data)
=============================================================
Run locally:
    streamlit run dashboard/app.py

All data is synthetic (fictional companies and tickers). Regenerate it with:
    python generate_demo_data.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Ensure stock-trading-system/ is on the path so data_loader / charts import cleanly
sys.path.insert(0, str(Path(__file__).parent))   # dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))  # stock-trading-system/

import streamlit as st
import pandas as pd

from data_loader import (
    load_pipeline_output,
    load_forecast_series,
    load_forecast_metadata,
    load_backtest_raw,
    get_action_badge,
)
from charts import (
    forecast_chart,
    sharpe_bar_chart,
    mean_return_bar_chart,
    portfolio_performance_chart,
    risk_reward_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trading Dashboard (Demo)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clear cache on every run to prevent stale data
st.cache_data.clear()

# ── Load pipeline data ────────────────────────────────────────────────────────
data = load_pipeline_output()

import update_cash as _uc

_SEED_POSITIONS = _uc.DEFAULT_FILE   # data/positions.json: the pristine sample holdings


def _session_positions_file(reset: bool = False) -> Path:
    """Each browser session edits its own temp copy of positions.json, so one
    viewer's Record Fills never shows up for another viewer."""
    f = st.session_state.get("_positions_file")
    if reset or not f or not Path(f).exists():
        folder = Path(tempfile.mkdtemp(prefix="demo_positions_"))
        f = str(folder / "positions.json")
        shutil.copy(_SEED_POSITIONS, f)
        st.session_state["_positions_file"] = f
    return Path(f)


_POS_FILE = _session_positions_file()

# Live cash comes from this session's positions file (updated by Record Fills)
try:
    _live_cash = _uc.load(_POS_FILE)["cash_available"]
except Exception:
    _live_cash = data.get("cash_remaining", 0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Trading Dashboard")
    st.caption("B. I. Parker Data Science")
    st.divider()

    mode_label = {"mock": "🟡 Mock (offline)", "live": "🟢 Live", "demo": "🧪 Demo (sample data)",
                  "no_data": "🔴 No data yet"}.get(data.get("mode", ""), "⚪ Unknown")
    st.metric("Pipeline Mode", mode_label)
    st.metric("Last Run",      data.get("date", "—"))
    st.metric("Budget",        f"${data.get('budget', 2500):,.0f}")
    st.metric("Cash Remaining",f"${_live_cash:,.2f}")
    st.divider()

    all_tickers = data.get("all_tickers", ["AMGN"])
    selected_ticker = st.selectbox("🔍 Ticker Detail View", all_tickers)
    st.divider()

    # Quick actions list
    st.subheader("Today's Actions")
    for t in data.get("tickers_to_buy", []):
        st.success(f"BUY  {t}")
    for t in data.get("tickers_to_sell", []):
        st.error(f"SELL  {t}")
    for t in data.get("tickers_to_hold", []):
        st.warning(f"HOLD  {t}")
    for t in data.get("tickers_to_monitor", []):
        st.info(f"WATCH  {t}")

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    if st.button("♻️ Reset demo holdings"):
        _session_positions_file(reset=True)
        st.rerun()

    if data.get("mode") == "no_data":
        st.error(
            "No pipeline data found.\n\n"
            "Run this command in your terminal:\n\n"
            "```\npython generate_demo_data.py\n```"
        )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Stock Trading System Dashboard")
st.caption(
    f"Daily Briefing — {data.get('date', '—')}  •  "
    f"B. I. Parker Data Science and Consulting LLC  •  "
    f"Pipeline: Stage 1 (Screen) → 2 (Forecast) → 3 (Sentiment) → 4 (Backtest) → 5 (Advise)"
)

st.info(
    "🧪 **DEMO — sample data.** Companies, tickers, prices, forecasts and results are "
    "fictional and randomly generated. Nothing here is real market data or investment advice."
)

# ── CRITICAL: Broker Account Monitoring Reminder ─────────────────────────────
st.warning(
    "🔔 **CRITICAL: Check Your Broker Account Daily**\n\n"
    "Trailing stop loss orders execute automatically at the broker. "
    "You must check your broker account DAILY to confirm if any positions were closed. "
    "When a stop triggers:\n"
    "1. Position closes automatically\n"
    "2. Cash is freed up\n"
    "3. Select new candidate stocks to maintain 4-7 positions\n"
    "4. Update tickers list and run pipeline when ready\n\n"
    "**Without daily broker monitoring, you may miss exits and fail to redeploy freed capital.**"
)
st.divider()

# ── Row 1: MeLLeA Daily Advisor Briefing ─────────────────────────────────────
st.subheader("🤖 MeLLeA Daily Advisor")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Positions to Buy",  len(data.get("tickers_to_buy",  [])))
col_b.metric("Positions Held",    len(data.get("tickers_to_hold", [])))
col_c.metric("Sell Signals",      len(data.get("tickers_to_sell", [])))
col_d.metric("Cash Remaining",    f"${_live_cash:,.2f}",
             delta=f"of ${data.get('budget', 2500):.0f} budget")

user_prompt = data.get("user_prompt", "")
if user_prompt:
    st.info(f"**Daily Briefing:** {user_prompt}")
st.divider()

# ── Event Calendar Alerts ─────────────────────────────────────────────────────
event_alerts = data.get("event_alerts", [])
if event_alerts:
    st.subheader("⚠️ Upcoming Events (next " + str(data.get("event_guard_window", 10)) + " days)")
    for alert in event_alerts:
        ticker   = alert.get("ticker", "")
        etype    = alert.get("event_type", "").replace("_", " ").title()
        edate    = alert.get("event_date", "")
        days     = alert.get("days_until_event", 0)
        sev      = alert.get("severity", "info")
        ctx      = alert.get("position_context", "").title()
        rec      = alert.get("recommendation", "")
        msg      = f"**{ticker}** [{ctx}] — {etype} on {edate} ({days} day{'s' if days != 1 else ''})  \n{rec}"
        if sev == "critical":
            st.error(msg)
        elif sev == "warning":
            st.warning(msg)
        else:
            st.info(msg)
    st.divider()

# ── Row 2: Portfolio Combined Scores ──────────────────────────────────────────
st.subheader("📊 MeLLeA Combined Scores — All Candidates")
combined_scores = data.get("combined_scores", {})
if combined_scores:
    st.plotly_chart(
        portfolio_performance_chart(combined_scores, data.get("actions_by_ticker", {})),
        width='stretch'
    )

    st.caption("The score only **ranks** candidates (who gets cash first). A stock is bought when "
               "its forecast rule says include, its backtest passes, and no real sentiment "
               "veto applies. Sized by risk and limited by cash.")

    _buy_plan = data.get("buy_plan", {}) or {}
    _blockers = data.get("monitor_reasons", {}) or {}

    def _decision_detail(t):
        if t in _buy_plan:
            p = _buy_plan[t]
            return (f"Buy {p['shares']} sh @ ${p['price']:.2f} = ${p['cost']:,.2f}; "
                    f"stop ${p['stop_price']:.2f}")
        return _blockers.get(t, "—")

    # Score table
    score_rows = []
    for t, score in combined_scores.items():
        action = data.get("actions_by_ticker", {}).get(t, "monitor")
        label, _ = get_action_badge(action)
        bt = data.get("backtest", {}).get(t, {})
        fc = data.get("forecast", {}).get(t, {})
        se = data.get("sentiment", {}).get(t, {})
        score_rows.append({
            "Ticker":        t,
            "Sector":        fc.get("sector", "—"),
            "Action":        label,
            "Decision detail": _decision_detail(t),
            "Rank score":    f"{score:.1f}",
            "Forecast Trend":fc.get("forecast_trend", "—"),
            "Forecast Ret%": f"{fc.get('predicted_return_pct', 0):+.1f}%",
            "Sentiment":     f"{se.get('sentiment_score', 0):.0f}/100",
            "Sharpe":        f"{bt.get('sharpe_at_recommended', 0):.3f}",
            "Stop":          bt.get("recommended_stop_label", "—"),
        })
    st.dataframe(pd.DataFrame(score_rows), width='stretch', hide_index=True)
else:
    st.info("Run `python generate_demo_data.py` to generate the sample data.")

st.divider()

# ── Row 3: Ticker Detail Tabs ─────────────────────────────────────────────────
st.subheader(f"🔬 Detail View — {selected_ticker}")

# Load raw data for selected ticker
forecast_series  = load_forecast_series(selected_ticker)
forecast_meta    = load_forecast_metadata(selected_ticker)
backtest_raw     = load_backtest_raw(selected_ticker)
forecast_signal  = data.get("forecast",  {}).get(selected_ticker, {})
sentiment_signal = data.get("sentiment", {}).get(selected_ticker, {})
backtest_signal  = data.get("backtest",  {}).get(selected_ticker, {})
action_str       = data.get("actions_by_ticker", {}).get(selected_ticker, "monitor")

# KPI strip for selected ticker
k1, k2, k3, k4 = st.columns(4)
current_price = forecast_signal.get("current_price") or \
                forecast_meta.get("current_metrics", {}).get("current_price", 0)
target_price  = forecast_signal.get("target_price", 0)
# stop_dollar_amount is the per-share stop DISTANCE (pct x price), not a price level
stop_distance = backtest_signal.get("stop_dollar_amount")
stop_price    = (current_price - stop_distance) if (stop_distance and current_price) else None
sentiment_sc  = sentiment_signal.get("sentiment_score", 0)

k1.metric("Current Price",   f"${current_price:.2f}")
k2.metric("90-Day Target",   f"${target_price:.2f}",
          delta=f"{forecast_signal.get('predicted_return_pct', 0):+.1f}%")
k3.metric("Trailing Stop",
          f"${stop_price:.2f}" if stop_price else "—",
          delta=f"-{backtest_signal.get('recommended_stop_pct', 0):.2f}%" if stop_price else None,
          delta_color="inverse")
k4.metric("Sentiment Score", f"{sentiment_sc:.1f} / 100")

# Action badge
action_label, action_color = get_action_badge(action_str)
badge_map = {"green": st.success, "red": st.error,
             "orange": st.warning, "blue": st.info}
badge_fn = badge_map.get(action_color, st.info)
badge_fn(f"**{selected_ticker} Recommendation: {action_label}**")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Forecast", "📉 Backtest", "📰 Sentiment", "🎯 Strategy", "🎲 Monte Carlo", "💼 Portfolio",
    "🧾 Record Fills"
])

# ── Tab 1: Forecast ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Prophet 90-Day Price Forecast")
    st.plotly_chart(
        forecast_chart(
            forecast_series,
            selected_ticker,
            stop_price=stop_price,
            entry_price=current_price if current_price else None,
            price_history=forecast_meta.get("price_history"),
            model_metrics=forecast_meta.get("model_metrics"),
            forecast_signal=forecast_signal if forecast_signal else None,
        ),
        width='stretch',
    )
    if forecast_meta:
        cm = forecast_meta.get("current_metrics", {})
        dc = forecast_meta.get("data_characteristics", {})
        fs = forecast_meta.get("forecast_summary", {})
        mm = forecast_meta.get("model_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Forecast Base Price", f"${cm.get('current_price') or 0:.2f}")
        c2.metric("Mean Forecast", f"${fs.get('mean') or 0:.2f}")
        c3.metric("Forecast High", f"${fs.get('max') or 0:.2f}")
        c4.metric("Forecast Δ",    f"{fs.get('change_pct') or 0:+.2f}%")
        c5.metric("MAPE",          f"{mm.get('mape') or 0:.2f}%")
        trend = dc.get("trend", {})
        st.caption(
            f"**Trend:** {trend.get('direction','—')} "
            f"(R²={trend.get('r_squared',0):.3f})  •  "
            f"**Volatility:** {dc.get('volatility',{}).get('level','—')}  •  "
            f"**Model:** {forecast_meta.get('selected_method','—').upper()}"
        )
    if forecast_signal:
        st.markdown(
            f"**MeLLeA Interpretation:** {forecast_signal.get('reasoning','—')}"
        )

# ── Tab 2: Backtest ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 3-Year ATR Trailing Stop Backtest")

    if backtest_raw:
        # Methodology note
        st.caption(backtest_raw.get("methodology", ""))

        # Stats strip
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("ATR(14)",       f"${backtest_raw.get('atr14', 0):.2f}"
                                    if "atr14" in backtest_raw else "—")
        b2.metric("ATR % Price",   f"{backtest_raw.get('atr_pct_of_price', 0):.2f}%"
                                    if "atr_pct_of_price" in backtest_raw else "—")
        b3.metric("Parkinson Vol", f"{backtest_raw.get('parkinson_sigma_pct', 0):.3f}%"
                                    if "parkinson_sigma_pct" in backtest_raw else "—")
        b4.metric("Stage 2",       "✅ PASS" if backtest_raw.get("stage2_pass") else "❌ FAIL")

        # Results table
        results = backtest_raw.get("results", {})
        recommended = backtest_raw.get("recommended_stop", "")
        if results:
            rows = []
            for label, m in results.items():
                rows.append({
                    "Stop Level":   label + (" ★" if label == recommended else ""),
                    "Sharpe":       f"{m['sharpe']:.3f}",
                    "Mean Ret%":    f"{m['mean_ret_pct']:+.2f}%",
                    "Median Ret%":  f"{m['median_ret_pct']:+.2f}%",
                    "P(Loss>5%)":   f"{m['p_loss_gt5']:.2f}%",
                    "P(Profit)":    f"{m['p_profit']:.1f}%",
                    "Stops/Yr":     m['avg_stops_yr'],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch', hide_index=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                sharpe_bar_chart(backtest_raw, selected_ticker),
                width='stretch',
            )
        with col_r:
            st.plotly_chart(
                mean_return_bar_chart(backtest_raw, selected_ticker),
                width='stretch',
            )
    else:
        st.info(f"No backtest data found for {selected_ticker}. "
                f"Run: `python generate_demo_data.py`")

    if backtest_signal:
        st.markdown(
            f"**MeLLeA Advice:** {backtest_signal.get('position_size_advice','—')}"
        )

# ── Tab 3: Sentiment ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Sentiment Analysis — EDGAR Financial Filings")

    # Portfolio-wide sentiment comparison
    st.markdown("**Portfolio Sentiment Scores**")
    sentiment_data = data.get("sentiment", {})
    all_tickers = data.get("all_tickers", [])

    # Build sentiment table for all tickers
    sentiment_rows = []
    for ticker in all_tickers:
        sent = sentiment_data.get(ticker, {})
        if sent:
            sentiment_rows.append({
                "Ticker": ticker,
                "Score": f"{sent.get('sentiment_score', 0):.1f}",
                "Label": sent.get("sentiment_label", "—").capitalize(),
                "Confidence": f"{sent.get('confidence', 0):.2%}",
                "Risk": sent.get("risk_level", "—").upper(),
            })

    if sentiment_rows:
        sentiment_df = pd.DataFrame(sentiment_rows)
        st.dataframe(sentiment_df, width='stretch', hide_index=True)
    else:
        st.warning("No sentiment data available")

    # Selected ticker details
    st.markdown("---")
    st.markdown(f"**Detailed Analysis: {selected_ticker}**")

    if sentiment_signal:
        # Metrics
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Sentiment Score",    f"{sentiment_signal.get('sentiment_score',0):.1f}/100")
        s2.metric("Analyst Rating",     sentiment_signal.get("analyst_rating","—").upper())
        s3.metric("Risk Level",         sentiment_signal.get("risk_level","—").upper())
        s4.metric("Position Multiplier",f"{sentiment_signal.get('position_multiplier',1.0):.2f}×")

        # Progress bar
        score = sentiment_signal.get("sentiment_score", 50)
        label = sentiment_signal.get("sentiment_label", "neutral")

        # Color code based on score
        if score >= 65:
            st.success(f"✅ Bullish Sentiment: {score:.1f}/100 — {label.upper().replace('_',' ')}")
        elif score >= 55:
            st.info(f"ℹ️ Slightly Bullish: {score:.1f}/100 — {label.upper().replace('_',' ')}")
        elif score >= 45:
            st.warning(f"⚠️ Neutral: {score:.1f}/100 — {label.upper().replace('_',' ')}")
        else:
            st.error(f"❌ Bearish Sentiment: {score:.1f}/100 — {label.upper().replace('_',' ')}")

        # Key risks
        key_risks = sentiment_signal.get("key_risks", [])
        if key_risks:
            st.markdown("**Key Risks Identified:**")
            for risk in key_risks:
                st.markdown(f"- {risk}")

        # Confidence and recommendation
        confidence = sentiment_signal.get("confidence", 0)
        rec = sentiment_signal.get('recommendation','—').upper()

        st.markdown("**Analysis Metadata**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{confidence:.1%}")
        col2.metric("Data Source", "EDGAR 10-Q/10-K + Yahoo analysts")
        col3.metric("Recommendation", rec)

        st.caption(f"Method: TextBlob NLP on financial filings  •  "
                   f"Updated: {data.get('date', '—')}")
    else:
        st.info(f"Limited sentiment data for {selected_ticker}. "
                f"EDGAR filings analysis in progress...")

# ── Tab 4: Strategy ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Trading Strategy Advice")

    # Combined score
    cs = data.get("combined_scores", {}).get(selected_ticker)
    _bp = (data.get("buy_plan", {}) or {}).get(selected_ticker)
    _why = (data.get("monitor_reasons", {}) or {}).get(selected_ticker)
    if _bp:
        st.success(f"**BUY** — {_bp['shares']} shares @ ${_bp['price']:.2f} "
                   f"(${_bp['cost']:,.2f}), stop ${_bp['stop_price']:.2f}. "
                   f"Forecast include + backtest pass + no sentiment veto + reward/risk pass.")
        if _bp.get("stop_distance") and _bp.get("target_price"):
            from datetime import date as _date
            from drift_check import buy_band, stop_check

            st.markdown("##### Drift check — use this advice when the price has moved")
            _k = selected_ticker
            d1, d2 = st.columns(2)
            _allow = d1.slider("Drift allowance (% of the stop distance, each side)",
                               0, 60, 25, key=f"allow_{_k}",
                               help="A volatile stock has a wide stop, so it tolerates more "
                                    "drift in dollars than a calm one.")
            _minrr = d2.number_input("Minimum reward/risk", 0.5, 5.0, 1.25, 0.05, key=f"rr_{_k}")
            try:
                _band = buy_band(_bp["price"], _bp["stop_distance"], _bp["target_price"],
                                 _bp.get("max_cost", _bp["cost"]), buy_date=_date.today(),
                                 allowance_frac=_allow / 100.0, min_rr=_minrr)
            except ValueError as _e:
                _band = None
                st.error(f"Cannot compute the band: {_e}")
            if _band:
                st.markdown("**Order 1 — BUY** (place a *limit* order at the ceiling)")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Skip if open below", f"${_band.floor:,.2f}")
                b2.metric("Buy limit (ceiling)", f"${_band.buy_limit:,.2f}")
                b3.metric("Shares at ceiling", _band.shares)
                b4.metric("Max cost", f"${_band.max_cost:,.2f}")
                if not _band.valid:
                    st.error("This buy fails the rules at today's numbers; do not place it.")
                for _n in _band.notes:
                    st.warning(_n)
                _live = st.number_input("Current price (optional, to test the band)",
                                        0.0, step=0.01, key=f"live_{_k}")
                if _live > 0 and _band.valid:
                    if _live < _band.floor:
                        st.error(f"${_live:,.2f} is below the floor: SKIP and re-run the pipeline. "
                                 f"A large drop can mean the situation has changed.")
                    elif _live > _band.ceiling:
                        st.error(f"${_live:,.2f} is above the ceiling: SKIP. Reward/risk or "
                                 f"drift is out of range, and you would be chasing the price.")
                    else:
                        st.success(f"${_live:,.2f} is inside the band: GO. "
                                   f"Buy limit ${_band.buy_limit:,.2f}.")

                st.markdown("**Order 2 — protective STOP-LIMIT**")
                st.error(
                    "⚠️ **Confirm with the broker that the purchase is FINALIZED (settled) "
                    "before placing the stop-limit order.** Placing it earlier risks a "
                    "Free Ride Violation. The date below is only an estimate "
                    "(weekdays, holidays ignored).")
                from drift_check import next_business_day as _nbd
                st.caption(f"If bought today ({_date.today():%a %m/%d}), the earliest likely "
                           f"stop date is {_nbd(_date.today()):%a %m/%d} "
                           f"(~{_band.unprotected_days} day(s) unprotected).")
                s1, s2 = st.columns(2)
                _fill = s1.number_input("Actual fill price", 0.0, step=0.01, key=f"fill_{_k}")
                _cur = s2.number_input("Price now (after settlement)", 0.0, step=0.01,
                                       key=f"now_{_k}")
                if _fill > 0 and _cur > 0:
                    _sc = stop_check(_fill, _cur, _bp["stop_distance"])
                    {"PLACE": st.success, "PLACE_WARN": st.warning,
                     "STOP_BREACHED": st.error}[_sc.verdict](f"**{_sc.verdict}** — {_sc.message}")
                else:
                    st.caption(f"Enter the fill and current price to get the stop. "
                               f"{_band.stop_after_fill}")
    elif _why:
        st.warning(f"**NOT BOUGHT** — {_why}")
    if cs:
        st.caption(f"Ranking score {cs:.1f} / 100 — orders candidates for cash; "
                   f"it is not a buy threshold.")

    # Strategy grid
    sg1, sg2, sg3, sg4 = st.columns(4)
    sg1.metric("Action",        action_label)
    sg2.metric("Entry Price",   f"${current_price:.2f}" if current_price else "—")
    sg3.metric("Price Target",  f"${target_price:.2f}"  if target_price  else "—")
    sg4.metric("Stop Loss",     f"${stop_price:.2f}"    if stop_price    else "—")

    sg5, sg6, sg7, sg8 = st.columns(4)
    sg5.metric("Stop %",
               f"{backtest_signal.get('recommended_stop_pct',0):.2f}%"
               if backtest_signal else "—")
    sg6.metric("Sharpe Ratio",
               f"{backtest_signal.get('sharpe_at_recommended',0):.3f}"
               if backtest_signal else "—")
    sg7.metric("Mean Annual Return",
               f"{backtest_signal.get('mean_annual_return_pct',0):+.2f}%"
               if backtest_signal else "—")
    sg8.metric("Sentiment Mult",
               f"{sentiment_signal.get('position_multiplier',1.0):.2f}×"
               if sentiment_signal else "—")

    # Risk/reward chart
    if current_price and stop_price and target_price:
        st.plotly_chart(
            risk_reward_chart(current_price, stop_price, target_price,
                              selected_ticker),
            width='stretch',
        )

    # Broker caution note
    st.warning(
        "⚠️ **Broker Order Reminder:** After purchasing, wait at least one "
        "full trading day (next business day) before placing the trailing stop "
        "sell order. Same-day stop orders risk a Free Ride Violation."
    )

# ── Tab 5: Monte Carlo Risk Simulation ────────────────────────────────────────
with tab5:
    st.markdown("#### Monte Carlo Risk Simulation (1,000 GBM Paths)")

    mc_data = data.get("monte_carlo", {}).get(selected_ticker)
    if mc_data and mc_data.get("forecast"):
        # Forecast & Volatility
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "90-Day Forecast",
            f"{mc_data['forecast']['expected_change_pct']:+.2f}%"
        )
        mc2.metric(
            "Daily Volatility",
            f"{mc_data['volatility']['daily_sigma_pct'] * 100:.2f}%"
        )
        mc3.metric(
            "Annualized Vol",
            f"{mc_data['volatility']['annualized_sigma_pct'] * 100:.2f}%"
        )
        mc4.metric(
            "ATR Stop %",
            f"{mc_data['strategy']['atr_stop_pct']:.2f}%"
        )

        # Results
        st.markdown("**Strategy Results**")
        res = mc_data.get("results", {})

        mcr1, mcr2, mcr3, mcr4 = st.columns(4)
        mcr1.metric("Win Rate", f"{res.get('win_rate_pct', 0):.1f}%")
        mcr2.metric("Mean Return", f"{res.get('mean_return_pct', 0):+.2f}%")
        mcr3.metric("Sharpe Ratio", f"{res.get('sharpe_ratio', 0):.3f}")
        mcr4.metric("Max Drawdown", f"{res.get('max_drawdown_pct', 0):.2f}%")

        # Risk metrics
        mcr5, mcr6, mcr7, mcr8 = st.columns(4)
        mcr5.metric("Confidence", f"{mc_data.get('confidence_score', 0)}/100")
        mcr6.metric("Recommendation", mc_data.get('recommendation', 'NEUTRAL').upper())
        mcr7.metric("P(Loss > 5%)", f"{res.get('prob_loss_gt_5pct', 0):.1f}%")
        mcr8.metric("5th Percentile", f"{res.get('percentile_5_pct', 0):+.2f}%")

        # Probability distribution
        st.markdown("**Return Distribution**")
        mc_chart_cols = st.columns(2)

        with mc_chart_cols[0]:
            # Return distribution
            if "percentile_" in str(res.keys()):
                dist_data = {
                    "Percentile": ["5th", "25th", "50th (Median)", "75th", "95th"],
                    "Return %": [
                        res.get('percentile_5_pct', 0),
                        res.get('percentile_25_pct', 0),
                        res.get('median_return_pct', 0),
                        res.get('percentile_75_pct', 0),
                        res.get('percentile_95_pct', 0),
                    ]
                }
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Bar(x=dist_data["Percentile"], y=dist_data["Return %"],
                           marker_color=['red' if x < 0 else 'green' for x in dist_data["Return %"]])
                ])
                fig.update_layout(
                    title=f"{selected_ticker} — Return Distribution",
                    xaxis_title="Probability Percentile",
                    yaxis_title="Expected Return %",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, width='stretch')

        with mc_chart_cols[1]:
            # Confidence gauge
            import plotly.graph_objects as go
            confidence = mc_data.get('confidence_score', 0)
            fig = go.Figure(data=[go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Score"},
                delta={'reference': 50},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': 'darkblue'},
                       'steps': [
                           {'range': [0, 30], 'color': '#ffcccc'},
                           {'range': [30, 70], 'color': '#ffffcc'},
                           {'range': [70, 100], 'color': '#ccffcc'}
                       ],
                       'threshold': {'line': {'color': 'red', 'width': 4},
                                   'thickness': 0.75,
                                   'value': 50}}
            )])
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, width='stretch')

        # Summary
        st.markdown("**Summary**")
        st.info(
            f"Based on {mc_data.get('forecast', {}).get('expected_change_pct', 0):+.1f}% expected return "
            f"over 90 days and {mc_data['volatility']['daily_sigma_pct'] * 100:.2f}% daily volatility, "
            f"the ATR {mc_data['strategy']['atr_stop_pct']:.1f}% stop loss strategy has a "
            f"{res.get('win_rate_pct', 0):.1f}% win rate with {res.get('max_drawdown_pct', 0):.1f}% "
            f"maximum drawdown risk."
        )
    else:
        st.warning(f"No Monte Carlo data available for {selected_ticker}. "
                   f"Run: `python generate_demo_data.py`")

# ── Tab 6: Portfolio Analysis ─────────────────────────────────────────────────
with tab6:
    st.markdown("#### Portfolio-Level Risk Analysis")

    portfolio_data = data.get("portfolio_analysis", {})

    if not portfolio_data:
        st.warning("No portfolio analysis data. Portfolio analysis runs only with 2-7 tickers.")
    else:
        # Portfolio MC
        pmc = portfolio_data.get("monte_carlo", {})
        if pmc:
            st.markdown("**Portfolio Monte Carlo (1,000 Correlated Paths)**")

            portfolio_size = pmc.get("portfolio_size", 0)
            tickers_in_portfolio = pmc.get("tickers", [])
            weights = pmc.get("weights", {})

            # Show portfolio composition
            comp_cols = st.columns(portfolio_size)
            for i, ticker in enumerate(tickers_in_portfolio):
                with comp_cols[i]:
                    st.metric(
                        ticker,
                        f"{weights.get(ticker, 0) * 100:.0f}%",
                        "of portfolio"
                    )

            # Portfolio results
            st.markdown("**Results**")
            pres = pmc.get("results", {})

            por1, por2, por3, por4 = st.columns(4)
            por1.metric("Mean Return", f"{pres.get('mean_return_pct', 0):+.1f}%")
            por2.metric("Win Rate", f"{pres.get('win_rate_pct', 0):.1f}%")
            por3.metric("Sharpe Ratio", f"{pres.get('sharpe_ratio', 0):.3f}")
            por4.metric("Max Drawdown", f"{pres.get('max_drawdown_pct', 0):.1f}%")

            por5, por6, por7, por8 = st.columns(4)
            por5.metric("5th Percentile", f"{pres.get('percentile_5_pct', 0):+.1f}%", "worst 5%")
            por6.metric("Median", f"{pres.get('median_return_pct', 0):+.1f}%")
            por7.metric("95th Percentile", f"{pres.get('percentile_95_pct', 0):+.1f}%", "best 5%")
            por8.metric("Confidence", f"{pmc.get('confidence_score', 0)}/100")

            # Recommendation
            rec = pmc.get("recommendation", "HOLD")
            if "CAUTION" in rec:
                st.error(f"⚠️ {rec}")
            elif "STRONG" in rec:
                st.success(f"✅ {rec}")
            else:
                st.info(f"ℹ️ {rec}")

        # Diversification
        div = portfolio_data.get("diversification", {})
        if div:
            st.markdown("---")
            st.markdown("**Diversification Analysis**")

            rating = div.get("overall_rating", "UNKNOWN")
            if rating == "WELL_DIVERSIFIED":
                st.success(f"✅ Rating: {rating}")
            elif rating == "ACCEPTABLE":
                st.warning(f"⚠️ Rating: {rating}")
            else:
                st.error(f"❌ Rating: {rating}")

            # Sector breakdown
            sector_div = div.get("sector_diversity", {})
            sectors = sector_div.get("sectors", {})

            if sectors:
                st.markdown("**Sector Breakdown**")
                sector_cols = st.columns(len(sectors))
                for i, (sector, count) in enumerate(sectors.items()):
                    with sector_cols[i]:
                        pct = (count / portfolio_size) * 100 if portfolio_size > 0 else 0
                        st.metric(sector, f"{count} ({pct:.0f}%)")

            # Issues
            issues = div.get("all_issues", [])
            if issues:
                st.markdown("**⚠️ Rebalancing Issues**")
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("No rebalancing issues — portfolio is well-balanced")

            # Recommendation
            rec = div.get("recommendation", "")
            st.info(rec)

        # Portfolio MC Charts
        if pmc and pmc.get('results'):
            st.markdown("---")
            st.markdown("**Portfolio Return Distribution**")

            pres = pmc.get('results', {})

            # Create return distribution chart
            import plotly.graph_objects as go

            dist_cols = st.columns(2)

            with dist_cols[0]:
                # Return percentile chart
                percentiles = ["5th", "25th", "Median", "75th", "95th"]
                returns = [
                    pres.get('percentile_5_pct', 0),
                    pres.get('percentile_25_pct', 0),
                    pres.get('median_return_pct', 0),
                    pres.get('percentile_75_pct', 0),
                    pres.get('percentile_95_pct', 0),
                ]

                fig = go.Figure(data=[
                    go.Bar(
                        x=percentiles,
                        y=returns,
                        marker_color=[
                            '#ef4444' if r < 0 else '#22c55e'
                            for r in returns
                        ],
                        text=[f"{r:+.1f}%" for r in returns],
                        textposition='outside',
                    )
                ])
                fig.update_layout(
                    title="Portfolio Return Distribution (90-day forecast)",
                    xaxis_title="Probability Percentile",
                    yaxis_title="Expected Return %",
                    height=400,
                    template="plotly_dark",
                    showlegend=False,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
                st.plotly_chart(fig, width='stretch')

            with dist_cols[1]:
                # Portfolio confidence gauge
                confidence = pmc.get('confidence_score', 0)
                rec_text = pmc.get('recommendation', 'HOLD')

                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Portfolio Confidence"},
                    delta={'reference': 50, 'suffix': " vs Neutral"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': 'darkblue'},
                        'steps': [
                            {'range': [0, 30], 'color': '#ffcccc'},
                            {'range': [30, 70], 'color': '#ffffcc'},
                            {'range': [70, 100], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                )])
                fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    annotations=[
                        dict(
                            text=rec_text,
                            x=0.5,
                            y=-0.15,
                            xref='paper',
                            yref='paper',
                            showarrow=False,
                            font=dict(size=12, color='#1f2937')
                        )
                    ]
                )
                st.plotly_chart(fig, width='stretch')

            # Risk metrics comparison
            st.markdown("**Risk Metrics Summary**")
            risk_cols = st.columns(4)

            with risk_cols[0]:
                st.metric(
                    "Portfolio Volatility",
                    f"{pres.get('std_return_pct', 0):.2f}%",
                    "90-day std dev"
                )

            with risk_cols[1]:
                st.metric(
                    "Best Case (95th %ile)",
                    f"{pres.get('percentile_95_pct', 0):+.1f}%",
                    "top 5% outcome"
                )

            with risk_cols[2]:
                st.metric(
                    "Worst Case (5th %ile)",
                    f"{pres.get('percentile_5_pct', 0):+.1f}%",
                    "worst 5% outcome"
                )

            with risk_cols[3]:
                st.metric(
                    "Risk/Reward Ratio",
                    f"{abs(pres.get('percentile_95_pct', 0) / pres.get('percentile_5_pct', 1)) if pres.get('percentile_5_pct', 1) != 0 else 0:.2f}x",
                    "upside/downside"
                )

            # Summary interpretation
            st.markdown("**Portfolio Summary**")
            mean_ret = pres.get('mean_return_pct', 0)
            win_rate = pres.get('win_rate_pct', 0)
            sharpe = pres.get('sharpe_ratio', 0)
            max_dd = abs(pres.get('max_drawdown_pct', 0))

            if mean_ret > 10:
                summary_color = "success"
                summary_icon = "✅"
            elif mean_ret > 0:
                summary_color = "info"
                summary_icon = "ℹ️"
            else:
                summary_color = "error"
                summary_icon = "⚠️"

            summary_text = (
                f"{summary_icon} **Portfolio Monte Carlo Analysis** ({pmc.get('n_simulations', 0)} simulations):\n\n"
                f"Based on your {pmc.get('portfolio_size', 0)}-stock portfolio with dollar-value weighting, "
                f"the 90-day outlook shows a **{mean_ret:+.1f}% mean return** with a **{win_rate:.1f}% win rate**. "
                f"The Sharpe ratio of {sharpe:.3f} indicates {'strong' if sharpe > 0.5 else 'weak'} risk-adjusted returns. "
                f"Maximum portfolio drawdown risk is **{max_dd:.1f}%**, which occurs in {pres.get('portfolio_stop_hits_pct', 0):.0f}% of simulations. "
                f"Overall portfolio confidence is **{confidence}/100** ({rec_text})."
            )

            getattr(st, summary_color)(summary_text)

# ── Tab 7: Record Fills (broker confirmations -> positions.json) ──────────────
with tab7:
    st.markdown("#### Record broker fills")
    st.caption("Enter what the broker actually filled. Nothing is recorded from the "
               "recommendations themselves. Holdings and cash change only here. "
               "(Demo: changes last only for your browser session. Use "
               "**Reset demo holdings** in the sidebar to start over.)")

    _pos_file = _POS_FILE
    _state = _uc.load(_pos_file)
    if st.session_state.get("fills_msg"):
        st.success(st.session_state.pop("fills_msg"))

    # ---- open positions -----------------------------------------------------
    st.markdown("**Open positions**")
    _held = _state["open_positions"]
    st.metric("Cash available", f"${_state['cash_available']:,.2f}")
    if _held:
        st.dataframe(pd.DataFrame([{
            "Stock": t, "Shares": p.get("shares"), "Cost basis": p.get("cost_basis"),
            "Cost": round((p.get("shares") or 0) * (p.get("cost_basis") or 0), 2),
            "Stop": p["stop"] if p.get("stop") is not None else "NOT PLACED",
            "Bought": p.get("added_date"),
        } for t, p in _held.items()]), hide_index=True, width='stretch')
        _nostop = [t for t, p in _held.items() if p.get("stop") is None]
        if _nostop:
            st.warning(f"No stop recorded yet for: {', '.join(_nostop)}. These positions are "
                       f"unprotected until the stop-limit is placed.")
    else:
        st.info("No open positions recorded.")

    # ---- record buy fills -----------------------------------------------------
    st.markdown("**1. Record purchases (Order 1 fills)**")
    _planned = [t for t in (data.get("buy_plan") or {}) if t not in _held]
    _template = "Stock\tShares\tPrice\n" + "".join(
        f"{t}\t{data['buy_plan'][t]['shares']}\t\n" for t in _planned)
    _up = st.file_uploader("Upload the broker download (CSV or TSV)", type=["csv", "tsv", "txt"],
                           key="fills_upload")
    _default = _up.getvalue().decode("utf-8-sig") if _up else _template
    _txt = st.text_area(
        "…or paste / edit here. Columns: Stock, Shares, Price (a Date column is optional). "
        "The template lists planned shares only: enter the price you were actually filled at "
        "and correct shares if they differ.",
        value=_default, height=170,
        key=f"fills_text_{(_up.file_id if _up else 'tpl')}")
    if _up:
        st.info("File loaded. Review the preview below, tick the confirmation box and click "
                "**Record fills** to update holdings and cash. Nothing changes until then.")

    _fills, _err = [], None
    try:
        _fills = _uc.parse_fills(_txt)
    except ValueError as _e:
        _err = str(_e)
    if _err:
        st.error(_err)
    else:
        _preview = _uc.load(_pos_file)
        try:
            _lines = _uc.import_fills(_preview, _fills)
            _spent = _state["cash_available"] - _preview["cash_available"]
            st.dataframe(pd.DataFrame(_fills), hide_index=True, width='stretch')
            st.caption(f"Preview: spends ${_spent:,.2f}; cash after = "
                       f"${_preview['cash_available']:,.2f}. "
                       + " ".join(l for l in _lines if l.startswith("SKIPPED")))
            _ok = st.checkbox("These are broker-confirmed fills", key="fills_confirm")
            if st.button("Record fills", type="primary", disabled=not _ok):
                _live_state = _uc.load(_pos_file)
                _done = _uc.import_fills(_live_state, _fills)
                _uc.save(_pos_file, _live_state)
                st.session_state["fills_msg"] = "Recorded:\n\n" + "\n\n".join(_done)
                st.rerun()
        except ValueError as _e:
            st.error(f"Cannot record: {_e}")

    # ---- record a placed stop -------------------------------------------------
    st.markdown("**2. Record a stop once you have placed it (Order 2)**")
    st.error("⚠️ **Confirm with the broker that the purchase is FINALIZED (settled) before "
             "placing the stop-limit order.** Use the Strategy tab's drift check to get the "
             "stop price from your actual fill.")
    _need = [t for t, p in _held.items() if p.get("stop") is None] or list(_held)
    if _need:
        r1, r2 = st.columns(2)
        _st_t = r1.selectbox("Position", _need, key="stop_ticker")
        _adv = (data.get("buy_plan") or {}).get(_st_t, {}).get("stop_price")
        _st_p = r2.number_input("Stop trigger price you placed", 0.0, step=0.01,
                                value=float(_adv or 0.0), key="stop_price_in")
        if st.button("Record stop"):
            try:
                _s2 = _uc.load(_pos_file)
                _uc.set_stop(_s2, _st_t, _st_p)
                _uc.save(_pos_file, _s2)
                st.success(f"Recorded stop ${_st_p:,.2f} for {_st_t}.")
            except ValueError as _e:
                st.error(str(_e))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Disclaimer:** DEMO WITH SAMPLE DATA. All companies and numbers are fictional. This dashboard is for informational purposes only and does not "
    "constitute financial advice. All analysis is based on historical data and "
    "quantitative models. Past performance does not guarantee future results. "
    "Always conduct your own due diligence.  |  B. I. Parker Data Science and Consulting LLC  "
    "|  Made with Claude (Anthropic)"
)

# Made with Claude
