# Stock Trading System: Demo Dashboard

A Streamlit demo of the stock trading dashboard that runs entirely on **sample data**.
All companies, tickers, prices, forecasts and results are fictional and randomly
generated. Nothing is real market data or investment advice.

The real system (Prophet forecasts, EDGAR/FMP sentiment, ATR backtests, Monte Carlo)
is not included. The demo just reads the same file layout that system produces.

## Run it

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## What you will see

Fictional universe of 8 stocks: 4 **buy**, 1 **hold**, 1 **sell**, 2 **monitor**
(one fails reward/risk, one has a forecast below the current price).

| Tab | Shows |
|---|---|
| Forecast | 90-day forecast with confidence band and residual distribution |
| Backtest | ATR trailing-stop results by stop level |
| Sentiment | Sentiment score, analyst rating, key risks |
| Strategy | Buy plan, drift-check band, stop-limit check, risk/reward |
| Monte Carlo | 1,000-path simulation per stock |
| Portfolio | Correlated portfolio simulation and diversification check |
| Record Fills | Paste broker fills; updates holdings and cash |

**Record Fills is per browser session.** Changes never touch the shared sample data
and disappear when the session ends. Use **Reset demo holdings** in the sidebar to
start over between presentations.

## Regenerate the sample data

```bash
python generate_demo_data.py                    # dated today
python generate_demo_data.py --as-of 2026-11-13 # fixed date
```

Output is seeded, so the same date gives identical data. Edit the `UNIVERSE` table at
the top of `generate_demo_data.py` to change companies, prices or outlooks. The script
asserts that each stock's designed role (buy/hold/sell/monitor) matches what the
rules actually decide.

## Deploy on Streamlit Community Cloud

Point it at this repo with main file path `dashboard/app.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Runs the real dashboard headlessly for every ticker and checks the Record Fills flow.

## Layout

```
dashboard/            app.py, charts.py, data_loader.py
generate_demo_data.py synthetic data generator
update_cash.py        fill parsing and cash/position bookkeeping
drift_check.py        buy band and stop-limit check
data/                 pipeline_output.json, positions.json (generated)
forecasts/            per-ticker forecast JSON (generated)
Analysis_Outcomes/    per-ticker backtest JSON (generated)
```
