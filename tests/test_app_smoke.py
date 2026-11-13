"""Run the real dashboard headlessly against the committed sample data."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "dashboard" / "app.py")
OUT = json.loads((ROOT / "data" / "pipeline_output.json").read_text(encoding="utf-8"))


def _run(ticker=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if ticker:
        at.sidebar.selectbox[0].select(ticker).run()
    return at


def test_loads_without_exception():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("ticker", OUT["all_tickers"])
def test_every_ticker_renders(ticker):
    at = _run(ticker)
    assert not at.exception, [e.value for e in at.exception]


def test_sample_data_roles():
    assert OUT["mode"] == "demo"
    assert set(OUT["tickers_to_buy"]) == set(OUT["buy_plan"])
    assert OUT["tickers_to_sell"] and OUT["tickers_to_hold"] and OUT["tickers_to_monitor"]
    spend = sum(p["cost"] for p in OUT["buy_plan"].values())
    assert spend <= OUT["cash_remaining"]


def test_record_fills_is_per_session_and_leaves_seed_untouched():
    seed = ROOT / "data" / "positions.json"
    before = seed.read_text(encoding="utf-8")
    cash0 = json.loads(before)["cash_available"]

    at = _run()
    nvlx = OUT["buy_plan"]["NVLX"]
    at.text_area(key="fills_text_tpl").set_value(
        f"Stock\tShares\tPrice\nNVLX\t{nvlx['shares']}\t{nvlx['price']}\n")
    at.run()
    at.checkbox(key="fills_confirm").check().run()
    [b for b in at.button if b.label == "Record fills"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]

    session_file = Path(at.session_state["_positions_file"])
    state = json.loads(session_file.read_text(encoding="utf-8"))
    assert "NVLX" in state["open_positions"]
    assert state["cash_available"] < cash0
    assert seed.read_text(encoding="utf-8") == before   # shared seed never modified

    [b for b in at.sidebar.button if "Reset" in b.label][0].click().run()
    fresh = json.loads(Path(at.session_state["_positions_file"]).read_text(encoding="utf-8"))
    assert "NVLX" not in fresh["open_positions"]
