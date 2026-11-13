import copy

import pytest

from update_cash import parse_fills, import_fills, apply_buy, set_stop


def fresh(cash=2250.0):
    return {"cash_available": cash, "open_positions": {}, "stop_loss_orders": {}}


TSV = "Stock\tShares\tPrice\nTRSP\t8\t$111.90\nVRNT\t2\t$101.71\n"


def test_parse_tsv_with_dollar_signs():
    f = parse_fills(TSV)
    assert f == [{"ticker": "TRSP", "shares": 8.0, "price": 111.9},
                 {"ticker": "VRNT", "shares": 2.0, "price": 101.71}]


def test_parse_csv_aliases_and_date_and_ignores_stop():
    f = parse_fills("Symbol,Quantity,Fill Price,Trade Date,Stop\nCRSN,8,\"$32.30\",11/16/2026,29.70\n")
    assert f == [{"ticker": "CRSN", "shares": 8.0, "price": 32.3, "date": "11/16/2026"}]


def test_parse_errors_are_specific():
    with pytest.raises(ValueError, match="missing column"):
        parse_fills("Stock\tShares\nTRSP\t8\n")
    with pytest.raises(ValueError, match="Row 2"):
        parse_fills("Stock\tShares\tPrice\nTRSP\teight\t10\n")
    with pytest.raises(ValueError, match="Row 2"):
        parse_fills("Stock\tShares\tPrice\nTRSP\t-1\t10\n")


def test_import_records_positions_and_cash():
    d = fresh()
    import_fills(d, parse_fills(TSV))
    assert d["open_positions"]["TRSP"]["shares"] == 8
    assert d["open_positions"]["TRSP"]["cost_basis"] == 111.9
    assert d["open_positions"]["TRSP"]["stop"] is None          # stop not yet placed
    assert d["cash_available"] == pytest.approx(2250 - 8 * 111.9 - 2 * 101.71)
    assert d["stop_loss_orders"] == {}


def test_import_twice_does_not_double_count():
    d = fresh()
    import_fills(d, parse_fills(TSV))
    after_first = copy.deepcopy(d)
    report = import_fills(d, parse_fills(TSV))
    assert all(r.startswith("SKIPPED") for r in report)
    assert d["open_positions"] == after_first["open_positions"]
    assert d["cash_available"] == after_first["cash_available"]


def test_import_is_all_or_nothing():
    d = fresh(cash=500.0)                                       # TRSP fits? 8*111.9=895 no
    before = copy.deepcopy(d)
    with pytest.raises(ValueError, match="exceeds cash"):
        import_fills(d, parse_fills("Stock\tShares\tPrice\nVRNT\t2\t101.71\nTRSP\t8\t111.90\n"))
    assert d == before                                          # VRNT not half-recorded


def test_second_buy_of_same_ticker_averages_cost():
    d = fresh()
    apply_buy(d, "TRSP", 4, 100.0)
    apply_buy(d, "TRSP", 4, 110.0)
    assert d["open_positions"]["TRSP"]["shares"] == 8
    assert d["open_positions"]["TRSP"]["cost_basis"] == 105.0


def test_set_stop():
    d = fresh()
    apply_buy(d, "TRSP", 1, 100.0)
    set_stop(d, "TRSP", 92.0)
    assert d["open_positions"]["TRSP"]["stop"] == 92.0 and d["stop_loss_orders"]["TRSP"] == 92.0
    with pytest.raises(ValueError):
        set_stop(d, "NOPE", 1.0)
