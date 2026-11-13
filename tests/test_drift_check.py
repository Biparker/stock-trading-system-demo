from datetime import date

import pytest

from drift_check import buy_band, stop_check, unprotected_days, next_business_day

TUE = date(2026, 11, 10)
FRI = date(2026, 11, 13)


def test_next_business_day_skips_weekend():
    assert next_business_day(FRI) == date(2026, 11, 16)   # Monday
    assert unprotected_days(FRI) == 3
    assert unprotected_days(TUE) == 1


def test_band_basic():
    # price 100, stop dist 8, target 130 -> allowance 2, rr ceiling 130-12=118
    b = buy_band(100, 8, 130, max_cost=1000, buy_date=TUE)
    assert b.valid
    assert b.floor == 98.0 and b.ceiling == 102.0 and b.buy_limit == 102.0
    assert b.shares == 9 and b.max_cost == 918.0           # 1000 // 102
    assert b.rr_at_price == pytest.approx(3.75)


def test_band_ceiling_capped_by_rr():
    # target 110, d 8 -> rr ceiling 110 - 1.25*8 = 100 (== price) so ceiling = price, valid
    b = buy_band(100, 8, 110, max_cost=1000, buy_date=TUE)
    assert b.valid and b.ceiling == 100.0


def test_band_invalid_when_rr_fails_at_price():
    b = buy_band(100, 8, 105, max_cost=1000, buy_date=TUE)
    assert not b.valid
    assert any("Reward/risk" in n for n in b.notes)


def test_friday_tightens_floor_and_warns():
    tue = buy_band(100, 8, 130, 1000, buy_date=TUE)
    fri = buy_band(100, 8, 130, 1000, buy_date=FRI)
    assert fri.floor > tue.floor                            # 99.0 vs 98.0
    assert fri.unprotected_days == 3
    assert any("Friday" in n for n in fri.notes)


def test_band_cash_cap_and_no_affordable_share():
    b = buy_band(100, 8, 130, max_cost=1000, cash_left=50, buy_date=TUE)
    assert not b.valid and b.shares == 0


def test_band_rejects_bad_input():
    with pytest.raises(ValueError):
        buy_band(0, 8, 130, 1000)
    with pytest.raises(ValueError):
        buy_band(100, 0, 130, 1000)


def test_stop_flat_price_places_at_fill_minus_distance():
    s = stop_check(fill_price=100, current_price=100, stop_distance=8)
    assert s.verdict == "PLACE" and s.stop_trigger == 92.0
    assert s.limit_price == pytest.approx(91.54)            # 92 * 0.995


def test_stop_trails_up_when_price_rose():
    s = stop_check(100, 110, 8)
    assert s.stop_trigger == 102.0 and "raised" in s.message


def test_stop_never_trails_down():
    s = stop_check(100, 95, 8)                              # still above 92
    assert s.stop_trigger == 92.0 and s.verdict == "PLACE"


def test_stop_thin_cushion_warns():
    s = stop_check(100, 93, 8)                              # cushion 1.0 < 0.25*8
    assert s.verdict == "PLACE_WARN"


def test_stop_breached_at_or_below_intended():
    for cur in (92.0, 90.0):
        s = stop_check(100, cur, 8)
        assert s.verdict == "STOP_BREACHED" and s.stop_trigger is None


def test_stop_rejects_bad_input():
    with pytest.raises(ValueError):
        stop_check(100, 0, 8)


from drift_check import rr_blocker


def test_rr_blocker_passes_good_trade():
    assert rr_blocker(100, 130, 8) is None             # rr 3.75


def test_rr_blocker_target_below_price():
    assert "not above" in rr_blocker(101.71, 97.57, 8.14)


def test_rr_blocker_low_reward_risk():
    assert "below the 1.25 minimum" in rr_blocker(107.62, 110.71, 7.46)


def test_rr_blocker_boundary_and_missing_data():
    assert rr_blocker(100, 110, 8) is None             # exactly 1.25 passes
    assert rr_blocker(100, 109.9, 8) is not None       # just under is blocked
    assert rr_blocker(100, 110, None) is None          # no stop distance: target>price only
    assert rr_blocker(100, None, 8) is None            # no target: other rules handle it
    assert rr_blocker(0, 130, 8) is None
