import numpy as np
import pandas as pd
import pytest
import torch

from src import config
from src.causal import PRE_DAYS, daily_panel, event_study, window_summary
from src.etl import build_panel, cloud_fraction, read_ercot_year, solar_elevation
from src.features import GBM_FEATURES, load_panel, tabular
from src.forecast import pinball


@pytest.fixture(scope="module")
def built():
    return build_panel()


def test_strict_xlsx_years_are_readable():
    df = read_ercot_year(config.RAW / "ercot" / "Native_Load_2012.xlsx")
    assert len(df) == 8784 and df["COAST"].between(5000, 25000).all()
    assert df["local_start"].iloc[0] == pd.Timestamp("2012-01-01 00:00")


def test_load_is_a_complete_hourly_series_and_zones_sum_to_ercot(built):
    _, load = built
    assert len(load) == len(pd.date_range(load.index.min(), load.index.max(), freq="h"))
    assert load.isna().sum().sum() == 0
    assert (load[config.ALL_ZONES].sum(axis=1) - load["ERCOT"]).abs().max() < 1.0


def test_hour_ending_becomes_utc_hour_start(built):
    _, load = built
    raw = read_ercot_year(config.RAW / "ercot" / "Native_Load_2017.xlsx")
    # 15 Jul 2017, hour ending 17:00 CDT = 16:00-17:00 local = 21:00 UTC start
    row = raw[raw.local_start == pd.Timestamp("2017-07-15 16:00")].iloc[0]
    assert load.loc[pd.Timestamp("2017-07-15 21:00"), "COAST"] == pytest.approx(row["COAST"])
    # 15 Jan 2017 is standard time: 16:00 local = 22:00 UTC
    row = raw[raw.local_start == pd.Timestamp("2017-01-15 16:00")].iloc[0]
    assert load.loc[pd.Timestamp("2017-01-15 22:00"), "COAST"] == pytest.approx(row["COAST"])


def test_solar_elevation_and_cloud_mapping():
    noon = pd.DatetimeIndex(["2017-06-21 18:00", "2017-12-21 18:00", "2017-06-21 06:00"])   # ~local noon, noon, 1am
    el = solar_elevation(noon, 29.76, -95.36)
    assert el[0] > 75 and 30 < el[1] < 42 and el[2] < 0
    assert cloud_fraction("sky is clear") == 0 and cloud_fraction("overcast clouds") == 1
    assert cloud_fraction("heavy intensity rain") > 0.9


def test_features_never_use_load_from_the_forecast_day():
    panel = load_panel()
    cut = pd.Timestamp("2016-03-01 06:00")
    changed = panel.copy()
    changed.loc[changed.ts >= cut, "load_mw"] *= 3.0          # corrupt everything from the origin onwards
    a = tabular(panel).query("origin == @cut").sort_values(["zone", "ts"])
    b = tabular(changed).query("origin == @cut").sort_values(["zone", "ts"])
    assert len(a) == 72
    np.testing.assert_allclose(a[GBM_FEATURES].to_numpy(), b[GBM_FEATURES].to_numpy())


def test_pinball_loss_matches_hand_calculation():
    pred = torch.tensor([[[9.0, 10.0, 12.0]]])
    y = torch.tensor([[11.0]])
    # errors 2, 1, -1 -> 0.1*2, 0.5*1, 0.1*1
    assert pinball(pred, y).item() == pytest.approx((0.2 + 0.5 + 0.1) / 3)


def test_event_study_recovers_a_planted_drop():
    d = daily_panel()
    start = pd.Timestamp("2015-05-12")                         # an ordinary week
    rel = (d.day - start).dt.days
    hit = (d.zone == "NCENT") & (rel >= 1) & (rel <= 7)
    before, _ = event_study(d, start, treated="NCENT")
    d.loc[hit, "load"] *= 0.90
    d["log_load"] = np.log(d.load)
    after, _ = event_study(d, start, treated="NCENT")
    week = lambda es: es[(es.rel_day >= 1) & (es.rel_day <= 7)].effect_pct.mean()
    assert week(after) - week(before) == pytest.approx(-10.0, abs=0.6)
    assert len(after) == PRE_DAYS + 21
    assert window_summary(after, d, start, 1, 7, treated="NCENT")["days"] == 7
