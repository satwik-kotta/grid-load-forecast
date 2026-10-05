"""Shared feature building for both forecasters.

Forecast set-up: at 06:00 UTC each day (midnight Central standard time) we
forecast the next 24 hourly loads for each zone. Load is known up to the
hour before the origin. Weather for the forecast day is taken from the
observed record, standing in for a weather forecast (see README, Limits).
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config

ORIGIN_HOUR_UTC = 6
WEATHER = ["temp_c", "humidity", "wind", "cloud", "raining", "solar_proxy"]


def load_panel():
    p = pd.read_parquet(config.PROCESSED / "panel.parquet")
    hol = USFederalHolidayCalendar().holidays(p.local.min().normalize(), p.local.max().normalize())
    p["holiday"] = p.local.dt.normalize().isin(hol).astype(float)
    p["hour"] = p.local.dt.hour
    p["dow"] = p.local.dt.dayofweek
    p["doy"] = p.local.dt.dayofyear
    return p


def split_of(ts):
    """train / val / test by forecast-day start (UTC)."""
    return np.where(ts <= pd.Timestamp(config.TRAIN_END) + pd.Timedelta(hours=23), "train",
                    np.where(ts <= pd.Timestamp(config.VAL_END) + pd.Timedelta(hours=23), "val", "test"))


def tabular(panel):
    """One row per (zone, target hour) with only information available at the origin."""
    out = []
    for zone, g in panel.groupby("zone"):
        g = g.set_index("ts").sort_index()
        f = g[WEATHER + ["holiday", "hour", "dow", "doy", "load_mw"]].copy()
        # hours since the forecast origin: 0..23
        f["lead"] = (f.index.hour - ORIGIN_HOUR_UTC) % 24
        origin = f.index - pd.to_timedelta(f["lead"], unit="h")
        f["origin"] = origin
        load = g["load_mw"]
        # level known at the origin: mean load over the 24h and 168h before it
        day_mean = load.rolling(24).mean().shift(1)
        week_mean = load.rolling(168).mean().shift(1)
        f["level_24"] = day_mean.reindex(origin).to_numpy()
        f["level_168"] = week_mean.reindex(origin).to_numpy()
        f["last_load"] = load.shift(1).reindex(origin).to_numpy()
        for lag in (24, 48, 168):                        # same hour on earlier days: always before the origin
            f[f"lag_{lag}"] = load.shift(lag)
        f["lag_24_ratio"] = f["lag_24"] / f["level_24"]
        f["lag_168_ratio"] = f["lag_168"] / f["level_168"]
        f["last_ratio"] = f["last_load"] / f["level_24"]
        f["temp_lag_24"] = g["temp_c"].shift(24)
        f["temp_delta_24"] = f["temp_c"] - f["temp_lag_24"]
        f["temp_roll_24"] = g["temp_c"].rolling(24).mean()
        f["temp_roll_72"] = g["temp_c"].rolling(72).mean()
        f["cdh"] = (f["temp_c"] - 18.3).clip(lower=0)      # cooling / heating degree hours
        f["hdh"] = (18.3 - f["temp_c"]).clip(lower=0)
        f["zone"] = zone
        f["y"] = np.log(f["load_mw"] / f["level_24"])      # forecast the shape relative to yesterday's level
        out.append(f.reset_index())
    t = pd.concat(out, ignore_index=True).dropna()
    t["zone_code"] = t["zone"].map({z: i for i, z in enumerate(config.ZONES)})
    t["split"] = split_of(t["origin"])
    full = t.groupby(["zone", "origin"])["lead"].transform("count") == 24
    return t[full].reset_index(drop=True)


GBM_FEATURES = WEATHER + ["holiday", "hour", "dow", "doy", "lead", "lag_24_ratio", "lag_168_ratio", "last_ratio",
                          "temp_lag_24", "temp_delta_24", "temp_roll_24", "temp_roll_72", "cdh", "hdh", "zone_code"]
