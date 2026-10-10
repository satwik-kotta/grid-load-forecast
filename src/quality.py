"""Data-quality and time-alignment checks on the hourly panel.

These run before anything downstream (forecasting, causal analysis, Athena tables)
reads the data. The AWS Glue job refuses to write curated files if any check fails.
"""
import pandas as pd

from . import config

WEATHER = ["temp_c", "humidity", "wind", "cloud", "raining", "sun_elev", "solar_proxy"]


def _check(name, failures, detail):
    return {"check": name, "passed": bool(failures == 0), "failures": int(failures), "detail": detail}


def run_checks(panel, load):
    """Return one result per check. `panel` and `load` are the outputs of etl.build_panel()."""
    out = []

    # --- completeness -------------------------------------------------------
    full = pd.date_range(load.index.min(), load.index.max(), freq="h")
    out.append(_check("ercot_hourly_complete", len(full.difference(load.index)) + load.index.duplicated().sum(),
                      f"{len(load):,} hours, {load.index.min()} to {load.index.max()} UTC"))
    gaps = 0
    for _, g in panel.groupby("zone"):
        span = pd.date_range(g.ts.min(), g.ts.max(), freq="h")
        gaps += len(span.difference(pd.DatetimeIndex(g.ts))) + g.ts.duplicated().sum()
    out.append(_check("panel_hourly_complete", gaps, f"{panel.groupby('zone').size().to_dict()} hours per zone"))

    # --- values -------------------------------------------------------------
    nulls = panel[["load_mw"] + WEATHER].isna().sum().sum() + load.isna().sum().sum()
    out.append(_check("no_nulls", nulls, "load and weather columns"))
    bad = (~panel.load_mw.gt(0)).sum() + (~panel.temp_c.between(-30, 50)).sum() + (~panel.humidity.between(0, 100)).sum()
    out.append(_check("values_in_range", bad, "load > 0 MW, temperature -30 to 50 C, humidity 0 to 100%"))

    # --- reconciliation -----------------------------------------------------
    diff = (load[config.ALL_ZONES].sum(axis=1) - load["ERCOT"]).abs()
    out.append(_check("zones_sum_to_ercot_total", (diff >= 1.0).sum(), f"largest gap {diff.max():.3f} MW"))

    # --- time alignment -----------------------------------------------------
    local = panel.ts.dt.tz_localize("UTC").dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None)
    out.append(_check("local_time_matches_utc", (local != panel.local).sum(), f"UTC to {config.LOCAL_TZ}"))

    # A local day has 24 hours, except 23 on the spring-forward day and 25 on the fall-back day.
    days = panel.groupby(["zone", panel.local.dt.normalize()]).size().rename("n").reset_index()
    edge = days.local.isin([days.local.min(), days.local.max()])          # first and last days are partial
    days = days[~edge]
    wrong = (~days.n.isin([23, 24, 25])).sum()
    wrong += ((days.n == 23) & (days.local.dt.month != 3)).sum() + ((days.n == 25) & (days.local.dt.month != 11)).sum()
    out.append(_check("daylight_saving_days", wrong,
                      f"{(days.n == 23).sum()} spring-forward and {(days.n == 25).sum()} fall-back zone-days"))

    # If load were shifted against the clock, summer demand would stop peaking in the local afternoon.
    summer = panel[panel.local.dt.month.isin([7, 8])]
    peak_hour = summer.loc[summer.groupby(["zone", summer.local.dt.normalize()]).load_mw.idxmax(), "local"].dt.hour
    share = peak_hour.between(13, 19).mean()
    out.append(_check("summer_peak_in_local_afternoon", int(share < 0.9),
                      f"{share:.1%} of July and August days peak between 13:00 and 19:59 local"))
    return out


def assert_clean(panel, load):
    results = run_checks(panel, load)
    failed = [r for r in results if not r["passed"]]
    if failed:
        raise ValueError("data-quality checks failed: " + "; ".join(f"{r['check']} ({r['failures']})" for r in failed))
    return results
