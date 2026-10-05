"""How much load did Hurricane Harvey remove from the Coast zone, beyond the weather itself?

    python -m src.causal

Harvey made landfall on 25 Aug 2017 and flooded Houston over the next days.
It was also cool and wet, and cool weather lowers summer load on its own.
A raw before/after comparison mixes the two. This script separates them.

Daily panel, three zones (Coast, North Central, South Central), Oct 2012 - Nov 2017:

    log(load) = zone + zone x weekday + zone x holiday + zone trend
              + year-month effects          (shocks common to all zones: economy, fuel prices, growth)
              + zone-specific weather response (cooling/heating degrees, squared terms, humidity, lags)
              + Coast x event-day dummies   (the estimate)

Checks: pre-event dummies (should be ~0), the same window in four earlier
years (placebo), and a synthetic control built from other ERCOT zones.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import nnls

from . import config
from .features import load_panel

LANDFALL = pd.Timestamp("2017-08-25")
PRE_DAYS, POST_DAYS = 7, 21
TREATED = "COAST"
BASE_T = 18.3


def daily_panel():
    p = load_panel()
    p["day"] = p.local.dt.normalize()
    p["cd"] = (p.temp_c - BASE_T).clip(lower=0)
    p["hd"] = (BASE_T - p.temp_c).clip(lower=0)
    d = p.groupby(["zone", "day"]).agg(load=("load_mw", "sum"), n=("load_mw", "size"), cdd=("cd", "mean"),
                                       hdd=("hd", "mean"), tmax=("temp_c", "max"), humidity=("humidity", "mean"),
                                       cloud=("cloud", "mean"), holiday=("holiday", "max")).reset_index()
    d = d[d.n == 24].copy()                              # drop the two daylight-saving days per year and partial ends
    d["log_load"] = np.log(d.load)
    d["dow"] = d.day.dt.dayofweek
    d["ym"] = d.day.dt.strftime("%Y-%m")
    d["trend"] = (d.day - d.day.min()).dt.days / 365.25
    d = d.sort_values(["zone", "day"])
    d["cdd_lag"] = d.groupby("zone").cdd.shift(1)
    d["hdd_lag"] = d.groupby("zone").hdd.shift(1)
    return d.dropna().reset_index(drop=True)


WEATHER_TERMS = ("C(zone):cdd + C(zone):I(cdd**2) + C(zone):hdd + C(zone):I(hdd**2) + C(zone):cdd_lag "
                 "+ C(zone):hdd_lag + C(zone):humidity + C(zone):cdd:humidity + C(zone):cloud")
BASE_TERMS = "C(zone) + C(zone):C(dow) + C(zone):holiday + C(zone):trend + C(ym)"


def event_study(d, event_start, weather=True, treated=TREATED):
    d = d.copy()
    rel = (d.day - event_start).dt.days
    names = []
    for k in range(-PRE_DAYS, POST_DAYS):
        name = f"ev_m{-k}" if k < 0 else f"ev_p{k}"
        d[name] = ((d.zone == treated) & (rel == k)).astype(float)
        names.append((k, name))
    formula = "log_load ~ " + BASE_TERMS + (" + " + WEATHER_TERMS if weather else "") + " + " + " + ".join(n for _, n in names)
    fit = smf.ols(formula, d).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    rows = [{"rel_day": k, "date": str((event_start + pd.Timedelta(days=k)).date()),
             "effect_pct": float(100 * (np.exp(fit.params[n]) - 1)),
             "lo_pct": float(100 * (np.exp(fit.params[n] - 1.96 * fit.bse[n]) - 1)),
             "hi_pct": float(100 * (np.exp(fit.params[n] + 1.96 * fit.bse[n]) - 1)),
             "coef": float(fit.params[n])} for k, n in names]
    return pd.DataFrame(rows), fit


def window_summary(es, d, event_start, lo, hi, treated=TREATED):
    """Average effect and energy over relative days [lo, hi]."""
    w = es[(es.rel_day >= lo) & (es.rel_day <= hi)]
    act = d[(d.zone == treated)].set_index("day").load.reindex(pd.to_datetime(w.date)).to_numpy()
    counterfactual = act / np.exp(w.coef.to_numpy())
    return {"days": int(len(w)), "avg_effect_pct": float(w.effect_pct.mean()),
            "mwh_change": float((act - counterfactual).sum()),
            "actual_mwh": float(act.sum()), "counterfactual_mwh": float(counterfactual.sum())}


def synthetic_control():
    """Coast daily load as a weighted mix of inland zones, fitted before the storm."""
    load = pd.read_parquet(config.PROCESSED / "ercot_all_zones.parquet")
    local = load.index.tz_localize("UTC").tz_convert(config.LOCAL_TZ).tz_localize(None)
    daily = load.groupby(local.normalize()).sum()[config.ALL_ZONES]
    donors = ["FWEST", "NORTH", "NCENT", "WEST", "EAST"]       # SOUTH and SCENT were in the storm's path
    pre = daily.loc["2017-06-01":LANDFALL - pd.Timedelta(days=PRE_DAYS + 1)]
    check = daily.loc[LANDFALL - pd.Timedelta(days=PRE_DAYS):LANDFALL - pd.Timedelta(days=1)]
    post = daily.loc[LANDFALL:LANDFALL + pd.Timedelta(days=POST_DAYS - 1)]
    scale = pre.mean()
    w, _ = nnls((pre[donors] / scale[donors]).to_numpy(), (pre[TREATED] / scale[TREATED]).to_numpy())
    synth = lambda x: (x[donors] / scale[donors]).to_numpy() @ w * scale[TREATED]
    gap = lambda x: 100 * (x[TREATED].to_numpy() / synth(x) - 1)
    return {"donors": donors, "weights": {z: round(float(v), 3) for z, v in zip(donors, w)},
            "pre_fit_mape": float(np.abs(gap(pre)).mean()),
            "holdout_week_gap_pct": float(gap(check).mean()),
            "gap_pct_by_day": [{"rel_day": i, "date": str(dt.date()), "gap_pct": float(g)}
                               for i, (dt, g) in enumerate(zip(post.index, gap(post)))]}


def main():
    d = daily_panel()
    es, fit = event_study(d, LANDFALL, weather=True)
    raw, _ = event_study(d, LANDFALL, weather=False)
    pre = es[es.rel_day < 0]

    placebos = []
    for year in (2013, 2014, 2015, 2016):
        start = LANDFALL.replace(year=year)
        pe, _ = event_study(d, start, weather=True)
        placebos.append({"year": year, **window_summary(pe, d, start, 1, 7)})

    main_w = window_summary(es, d, LANDFALL, 1, 7)
    raw_w = window_summary(raw, d, LANDFALL, 1, 7)
    full_w = window_summary(es, d, LANDFALL, 0, POST_DAYS - 1)
    sc = synthetic_control()
    sc_week = float(np.mean([g["gap_pct"] for g in sc["gap_pct_by_day"] if 1 <= g["rel_day"] <= 7]))

    coast = d[d.zone == TREATED].set_index("day")
    summary = {
        "event": "Hurricane Harvey", "landfall": str(LANDFALL.date()), "zone": config.ZONE_NAMES[TREATED],
        "panel": {"zone_days": int(len(d)), "zones": sorted(d.zone.unique()), "r2": float(fit.rsquared),
                  "start": str(d.day.min().date()), "end": str(d.day.max().date())},
        "first_week_after_landfall": {
            "raw_drop_pct": raw_w["avg_effect_pct"], "weather_adjusted_drop_pct": main_w["avg_effect_pct"],
            "share_of_raw_drop_explained_by_weather": float(1 - main_w["avg_effect_pct"] / raw_w["avg_effect_pct"]),
            "mwh_lost_weather_adjusted": -main_w["mwh_change"], "mwh_lost_raw": -raw_w["mwh_change"]},
        "three_weeks": {"avg_effect_pct": full_w["avg_effect_pct"], "mwh_lost": -full_w["mwh_change"]},
        "worst_day": es.loc[es.effect_pct.idxmin(), ["date", "effect_pct", "lo_pct", "hi_pct"]].to_dict(),
        "pre_trend": {"mean_pct": float(pre.effect_pct.mean()), "max_abs_pct": float(pre.effect_pct.abs().max()),
                      "days_with_ci_excluding_zero": int(((pre.lo_pct > 0) | (pre.hi_pct < 0)).sum())},
        "placebo_years_first_week_pct": {str(p["year"]): p["avg_effect_pct"] for p in placebos},
        "synthetic_control": {"first_week_gap_pct": sc_week, **{k: sc[k] for k in ("weights", "pre_fit_mape", "holdout_week_gap_pct")}},
        "weather_during_event": {"mean_tmax_week_before": float(coast.loc[LANDFALL - pd.Timedelta(days=7):LANDFALL - pd.Timedelta(days=1)].tmax.mean()),
                                 "mean_tmax_week_after": float(coast.loc[LANDFALL + pd.Timedelta(days=1):LANDFALL + pd.Timedelta(days=7)].tmax.mean())},
    }
    es = es.merge(raw[["rel_day", "effect_pct"]].rename(columns={"effect_pct": "raw_pct"}), on="rel_day")
    es = es.merge(pd.DataFrame(sc["gap_pct_by_day"])[["rel_day", "gap_pct"]].rename(columns={"gap_pct": "synth_pct"}),
                  on="rel_day", how="left")
    config.ARTIFACTS.mkdir(exist_ok=True)
    es.drop(columns="coef").to_csv(config.ARTIFACTS / "harvey_event_study.csv", index=False)
    (config.ARTIFACTS / "harvey_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(es.drop(columns="coef").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
