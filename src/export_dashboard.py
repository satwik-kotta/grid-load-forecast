"""Write docs/data.js for the static dashboard.

    python -m src.export_dashboard
"""
import json

import numpy as np
import pandas as pd

from . import config
from .features import load_panel


def main():
    f = pd.read_parquet(config.ARTIFACTS / "forecasts.parquet")
    metrics = json.loads((config.ARTIFACTS / "forecast_metrics.json").read_text())
    harvey = json.loads((config.ARTIFACTS / "harvey_summary.json").read_text())
    es = pd.read_csv(config.ARTIFACTS / "harvey_event_study.csv")
    panel = load_panel()
    test = f[f.split == "test"].sort_values(["zone", "ts"])
    local = test.ts.dt.tz_localize("UTC").dt.tz_convert(config.LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M")

    train = panel[panel.ts <= pd.Timestamp(config.TRAIN_END) + pd.Timedelta(hours=23)]
    p99 = train.groupby("zone").load_mw.quantile(0.99)
    zones = {}
    for z, g in test.groupby("zone"):
        r = lambda c: g[c].round(0).astype(int).tolist()
        day = g.groupby("origin").agg(peak_fc=("transformer", "max"), peak_act=("actual", "max"), tmax=("temp_c", "max"))
        zones[z] = {"name": config.ZONE_NAMES[z], "p99_mw": float(p99[z]), "t": local[g.index].tolist(),
                    "actual": r("actual"), "transformer": r("tf_p50"), "p10": r("tf_p10"), "p90": r("tf_p90"),
                    "gbm": r("gbm"), "naive": r("naive_week"),
                    "days": [str((d + pd.Timedelta(hours=6)).date()) for d in day.index],
                    "strain_fc": (100 * day.peak_fc / p99[z]).round(1).tolist(),
                    "strain_act": (100 * day.peak_act / p99[z]).round(1).tolist(),
                    "tmax": day.tmax.round(1).tolist()}

    # temperature response: mean load (as % of zone average) by 2-degree temperature bin, training data
    bins = np.arange(-6, 42, 2)
    curve = {}
    for z, g in train.groupby("zone"):
        b = pd.cut(g.temp_c, bins)
        m = g.groupby(b, observed=True).load_mw.agg(["mean", "size"])
        m = m[m["size"] >= 50]
        curve[z] = [[float(i.mid), round(float(100 * v / g.load_mw.mean()), 1)] for i, v in m["mean"].items()]

    data = {"metrics": metrics, "harvey": harvey, "event_study": json.loads(es.round(2).to_json(orient="records")),
            "zones": zones, "temp_curve": curve, "zone_names": config.ZONE_NAMES}
    config.DOCS.mkdir(exist_ok=True)
    (config.DOCS / "data.js").write_text("window.GRID_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote docs/data.js ({(config.DOCS / 'data.js').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
