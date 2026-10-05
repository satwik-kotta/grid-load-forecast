"""Train the forecasters and score them on the held-out test period.

    python -m src.forecast

Models
  naive_week   same hour, same weekday last week
  naive_day    same hour yesterday
  gbm          LightGBM on lagged load, weather and calendar features
  transformer  encoder-decoder transformer: 168h of history in, 24h out,
               with known-future weather/calendar covariates and P10/P50/P90 outputs
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config
from .features import GBM_FEATURES, ORIGIN_HOUR_UTC, WEATHER, load_panel, split_of, tabular

QUANTILES = [0.1, 0.5, 0.9]


# ------------------------------------------------------------------ LightGBM
def fit_gbm(t):
    tr, va = t[t.split == "train"], t[t.split == "val"]
    model = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=config.SEED, verbose=-1)
    model.fit(tr[GBM_FEATURES], tr["y"], eval_set=[(va[GBM_FEATURES], va["y"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


# ------------------------------------------------------------------ Transformer
ENC_COV = WEATHER + ["holiday"]
CAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


def sequence_arrays(panel):
    """Per zone: aligned hourly arrays the windows are cut from."""
    zones = {}
    stats = {}
    train_mask = panel.ts <= pd.Timestamp(config.TRAIN_END) + pd.Timedelta(hours=23)
    for c in WEATHER:
        stats[c] = (float(panel.loc[train_mask, c].mean()), float(panel.loc[train_mask, c].std()))
    for zi, (zone, g) in enumerate(panel.groupby("zone")):
        g = g.sort_values("ts").reset_index(drop=True)
        cov = np.column_stack([(g[c] - stats[c][0]) / stats[c][1] for c in WEATHER] + [g["holiday"]])
        cal = np.column_stack([np.sin(2 * np.pi * g.hour / 24), np.cos(2 * np.pi * g.hour / 24),
                               np.sin(2 * np.pi * g.dow / 7), np.cos(2 * np.pi * g.dow / 7),
                               np.sin(2 * np.pi * g.doy / 365.25), np.cos(2 * np.pi * g.doy / 365.25)])
        zones[zone] = {"ts": g.ts.to_numpy(), "load": g.load_mw.to_numpy(dtype="float32"),
                       "cov": cov.astype("float32"), "cal": cal.astype("float32"), "zi": zi}
    return zones


class Windows(torch.utils.data.Dataset):
    def __init__(self, zones, origins):
        self.zones, self.items = zones, origins          # origins: [(zone, index of first target hour)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        zone, o = self.items[i]
        z = self.zones[zone]
        H, F = config.HISTORY, config.HORIZON
        hist = z["load"][o - H:o]
        level = hist[-24:].mean()                         # same normaliser the GBM uses
        enc = np.column_stack([np.log(hist / level), z["cov"][o - H:o], z["cal"][o - H:o]])
        dec = np.column_stack([z["cov"][o:o + F], z["cal"][o:o + F]])
        y = np.log(z["load"][o:o + F] / level)
        return (torch.from_numpy(enc.astype("float32")), torch.from_numpy(dec.astype("float32")),
                torch.tensor(z["zi"]), torch.from_numpy(y.astype("float32")), torch.tensor(level))


class LoadTransformer(nn.Module):
    def __init__(self, n_enc, n_dec, n_zones, d=64, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.enc_in, self.dec_in = nn.Linear(n_enc, d), nn.Linear(n_dec, d)
        self.zone = nn.Embedding(n_zones, d)
        self.pos = nn.Parameter(torch.randn(config.HISTORY + config.HORIZON, d) * 0.02)
        self.core = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=layers, num_decoder_layers=layers,
                                   dim_feedforward=4 * d, dropout=drop, batch_first=True)
        self.out = nn.Linear(d, len(QUANTILES))

    def forward(self, enc, dec, zone):
        z = self.zone(zone).unsqueeze(1)
        H = enc.shape[1]
        src = self.enc_in(enc) + self.pos[:H] + z
        tgt = self.dec_in(dec) + self.pos[H:H + dec.shape[1]] + z
        return self.out(self.core(src, tgt))              # no causal mask: decoder inputs are all known in advance


def pinball(pred, y):
    q = torch.tensor(QUANTILES, device=pred.device)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def origin_items(zones, lo, hi, every=24):
    """Window start indices whose target day starts in [lo, hi]. every=24 -> only 06:00 UTC origins."""
    items = []
    for zone, z in zones.items():
        ts = pd.DatetimeIndex(z["ts"])
        ok = np.where((ts >= lo) & (ts <= hi))[0]
        ok = ok[(ok >= config.HISTORY) & (ok + config.HORIZON <= len(ts))]
        if every == 24:
            ok = ok[ts[ok].hour == ORIGIN_HOUR_UTC]
        else:
            ok = ok[(ts[ok].hour - ORIGIN_HOUR_UTC) % every == 0]
        items += [(zone, int(i)) for i in ok]
    return items


def predict_tf(model, zones, items):
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(Windows(zones, items), batch_size=512)
    k = 0
    with torch.no_grad():
        for enc, dec, zi, y, level in loader:
            q = torch.exp(model(enc, dec, zi)) * level.view(-1, 1, 1)
            q = torch.sort(q, dim=-1).values.numpy()       # enforce P10 <= P50 <= P90
            for b in range(len(q)):
                zone, o = items[k]; k += 1
                rows.append(pd.DataFrame({"zone": zone, "ts": zones[zone]["ts"][o:o + config.HORIZON],
                                          "tf_p10": q[b, :, 0], "tf_p50": q[b, :, 1], "tf_p90": q[b, :, 2]}))
    return pd.concat(rows, ignore_index=True)


def fit_transformer(panel, epochs=10, log=print):
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    zones = sequence_arrays(panel)
    t0 = pd.Timestamp(panel.ts.min())
    tr_end = pd.Timestamp(config.TRAIN_END) + pd.Timedelta(hours=23)
    va_end = pd.Timestamp(config.VAL_END) + pd.Timedelta(hours=23)
    train = origin_items(zones, t0, tr_end - pd.Timedelta(hours=config.HORIZON), every=6)   # four origins a day for training
    val = origin_items(zones, tr_end + pd.Timedelta(hours=1), va_end)
    model = LoadTransformer(1 + len(ENC_COV) + len(CAL), len(ENC_COV) + len(CAL), len(zones))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(Windows(zones, train), batch_size=128, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(loader))
    vload = torch.utils.data.DataLoader(Windows(zones, val), batch_size=512)
    best, best_state, history = 1e9, None, []
    for ep in range(epochs):
        model.train(); t = time.time(); tot = 0.0
        for enc, dec, zi, y, _ in loader:
            opt.zero_grad()
            loss = pinball(model(enc, dec, zi), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([pinball(model(e, d, z), y).item() for e, d, z, y, _ in vload]))
        history.append({"epoch": ep + 1, "train": tot / len(loader), "val": vl})
        log(f"  epoch {ep + 1:2d} train {tot / len(loader):.4f} val {vl:.4f} ({time.time() - t:.0f}s)")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, zones, history, len(train)


# ------------------------------------------------------------------ scoring
def score(df, col):
    e = df[col] - df["actual"]
    daily = df.groupby(["zone", "origin"]).agg(a=("actual", "max"), p=(col, "max"))
    return {"mape": float((e.abs() / df["actual"]).mean() * 100), "mae_mw": float(e.abs().mean()),
            "rmse_mw": float(np.sqrt((e ** 2).mean())), "bias_mw": float(e.mean()),
            "peak_mape": float(((daily.p - daily.a).abs() / daily.a).mean() * 100)}


def main():
    panel = load_panel()
    t = tabular(panel)
    print(f"tabular rows: {len(t):,}; days per split:",
          (t.groupby("split")["origin"].nunique()).to_dict())

    gbm = fit_gbm(t)
    print(f"gbm trees: {gbm.best_iteration_}")
    f = t[t.split != "train"][["zone", "ts", "origin", "lead", "split", "load_mw", "level_24", "lag_24", "lag_168",
                               "temp_c"]].copy()
    f = f.rename(columns={"load_mw": "actual", "lag_168": "naive_week", "lag_24": "naive_day"})
    f["gbm"] = np.exp(gbm.predict(t.loc[f.index, GBM_FEATURES])) * f["level_24"]

    print("training transformer")
    model, zones, history, n_train = fit_transformer(panel)
    items = origin_items(zones, pd.Timestamp(config.TRAIN_END) + pd.Timedelta(hours=24), pd.Timestamp(panel.ts.max()))
    f = f.merge(predict_tf(model, zones, items), on=["zone", "ts"], how="inner")
    f["transformer"] = f["tf_p50"]

    models = ["naive_week", "naive_day", "gbm", "transformer"]
    metrics = {"setup": {"origin": "06:00 UTC daily", "horizon_hours": config.HORIZON,
                         "train_days": int(t[t.split == "train"].origin.nunique()),
                         "val_days": int(f[f.split == "val"].origin.nunique()),
                         "test_days": int(f[f.split == "test"].origin.nunique()),
                         "test_start": str(f[f.split == "test"].ts.min()), "test_end": str(f[f.split == "test"].ts.max()),
                         "transformer_train_windows": n_train,
                         "transformer_params": int(sum(p.numel() for p in model.parameters())),
                         "gbm_trees": int(gbm.best_iteration_)},
               "transformer_history": history}
    for split in ("val", "test"):
        d = f[f.split == split]
        metrics[split] = {m: score(d, m) for m in models}
        metrics[split]["transformer"]["p10_p90_coverage"] = float(((d.actual >= d.tf_p10) & (d.actual <= d.tf_p90)).mean())
        metrics[split]["by_zone"] = {z: {m: score(g, m)["mape"] for m in models} for z, g in d.groupby("zone")}
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=GBM_FEATURES)
    metrics["gbm_top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict()

    config.ARTIFACTS.mkdir(exist_ok=True)
    f.drop(columns=["level_24"]).to_parquet(config.ARTIFACTS / "forecasts.parquet", index=False)
    (config.ARTIFACTS / "forecast_metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save(model.state_dict(), config.ARTIFACTS / "transformer.pt")
    for split in ("val", "test"):
        print(split)
        for m in models:
            s = metrics[split][m]
            print(f"  {m:12s} MAPE {s['mape']:5.2f}%  MAE {s['mae_mw']:6.0f} MW  peak MAPE {s['peak_mape']:5.2f}%"
                  + (f"  P10-P90 coverage {s['p10_p90_coverage']:.2f}" if m == "transformer" else ""))


if __name__ == "__main__":
    main()
