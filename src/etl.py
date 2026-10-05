"""Build one clean hourly panel from ERCOT load files and city weather.

    python -m src.etl

ERCOT publishes load by "hour ending" in Central Prevailing Time; the weather
feed is in UTC. Everything is aligned on the UTC hour the interval *starts*.
"""
import re
import zipfile

import numpy as np
import pandas as pd

from . import config

RENAME = {"FAR_WEST": "FWEST", "NORTH_C": "NCENT", "SOUTHERN": "SOUTH", "SOUTH_C": "SCENT"}
CELL = re.compile(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(?:<f>.*?</f>)?<v>(.*?)</v></c>)', re.S)
ROW = re.compile(r"<row [^>]*>(.*?)</row>", re.S)


def read_strict_xlsx(path):
    """Minimal reader for 'Strict Open XML' workbooks, which openpyxl cannot open."""
    with zipfile.ZipFile(path) as z:
        shared = re.findall(r"<t[^>]*>(.*?)</t>", z.read("xl/sharedStrings.xml").decode())
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for body in ROW.findall(sheet):
        rec = {}
        for col, attrs, val in CELL.findall(body):
            if val == "":
                continue
            rec[col] = shared[int(val)] if 't="s"' in attrs else val
        rows.append(rec)
    cols = sorted(rows[0], key=lambda c: (len(c), c))
    df = pd.DataFrame([[r.get(c) for c in cols] for r in rows[1:]], columns=[rows[0][c] for c in cols])
    return df


def read_ercot_year(path):
    try:
        df = pd.read_excel(path)
    except ValueError:
        df = read_strict_xlsx(path)
    df = df.rename(columns={df.columns[0]: "hour_ending"}).rename(columns=RENAME)
    df = df.dropna(subset=["hour_ending"])
    he = df["hour_ending"].astype(str).str.strip()
    dst_flag = he.str.contains("DST")                      # ERCOT marks the repeated fall-back hour
    he = he.str.replace("DST", "", regex=False).str.strip()
    is24 = he.str.contains(r"\b24:00")                     # "12/31/2017 24:00" = midnight next day
    ts = pd.to_datetime(he.str.replace("24:00", "00:00", regex=False), format="mixed").dt.round("h")
    ts = ts + pd.to_timedelta(is24.astype(int), unit="D")
    out = df[config.ALL_ZONES + ["ERCOT"]].apply(pd.to_numeric, errors="coerce")
    out.insert(0, "local_start", ts - pd.Timedelta(hours=1))
    out.insert(1, "dst_flag", dst_flag.to_numpy())
    return out


def to_utc(df):
    """Local hour-start -> UTC, resolving the two daylight-saving edge cases."""
    df = df.sort_values("local_start", kind="mergesort").reset_index(drop=True)
    dup = df["local_start"].duplicated(keep=False)
    # Fall back: 01:00 occurs twice. The first is daylight time, the second standard.
    first = dup & ~df["local_start"].duplicated(keep="first")
    ambiguous = np.where(dup, first, True)
    ambiguous = np.where(df["dst_flag"] & ~dup, False, ambiguous)
    # Spring forward: the 2012-2016 files label the 01:00-02:00 hour as ending 03:00,
    # which puts its start at a local time that does not exist. Shift it back one hour.
    loc = df["local_start"].dt.tz_localize(config.LOCAL_TZ, ambiguous=ambiguous,
                                           nonexistent=pd.Timedelta(hours=-1))
    df = df.assign(ts=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    df = df.drop(columns=["local_start", "dst_flag"]).drop_duplicates("ts").set_index("ts").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "ts"
    return df.interpolate(limit=2)             # the repeated fall-back hour is blank in some years


def load_ercot():
    frames = [read_ercot_year(p) for p in sorted((config.RAW / "ercot").glob("Native_Load_*.xlsx"))]
    return to_utc(pd.concat(frames, ignore_index=True))


CLOUD = {  # OpenWeatherMap description -> rough sky cover fraction
    "sky is clear": 0.0, "few clouds": 0.2, "scattered clouds": 0.4, "broken clouds": 0.75,
    "overcast clouds": 1.0, "haze": 0.4, "smoke": 0.5, "dust": 0.5, "mist": 0.7, "fog": 0.9,
}


def cloud_fraction(desc):
    if not isinstance(desc, str):
        return np.nan
    if desc in CLOUD:
        return CLOUD[desc]
    return 0.95 if any(w in desc for w in ("rain", "thunder", "drizzle", "snow", "squall", "sleet")) else 0.5


def solar_elevation(ts_utc, lat, lon):
    """Sun elevation in degrees from time and position (no measured irradiance in the data)."""
    doy = ts_utc.dayofyear.to_numpy()
    hour = ts_utc.hour.to_numpy() + 0.5                      # middle of the hour
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    hour_angle = np.radians(15.0 * (hour + lon / 15.0 - 12.0))
    lat_r = np.radians(lat)
    sin_el = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def load_weather():
    w = config.RAW / "weather"
    read = lambda name: pd.read_csv(w / f"{name}.csv", parse_dates=["datetime"]).set_index("datetime")
    temp, hum, wind, desc = (read(n) for n in ("temperature", "humidity", "wind_speed", "weather_description"))
    attrs = pd.read_csv(w / "city_attributes.csv").set_index("City")
    frames = []
    for zone, city in config.ZONES.items():
        f = pd.DataFrame({"temp_c": temp[city] - 273.15, "humidity": hum[city], "wind": wind[city],
                          "cloud": desc[city].map(cloud_fraction), "raining": desc[city].astype(str).str.contains(
                              "rain|thunder|drizzle|squall").astype(float)})
        f = f.iloc[1:]                                        # first row of the feed is empty
        gaps = f["temp_c"].isna().sum()
        f = f.interpolate(limit=6).ffill().bfill()
        el = solar_elevation(f.index, attrs.loc[city, "Latitude"], attrs.loc[city, "Longitude"])
        f["sun_elev"] = np.clip(el, 0, None)
        f["solar_proxy"] = np.sin(np.radians(f["sun_elev"])) * (1 - 0.75 * f["cloud"] ** 3)   # Kasten-Czeplak
        f["zone"] = zone
        f.attrs["gaps"] = int(gaps)
        frames.append(f)
    return frames


def build_panel():
    load = load_ercot()
    rows = []
    for f in load_weather():
        zone = f["zone"].iloc[0]
        j = f.join(load[[zone]].rename(columns={zone: "load_mw"}), how="inner")
        rows.append(j)
    panel = pd.concat(rows).reset_index().rename(columns={"index": "ts", "datetime": "ts"})
    panel["local"] = panel["ts"].dt.tz_localize("UTC").dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None)
    return panel.sort_values(["zone", "ts"]).reset_index(drop=True), load


def main():
    panel, load = build_panel()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PROCESSED / "panel.parquet", index=False)
    load.to_parquet(config.PROCESSED / "ercot_all_zones.parquet")
    print(f"ERCOT hours: {len(load):,} ({load.index.min()} -> {load.index.max()} UTC), "
          f"missing hours: {len(pd.date_range(load.index.min(), load.index.max(), freq='h')) - len(load)}")
    for z, g in panel.groupby("zone"):
        full = pd.date_range(g.ts.min(), g.ts.max(), freq="h")
        print(f"{z}: {len(g):,} hours {g.ts.min()} -> {g.ts.max()}, gaps {len(full) - len(g)}, "
              f"load {g.load_mw.min():,.0f}-{g.load_mw.max():,.0f} MW, null load {g.load_mw.isna().sum()}, "
              f"corr(temp, load) {g.temp_c.corr(g.load_mw):.2f}")


if __name__ == "__main__":
    main()
