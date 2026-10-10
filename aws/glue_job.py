"""AWS Glue Python shell job: raw files in S3 -> checked, partitioned Parquet in S3.

    s3://<bucket>/raw/ercot/*.xlsx, raw/weather/*.csv
        -> same transform as `python -m src.etl`
        -> data-quality and time-alignment checks (the job fails here and writes nothing if one breaks)
        -> s3://<bucket>/curated/panel/zone=<Z>/year=<Y>/part.parquet
           s3://<bucket>/curated/ercot_zones/year=<Y>/part.parquet
           s3://<bucket>/curated/quality/report.json

Athena reads the curated prefix (see aws/athena/). Job arguments: --bucket, and optionally
--raw_prefix and --curated_prefix. The `src` package is shipped with --extra-py-files.
"""
import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import boto3
import pandas as pd

from src import config, etl, quality


def download_raw(s3, bucket, prefix, dest):
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
            n += 1
    return n


def clear_prefix(s3, bucket, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})


def put_parquet(s3, bucket, key, df):
    for c in df.select_dtypes("datetime").columns:          # Athena cannot read nanosecond timestamps
        df = df.assign(**{c: df[c].astype("datetime64[ms]")})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--raw_prefix", default="raw/")
    p.add_argument("--curated_prefix", default="curated/")
    args, _ = p.parse_known_args(argv)                      # Glue passes extra arguments of its own
    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp:
        files = download_raw(s3, args.bucket, args.raw_prefix, Path(tmp))
        print(f"downloaded {files} raw files from s3://{args.bucket}/{args.raw_prefix}")
        original, config.RAW = config.RAW, Path(tmp)
        try:
            panel, load = etl.build_panel()
        finally:
            config.RAW = original

    results = quality.run_checks(panel, load)
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['check']}: {r['detail']}")
    failed = [r["check"] for r in results if not r["passed"]]
    if failed:
        raise RuntimeError(f"data-quality checks failed, nothing written: {failed}")

    out = args.curated_prefix
    clear_prefix(s3, args.bucket, out)                      # full refresh, so reruns do not duplicate rows
    panel = panel.rename(columns={"local": "local_ts"})
    for (zone, year), g in panel.groupby(["zone", panel.ts.dt.year]):
        put_parquet(s3, args.bucket, f"{out}panel/zone={zone}/year={year}/part.parquet", g.drop(columns="zone"))
    zones = load.rename(columns=str.lower).reset_index()
    for year, g in zones.groupby(zones.ts.dt.year):
        put_parquet(s3, args.bucket, f"{out}ercot_zones/year={year}/part.parquet", g)
    report = {"rows": {"panel": len(panel), "ercot_zones": len(zones)}, "checks": results}
    s3.put_object(Bucket=args.bucket, Key=f"{out}quality/report.json", Body=json.dumps(report, indent=2).encode())
    print(f"wrote {len(panel):,} panel rows and {len(zones):,} ERCOT rows to s3://{args.bucket}/{out}")
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
