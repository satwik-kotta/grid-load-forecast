"""The AWS path, run offline: S3 is mocked with moto and the Athena SQL is executed with DuckDB."""
import io
import json

import boto3
import duckdb
import pandas as pd
import pytest
from moto import mock_aws

from aws import glue_job, run_athena
from src import config, quality
from src.etl import build_panel

BUCKET = "grid-load-test"
RAW = config.RAW


@pytest.fixture(scope="module")
def built():
    return build_panel()


@pytest.fixture(scope="module")
def s3_after_job():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for f in RAW.rglob("*"):
            if f.is_file():
                s3.upload_file(str(f), BUCKET, f"raw/{f.relative_to(RAW).as_posix()}")
        report = glue_job.main(["--bucket", BUCKET, "--job-bookmark-option", "job-bookmark-disable"])
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="curated/")["Contents"]]
        frames = {k: pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()))
                  for k in keys if k.endswith(".parquet")}
        stored = json.loads(s3.get_object(Bucket=BUCKET, Key="curated/quality/report.json")["Body"].read())
    return report, frames, stored


def test_every_check_passes_on_the_real_data(built):
    results = quality.run_checks(*built)
    assert len(results) == 8 and all(r["passed"] for r in results), results


def test_checks_catch_a_one_hour_shift_a_dropped_hour_and_a_broken_total(built):
    panel, load = built
    failed = lambda p, l: {r["check"] for r in quality.run_checks(p, l) if not r["passed"]}

    shifted = panel.assign(local=panel.local + pd.Timedelta(hours=1))
    assert "local_time_matches_utc" in failed(shifted, load)

    assert "panel_hourly_complete" in failed(panel.drop(index=panel.index[1000]), load)

    broken = load.copy()
    broken.iloc[500, broken.columns.get_loc("COAST")] += 250
    assert "zones_sum_to_ercot_total" in failed(panel, broken)

    # Load joined six hours off the clock: every row is present, but summer no longer peaks in the afternoon.
    off = panel.assign(load_mw=panel.groupby("zone").load_mw.shift(6).bfill())
    assert "summer_peak_in_local_afternoon" in failed(off, load)

    with pytest.raises(ValueError):
        quality.assert_clean(shifted, load)


def test_glue_job_writes_partitioned_parquet_and_a_report(s3_after_job):
    report, frames, stored = s3_after_job
    panel_keys = [k for k in frames if k.startswith("curated/panel/")]
    assert len(panel_keys) == 18                                   # 3 zones x 6 years
    assert "curated/panel/zone=COAST/year=2017/part.parquet" in frames
    assert sum(len(frames[k]) for k in panel_keys) == 3 * 45252 == report["rows"]["panel"]
    assert len([k for k in frames if k.startswith("curated/ercot_zones/")]) == 7
    one = frames["curated/panel/zone=COAST/year=2017/part.parquet"]
    assert str(one.ts.dtype) == "datetime64[ms]" and "local_ts" in one and "zone" not in one
    assert stored["rows"] == report["rows"] and all(c["passed"] for c in stored["checks"])


def test_glue_job_writes_nothing_when_a_check_fails(monkeypatch):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        panel, load = build_panel()
        monkeypatch.setattr(glue_job.etl, "build_panel", lambda: (panel.drop(index=panel.index[5]), load))
        with pytest.raises(RuntimeError, match="nothing written"):
            glue_job.main(["--bucket", BUCKET])
        assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET, Prefix="curated/")
    assert config.RAW == RAW


def test_athena_sql_runs_and_reports_no_failures(s3_after_job, tmp_path):
    _, frames, _ = s3_after_job
    for key, df in frames.items():
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    db = duckdb.connect()
    db.execute("CREATE SCHEMA grid_load")
    for table in ("panel", "ercot_zones"):
        db.execute(f"CREATE VIEW grid_load.{table} AS SELECT * FROM "
                   f"read_parquet('{tmp_path}/curated/{table}/**/*.parquet', hive_partitioning=true)")

    checks = db.execute(run_athena.statements("02_quality_checks.sql", BUCKET)[0]).fetchall()
    assert len(checks) == 5 and all(n == 0 for _, n in checks), checks

    db.execute(run_athena.statements("03_daily_peak_view.sql", BUCKET)[0])
    peak, hours = db.execute("SELECT MAX(peak_mw), MAX(hours) FROM grid_load.daily_peak WHERE zone = 'COAST'").fetchone()
    assert 20000 < peak < 20200 and hours == 25

    ddl = run_athena.statements("01_tables.sql", BUCKET)
    assert len(ddl) == 5 and f"s3://{BUCKET}/curated/panel/" in ddl[1]
