"""Create the Athena tables, run the quality checks on them, and build the daily view.

    python aws/run_athena.py --bucket <bucket> [--workgroup primary]

Exits with status 1 if any check reports failures.
"""
import argparse
import sys
import time
from pathlib import Path

import boto3

SQL = Path(__file__).parent / "athena"


def run(athena, sql, bucket, workgroup):
    qid = athena.start_query_execution(
        QueryString=sql, WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": f"s3://{bucket}/athena-results/"})["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(f"{status['State']}: {status.get('StateChangeReason')}\n{sql}")
    return qid


def statements(name, bucket):
    text = (SQL / name).read_text().replace("${BUCKET}", bucket)
    lines = [l for l in text.splitlines() if not l.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--workgroup", default="primary")
    a = p.parse_args()
    athena = boto3.client("athena")

    for sql in statements("01_tables.sql", a.bucket):
        run(athena, sql, a.bucket, a.workgroup)
    qid = run(athena, statements("02_quality_checks.sql", a.bucket)[0], a.bucket, a.workgroup)
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"][1:]
    failures = 0
    for r in rows:
        name, n = (c.get("VarCharValue") for c in r["Data"])
        failures += int(n)
        print(f"{'PASS' if int(n) == 0 else 'FAIL'}  {name}: {n}")
    if failures:
        sys.exit(1)
    for sql in statements("03_daily_peak_view.sql", a.bucket):
        run(athena, sql, a.bucket, a.workgroup)
    print("tables, checks and grid_load.daily_peak view are ready")


if __name__ == "__main__":
    main()
