#!/usr/bin/env bash
# Upload the raw files and code, create or update the Glue job, run it, then set up Athena.
#
#   GLUE_ROLE_ARN=arn:aws:iam::<account>:role/<glue-role> ./aws/deploy.sh <bucket>
#
# The role needs read/write on the bucket and the AWSGlueServiceRole managed policy.
set -euo pipefail
BUCKET=${1:?usage: GLUE_ROLE_ARN=... ./aws/deploy.sh <bucket>}
ROLE=${GLUE_ROLE_ARN:?set GLUE_ROLE_ARN to the IAM role the Glue job runs as}
JOB=grid-load-etl
cd "$(dirname "$0")/.."

aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3 mb "s3://$BUCKET"
aws s3 sync data/raw "s3://$BUCKET/raw/" --exclude ".DS_Store"
rm -f /tmp/grid_src.zip && zip -qr /tmp/grid_src.zip src -x "*__pycache__*"
aws s3 cp /tmp/grid_src.zip "s3://$BUCKET/code/src.zip"
aws s3 cp aws/glue_job.py "s3://$BUCKET/code/glue_job.py"

COMMAND="Name=pythonshell,PythonVersion=3.9,ScriptLocation=s3://$BUCKET/code/glue_job.py"
ARGS="{\"--bucket\":\"$BUCKET\",\"--extra-py-files\":\"s3://$BUCKET/code/src.zip\",\"--additional-python-modules\":\"pandas==2.2.2,pyarrow==15.0.2,openpyxl==3.1.5\"}"
if aws glue get-job --job-name "$JOB" >/dev/null 2>&1; then
  aws glue update-job --job-name "$JOB" --job-update "Role=$ROLE,Command={$COMMAND},DefaultArguments=$ARGS,MaxCapacity=1.0"
else
  aws glue create-job --name "$JOB" --role "$ROLE" --command "$COMMAND" --default-arguments "$ARGS" --max-capacity 1.0
fi

RUN=$(aws glue start-job-run --job-name "$JOB" --query JobRunId --output text)
echo "Glue run $RUN started"
while :; do
  STATE=$(aws glue get-job-run --job-name "$JOB" --run-id "$RUN" --query JobRun.JobRunState --output text)
  case "$STATE" in
    SUCCEEDED) break ;;
    FAILED|STOPPED|TIMEOUT|ERROR) echo "Glue run $STATE"; aws glue get-job-run --job-name "$JOB" --run-id "$RUN" --query JobRun.ErrorMessage --output text; exit 1 ;;
  esac
  sleep 15
done
echo "Glue run succeeded"
python aws/run_athena.py --bucket "$BUCKET"
