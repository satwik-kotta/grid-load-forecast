#!/usr/bin/env bash
# Create the IAM role the Glue job runs as, limited to one bucket. Prints the role ARN.
#
#   ./aws/setup_role.sh <bucket>
set -euo pipefail
BUCKET=${1:?usage: ./aws/setup_role.sh <bucket>}
ROLE=grid-load-glue
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"glue.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
S3="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"},{\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetBucketLocation\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"}]}"
aws iam get-role --role-name "$ROLE" >/dev/null 2>&1 || aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" >/dev/null
aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
aws iam put-role-policy --role-name "$ROLE" --policy-name grid-load-bucket --policy-document "$S3"
aws iam get-role --role-name "$ROLE" --query Role.Arn --output text
