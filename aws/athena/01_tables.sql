-- External tables over the Parquet files the Glue job writes. ${BUCKET} is filled in by run_athena.py.
CREATE DATABASE IF NOT EXISTS grid_load;

CREATE EXTERNAL TABLE IF NOT EXISTS grid_load.panel (
  ts          timestamp COMMENT 'UTC hour the interval starts',
  temp_c      double,
  humidity    double,
  wind        double,
  cloud       double,
  raining     double,
  sun_elev    double,
  solar_proxy double,
  load_mw     double,
  local_ts    timestamp COMMENT 'Central prevailing time'
)
PARTITIONED BY (zone string, year int)
STORED AS PARQUET
LOCATION 's3://${BUCKET}/curated/panel/';

MSCK REPAIR TABLE grid_load.panel;

CREATE EXTERNAL TABLE IF NOT EXISTS grid_load.ercot_zones (
  ts    timestamp COMMENT 'UTC hour the interval starts',
  coast double, east double, fwest double, north double,
  ncent double, south double, scent double, west double,
  ercot double COMMENT 'system total as published'
)
PARTITIONED BY (year int)
STORED AS PARQUET
LOCATION 's3://${BUCKET}/curated/ercot_zones/';

MSCK REPAIR TABLE grid_load.ercot_zones;
