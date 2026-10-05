from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"

# ERCOT weather zone -> the city whose weather station represents it.
ZONES = {"COAST": "Houston", "NCENT": "Dallas", "SCENT": "San Antonio"}
ZONE_NAMES = {"COAST": "Coast (Houston)", "NCENT": "North Central (Dallas)", "SCENT": "South Central (San Antonio)"}
ALL_ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
LOCAL_TZ = "America/Chicago"

TRAIN_END = "2016-06-30"      # train: Oct 2012 - Jun 2016
VAL_END = "2016-12-31"        # validation: Jul - Dec 2016 (model selection, early stopping)
                              # test: Jan - Nov 2017, touched once
HISTORY = 168                 # hours of history the transformer sees
HORIZON = 24                  # hours forecast, issued at local midnight for the day ahead
SEED = 11
