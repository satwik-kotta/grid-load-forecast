# Data

`raw/ercot/` ERCOT hourly native load by weather zone, 2012-2017, as published at
https://www.ercot.com/gridinfo/load/load_hist. Copied from the mirror in
https://github.com/mikemueller-data/ercot-load-forecasting. Four of the six files are in
Strict Open XML format; `src/etl.py` has a small reader for them.

`raw/weather/` hourly temperature, humidity, wind speed and sky description for Houston,
Dallas and San Antonio, Oct 2012 - Nov 2017. Extracted from the Kaggle dataset
"Historical Hourly Weather Data 2012-2017" (OpenWeatherMap observations, ODbL licence).
About 3.5% of hours repeat the previous reading, with flat stretches up to 42 hours where
the feed dropped out.

There is no measured cloud cover or solar irradiance in this feed. Sky cover is mapped from
the text description and sun elevation is computed from time and location.
