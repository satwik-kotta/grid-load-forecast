-- Daily summary used for analytics: the peak is the number a grid operator plans around.
CREATE OR REPLACE VIEW grid_load.daily_peak AS
SELECT zone,
       CAST(local_ts AS date) AS local_day,
       MAX(load_mw)           AS peak_mw,
       AVG(load_mw)           AS avg_mw,
       MAX(temp_c)            AS max_temp_c,
       COUNT(*)               AS hours
FROM grid_load.panel
GROUP BY zone, CAST(local_ts AS date)
