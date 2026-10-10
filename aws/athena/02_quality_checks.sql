-- Second line of checks, run on the curated tables themselves. Every row should report 0 failures.
SELECT 'panel: missing or duplicate hours' AS check_name,
       COALESCE(SUM(ABS(expected - n) + (n - distinct_n)), 0) AS failures
FROM (SELECT zone, COUNT(*) AS n, COUNT(DISTINCT ts) AS distinct_n,
             date_diff('hour', MIN(ts), MAX(ts)) + 1 AS expected
      FROM grid_load.panel GROUP BY zone) AS z
UNION ALL
SELECT 'panel: null load or weather', COUNT(*)
FROM grid_load.panel
WHERE load_mw IS NULL OR temp_c IS NULL OR humidity IS NULL OR wind IS NULL
UNION ALL
SELECT 'ercot_zones: zones do not sum to the system total', COUNT(*)
FROM grid_load.ercot_zones
WHERE ABS(coast + east + fwest + north + ncent + south + scent + west - ercot) >= 1
UNION ALL
SELECT 'panel: load differs from ercot_zones at the same UTC hour', COUNT(*)
FROM grid_load.panel AS p
JOIN grid_load.ercot_zones AS e ON p.ts = e.ts
WHERE ABS(p.load_mw - CASE p.zone WHEN 'COAST' THEN e.coast WHEN 'NCENT' THEN e.ncent WHEN 'SCENT' THEN e.scent END) > 0.001
UNION ALL
SELECT 'panel: local days longer than 25 hours', COUNT(*)
FROM (SELECT zone, CAST(local_ts AS date) AS local_day, COUNT(*) AS n
      FROM grid_load.panel GROUP BY zone, CAST(local_ts AS date)) AS d
WHERE n > 25
