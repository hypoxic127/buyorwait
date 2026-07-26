-- ============================================================================
-- BuyOrWait — live review-bombing alerts
-- Run after docs/evergreen.sql (this depends on v_daily_all).
--   bq query --use_legacy_sql=false < docs/alerts_live.sql
--
-- pipeline.py detects bombing on the 2023 snapshot. Block 1 re-runs the SAME
-- rule in SQL over snapshot+delta merged daily data, so events that happen
-- after the snapshot are detected too; block 2 unions old and new so the app
-- and attribute.py read one object (v_alerts_all).
--
-- Block 1 is also the body of a BigQuery SCHEDULED QUERY — set it to run daily
-- at 03:30 Asia/Singapore, right after fetch_recent.py tops up daily_delta.
-- ============================================================================

-- 1) Recent alerts, recomputed nightly ---------------------------------------
CREATE OR REPLACE TABLE `steam_intel.alerts_recent` AS
WITH d AS (
  SELECT appid, day, n, pos, SAFE_DIVIDE(n - pos, n) AS neg_rate
  FROM `steam_intel.v_daily_all`
  WHERE day >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
),
w AS (
  SELECT *,
    AVG(neg_rate)    OVER win30 AS base_m,
    STDDEV(neg_rate) OVER win30 AS base_s,
    AVG(n)           OVER win30 AS base_n,
    COUNT(*)         OVER win30 AS hist
  FROM d
  WINDOW win30 AS (PARTITION BY appid ORDER BY day
                   ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
)
SELECT appid, day, n, neg_rate,
       (neg_rate - base_m) / GREATEST(base_s, 1e-4) AS z,
       base_m AS base_neg_rate
FROM w
WHERE hist >= 7                                              -- enough history
  AND (neg_rate - base_m) / GREATEST(base_s, 1e-4) > 3       -- z > 3
  AND n > 2 * base_n;                                        -- and unusual volume

-- 2) One object for the app: snapshot alerts + nightly alerts -----------------
--    alerts.date is epoch NANOSECONDS (Parquet load); alerts_recent.day is DATE.
CREATE OR REPLACE VIEW `steam_intel.v_alerts_all` AS
SELECT a.appid, s.game,
       DATE(TIMESTAMP_SECONDS(DIV(a.date, 1000000000))) AS day,
       a.n, a.neg_rate, a.z, a.base_neg_rate
FROM `steam_intel.alerts` a
LEFT JOIN `steam_intel.game_scores` s USING (appid)
UNION ALL
SELECT r.appid, s.game, r.day, r.n, r.neg_rate, r.z, r.base_neg_rate
FROM `steam_intel.alerts_recent` r
LEFT JOIN `steam_intel.game_scores` s USING (appid);

-- 3) Cause attribution, written by pipeline/attribute.py ---------------------
CREATE TABLE IF NOT EXISTS `steam_intel.alert_causes` (
  appid        INT64,
  first_day    DATE,
  latest_day   DATE,
  n_neg        INT64,     -- negative reviews analysed in the window
  terms        STRING,    -- distinctive terms vs the 60-day baseline
  summary      STRING,    -- one-line cause, written by Gemini
  generated_at TIMESTAMP
);

-- 4) Anonymous usage events written by the app -------------------------------
CREATE TABLE IF NOT EXISTS `steam_intel.usage_events` (
  ts    TIMESTAMP,
  event STRING,
  game  STRING,
  appid INT64
);
