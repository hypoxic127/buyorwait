-- ============================================================================
-- BuyOrWait — live alerts on merged data + attribution table (Final Sprint D3)
-- Run blocks 1-3 once in the BigQuery console.
-- Block 1 is ALSO the body of a BigQuery *scheduled query* (daily 03:30 +08,
-- i.e. 19:30 UTC, right after the nightly fetch): console → query → Schedule.
-- ============================================================================

-- 1) Recent alerts recomputed nightly from snapshot+delta merged daily -------
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
WHERE hist >= 7
  AND (neg_rate - base_m) / GREATEST(base_s, 1e-4) > 3
  AND n > 2 * base_n;

-- 2) Unified alerts view: snapshot (epoch-nanos) + recent (DATE) -------------
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

-- 3) Attribution results written by pipeline/attribute.py --------------------
CREATE TABLE IF NOT EXISTS `steam_intel.alert_causes` (
  appid        INT64,
  first_day    DATE,
  latest_day   DATE,
  n_neg        INT64,     -- negative reviews analyzed in the window
  terms        STRING,    -- comma-separated top distinctive terms
  summary      STRING,    -- one-line cause written by Gemini
  generated_at TIMESTAMP
);
