-- ============================================================================
-- BuyOrWait Evergreen Scores (Final Sprint, D1)
-- Run in BigQuery console (or: bq query --use_legacy_sql=false < docs/evergreen.sql)
-- Replace PROJECT with your project id if your default project differs.
--
-- Idea: pipeline.py now stores UNDECAYED per-day weight sums (w_sum, wv_sum)
-- in game_daily. Because the decay is exponential in age only, the playtime-
-- weighted score for ANY reference date is:
--     score(ref) = SUM(wv_sum_day * EXP(-(ref-day)/90))
--                / SUM(w_sum_day  * EXP(-(ref-day)/90)) * 100
-- History never needs rewriting; new days are just appended (daily_delta).
-- ============================================================================

-- 1) Incremental table fed nightly by pipeline/fetch_recent.py --------------
CREATE TABLE IF NOT EXISTS `steam_intel.daily_delta` (
  appid      INT64,
  day        DATE,
  n          INT64,
  pos        INT64,
  w_sum      FLOAT64,
  wv_sum     FLOAT64,
  fetched_at TIMESTAMP
);

-- 2) Unified daily view: snapshot + delta (delta wins on overlapping days) --
CREATE OR REPLACE VIEW `steam_intel.v_daily_all` AS
WITH base AS (
  SELECT appid,
         CASE 
           WHEN SAFE_CAST(date AS INT64) IS NOT NULL THEN DATE(TIMESTAMP_SECONDS(DIV(CAST(date AS INT64), 1000000000)))
           ELSE SAFE_CAST(date AS DATE)
         END AS day,
         n, pos, w_sum, wv_sum
  FROM `steam_intel.game_daily`
),
delta AS (
  SELECT appid, day, n, pos, w_sum, wv_sum
  FROM `steam_intel.daily_delta`
)
SELECT * FROM base b
WHERE NOT EXISTS (SELECT 1 FROM delta d WHERE d.appid = b.appid AND d.day = b.day)
UNION ALL
SELECT * FROM delta;

-- 3) Evergreen scores: anchored to CURRENT_DATE, always fresh ---------------
CREATE OR REPLACE VIEW `steam_intel.v_scores_live` AS
SELECT
  a.appid,
  s.game,
  a.score_live,
  s.raw_pos_rate,
  a.recent_n_live,
  a.recent_pos_rate_live,
  a.n_reviews_total,
  a.last_review_day
FROM (
  SELECT
    appid,
    SAFE_DIVIDE(
      SUM(wv_sum * EXP(-DATE_DIFF(CURRENT_DATE(), day, DAY) / 90.0)),
      SUM(w_sum  * EXP(-DATE_DIFF(CURRENT_DATE(), day, DAY) / 90.0))
    ) * 100 AS score_live,
    SUM(n) AS n_reviews_total,
    SUM(IF(DATE_DIFF(CURRENT_DATE(), day, DAY) <= 90, n, 0))  AS recent_n_live,
    SAFE_DIVIDE(
      SUM(IF(DATE_DIFF(CURRENT_DATE(), day, DAY) <= 90, pos, 0)),
      SUM(IF(DATE_DIFF(CURRENT_DATE(), day, DAY) <= 90, n, 0))
    ) * 100 AS recent_pos_rate_live,
    MAX(day) AS last_review_day
  FROM `steam_intel.v_daily_all`
  GROUP BY appid
) a
LEFT JOIN `steam_intel.game_scores` s USING (appid);

-- 4) Freshness badge for the app --------------------------------------------
CREATE OR REPLACE VIEW `steam_intel.v_freshness` AS
SELECT
  MAX(fetched_at)          AS last_sync,
  COUNT(DISTINCT appid)    AS tracked_games,
  SUM(n)                   AS delta_reviews
FROM `steam_intel.daily_delta`;

-- ============================================================================
-- 5) ALIGNMENT CHECK (run once after reloading game_daily with w_sum/wv_sum).
-- Anchored at the snapshot's own max day, the evergreen formula must
-- reproduce pipeline scores. Expect: max_abs_diff < 0.5, most rows ≈ 0.
-- (Small drift is float32-sum vs float64-sum precision, not a logic bug.)
-- ============================================================================
WITH d AS (
  SELECT appid,
         CASE 
           WHEN SAFE_CAST(date AS INT64) IS NOT NULL THEN DATE(TIMESTAMP_SECONDS(DIV(CAST(date AS INT64), 1000000000)))
           ELSE SAFE_CAST(date AS DATE)
         END AS day,
         w_sum, wv_sum
  FROM `steam_intel.game_daily`
),
ref AS (SELECT MAX(day) AS r FROM d),
recalc AS (
  SELECT appid,
         SAFE_DIVIDE(
           SUM(wv_sum * EXP(-DATE_DIFF((SELECT r FROM ref), day, DAY) / 90.0)),
           SUM(w_sum  * EXP(-DATE_DIFF((SELECT r FROM ref), day, DAY) / 90.0))
         ) * 100 AS score_recalc
  FROM d GROUP BY appid
)
SELECT
  COUNT(*)                          AS games,
  MAX(ABS(s.score - r.score_recalc)) AS max_abs_diff,
  AVG(ABS(s.score - r.score_recalc)) AS mean_abs_diff
FROM `steam_intel.game_scores` s
JOIN recalc r USING (appid)
WHERE s.n_reviews >= 10;
