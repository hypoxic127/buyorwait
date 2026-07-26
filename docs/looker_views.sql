-- Convenience views for Looker Studio.
-- The raw `date` columns are epoch nanoseconds (INT64, from Parquet load);
-- these views expose them as proper DATE so Looker's date controls work.
-- Run once:  bq query --use_legacy_sql=false < docs/looker_views.sql

CREATE OR REPLACE VIEW `steam_intel.v_game_daily` AS
SELECT appid,
       CASE 
         WHEN SAFE_CAST(date AS INT64) IS NOT NULL THEN DATE(TIMESTAMP_SECONDS(DIV(CAST(date AS INT64), 1000000000)))
         ELSE SAFE_CAST(date AS DATE)
       END AS day,
       n, pos, neg, pos_rate, neg_rate
FROM `steam_intel.game_daily`;

CREATE OR REPLACE VIEW `steam_intel.v_alerts` AS
SELECT game, appid,
       CASE 
         WHEN SAFE_CAST(date AS INT64) IS NOT NULL THEN DATE(TIMESTAMP_SECONDS(DIV(CAST(date AS INT64), 1000000000)))
         ELSE SAFE_CAST(date AS DATE)
       END AS day,
       n, neg_rate, base_neg_rate, z
FROM `steam_intel.alerts`;
