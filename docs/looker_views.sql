-- Convenience views for Looker Studio.
-- Looker Studio requires proper DATE fields for time series & date-range controls.
-- Runs on dataset: steam_intel

CREATE OR REPLACE VIEW `steam_intel.v_game_daily` AS
SELECT appid,
       DATE(date) AS day,
       n, pos, neg, pos_rate, neg_rate
FROM `steam_intel.game_daily`;

CREATE OR REPLACE VIEW `steam_intel.v_alerts` AS
SELECT game, appid,
       DATE(date) AS day,
       n, neg_rate, base_neg_rate, z
FROM `steam_intel.alerts`;
