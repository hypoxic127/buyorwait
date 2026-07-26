# Looker Studio Executive Dashboard — Setup & Integration Guide

An executive analytics dashboard built on top of GCP BigQuery aggregated tables (`steam_intel`), providing real-time data visualization, trend exploration, and Review Bombing alert monitoring with 0 additional server code.

---

## 1. Automated BigQuery View Provisioning

Looker Studio requires native `DATE` formats for time-series aggregation and date-range controls. The convenience views `v_game_daily` and `v_alerts` have been created in BigQuery dataset `steam_intel`:

```sql
CREATE OR REPLACE VIEW `steam_intel.v_game_daily` AS
SELECT appid, DATE(date) AS day, n, pos, neg, pos_rate, neg_rate
FROM `steam_intel.game_daily`;

CREATE OR REPLACE VIEW `steam_intel.v_alerts` AS
SELECT game, appid, DATE(date) AS day, n, neg_rate, base_neg_rate, z
FROM `steam_intel.alerts`;
```

To re-apply or update these views in BigQuery at any time:
```bash
python -c "
from google.cloud import bigquery
client = bigquery.Client(project='buyorwait-2026')
with open('docs/looker_views.sql', 'r', encoding='utf-8') as f:
    client.query(f.read()).result()
print('BigQuery views updated successfully!')
"
```

---

## 2. Connecting Data Sources in Looker Studio

1. Open [lookerstudio.google.com](https://lookerstudio.google.com) → **Create** → **Report**.
2. Select the **BigQuery** connector → Project `buyorwait-2026` → Dataset `steam_intel` → Table `game_scores` → **Add to report**.
3. Click **Resource** → **Manage added data sources** → **Add a Data Source** → Select BigQuery for both:
   - `steam_intel.v_game_daily`
   - `steam_intel.v_alerts`

---

## 3. Recommended One-Page Dashboard Layout

### 📌 Header — Executive Key Metrics (Scorecards)
- **Scored Games Count**: Metric = `COUNT(appid)` on `game_scores` (Title: *Games Scored*)
- **Total Reviews Analyzed**: Metric = `SUM(n_reviews)` on `game_scores` (Title: *Reviews Analyzed* ~114M)
- **Review Bombing Anomalies**: Metric = `COUNT(day)` on `v_alerts` (Title: *Bombing Alert Days*)

### 📊 Left Column — Leaderboard & Game Ratings
- **Chart Type**: Table with Bars
- **Data Source**: `game_scores`
- **Dimensions**: `game`
- **Metrics**: `score`, `recent_pos_rate`, `n_reviews`
- **Sorting**: `score` (Descending)
- **Filter**: `n_reviews > 50000` (Filters out niche titles with low sample size)

### 📈 Right Column — Sentiment Trend Over Time
- **Chart Type**: Time Series / Smooth Line Chart
- **Data Source**: `v_game_daily`
- **Dimension**: `day`
- **Metric**: `AVG(pos_rate)`
- **Breakdown Dimension**: `appid` (or joined game title)
- **Control Widgets**: Add a **Date Range Control** and a **Drop-down List Control** on `appid`

### 🚨 Bottom Row — Review Bombing Incident Stream
- **Chart Type**: Table
- **Data Source**: `v_alerts`
- **Dimensions**: `game`, `day`
- **Metrics**: `n` (Review Volume), `neg_rate` (Negative Share), `z` (Anomaly Score)
- **Sorting**: `day` (Descending)

---

## 4. Public Access & Publishing

1. In Looker Studio, click **Share** → **Manage access** → Set to **"Anyone with the link can view"**.
2. Go to **Share** → **Report settings** → Ensure *"Viewers can use their own credentials"* is **Unchecked** (use owner credentials so viewers don't need GCP BigQuery IAM permissions).
3. Copy the public link and paste it into `README.md` under the **📊 Looker Studio Dashboard** section.
