# Looker Studio Dashboard — Setup Guide (~10 min)

An executive dashboard on top of the same BigQuery aggregated tables, adding one more GCP component with zero code. Result: a public link to put in the README next to the Live Demo.

## 1. Create the convenience views (once)

The `date` columns in `game_daily` / `alerts` are epoch **nanoseconds** (INT64 from the Parquet load). Looker Studio needs real `DATE` fields for its date-range controls, so create the two views first:

```bash
bq query --use_legacy_sql=false < docs/looker_views.sql
```

This creates `steam_intel.v_game_daily` and `steam_intel.v_alerts`. (`game_scores` has no date column — connect it directly.)

## 2. Connect the data

1. Open [lookerstudio.google.com](https://lookerstudio.google.com) → **Create** → **Report**.
2. Pick the **BigQuery** connector → your project → dataset `steam_intel` → table `game_scores` → **Add**.
3. **Resource → Manage added data sources → Add a data source** → BigQuery again for `v_game_daily` and `v_alerts`.

## 3. Suggested layout (one page, three sections)

**Header — scorecards** (data source: `game_scores` / `v_alerts`)
- Scorecard: `Record Count` on `game_scores` → rename "Games scored"
- Scorecard: `SUM(n_reviews)` → "Reviews analyzed" (should read ~114M)
- Scorecard: `Record Count` on `v_alerts` → "Bombing alert days"

**Left — Top games table** (data source: `game_scores`)
- Chart: **Table with bars**; dimension `game`; metrics `score`, `recent_pos_rate`, `n_reviews`
- Sort: `score` descending; add filter `n_reviews > 50000` (chart filter) to hide tiny games

**Right — sentiment over time** (data source: `v_game_daily`)
- Chart: **Time series**; dimension `day`; metric `AVG(pos_rate)`; optional breakdown by `appid`
- Add a **date-range control** and a **drop-down control** on `appid` (or join game names via blend with `game_scores`)
- ⚠ The data ends on **2023-10-30** — keep the date-range control default on **Auto**; a preset like "Last 28 days" resolves against today and renders empty charts

**Bottom — bombing alerts table** (data source: `v_alerts`)
- Chart: **Table**; dimensions `game`, `day`; metrics `n`, `neg_rate`, `z`
- Sort: `day` descending

## 4. Publish

1. **Share → Manage access → Anyone with the link → Viewer**.
2. Under **Share → Report settings**, keep "Viewers can use their own credentials" **off** (owner's credentials) so judges don't need BigQuery access.
3. Copy the report link into `README.md` (the `📊 Looker Studio` placeholder on line 5).

Cost note: the views scan a few MB per refresh — comfortably inside the BigQuery free tier.
