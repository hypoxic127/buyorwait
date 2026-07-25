# -*- coding: utf-8 -*-
"""Nightly incremental fetcher: Steam appreviews API -> BigQuery daily_delta.

For the top-N most-reviewed games, pull reviews from the last RECENT_DAYS days
(filter=recent, cursor pagination), aggregate to (appid, day) with the same
undecayed weight sums as pipeline.py, and MERGE into steam_intel.daily_delta.
Idempotent: re-running a night simply overwrites the same (appid, day) rows.

Run on the GCE VM or as a Cloud Run Job:
  GCP_PROJECT=your-project python3 fetch_recent.py
Env:
  GCP_PROJECT (required)   BQ_DATASET (default steam_intel)
  TOP_N (default 2000)     RECENT_DAYS (default 90)
  MAX_PAGES_PER_APP (default 5, 100 reviews/page)
  SLEEP_S (default 0.5)    APPIDS (optional comma list, overrides TOP_N query)
"""
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from google.cloud import bigquery

PROJECT = (os.environ.get("GCP_PROJECT") or "").split()[0] if os.environ.get("GCP_PROJECT") else None
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
TOP_N = int(os.environ.get("TOP_N", 2000))
RECENT_DAYS = int(os.environ.get("RECENT_DAYS", 90))
MAX_PAGES = int(os.environ.get("MAX_PAGES_PER_APP", 5))
SLEEP_S = float(os.environ.get("SLEEP_S", 0.5))

# Generate a unique run-level token to defeat concurrent staging collisions
RUN_UUID = uuid.uuid4().hex[:8]

API = "https://store.steampowered.com/appreviews/{appid}"
UA = {"User-Agent": "BuyOrWait/1.0 (hackathon research; contact in repo)"}

if not PROJECT:
    sys.exit("[!] Set GCP_PROJECT")

bq = bigquery.Client(project=PROJECT)
T_DELTA = f"{PROJECT}.{DATASET}.daily_delta"
T_SCORES = f"{PROJECT}.{DATASET}.game_scores"


def target_appids() -> list[int]:
    manual = os.environ.get("APPIDS")
    if manual:
        import re
        apps = [int(x) for x in re.split(r"[,\s]+", manual) if x.strip()]
    else:
        q = f"""SELECT appid FROM `{T_SCORES}`
                ORDER BY n_reviews DESC LIMIT {TOP_N}"""
        apps = [r.appid for r in bq.query(q).result()]

    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))

    if task_count > 1:
        # Partition apps evenly across parallel tasks
        apps = [a for i, a in enumerate(apps) if i % task_count == task_index]
        print(f"Task {task_index}/{task_count}: assigned {len(apps)} apps.")

    return apps


def app_last_fetched_dates() -> dict[int, datetime]:
    """Query BigQuery for the latest day fetched for each appid."""
    try:
        q = f"SELECT appid, MAX(day) AS max_day FROM `{T_DELTA}` GROUP BY appid"
        df = bq.query(q).to_dataframe()
        return {int(row.appid): datetime.combine(row.max_day, datetime.min.time(), tzinfo=timezone.utc)
                for row in df.itertuples()}
    except Exception as e:
        print(f"[!] Could not query existing max_day: {e}")
        return {}


def fetch_app(appid: int, default_cutoff_ts: int, last_fetched_map: dict[int, datetime]) -> pd.DataFrame:
    """Fetch only missing new reviews since last_fetched_day. Empty df if none/error."""
    if appid in last_fetched_map:
        cutoff_ts = int((last_fetched_map[appid] - timedelta(days=1)).timestamp())
    else:
        cutoff_ts = default_cutoff_ts

    rows, seen, cursor = [], set(), "*"
    p = 0
    while p < MAX_PAGES:
        retries = 0
        r = None
        while retries <= 3:
            try:
                r = requests.get(
                    API.format(appid=appid), headers=UA, timeout=12,
                    params={"json": 1, "filter": "recent", "language": "all",
                            "purchase_type": "all", "num_per_page": 100,
                            "cursor": cursor})
                if r.status_code == 429:
                    retries += 1
                    sleep_time = (2 ** retries) + random.uniform(0.5, 1.5)
                    print(f"  [!] Rate limited (429) for appid {appid}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                r.raise_for_status()
                break
            except Exception as e:
                print(f"  [!] Network error on appid {appid}: {e}")
                retries += 1
                time.sleep(1)

        if not r or r.status_code != 200:
            break

        try:
            js = r.json()
        except Exception as e:
            print(f"  [!] JSON Parse error on appid {appid}: {e}")
            break

        reviews = js.get("reviews") or []
        if not reviews:
            break
        oldest = None
        for rv in reviews:
            rid = rv.get("recommendationid")
            ts = int(rv.get("timestamp_created", 0))
            oldest = ts if oldest is None else min(oldest, ts)
            if rid in seen or ts < cutoff_ts:
                continue
            seen.add(rid)
            rows.append((ts,
                         1 if rv.get("voted_up") else 0,
                         float((rv.get("author") or {}).get("playtime_at_review", 0) or 0)))
        cursor = js.get("cursor", "")
        if not cursor or (oldest is not None and oldest < cutoff_ts):
            break
        p += 1
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "voted_up", "playtime"])
    df["day"] = pd.to_datetime(df["ts"], unit="s").dt.floor("D")
    if len(df) and (oldest is not None and oldest >= cutoff_ts):
        df = df[df["day"] > df["day"].min()]
    if df.empty:
        return df
    lw = np.log1p(df["playtime"])
    df["_lw"], df["_lwv"] = lw, lw * df["voted_up"]
    agg = (df.groupby("day")
             .agg(n=("voted_up", "size"), pos=("voted_up", "sum"),
                  w_sum=("_lw", "sum"), wv_sum=("_lwv", "sum"))
             .reset_index())
    agg.insert(0, "appid", appid)
    return agg


def merge_into_bq(all_agg: pd.DataFrame):
    all_agg["day"] = pd.to_datetime(all_agg["day"]).dt.date
    all_agg["fetched_at"] = datetime.now(timezone.utc)
    task_index = os.environ.get("CLOUD_RUN_TASK_INDEX", "0")
    staging = f"{PROJECT}.{DATASET}._staging_delta_{task_index}_{RUN_UUID}"
    schema = [
        bigquery.SchemaField("appid", "INT64"),
        bigquery.SchemaField("day", "DATE"),
        bigquery.SchemaField("n", "INT64"),
        bigquery.SchemaField("pos", "INT64"),
        bigquery.SchemaField("w_sum", "FLOAT64"),
        bigquery.SchemaField("wv_sum", "FLOAT64"),
        bigquery.SchemaField("fetched_at", "TIMESTAMP"),
    ]
    try:
        job = bq.load_table_from_dataframe(
            all_agg, staging,
            job_config=bigquery.LoadJobConfig(
                schema=schema, write_disposition="WRITE_TRUNCATE"))
        job.result()
        bq.query(f"""
            MERGE `{T_DELTA}` t USING `{staging}` s
            ON t.appid = s.appid AND t.day = s.day
            WHEN MATCHED THEN UPDATE SET
              n = s.n, pos = s.pos, w_sum = s.w_sum,
              wv_sum = s.wv_sum, fetched_at = s.fetched_at
            WHEN NOT MATCHED THEN INSERT ROW
        """).result()
    finally:
        print(f"Cleaning up staging table: {staging}")
        bq.delete_table(staging, not_found_ok=True)


def main():
    t0 = time.time()
    cutoff_ts = int((datetime.now(timezone.utc)
                     - timedelta(days=RECENT_DAYS)).timestamp())
    apps = target_appids()
    last_fetched_map = app_last_fetched_dates()
    print(f"=== fetch_recent | {len(apps)} apps | last {RECENT_DAYS}d "
          f"| {datetime.now():%F %T} | Run: {RUN_UUID} ===")
    parts, reviews_total = [], 0
    for i, appid in enumerate(apps, 1):
        agg = fetch_app(appid, cutoff_ts, last_fetched_map)
        if not agg.empty:
            parts.append(agg)
            reviews_total += int(agg["n"].sum())
        if i % 50 == 0 or i == len(apps):
            print(f"  {i}/{len(apps)} apps | +{reviews_total:,} reviews "
                  f"| {time.time()-t0:5.0f}s", flush=True)
        time.sleep(0.05)
    if not parts:
        print("[!] Nothing fetched — check network / appids"); return
    all_agg = pd.concat(parts, ignore_index=True)
    merge_into_bq(all_agg)
    print(f"[√] MERGEd {len(all_agg):,} (appid, day) rows "
          f"({reviews_total:,} reviews) into {T_DELTA} "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
