# -*- coding: utf-8 -*-
"""Reviewer-composition X-ray: who is actually writing each game's reviews?

One chunked pass over the raw Kaggle CSV -> per-game shares of
  * steam_purchase        (bought on Steam vs external key)
  * received_for_free     (free keys — astroturf risk signal)
  * written_during_early_access
Writes steam_intel.game_composition. Standalone: touches nothing else.

Run on the VM (~15-20 min):  GCP_PROJECT=buyorwait-2026 python3 composition.py
"""
import glob
import os
import sys
import time

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
RAW_DIR = os.path.expanduser(os.environ.get("RAW_DIR", "~/raw"))
if not PROJECT:
    sys.exit("[!] Set GCP_PROJECT")
bq = bigquery.Client(project=PROJECT)

COLS = ["appid", "steam_purchase", "received_for_free",
        "written_during_early_access"]


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1")).astype("int32")


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not paths:
        sys.exit(f"[!] no raw CSV under {RAW_DIR}")
    agg, t0 = None, time.time()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=COLS, chunksize=3_000_000,
                                 on_bad_lines="skip", low_memory=True):
            chunk["appid"] = pd.to_numeric(chunk["appid"], errors="coerce")
            chunk = chunk.dropna(subset=["appid"])
            g = pd.DataFrame({
                "appid": chunk["appid"].astype("int64"),
                "n": 1,
                "purchase": as_bool(chunk["steam_purchase"]),
                "free": as_bool(chunk["received_for_free"]),
                "ea": as_bool(chunk["written_during_early_access"]),
            }).groupby("appid").sum()
            agg = g if agg is None else agg.add(g, fill_value=0)
        print(f"  scanned {path} ({time.time()-t0:.0f}s)", flush=True)

    out = agg.reset_index()
    for c in ("purchase", "free", "ea"):
        out[f"{c}_pct"] = (out[c] / out["n"] * 100).round(1)
    out = out[["appid", "n", "purchase_pct", "free_pct", "ea_pct"]]
    out = out[out["n"] >= 50]          # composition is meaningless below this
    bq.load_table_from_dataframe(
        out, f"{PROJECT}.{DATASET}.game_composition",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()
    print(f"[√] {len(out):,} games -> game_composition ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
