# -*- coding: utf-8 -*-
"""BuyOrWait core pipeline: Clean -> Daily Aggregation -> Weighted Score -> Bomb Detection, timing each stage.

Run the same code with two modes:
  python pipeline.py cpu                    # pandas baseline (CPU)
  python -m cudf.pandas pipeline.py gpu     # GPU acceleration (zero code change)
Subset debugging: MAX_FILES=3 python -m cudf.pandas pipeline.py gpu
Environment variables: DATA_DIR (default: slim_parquet), MAX_FILES (default: 0 = all files)
Outputs: out_game_daily.parquet / out_game_scores.parquet / out_alerts.parquet
         benchmark_results.csv (append-only)
Metric definitions (refer to Proposal §3):
  w_i = log(1 + playtime_at_review) * exp(-age_days / 90)
  Bombing = daily negative review rate z-score (over 30-day rolling window) > 3 
            AND daily review volume > 2x of 30-day rolling average
"""
import sys

# RMM Memory Pool: MUST be initialized before importing pandas/cuDF to avoid memory corruption
# (such as "corrupted double-linked list" or double free errors) due to allocator mismatches.
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    pass

import glob
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", "slim_parquet")
MAX_FILES = int(os.environ.get("MAX_FILES", 0))
BENCH_CSV = "benchmark_results.csv"
HALF_LIFE_DAYS = 90.0
ROLL = 30          # Rolling window for bomb detection (rows ≈ days)
MIN_HIST = 7       # Minimum history of 7 days required for detection

_timings: list = []


@contextmanager
def timer(stage: str):
    t = time.perf_counter()
    yield
    dt = time.perf_counter() - t
    _timings.append((stage, dt))
    print(f"  [{MODE}] {stage:<18s} {dt:9.2f}s", flush=True)


def load_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "part_*.parquet")))
    if not files:
        sys.exit(f"[!] No part_*.parquet found under {DATA_DIR}. Please run convert_to_parquet.py first.")
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
        print(f"[i] Subset mode: only reading the first {len(files)} file(s)")
    return files


def main():
    print(f"=== BuyOrWait pipeline | mode={MODE} | {datetime.now():%F %T} ===")
    files = load_files()

    with timer("read_parquet"):
        cols = ["appid", "voted_up", "playtime_at_review", "timestamp_created"]
        df = pd.read_parquet(files, columns=cols)
    rows = len(df)
    print(f"[i] {rows:,} rows")

    with timer("clean_cast"):
        df = df.dropna(subset=["appid", "voted_up", "timestamp_created"])
        df["appid"] = df["appid"].astype("int32")
        df["voted_up"] = df["voted_up"].astype("int8")
        df["playtime_at_review"] = df["playtime_at_review"].fillna(0).astype("float32")
        # Integer day index instead of timedelta operations - fully native on GPU
        df["day_i"] = (df["timestamp_created"] // 86400).astype("int32")
        df = df.drop(columns=["timestamp_created"])
        # Evergreen-score basis: UNDECAYED row-level weights. Once summed per day,
        # the weighted score can be recomputed for ANY reference date in BigQuery
        # via EXP(-age/90) at query time (see docs/evergreen.sql).
        df["_lw"] = np.log1p(df["playtime_at_review"]).astype("float32")
        df["_lwv"] = df["_lw"] * df["voted_up"]

    with timer("daily_groupby"):
        daily = (df.groupby(["appid", "day_i"], sort=False)
                   .agg(n=("voted_up", "size"), pos=("voted_up", "sum"),
                        w_sum=("_lw", "sum"), wv_sum=("_lwv", "sum"))
                   .reset_index())
        daily["date"] = pd.to_datetime(daily["day_i"].astype("int64") * 86400, unit="s")
        daily["neg"] = daily["n"] - daily["pos"]
        daily["pos_rate"] = (daily["pos"] / daily["n"]).astype("float32")
        daily["neg_rate"] = (daily["neg"] / daily["n"]).astype("float32")

    with timer("weighted_score"):
        # Day age calculated via int32 arithmetic, weights via float32 ufunc.
        # Recent metrics merged into a single groupby to prevent timedelta
        # and double-groupby + merge from triggering cuDF fallback/copy overhead.
        ref_day = int(df["day_i"].max())
        age = (ref_day - df["day_i"]).astype("float32")
        df["_w"] = (df["_lw"] * np.exp(-age / HALF_LIFE_DAYS)).astype("float32")
        df["_wv"] = df["_w"] * df["voted_up"]
        df["_rn"] = (age <= HALF_LIFE_DAYS).astype("int8")   # Whether review is within recent 90 days
        df["_rp"] = df["_rn"] * df["voted_up"]
        # Refund-zone risk: negative reviews written with < 2h playtime
        df["_nn"] = (1 - df["voted_up"]).astype("int8")
        df["_rz"] = ((df["playtime_at_review"] < 120) & (df["voted_up"] == 0)).astype("int8")
        df["_pp"] = df["playtime_at_review"].where(df["voted_up"] == 1)  # playtime of happy players
        scores = (df.groupby("appid")
                    .agg(w_sum=("_w", "sum"), wv_sum=("_wv", "sum"),
                         n_reviews=("voted_up", "size"), n_pos=("voted_up", "sum"),
                         recent_n=("_rn", "sum"), recent_pos=("_rp", "sum"),
                         n_neg=("_nn", "sum"), rz_neg=("_rz", "sum"),
                         pos_med_pt=("_pp", "median"))
                     .reset_index())
        scores["score"] = (scores["wv_sum"] / scores["w_sum"] * 100).astype("float32")
        scores["raw_pos_rate"] = (scores["n_pos"] / scores["n_reviews"] * 100).astype("float32")
        scores["recent_pos_rate"] = (scores["recent_pos"]
                                     / scores["recent_n"].where(scores["recent_n"] > 0)
                                     * 100).astype("float32")
        scores["refund_zone_pct"] = (scores["rz_neg"]
                                     / scores["n_neg"].where(scores["n_neg"] > 0)
                                     * 100).astype("float32")
        scores["pos_median_hours"] = (scores["pos_med_pt"] / 60).astype("float32")
        scores = scores.drop(columns=["w_sum", "wv_sum", "n_pos", "recent_pos",
                                      "n_neg", "rz_neg", "pos_med_pt"])

    with timer("bomb_detect"):
        d = daily.sort_values(["appid", "date"], ignore_index=True)
        # Pure vectorized group rolling: after sorting, window is valid if row 30 positions back belongs to the same appid
        base_n = d["n"].rolling(ROLL, min_periods=MIN_HIST).mean().shift(1)
        base_m = d["neg_rate"].rolling(ROLL, min_periods=MIN_HIST).mean().shift(1)
        base_s = d["neg_rate"].rolling(ROLL, min_periods=MIN_HIST).std().shift(1)
        same_game = d["appid"] == d["appid"].shift(MIN_HIST + 1)
        z = (d["neg_rate"] - base_m) / base_s.clip(lower=1e-4)
        is_bomb = same_game & (z > 3) & (d["n"] > 2 * base_n)
        alerts = d.loc[is_bomb, ["appid", "date", "n", "neg_rate"]].copy()
        alerts["z"] = z[is_bomb].astype("float32")
        alerts["base_neg_rate"] = base_m[is_bomb].astype("float32")

    with timer("write_outputs"):
        names_f = os.path.join(DATA_DIR, "names.parquet")
        if os.path.exists(names_f):
            names = pd.read_parquet(names_f)
            scores = scores.merge(names, on="appid", how="left")
            alerts = alerts.merge(names, on="appid", how="left")
        
        out_dir = os.environ.get("OUT_DIR", DATA_DIR)
        os.makedirs(out_dir, exist_ok=True)
        daily.to_parquet(os.path.join(out_dir, "out_game_daily.parquet"), index=False)
        scores.to_parquet(os.path.join(out_dir, "out_game_scores.parquet"), index=False)
        alerts.to_parquet(os.path.join(out_dir, "out_alerts.parquet"), index=False)

    total = sum(t for _, t in _timings)
    _timings.append(("end_to_end", total))
    print(f"\n[√] End-to-End: {total:.2f}s | Scores: {len(scores):,} games | Alerts: {len(alerts):,} logs")

    out_dir = os.environ.get("OUT_DIR", DATA_DIR)
    bench_csv = os.path.join(out_dir, BENCH_CSV)
    bench = pd.DataFrame(_timings, columns=["stage", "seconds"])
    bench.insert(0, "mode", MODE)
    bench.insert(0, "run_ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    bench["rows"] = rows
    bench.to_csv(bench_csv, mode="a", index=False,
                 header=not os.path.exists(bench_csv))
    print(f"[√] Timing appended to -> {bench_csv}")



if __name__ == "__main__":
    main()
