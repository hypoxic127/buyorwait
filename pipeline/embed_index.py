# -*- coding: utf-8 -*-
"""Build the semantic layer: review texts -> GPU embeddings -> BigQuery vectors.

Corpus (deliberately scoped — "index the text that matters for decisions"):
  * Top TOP_GAMES games by review count: up to PER_GAME/2 positive + PER_GAME/2
    negative English reviews each, ranked by helpful votes.
  * Plus every game in alert_causes (bombing episodes) — negative reviews only.
Embeddings: fastembed BAAI/bge-small-en-v1.5 (384-d) on the L4 via CUDA
(pip install fastembed-gpu). Same model runs on CPU inside Cloud Run for
query-time embedding, so corpus and queries share one vector space.

Output: steam_intel.review_vectors + IVF vector index (docs/vector.sql).
Run on the VM:  GCP_PROJECT=buyorwait-2026 python3 embed_index.py   (~25-40 min)
Env: TOP_GAMES=300, PER_GAME=800, RAW_DIR=~/raw, BQ_DATASET=steam_intel
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
RAW_DIR = os.path.expanduser(os.environ.get("RAW_DIR", "~/raw"))
TOP_GAMES = int(os.environ.get("TOP_GAMES", 300))
PER_GAME = int(os.environ.get("PER_GAME", 800))
MODEL = "BAAI/bge-small-en-v1.5"

if not PROJECT:
    sys.exit("[!] Set GCP_PROJECT")
bq = bigquery.Client(project=PROJECT)


def target_sets():
    top = [r.appid for r in bq.query(
        f"SELECT appid FROM `{PROJECT}.{DATASET}.game_scores` "
        f"ORDER BY n_reviews DESC LIMIT {TOP_GAMES}").result()]
    try:
        ep = [r.appid for r in bq.query(
            f"SELECT DISTINCT appid FROM `{PROJECT}.{DATASET}.alert_causes`").result()]
    except Exception:
        ep = []
    return set(top), set(ep)


def build_corpus(top_ids: set, ep_ids: set) -> pd.DataFrame:
    """One chunked CSV pass; keep per-game top-K helpful reviews per polarity."""
    all_ids = top_ids | ep_ids
    usecols = ["appid", "review", "language", "timestamp_created",
               "voted_up", "votes_up", "author_playtime_at_review"]
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not paths:
        sys.exit(f"[!] no raw CSV under {RAW_DIR}")
    kept, t0, half = [], time.time(), PER_GAME // 2

    def compact(frames):
        df = pd.concat(frames, ignore_index=True)
        df = (df.sort_values("votes_up", ascending=False)
                .groupby(["appid", "pol"], sort=False).head(half)
                .reset_index(drop=True))
        return [df]

    for path in paths:
        for i, chunk in enumerate(pd.read_csv(path, usecols=usecols,
                                              chunksize=2_000_000,
                                              on_bad_lines="skip",
                                              low_memory=True)):
            c = chunk[chunk["appid"].isin(all_ids)
                      & (chunk["language"] == "english")].copy()
            if not len(c):
                continue
            c["review"] = c["review"].astype(str).str.strip()
            c = c[c["review"].str.len().between(25, 4000)]
            c["pol"] = c["voted_up"].astype(str).str.lower().isin(
                ("true", "1")).astype("int8")
            # episode-only games: keep negatives only
            c = c[c["appid"].isin(top_ids) | (c["pol"] == 0)]
            c["votes_up"] = pd.to_numeric(c["votes_up"], errors="coerce").fillna(0)
            kept.append(c[["appid", "review", "timestamp_created",
                           "pol", "votes_up", "author_playtime_at_review"]])
            if len(kept) >= 10:
                kept = compact(kept)
        print(f"  scanned {path} ({time.time()-t0:.0f}s)", flush=True)
    corpus = compact(kept)[0]
    corpus["day"] = pd.to_datetime(pd.to_numeric(
        corpus["timestamp_created"], errors="coerce"), unit="s").dt.date
    corpus["text"] = corpus["review"].str.slice(0, 700)
    corpus["review_id"] = [hashlib.md5(f"{a}{t[:60]}".encode()).hexdigest()[:16]
                           for a, t in zip(corpus["appid"], corpus["text"])]
    corpus = corpus.drop_duplicates("review_id")
    print(f"[i] corpus: {len(corpus):,} reviews across "
          f"{corpus['appid'].nunique()} games ({time.time()-t0:.0f}s)")
    return corpus


def embed(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding
    try:
        model = TextEmbedding(MODEL, providers=["CUDAExecutionProvider"])
        print("[i] embedding on GPU (CUDAExecutionProvider)")
    except Exception as e:
        model = TextEmbedding(MODEL)
        print(f"[i] GPU EP unavailable ({e}) — CPU fallback")
    t0, out = time.time(), []
    for i in range(0, len(texts), 5000):
        out.append(np.array(list(model.embed(texts[i:i + 5000]))))
        print(f"  embedded {min(i+5000, len(texts)):,}/{len(texts):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return np.vstack(out)


def load_bq(corpus: pd.DataFrame, vecs: np.ndarray):
    df = pd.DataFrame({
        "review_id": corpus["review_id"].values,
        "appid": corpus["appid"].astype("int64").values,
        "voted_up": corpus["pol"].astype(bool).values,
        "votes_up": corpus["votes_up"].astype("int64").values,
        "playtime_h": (pd.to_numeric(corpus["author_playtime_at_review"],
                                     errors="coerce").fillna(0) / 60).values,
        "day": corpus["day"].values,
        "text": corpus["text"].values,
        "embedding": [v.tolist() for v in vecs],
    })
    schema = [
        bigquery.SchemaField("review_id", "STRING"),
        bigquery.SchemaField("appid", "INT64"),
        bigquery.SchemaField("voted_up", "BOOL"),
        bigquery.SchemaField("votes_up", "INT64"),
        bigquery.SchemaField("playtime_h", "FLOAT64"),
        bigquery.SchemaField("day", "DATE"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    tbl = f"{PROJECT}.{DATASET}.review_vectors"
    bq.load_table_from_dataframe(df, tbl, job_config=bigquery.LoadJobConfig(
        schema=schema, write_disposition="WRITE_TRUNCATE")).result()
    print(f"[√] loaded {len(df):,} vectors -> {tbl}")
    bq.query(f"""
        CREATE VECTOR INDEX IF NOT EXISTS idx_review_vectors
        ON `{tbl}`(embedding)
        OPTIONS(index_type='IVF', distance_type='COSINE')""").result()
    print("[√] vector index requested (builds async; brute-force works meanwhile)")


def main():
    top_ids, ep_ids = target_sets()
    corpus = build_corpus(top_ids, ep_ids)
    vecs = embed(corpus["text"].tolist())
    load_bq(corpus, vecs)


if __name__ == "__main__":
    main()
