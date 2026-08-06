# -*- coding: utf-8 -*-
"""Bombing-cause attribution: WHY was each game review-bombed?

For every significant bombing episode (from v_alerts_all), compare negative-
review vocabulary inside the episode window vs a 60-day baseline, keep the
top distinctive terms + example quotes, optionally ask Gemini for a one-line
cause, and MERGE everything into steam_intel.alert_causes.

Text sources:
  - Episodes ending on/before SNAPSHOT_END: one chunked pass over the raw
    Kaggle CSV (review text lives only there).
  - Later episodes: pulled live from the Steam appreviews API.

Run on the VM (CSV scan ~15-25 min, one-off):
  GCP_PROJECT=buyorwait-2026 python3 attribute.py
Env: GCP_PROJECT (required), BQ_DATASET=steam_intel, RAW_DIR=~/raw,
     Z_MIN=3, N_MIN=30 (must match the app's episode definition),
     EPISODES_MAX=400, SUMMARY_MAX=150 (Gemini budget), GEMINI_MODEL,
     SNAPSHOT_END=2023-10-30, BASELINE_DAYS=60
"""
import glob
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
RAW_DIR = os.path.expanduser(os.environ.get("RAW_DIR", "~/raw"))
Z_MIN = float(os.environ.get("Z_MIN", 3))
N_MIN = int(os.environ.get("N_MIN", 30))
EPISODES_MAX = int(os.environ.get("EPISODES_MAX", 400))
SUMMARY_MAX = int(os.environ.get("SUMMARY_MAX", 150))
SNAPSHOT_END = pd.Timestamp(os.environ.get("SNAPSHOT_END", "2023-10-30"))
BASELINE_DAYS = int(os.environ.get("BASELINE_DAYS", 60))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

STOP = set("""the and for was but not you all are this that with have has had
they them their there its it's just like game games really only get got very
much more even still than then will would can could into out about a an of to
in on is it as at be or so if my me we he she his her do did does don doesn
didn isn aren won because been being play played playing player players time
hours hour make makes made no yes one two now new old way from too also lot
who what when where why how which your our us him think know go going dont
cant im ive isnt wasnt werent youre theyre thats""".split())
TOKEN = re.compile(r"[a-zA-Z']{3,}")

if not PROJECT:
    sys.exit("[!] Set GCP_PROJECT")
bq = bigquery.Client(project=PROJECT)


def episodes() -> pd.DataFrame:
    """Same episode definition as the app's Bombing tab (z>=Z_MIN, n>=N_MIN)."""
    q = f"""
    SELECT appid, ANY_VALUE(game) AS game, MIN(day) AS first_day,
           MAX(day) AS latest_day, MAX(n) AS peak_n,
           MAX(LEAST(z, 99.9)) AS peak_z, SUM(n) AS window_n
    FROM `{PROJECT}.{DATASET}.v_alerts_all`
    WHERE z >= {Z_MIN} AND n >= {N_MIN}
    GROUP BY appid
    ORDER BY peak_n * peak_z DESC
    LIMIT {EPISODES_MAX}"""
    df = bq.query(q).to_dataframe()
    df["first_day"] = pd.to_datetime(df["first_day"])
    df["latest_day"] = pd.to_datetime(df["latest_day"])
    print(f"[i] {len(df)} episodes to attribute")
    return df


# --------------------------------------------------------------- text sources
def texts_from_csv(eps: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """One chunked pass over the raw CSV; keep negative english reviews that
    fall in any episode's alert-or-baseline window. Returns {appid: df}."""
    windows = {int(r.appid): (r.first_day - pd.Timedelta(days=BASELINE_DAYS),
                              r.latest_day)
               for r in eps.itertuples()}
    ids = set(windows)
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not paths:
        sys.exit(f"[!] no raw CSV under {RAW_DIR}")
    usecols = ["appid", "review", "language", "timestamp_created", "voted_up"]
    keep, t0 = [], time.time()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=2_000_000,
                                 on_bad_lines="skip", low_memory=True):
            part = chunk[chunk["appid"].isin(ids)
                         & (chunk["language"] == "english")]
            if not len(part):
                continue
            part = part[~part["voted_up"].astype(str).str.lower()
                        .isin(("true", "1"))]
            if len(part):
                keep.append(part[["appid", "review", "timestamp_created"]])
        print(f"  scanned {path} ({time.time()-t0:.0f}s)", flush=True)
    if not keep:
        return {}
    df = pd.concat(keep, ignore_index=True)
    df["day"] = pd.to_datetime(pd.to_numeric(df["timestamp_created"],
                                             errors="coerce"), unit="s")
    out = {}
    for appid, (lo, hi) in windows.items():
        g = df[(df["appid"] == appid) & (df["day"] >= lo) & (df["day"] <= hi)]
        if len(g):
            out[appid] = g[["review", "day"]]
    print(f"[i] CSV pass done: text for {len(out)} episodes "
          f"({len(df):,} negative rows kept, {time.time()-t0:.0f}s)")
    return out


def texts_from_api(appid: int, first_day: pd.Timestamp) -> pd.DataFrame:
    """Negative reviews since (first_day - baseline) for post-snapshot episodes."""
    cutoff = int((first_day - pd.Timedelta(days=BASELINE_DAYS)).timestamp())
    rows, cursor = [], "*"
    for _ in range(30):
        try:
            r = requests.get(
                f"https://store.steampowered.com/appreviews/{appid}",
                params={"json": 1, "filter": "recent", "language": "english",
                        "purchase_type": "all", "num_per_page": 100,
                        "cursor": cursor},
                headers={"User-Agent": "BuyOrWait/1.0"}, timeout=20)
            js = r.json()
        except Exception:
            break
        revs = js.get("reviews") or []
        if not revs:
            break
        oldest = min(rv["timestamp_created"] for rv in revs)
        rows += [(rv["review"], rv["timestamp_created"]) for rv in revs
                 if not rv.get("voted_up") and rv["timestamp_created"] >= cutoff]
        cursor = js.get("cursor", "")
        if not cursor or oldest < cutoff:
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows, columns=["review", "ts"])
    if len(df):
        df["day"] = pd.to_datetime(df["ts"], unit="s")
    return df


# ----------------------------------------------------------- term extraction
def top_terms(texts: pd.DataFrame, first_day, latest_day, k=8):
    alert = texts[(texts["day"] >= first_day) & (texts["day"] <= latest_day)]
    base = texts[texts["day"] < first_day]
    if len(alert) < 30:
        return None, len(alert), []

    def counts(frame):
        c = Counter()
        for t in frame["review"].astype(str):
            c.update(w for w in (x.lower() for x in TOKEN.findall(t))
                     if w not in STOP)
        return c

    ca, cb = counts(alert), counts(base)
    na, nb = max(sum(ca.values()), 1), max(sum(cb.values()), 1)
    scored = sorted(
        ((((fa / na) / ((cb.get(t, 0) + 1) / nb)) * (fa ** .5), t, fa)
         for t, fa in ca.items() if fa >= 5),
        reverse=True)
    terms = [t for _, t, _ in scored[:k]]
    examples = []
    for t in terms[:3]:
        ex = next((x[:160].replace("\n", " ")
                   for x in alert["review"].astype(str) if t in x.lower()), None)
        if ex:
            examples.append(ex)
    return ", ".join(terms), len(alert), examples


# ------------------------------------------------------------------- Gemini
_gem = None
def gemini_summary(game, terms, examples):
    global _gem
    try:
        if _gem is None:
            from google import genai
            key = os.environ.get("GEMINI_API_KEY")
            _gem = (genai.Client(api_key=key) if key else
                    genai.Client(vertexai=True, project=PROJECT,
                                 location=os.environ.get("VERTEX_LOCATION", "global")))
        prompt = (f"Steam game '{game}' was review-bombed. Distinctive terms from "
                  f"negative reviews: {terms}. Example reviews: {examples}. "
                  "In ONE short sentence (max 18 words), name the concrete cause(s). "
                  "No preamble, no hedging.")
        return _gem.models.generate_content(model=GEMINI_MODEL,
                                            contents=prompt).text.strip()
    except Exception as e:
        print(f"  [!] Gemini skipped: {e}")
        return None


# --------------------------------------------------------------------- main
def main():
    eps = episodes()
    hist = eps[eps["latest_day"] <= SNAPSHOT_END]
    recent = eps[eps["latest_day"] > SNAPSHOT_END]
    texts = texts_from_csv(hist) if len(hist) else {}

    out, done = [], 0
    for r in eps.itertuples():
        appid = int(r.appid)
        t = texts.get(appid)
        if t is None and r.latest_day > SNAPSHOT_END:
            t = texts_from_api(appid, r.first_day)
        if t is None or not len(t):
            continue
        terms, n_neg, examples = top_terms(t, r.first_day, r.latest_day)
        if not terms:
            continue
        summary = (gemini_summary(r.game, terms, examples)
                   if done < SUMMARY_MAX else None)
        out.append({"appid": appid, "first_day": r.first_day.date(),
                    "latest_day": r.latest_day.date(), "n_neg": int(n_neg),
                    "terms": terms, "summary": summary,
                    "generated_at": datetime.now(timezone.utc)})
        done += 1
        if done % 25 == 0:
            print(f"  attributed {done} episodes…", flush=True)

    if not out:
        sys.exit("[!] nothing attributed")
    df = pd.DataFrame(out)
    staging = f"{PROJECT}.{DATASET}._staging_causes"
    bq.load_table_from_dataframe(
        df, staging, job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE")).result()
    bq.query(f"""
        MERGE `{PROJECT}.{DATASET}.alert_causes` t
        USING `{staging}` s ON t.appid = s.appid AND t.first_day = s.first_day
        WHEN MATCHED THEN UPDATE SET latest_day=s.latest_day, n_neg=s.n_neg,
             terms=s.terms, summary=s.summary, generated_at=s.generated_at
        WHEN NOT MATCHED THEN INSERT ROW""").result()
    bq.delete_table(staging, not_found_ok=True)
    print(f"[√] {len(df)} episodes attributed "
          f"({df['summary'].notna().sum()} with Gemini summaries) -> alert_causes")


if __name__ == "__main__":
    main()
