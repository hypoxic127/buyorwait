# -*- coding: utf-8 -*-
"""D1 feasibility check: can we extract WHY a game was review-bombed?

Scans the raw Kaggle CSV (review text still intact there) for ONE game,
compares negative-review vocabulary in the alert window vs a baseline
window, prints the top distinctive keywords + example lines.
If the keywords look meaningful (e.g. "kernel", "anticheat", "price"),
green-light the full attribution feature for D3. If they're generic
("game", "bad"), switch to the Gemini-summary fallback.

Usage (on the VM, ~/raw must still contain the Kaggle CSV):
  python3 verify_attribution.py --appid 2357570 --start 2023-08-10 --end 2023-08-31
  (default = Overwatch 2's Steam launch bombing, present in the snapshot)
Optional: --csv ~/raw/all_reviews.csv  --lang english  --baseline-days 60
"""
import argparse
import glob
import os
import re
from collections import Counter

import pandas as pd

STOP = set("""the and for was but not you all are this that with have has had
they them their there its it's just like game games really only get got very
much more even still than then will would can could into out about a an of to
in on is it as at be or so if my me we he she his her do did does don doesn
didn isn aren won because been being play played playing player players time
hours hour make makes made no yes one two now new old way from too also lot
who what when where why how which your our us him think know go going""".split())

TOKEN = re.compile(r"[a-zA-Z']{3,}")


def load_game(csv_paths, appid, lang):
    keep = []
    usecols = ["appid", "review", "language", "timestamp_created", "voted_up"]
    for path in csv_paths:
        print(f"[i] scanning {path} …", flush=True)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=2_000_000,
                                 on_bad_lines="skip", low_memory=True):
            part = chunk[chunk["appid"] == appid]
            if lang != "all":
                part = part[part["language"] == lang]
            if len(part):
                keep.append(part)
    if not keep:
        raise SystemExit(f"[!] appid {appid} not found — check the CSV path")
    df = pd.concat(keep, ignore_index=True)
    df["day"] = pd.to_datetime(pd.to_numeric(df["timestamp_created"],
                                             errors="coerce"), unit="s")
    df["neg"] = ~df["voted_up"].astype(str).str.lower().isin(("true", "1"))
    print(f"[i] {len(df):,} reviews loaded for appid {appid}")
    return df


def top_terms(df, start, end, baseline_days, k=25):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    alert = df[(df["day"] >= start) & (df["day"] <= end) & df["neg"]]
    base = df[(df["day"] >= start - pd.Timedelta(days=baseline_days))
              & (df["day"] < start) & df["neg"]]
    print(f"[i] negative reviews — alert window: {len(alert):,}, "
          f"baseline: {len(base):,}")
    if len(alert) < 50:
        print("[!] alert window very small; results may be noisy")

    def counts(frame):
        c = Counter()
        for text in frame["review"].astype(str):
            c.update(t for t in (w.lower() for w in TOKEN.findall(text))
                     if t not in STOP)
        return c

    ca, cb = counts(alert), counts(base)
    na, nb = max(sum(ca.values()), 1), max(sum(cb.values()), 1)
    scored = []
    for term, fa in ca.items():
        if fa < 10:
            continue
        lift = (fa / na) / ((cb.get(term, 0) + 1) / nb)
        scored.append((lift * (fa ** 0.5), lift, fa, term))
    scored.sort(reverse=True)

    print(f"\n=== TOP {k} distinctive terms in the bombing window ===")
    print(f"{'term':<20s} {'lift':>8s} {'count':>7s}   example")
    shown = 0
    for _, lift, fa, term in scored:
        if shown >= k:
            break
        ex = next((t[:110].replace("\n", " ") for t in
                   alert["review"].astype(str) if term in t.lower()), "")
        print(f"{term:<20s} {lift:>8.1f} {fa:>7,d}   {ex}")
        shown += 1
    print("\n[?] Meaningful terms (specific complaints)?  -> build attribution (D3)"
          "\n    Generic terms only?                      -> Gemini-summary fallback")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--appid", type=int, default=2357570)   # Overwatch 2
    ap.add_argument("--start", default="2023-08-10")        # OW2 Steam launch
    ap.add_argument("--end", default="2023-08-31")
    ap.add_argument("--csv", default=os.path.expanduser("~/raw"))
    ap.add_argument("--lang", default="english")
    ap.add_argument("--baseline-days", type=int, default=60)
    a = ap.parse_args()
    paths = ([a.csv] if os.path.isfile(a.csv) else
             sorted(glob.glob(os.path.join(a.csv, "**", "*.csv"),
                              recursive=True)))
    if not paths:
        raise SystemExit(f"[!] no CSV under {a.csv} — raw data deleted? "
                         "Re-download or use the API fallback on a recent event.")
    df = load_game(paths, a.appid, a.lang)
    top_terms(df, a.start, a.end, a.baseline_days)
