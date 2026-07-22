# -*- coding: utf-8 -*-
"""Detector evaluation: does our z-score bombing detector catch DOCUMENTED
review-bombing incidents?

Ground truth: ~19 publicly documented Steam review-bombing events (sources:
Wikipedia 'Review bomb' article + contemporaneous gaming press). For each,
check whether v_alerts_all contains an episode within the event window (±14d).

Outputs:
  1. Recall table (✓/✗ per incident, peak z, alert days) — README/PPT ready
  2. Top-25 largest detected episodes NOT in the list — 10-minute manual
     precision check (google each; most should be real incidents)

Run anywhere with BQ access:  GCP_PROJECT=buyorwait-2026 python3 eval_detector.py
"""
import os
import sys

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
if not PROJECT:
    sys.exit("[!] Set GCP_PROJECT")
bq = bigquery.Client(project=PROJECT)

# (game, appid, window_start, window_end, documented cause)
EVENTS = [
    ("The Elder Scrolls V: Skyrim",   72850,  "2015-04-23", "2015-05-10", "paid mods"),
    ("Grand Theft Auto V",            271590, "2017-06-14", "2017-07-05", "OpenIV mod takedown"),
    ("Firewatch",                     383870, "2017-09-10", "2017-09-25", "PewDiePie DMCA"),
    ("Middle-earth: Shadow of War",   356190, "2017-10-09", "2017-11-01", "loot boxes"),
    ("Total War: ROME II",            214950, "2018-09-20", "2018-10-15", "female generals"),
    ("Metro 2033 Redux",              286690, "2019-01-28", "2019-02-20", "Epic exclusivity"),
    ("Metro: Last Light Redux",       287390, "2019-01-28", "2019-02-20", "Epic exclusivity"),
    ("Borderlands 2",                 49520,  "2019-04-03", "2019-04-20", "BL3 Epic exclusive"),
    ("Mordhau",                       629760, "2019-05-20", "2019-06-10", "community controversy"),
    ("NBA 2K20",                      1089350,"2019-09-05", "2019-09-30", "casino-style MTX"),
    ("Monster Hunter: World",         582010, "2020-12-04", "2020-12-25", "movie backlash (CN)"),
    ("Cyberpunk 2077",                1091500,"2020-12-10", "2021-01-05", "broken launch"),
    ("eFootball 2022",                1665460,"2021-09-30", "2021-10-20", "disastrous launch"),
    ("Battlefield 2042",              1517290,"2021-11-19", "2021-12-15", "launch state"),
    ("Team Fortress 2",               440,    "2022-05-26", "2022-06-15", "#SaveTF2 bot crisis"),
    ("Total War: WARHAMMER III",      1142710,"2022-08-23", "2022-09-15", "DLC pricing"),
    ("The Last of Us Part I",         1888930,"2023-03-28", "2023-04-20", "PC port quality"),
    ("Overwatch 2",                   2357570,"2023-08-10", "2023-09-05", "PVE cancelled / MTX"),
    ("Counter-Strike 2",              730,    "2023-09-27", "2023-10-25", "CS:GO replaced"),
]
PAD_DAYS = 14


def main():
    rows, hit = [], 0
    for game, appid, start, end, cause in EVENTS:
        df = bq.query(f"""
            SELECT COUNT(*) AS days, MAX(LEAST(z, 99.9)) AS peak_z, MAX(n) AS peak_n
            FROM `{PROJECT}.{DATASET}.v_alerts_all`
            WHERE appid = {appid}
              AND day BETWEEN DATE_SUB('{start}', INTERVAL {PAD_DAYS} DAY)
                          AND DATE_ADD('{end}',   INTERVAL {PAD_DAYS} DAY)
        """).to_dataframe().iloc[0]
        ok = df.days > 0
        hit += int(ok)
        rows.append((("✅" if ok else "❌"), game, cause, start,
                     int(df.days), None if pd.isna(df.peak_z) else round(float(df.peak_z), 1)))
    out = pd.DataFrame(rows, columns=["hit", "game", "documented cause",
                                      "event start", "alert days", "peak z"])
    print("\n=== RECALL vs documented incidents ===")
    print(out.to_string(index=False))
    print(f"\n>>> RECALL: {hit}/{len(EVENTS)} documented incidents detected "
          f"({hit/len(EVENTS)*100:.0f}%)\n")
    print("README/PPT line:")
    print(f'  "Validated against {len(EVENTS)} publicly documented review-bombing '
          f'incidents (2015-2023): the detector catches {hit}/{len(EVENTS)}."\n')

    known = {a for _, a, *_ in EVENTS}
    top = bq.query(f"""
        SELECT ANY_VALUE(game) AS game, appid, MIN(day) AS first_day,
               MAX(n) AS peak_n, MAX(LEAST(z,99.9)) AS peak_z
        FROM `{PROJECT}.{DATASET}.v_alerts_all`
        WHERE n >= 100 GROUP BY appid
        ORDER BY MAX(n) DESC LIMIT 30
    """).to_dataframe()
    top = top[~top["appid"].isin(known)].head(25)
    print("=== Top detected episodes NOT in the ground-truth list ===")
    print("(manual precision check: google '<game> review bomb <year>' — "
          "tick how many are real incidents)")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
