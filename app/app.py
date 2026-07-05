# -*- coding: utf-8 -*-
"""BuyOrWait Streamlit App: Purchase Decision / Bombing Alert / Ask Gemini / Why GPU.
Queries only aggregated small tables in BigQuery, never touches raw data.
The Purchase Decision tab overlays a 🔴 Live check pulled from the public Steam
Web API (appreviews/storesearch) so any game — even post-snapshot releases —
can be compared against the 2023-10 snapshot scores.
Environment variables: GCP_PROJECT (required), BQ_DATASET (default: steam_intel)
  Ask Gemini tab: GEMINI_API_KEY (Google AI Studio key), or leave unset to use
  Vertex AI with the runtime service account (GEMINI_MODEL / VERTEX_LOCATION optional)
"""
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT", "buyorwait-2026")   # Change to your GCP project ID or set via env var
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title="BuyOrWait", page_icon="🎮", layout="wide")


@st.cache_resource
def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


@st.cache_data(ttl=600, show_spinner="Querying BigQuery...")
def q(sql: str, **params) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter(k, "STRING" if isinstance(v, str) else "FLOAT64"
                                      if isinstance(v, float) else "INT64", v)
        for k, v in params.items()
    ])
    return _client().query(sql, job_config=cfg).to_dataframe()


def T(name: str) -> str:
    return f"`{PROJECT}.{DATASET}.{name}`"


def verdict(score: float, recent: float | None) -> str:
    base = "🟢 Buy" if score >= 70 else ("🟡 Wait" if score >= 40 else "🔴 Skip")
    if recent is not None and not pd.isna(recent) and recent + 15 < score:
        base += " (⚠ Recent reviews are significantly lower than overall score)"
    return base


# ---- 🔴 Live check: today's sentiment straight from the public Steam Web API ----
STEAM_HDRS = {"User-Agent": "BuyOrWait/1.0 (hackathon demo)"}


@st.cache_data(ttl=300, show_spinner="Searching Steam live...")
def steam_search(term: str) -> pd.DataFrame:
    r = requests.get("https://store.steampowered.com/api/storesearch/",
                     params={"term": term, "l": "english", "cc": "US"},
                     headers=STEAM_HDRS, timeout=10)
    r.raise_for_status()
    apps = [it for it in r.json().get("items", []) if it.get("type") == "app"]
    return pd.DataFrame([{"appid": it["id"], "game": it["name"]} for it in apps])


@st.cache_data(ttl=300, show_spinner="Contacting Steam API...")
def steam_live(appid: int, pages: int = 2) -> dict | None:
    """Overall totals plus a sample of the newest reviews for one game."""
    base = f"https://store.steampowered.com/appreviews/{appid}"
    common = {"json": 1, "language": "all", "purchase_type": "all"}
    js = requests.get(base, params={**common, "num_per_page": 0},
                      headers=STEAM_HDRS, timeout=10).json()
    summ = js.get("query_summary", {})
    if js.get("success") != 1 or not summ.get("total_reviews"):
        return None
    votes, newest, cursor = [], 0, "*"
    for _ in range(pages):
        js2 = requests.get(base, params={**common, "filter": "recent",
                                         "num_per_page": 100, "cursor": cursor},
                           headers=STEAM_HDRS, timeout=10).json()
        revs = js2.get("reviews", [])
        if not revs:
            break
        votes += [rv["voted_up"] for rv in revs]
        newest = max(newest, max(rv["timestamp_created"] for rv in revs))
        cursor = js2.get("cursor", "")
        if len(revs) < 100 or not cursor:
            break
    return {
        "desc": summ.get("review_score_desc", "—"),
        "total": summ["total_reviews"],
        "total_pos": summ.get("total_positive", 0) / summ["total_reviews"] * 100,
        "sample_n": len(votes),
        "sample_pos": sum(votes) / len(votes) * 100 if votes else None,
        "newest": (datetime.fromtimestamp(newest, tz=timezone.utc).strftime("%Y-%m-%d")
                   if newest else None),
    }


def live_panel(appid: int, snap_recent: float | None):
    """Render live Steam metrics; never break the snapshot view if Steam is down."""
    try:
        live = steam_live(int(appid))
    except Exception as e:
        st.info(f"Steam API unreachable right now ({e}) — snapshot data above is unaffected.")
        return
    if live is None:
        st.info("Steam reports no reviews for this app.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall rating (live)", live["desc"],
              f"{live['total_pos']:.0f}% positive", delta_color="off")
    c2.metric("Total reviews (live)", f"{live['total']:,}")
    c3.metric(f"Newest {live['sample_n']} reviews",
              "—" if live["sample_pos"] is None else f"{live['sample_pos']:.0f}% 👍",
              None if (live["sample_pos"] is None or snap_recent is None)
              else f"{live['sample_pos'] - snap_recent:+.0f}% vs snapshot recent 90d")
    c4.metric("Newest review", live["newest"] or "—")
    st.caption("Fetched seconds ago from the public Steam appreviews API — the same feed an "
               "incremental ingestion job would stream into BigQuery to keep scores current.")


st.title("🎮 BuyOrWait — To Buy or Not to Buy")
st.caption("114M Steam reviews (snapshot through Oct 2023) · Playtime-weighted + 90-day half-life decay · RAPIDS cudf.pandas on NVIDIA L4 (GCE)")

tab_buy, tab_alert, tab_ask, tab_gpu = st.tabs(
    ["🛒 Purchase Decision", "🚨 Bombing Alert", "💬 Ask Gemini", "⚡ Why GPU"])

# ---------------------------------------------------------------- Purchase Decision
with tab_buy:
    asof = q(f"SELECT FORMAT_DATE('%Y-%m-%d', DATE(TIMESTAMP_SECONDS(DIV(MAX(date), 1000000000)))) AS d "
             f"FROM {T('game_daily')}")["d"].iloc[0]
    st.caption(f"Scores as of **{asof}** (dataset snapshot). The pipeline anchors 'today' to the newest review in the data; "
               "wire the Steam API for live incremental updates.")
    kw = st.text_input("Search Game Name", placeholder="e.g., Cyberpunk / Overwatch / Black Myth: Wukong")
    if kw:
        hits = q(f"""
            SELECT appid, game, score, raw_pos_rate, recent_pos_rate, n_reviews
            FROM {T('game_scores')}
            WHERE LOWER(game) LIKE CONCAT('%', LOWER(@kw), '%')
            ORDER BY n_reviews DESC LIMIT 20""", kw=kw)
        if hits.empty:
            st.info("Not in the 2023-10 snapshot — searching Steam live instead…")
            try:
                live_hits = steam_search(kw)
            except Exception as e:
                live_hits = pd.DataFrame()
                st.warning(f"Steam search failed: {e}")
            if live_hits.empty:
                st.info("No games found on Steam either, try another keyword.")
            else:
                opts = {f"{g}  (#{a})": int(a)
                        for g, a in zip(live_hits["game"], live_hits["appid"])}
                pick = st.selectbox("Found on Steam (live):", list(opts))
                st.subheader("🔴 Live check — Steam right now")
                live_panel(opts[pick], None)
                st.caption("This game post-dates the snapshot, so it has no Purchase Confidence "
                           "Score yet — rerunning the pipeline with fresh reviews would score it.")
        else:
            row = hits.iloc[0]
            if len(hits) > 1:
                label = st.selectbox("Multiple matches found, select one:", hits["game"] + "  (#" + hits["appid"].astype(str) + ")")
                row = hits.iloc[list(hits.index)[int(
                    (hits["game"] + "  (#" + hits["appid"].astype(str) + ")").tolist().index(label))]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Purchase Confidence (0-100)", f"{row.score:.0f}")
            c2.metric("Verdict", verdict(row.score, row.recent_pos_rate))
            c3.metric("Recent 90d Positive Rate", "—" if pd.isna(row.recent_pos_rate) else f"{row.recent_pos_rate:.0f}%",
                      delta=None if pd.isna(row.recent_pos_rate) else f"{row.recent_pos_rate - row.raw_pos_rate:+.0f}% vs overall")
            c4.metric("Review Count", f"{int(row.n_reviews):,}")

            daily = q(f"""
                SELECT date, n, pos_rate FROM {T('game_daily')}
                WHERE appid = @a ORDER BY date""", a=int(row.appid))
            if not daily.empty:
                daily["date"] = pd.to_datetime(daily["date"])
                daily = daily.set_index("date")
                daily["Positive Rate (7d rolling avg)"] = daily["pos_rate"].rolling(7, min_periods=1).mean() * 100
                st.line_chart(daily["Positive Rate (7d rolling avg)"], height=260)
                st.bar_chart(daily["n"].rename("Daily Review Count"), height=160)

            st.divider()
            st.subheader("🔴 Live check — Steam right now")
            live_panel(int(row.appid),
                       None if pd.isna(row.recent_pos_rate) else float(row.recent_pos_rate))

# ---------------------------------------------------------------- Bombing Alert
with tab_alert:
    c1, c2 = st.columns(2)
    zmin = c1.slider("Minimum z-score", 3.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Min reviews on alert day (filters tiny-sample noise)", 3, 200, 30, 1)
    # One row per game = one bombing episode (multi-day alerts aggregated)
    alerts = q(f"""
        SELECT MAX(game)                  AS game,
               appid,
               DATE(TIMESTAMP_SECONDS(DIV(MIN(date), 1000000000)))            AS first_day,
               DATE(TIMESTAMP_SECONDS(DIV(MAX(date), 1000000000)))            AS latest_day,
               COUNT(*)                   AS alert_days,
               MAX(n)                     AS peak_daily_reviews,
               ROUND(MAX(neg_rate)*100,1) AS peak_neg_pct,
               ROUND(AVG(base_neg_rate)*100,1) AS baseline_neg_pct,
               ROUND(MAX(LEAST(z, 99.9)),1)    AS peak_z
        FROM {T('alerts')}
        WHERE z >= @z AND n >= @minn
        GROUP BY appid
        ORDER BY latest_day DESC, peak_daily_reviews DESC
        LIMIT 500""", z=float(zmin), minn=int(minn))
    st.caption(f"{len(alerts)} bombing episodes (one row per game, up to 500). "
               "Criterion: negative review rate z > 3 AND daily review count > 2x of 30d rolling average; z capped at 99.9.")
    st.dataframe(alerts, use_container_width=True, height=480)

# ---------------------------------------------------------------- Ask Gemini
SCHEMA_PROMPT = f"""You translate questions about Steam game reviews into BigQuery Standard SQL.

Tables:
1. {T('game_scores')} — one row per game.
   appid INT64, game STRING, score FLOAT64 (purchase confidence, 0-100),
   raw_pos_rate FLOAT64 (all-time positive rate in percent, 0-100),
   recent_pos_rate FLOAT64 (last-90-days positive rate in percent, NULL if no recent reviews),
   n_reviews INT64 (total reviews), recent_n INT64 (reviews in the last 90 days).
2. {T('game_daily')} — one row per game per day.
   appid INT64, date INT64 (epoch NANOSECONDS — convert with DATE(TIMESTAMP_SECONDS(DIV(date, 1000000000)))),
   n INT64 (reviews that day), pos INT64, neg INT64, pos_rate FLOAT64 (0-1), neg_rate FLOAT64 (0-1).
3. {T('alerts')} — one row per game per review-bombing day.
   appid INT64, game STRING, date INT64 (epoch nanoseconds, same conversion as above),
   n INT64 (reviews that day), neg_rate FLOAT64 (0-1), z FLOAT64 (severity z-score),
   base_neg_rate FLOAT64 (0-1, 30-day baseline).
4. {T('benchmark_results')} — CPU vs GPU pipeline timings.
   run_ts STRING, mode STRING ('cpu' or 'gpu'), stage STRING, seconds FLOAT64, rows INT64.

Rules:
- Output exactly ONE BigQuery Standard SQL SELECT (or WITH ... SELECT) statement — no markdown, no comments, no explanation.
- Read-only. Never generate INSERT/UPDATE/DELETE/DDL.
- The data is a static snapshot: reviews end on 2023-10-30. NEVER use CURRENT_DATE() or
  CURRENT_TIMESTAMP(); interpret "recent", "latest", "this year" etc. relative to 2023-10-30.
- Match game names case-insensitively: LOWER(game) LIKE '%...%'.
- End with LIMIT 100 unless the question implies a different limit.
"""

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|call|export)\b", re.I)


def guard_sql(sql: str) -> str | None:
    """Return an error message if the generated SQL is not a read-only SELECT."""
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return "Generated statement is not a SELECT — refused to run it."
    if _WRITE_KEYWORDS.search(sql):
        return "Generated SQL contains a write/DDL keyword — refused to run it."
    return None


@st.cache_data(ttl=600, show_spinner="Gemini is writing SQL...")
def nl_to_sql(question: str) -> str:
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    client = (genai.Client(api_key=key) if key else
              genai.Client(vertexai=True, project=PROJECT,
                           location=os.environ.get("VERTEX_LOCATION", "global")))
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=f"{SCHEMA_PROMPT}\nQuestion: {question}\nSQL:")
    sql = resp.text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip().rstrip(";")
    return sql


@st.cache_data(ttl=600, show_spinner="Querying BigQuery...")
def run_sql(sql: str) -> pd.DataFrame:
    # Cost guard: aggregated tables are tiny; refuse anything that would scan > 1 GB
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=1024 ** 3)
    return _client().query(sql, job_config=cfg).to_dataframe()


with tab_ask:
    st.caption(f"Ask in plain English — Gemini ({GEMINI_MODEL}) writes BigQuery SQL over the "
               "aggregated tables and runs it read-only. Try an example:")
    examples = [
        "Top 10 games by purchase confidence with at least 100k reviews",
        "Which 5 games had the most review-bombing days, and when was the latest?",
        "How much faster is the GPU than the CPU for each pipeline stage?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["nl_question"] = ex
    question = st.text_input("Your question", key="nl_question",
                             placeholder="e.g., Which games recovered from a bad launch?")
    if question:
        try:
            sql = nl_to_sql(question)
        except Exception as e:
            st.error(f"Gemini is not available: {e}")
            st.info("Set the GEMINI_API_KEY env var (Google AI Studio), or enable "
                    "`aiplatform.googleapis.com` and grant the service account "
                    "`roles/aiplatform.user` to use Vertex AI without a key.")
        else:
            err = guard_sql(sql)
            with st.expander("Generated SQL", expanded=False):
                st.code(sql, language="sql")
            if err:
                st.error(err)
            else:
                try:
                    out = run_sql(sql)
                except Exception as e:
                    st.error(f"BigQuery rejected the query: {e}")
                else:
                    st.dataframe(out, use_container_width=True)
                    st.caption(f"{len(out)} rows · query generated by {GEMINI_MODEL}, "
                               "validated as read-only, capped at 1 GB scanned.")

# ---------------------------------------------------------------- Why GPU
with tab_gpu:
    bm = q(f"SELECT * FROM {T('benchmark_results')}")
    if bm.empty:
        st.info("No benchmark data yet — please run the dual benchmark first according to instructions in H7-9.")
    else:
        # Get the latest run for each mode
        last = (bm.sort_values("run_ts").groupby(["mode", "stage"], as_index=False).last())
        pv = last.pivot(index="stage", columns="mode", values="seconds")
        if {"cpu", "gpu"} <= set(pv.columns):
            pv["speedup"] = (pv["cpu"] / pv["gpu"]).round(1)
        order = ["read_parquet", "clean_cast", "daily_groupby",
                 "weighted_score", "bomb_detect", "write_outputs", "end_to_end"]
        pv = pv.reindex([s for s in order if s in pv.index])
        rows_note = int(last["rows"].max())
        st.subheader(f"Same GCE g2-standard-8 instance: 8 vCPUs (pandas) vs NVIDIA L4 GPU (cudf.pandas)")
        st.caption(f"Data scale: {rows_note:,} rows; dual-run on the same machine, zero code change (`python -m cudf.pandas`).")
        st.dataframe(pv.style.format("{:.2f}", subset=[c for c in ("cpu", "gpu") if c in pv.columns]),
                     use_container_width=True)
        st.bar_chart(pv[[c for c in ("cpu", "gpu") if c in pv.columns]], height=320)
        if "speedup" in pv.columns:
            e2e = pv.loc["end_to_end", "speedup"] if "end_to_end" in pv.index else None
            if e2e:
                st.success(f"End-to-end speedup: **{e2e}×** — recalculation reduced from minutes to seconds. "
                           f"Bombing alerts can now be refreshed hourly instead of daily.")

st.divider()
st.caption("Data: Kaggle Steam Reviews — 114M reviews, snapshot through 2023 (public Steam API) | Architecture: GCS + BigQuery + Cloud Run + RAPIDS on L4 + Gemini (Vertex AI) | App only queries aggregated tables, latency < 2s")
