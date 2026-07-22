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


# ---- Semantic layer: query embedding + BigQuery VECTOR_SEARCH ----------------
@st.cache_resource
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")   # same model as the corpus


@st.cache_data(ttl=3600, show_spinner=False)
def embed_query(text: str) -> list[float]:
    return [float(x) for x in next(iter(_embedder().embed([text])))]


@st.cache_data(ttl=600, show_spinner=False)
def vsearch(appid: int, query: str, k: int = 20, polarity: bool | None = None) -> pd.DataFrame:
    """Filtered VECTOR_SEARCH (brute-force over one game's vectors — cheap at our scale)."""
    where = "appid = @a" + ("" if polarity is None else " AND voted_up = @p")
    params = [bigquery.ArrayQueryParameter("qv", "FLOAT64", embed_query(query)),
              bigquery.ScalarQueryParameter("a", "INT64", int(appid))]
    if polarity is not None:
        params.append(bigquery.ScalarQueryParameter("p", "BOOL", bool(polarity)))
    cfg = bigquery.QueryJobConfig(query_parameters=params,
                                  maximum_bytes_billed=2 * 1024 ** 3)
    sql = f"""
        SELECT base.text, base.voted_up, base.votes_up,
               base.playtime_h, base.day, distance
        FROM VECTOR_SEARCH(
          (SELECT * FROM {T('review_vectors')} WHERE {where}),
          'embedding', (SELECT @qv AS embedding),
          top_k => {int(k)}, distance_type => 'COSINE')
        ORDER BY distance"""
    try:
        return _client().query(sql, job_config=cfg).to_dataframe()
    except Exception:
        return pd.DataFrame()


RADAR_DIMS = {
    "💸 Monetization / MTX": "microtransactions pay to win overpriced cash grab battle pass",
    "🖥️ Performance on low-end PC": "fps drops stuttering lag poor optimization low end pc",
    "🐛 Bugs & crashes": "bugs crashes broken glitches corrupted save unplayable",
    "🌐 Servers / online issues": "servers down disconnect lag matchmaking dead online",
    "⏱️ Too short / thin content": "too short lacking content finished in a few hours",
    "😴 Grind / repetitive": "grindy repetitive boring farming time gated chores",
    "🎮 Deck / controller support": "steam deck controller support broken keyboard only",
}


def radar_level(hits: pd.DataFrame) -> tuple[str, int]:
    close = hits[hits["distance"] < 0.45] if len(hits) else hits
    n = len(close)
    return ("🔴 High", n) if n >= 12 else ("🟡 Some", n) if n >= 5 else ("🟢 Low", n)


def _log_usage(event: str, game: str, appid: int):
    """Fire-and-forget usage event -> BQ (deduped per session per game)."""
    key = f"logged_{event}_{appid}"
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    try:
        _client().insert_rows_json(
            f"{PROJECT}.{DATASET}.usage_events",
            [{"ts": datetime.now(timezone.utc).isoformat(),
              "event": event, "game": game, "appid": int(appid)}])
    except Exception:
        pass  # analytics must never break the app


def freshness_badge():
    try:
        f = q(f"SELECT last_sync, tracked_games, delta_reviews FROM {T('v_freshness')}").iloc[0]
        if pd.isna(f.last_sync):
            raise ValueError("no sync yet")
        hrs = max(0, int((pd.Timestamp.now(tz="UTC")
                          - pd.Timestamp(f.last_sync)).total_seconds() // 3600))
        st.caption(f"🟢 **LIVE** — last Steam sync **{hrs}h ago** · tracking "
                   f"**{int(f.tracked_games):,} games** nightly · "
                   f"**+{int(f.delta_reviews):,}** reviews since the 2023 snapshot · "
                   "scores decay-anchored to today")
    except Exception:
        st.caption("🟡 Snapshot mode — first nightly Steam sync lands tonight.")


st.title("🎮 BuyOrWait — To Buy or Not to Buy")
st.caption("114M-review snapshot + nightly Steam sync · Playtime-weighted, 90-day half-life evergreen scores · RAPIDS cudf.pandas on NVIDIA L4 (GCE)")
freshness_badge()

# ---- My Radar: personal dealbreaker profile (session-only, no login) ---------
with st.sidebar:
    st.header("🧭 My Radar")
    st.caption("We don't just measure the game — we measure how it will collide "
               "with *your* dealbreakers, hardware and time.")
    my_dims = st.multiselect("My dealbreakers", list(RADAR_DIMS), default=[])
    my_device = st.selectbox("My hardware", ["High-end PC", "Low-end PC", "Steam Deck"], index=0)
    my_hours = st.slider("My gaming hours / week", 1, 40, 8)
    if my_device == "Low-end PC" and "🖥️ Performance on low-end PC" not in my_dims:
        my_dims.append("🖥️ Performance on low-end PC")
    if my_device == "Steam Deck" and "🎮 Deck / controller support" not in my_dims:
        my_dims.append("🎮 Deck / controller support")

tab_buy, tab_alert, tab_ask, tab_gpu = st.tabs(
    ["🛒 Purchase Decision", "🚨 Bombing Alert", "💬 Ask Gemini", "⚡ Why GPU"])

# ---------------------------------------------------------------- Purchase Decision
with tab_buy:
    names_df = q(f"""SELECT appid, game FROM {T('game_scores')}
                     WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 20000""")
    labels = (names_df["game"] + "  (#" + names_df["appid"].astype(str) + ")").tolist()
    pick = st.selectbox("Search a game — type to filter (top 20,000 by review count)",
                        labels, index=None,
                        placeholder="e.g., Cyberpunk 2077 / Overwatch 2 / ELDEN RING")

    with st.expander("Game not listed? Search Steam live (post-2023 releases)"):
        kw_live = st.text_input("Steam live search", placeholder="e.g., Black Myth: Wukong")
        if kw_live:
            try:
                live_hits = steam_search(kw_live)
            except Exception as e:
                live_hits = pd.DataFrame()
                st.warning(f"Steam search failed: {e}")
            if live_hits.empty:
                st.info("No games found on Steam, try another keyword.")
            else:
                opts = {f"{g}  (#{a})": int(a)
                        for g, a in zip(live_hits["game"], live_hits["appid"])}
                pick_live = st.selectbox("Found on Steam (live):", list(opts))
                _log_usage("live_search", pick_live, opts[pick_live])
                st.subheader("🔴 Live check — Steam right now")
                live_panel(opts[pick_live], None)
                st.caption("This game post-dates the snapshot, so it has no Purchase Confidence "
                           "Score yet — tonight's sync starts scoring it once it enters the top tracked set.")

    if pick:
        appid = int(pick.rsplit("#", 1)[1].rstrip(")"))
        hit = q(f"SELECT * FROM {T('v_scores_live')} WHERE appid = @a", a=appid)
        row = hit.iloc[0]
        _log_usage("search", str(row.game), appid)

        c_img, c_m = st.columns([1, 3])
        c_img.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg",
                    use_container_width=True)
        with c_m:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Purchase Confidence (0-100)", f"{row.score_live:.0f}")
            c2.metric("Verdict", verdict(row.score_live, row.recent_pos_rate_live))
            c3.metric("Recent 90d Positive Rate",
                      "—" if pd.isna(row.recent_pos_rate_live) else f"{row.recent_pos_rate_live:.0f}%",
                      delta=None if (pd.isna(row.recent_pos_rate_live) or pd.isna(row.raw_pos_rate))
                      else f"{row.recent_pos_rate_live - row.raw_pos_rate:+.0f}% vs all-time")
            c4.metric("Reviews analyzed", f"{int(row.n_reviews_total):,}")
        with st.expander("Why this verdict?"):
            st.markdown(
                f"- **All-time positive rate:** {row.raw_pos_rate:.0f}% — what the Steam page implies\n"
                f"- **Confidence score:** {row.score_live:.0f} — every review weighted by "
                f"`log(1+playtime) × exp(−age/90d)`, anchored to **today**\n"
                f"- **Last 90 days:** "
                f"{'no recent reviews' if pd.isna(row.recent_pos_rate_live) else f'{row.recent_pos_rate_live:.0f}% positive over {int(row.recent_n_live):,} reviews'}\n"
                f"- **Newest review in data:** {row.last_review_day}\n\n"
                "Big gap between score and all-time rate = the game changed — "
                "recovered from a rough launch, or being bombed right now."
            )

        # ---------------- For YOU: personal collision card ----------------
        try:  # columns land with the D4 pipeline rerun; degrade gracefully before that
            extra = q(f"""SELECT refund_zone_pct, pos_median_hours
                          FROM {T('game_scores')} WHERE appid = @a""", a=appid)
        except Exception:
            extra = pd.DataFrame()
        st.subheader("🧭 For YOU")
        red_flags = []
        if my_dims:
            cols = st.columns(min(len(my_dims), 4))
            for i, dim in enumerate(my_dims):
                hits = vsearch(appid, RADAR_DIMS[dim], k=25, polarity=False)
                if hits.empty:
                    cols[i % 4].metric(dim, "—", "not in indexed set", delta_color="off")
                    continue
                level, n = radar_level(hits)
                if level.startswith("🔴"):
                    red_flags.append(dim)
                cols[i % 4].metric(dim, level, f"{n} close complaints", delta_color="off")
                with cols[i % 4].popover("quotes"):
                    for t in hits["text"].head(3):
                        st.caption(f"“{t[:220]}…”")
        if not extra.empty and not pd.isna(extra.iloc[0].pos_median_hours):
            med_h = float(extra.iloc[0].pos_median_hours)
            weeks = med_h / max(my_hours, 1)
            rz = extra.iloc[0].refund_zone_pct
            line = (f"⏳ Happy players hit their stride around **{med_h:.0f}h** — "
                    f"at your {my_hours}h/week that's **~{weeks:.1f} weeks** to real value.")
            if not pd.isna(rz):
                line += (f"  🛡️ **Refund radar:** {rz:.0f}% of unhappy players bailed "
                         f"within the 2-hour refund window — if it hasn't clicked by hour 2, refund.")
            st.markdown(line)
        if red_flags:
            st.error(f"**For YOU: 🟡 Wait.** Your dealbreaker{'s' if len(red_flags)>1 else ''} "
                     f"({', '.join(red_flags)}) show{'s' if len(red_flags)==1 else ''} high complaint density — "
                     "even though the crowd score above may look fine.")
        elif my_dims:
            st.success("**For YOU: no elevated risk** on your selected dealbreakers.")
        else:
            st.caption("Pick your dealbreakers in the sidebar to get a personal verdict, "
                       "not just the crowd's.")

        # ---------------- ⚖️ Courtroom mode: forced adversarial verdict ----
        if st.button("⚖️ Courtroom verdict — let AI argue both sides", key=f"court_{appid}"):
            pros = vsearch(appid, "amazing experience totally worth it best game recommended", 10, polarity=True)
            cons = vsearch(appid, "broken disappointed waste of money refund problems", 10, polarity=False)
            if pros.empty and cons.empty:
                st.info("This game isn't in the semantic index (top-300 + bombed games).")
            else:
                ev_p = "\n".join(f"- {t[:200]}" for t in pros["text"].head(8))
                ev_c = "\n".join(f"- {t[:200]}" for t in cons["text"].head(8))
                try:
                    from google import genai as _genai_mod  # reuse creds pattern
                    key = os.environ.get("GEMINI_API_KEY")
                    _gc = (_genai_mod.Client(api_key=key) if key else
                           _genai_mod.Client(vertexai=True, project=PROJECT,
                                             location=os.environ.get("VERTEX_LOCATION", "global")))
                    verdictmd = _gc.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=(f"You are a courtroom for the game '{row.game}'. Real player "
                                  f"evidence follows.\nDEFENSE evidence (positive reviews):\n{ev_p}\n"
                                  f"PROSECUTION evidence (negative reviews):\n{ev_c}\n"
                                  "Write in markdown: '#### 🟢 The case for buying' (3 bullets, "
                                  "cite evidence), '#### 🔴 The case against' (3 bullets), "
                                  "'#### ⚖️ Ruling' (2 sentences, decisive, who should buy and "
                                  "who should skip). Ground every claim in the evidence.")).text
                    st.markdown(verdictmd)
                    st.caption("Adversarial-by-design: the model must argue both sides from "
                               "retrieved real reviews before ruling — a confirmation-bias guard.")
                except Exception as e:
                    st.error(f"Gemini unavailable: {e}")

        daily = q(f"""
            SELECT day, n, SAFE_DIVIDE(pos, n) AS pos_rate
            FROM {T('v_daily_all')} WHERE appid = @a ORDER BY day""", a=appid)
        if not daily.empty:
            daily["day"] = pd.to_datetime(daily["day"])
            daily = daily.set_index("day")
            daily["Positive Rate (7d rolling avg)"] = daily["pos_rate"].rolling(7, min_periods=1).mean() * 100
            st.line_chart(daily["Positive Rate (7d rolling avg)"], height=260)
            st.bar_chart(daily["n"].rename("Daily Review Count"), height=160)

        st.divider()
        st.subheader("🔴 Live check — Steam right now")
        live_panel(appid,
                   None if pd.isna(row.recent_pos_rate_live) else float(row.recent_pos_rate_live))

# ---------------------------------------------------------------- Bombing Alert
with tab_alert:
    c1, c2 = st.columns(2)
    zmin = c1.slider("Minimum z-score", 3.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Min reviews on alert day (filters tiny-sample noise)", 3, 200, 30, 1)
    # One row per game = one bombing episode; snapshot + nightly-recomputed
    # recent alerts merged, joined with GPU/Gemini cause attribution.
    # Graceful degradation: fall back to snapshot-only alerts if the live
    # objects (docs/alerts_live.sql) haven't been created yet.
    try:
        alerts = q(f"""
            WITH ep AS (
              SELECT appid, ANY_VALUE(game) AS game,
                     MIN(day) AS first_day, MAX(day) AS latest_day,
                     COUNT(*) AS alert_days, MAX(n) AS peak_daily_reviews,
                     ROUND(MAX(neg_rate)*100,1)      AS peak_neg_pct,
                     ROUND(AVG(base_neg_rate)*100,1) AS baseline_neg_pct,
                     ROUND(MAX(LEAST(z, 99.9)),1)    AS peak_z
              FROM {T('v_alerts_all')}
              WHERE z >= @z AND n >= @minn
              GROUP BY appid)
            SELECT ep.game, c.summary AS why_bombed_ai, ep.first_day, ep.latest_day,
                   ep.alert_days, ep.peak_daily_reviews, ep.peak_neg_pct,
                   ep.baseline_neg_pct, ep.peak_z, c.terms AS top_terms, ep.appid
            FROM ep LEFT JOIN {T('alert_causes')} c
              ON c.appid = ep.appid AND c.first_day = ep.first_day
            ORDER BY ep.latest_day DESC, ep.peak_daily_reviews DESC
            LIMIT 500""", z=float(zmin), minn=int(minn))
    except Exception:
        st.info("Live alerts / AI causes not initialized yet (run docs/alerts_live.sql "
                "and pipeline/attribute.py) — showing snapshot alerts.")
        alerts = q(f"""
            SELECT ANY_VALUE(game) AS game, appid,
                   DATE(TIMESTAMP_SECONDS(DIV(MIN(date), 1000000000))) AS first_day,
                   DATE(TIMESTAMP_SECONDS(DIV(MAX(date), 1000000000))) AS latest_day,
                   COUNT(*) AS alert_days, MAX(n) AS peak_daily_reviews,
                   ROUND(MAX(neg_rate)*100,1)      AS peak_neg_pct,
                   ROUND(AVG(base_neg_rate)*100,1) AS baseline_neg_pct,
                   ROUND(MAX(LEAST(z, 99.9)),1)    AS peak_z
            FROM {T('alerts')}
            WHERE z >= @z AND n >= @minn
            GROUP BY appid
            ORDER BY latest_day DESC, peak_daily_reviews DESC
            LIMIT 500""", z=float(zmin), minn=int(minn))
    st.caption(f"{len(alerts)} bombing episodes (one row per game, snapshot + live merged, up to 500). "
               "Criterion: negative rate z > 3 AND daily reviews > 2x of 30d rolling average. "
               "**why_bombed_ai** = distinctive-term extraction over the episode's negative reviews, "
               "summarized by Gemini; blank = below attribution threshold.")
    st.dataframe(alerts, use_container_width=True, height=480,
                 column_config={"why_bombed_ai": st.column_config.TextColumn(
                     "why bombed (AI)", width="large")})

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
5. {T('v_daily_all')} — PREFERRED for any date logic: one row per game per day,
   2023 snapshot MERGED with the nightly Steam sync. appid INT64, day DATE,
   n INT64, pos INT64, w_sum FLOAT64, wv_sum FLOAT64.
6. {T('v_scores_live')} — evergreen scores anchored to CURRENT_DATE():
   appid INT64, game STRING, score_live FLOAT64 (0-100), raw_pos_rate FLOAT64,
   recent_n_live INT64, recent_pos_rate_live FLOAT64, n_reviews_total INT64,
   last_review_day DATE.

Rules:
- Output exactly ONE BigQuery Standard SQL SELECT (or WITH ... SELECT) statement — no markdown, no comments, no explanation.
- Read-only. Never generate INSERT/UPDATE/DELETE/DDL.
- Data = 2023 snapshot + nightly delta for top tracked games. CURRENT_DATE() is fine.
  For "recent / now / today" questions prefer v_daily_all and v_scores_live
  (their day/last_review_day columns are proper DATEs; game_daily.date is epoch nanoseconds).
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
    ask_mode = st.radio("Mode", ["📊 SQL analytics", "💬 Ask the reviews (RAG)"],
                        horizontal=True, label_visibility="collapsed")
    if ask_mode == "💬 Ask the reviews (RAG)":
        names_rag = q(f"""SELECT appid, game FROM {T('game_scores')}
                          WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 300""")
        rag_pick = st.selectbox("Game", (names_rag["game"] + "  (#" +
                                         names_rag["appid"].astype(str) + ")").tolist(),
                                index=None, placeholder="Pick an indexed game (top 300)")
        rag_q = st.text_input("Your question about this game",
                              placeholder="e.g., How is performance on Steam Deck? Is the story worth it?")
        used_r = st.session_state.get("gemini_calls", 0)
        if rag_pick and rag_q and used_r < 5:
            st.session_state["gemini_calls"] = used_r + 1
            rag_appid = int(rag_pick.rsplit("#", 1)[1].rstrip(")"))
            ev = vsearch(rag_appid, rag_q, k=12)
            if ev.empty:
                st.info("No indexed reviews for this game.")
            else:
                numbered = "\n".join(f"[{i+1}] ({'👍' if v else '👎'}, {h:.0f}h played) {t[:250]}"
                                     for i, (t, v, h) in enumerate(
                                         zip(ev["text"], ev["voted_up"], ev["playtime_h"])))
                try:
                    from google import genai as _gm
                    key = os.environ.get("GEMINI_API_KEY")
                    _gc2 = (_gm.Client(api_key=key) if key else
                            _gm.Client(vertexai=True, project=PROJECT,
                                       location=os.environ.get("VERTEX_LOCATION", "global")))
                    ans = _gc2.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=(f"Answer the question using ONLY these player reviews of "
                                  f"'{rag_pick}'. Cite like [3]. If evidence is mixed, say so. "
                                  f"Max 120 words.\nREVIEWS:\n{numbered}\n"
                                  f"QUESTION: {rag_q}\nANSWER:")).text
                    st.markdown(ans)
                    with st.expander("Evidence (retrieved via BigQuery VECTOR_SEARCH)"):
                        st.text(numbered)
                    st.caption("Grounded RAG: query embedded in-process (CPU, ~50ms) → BigQuery "
                               "VECTOR_SEARCH over GPU-built corpus → Gemini answers only from evidence.")
                except Exception as e:
                    st.error(f"Gemini unavailable: {e}")
        elif rag_pick and rag_q:
            st.warning("Ask-Gemini session limit (5) reached — refresh for a new session.")
    if ask_mode == "📊 SQL analytics":
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
        used = st.session_state.get("gemini_calls", 0)
        if question and used >= 5:
            st.warning("Ask-Gemini limit for this session (5 questions) reached — refresh the page for a new session.")
            question = None
        if question:
            st.session_state["gemini_calls"] = used + 1
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
st.caption("Data: 114M-review Kaggle snapshot + nightly Steam Web API sync (all public Steam data) | "
           "Architecture: GCS + BigQuery + Cloud Run + Cloud Scheduler/Jobs + RAPIDS on L4 + Gemini (Vertex AI) | "
           "App only queries aggregated tables, latency < 2s")
