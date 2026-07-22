# -*- coding: utf-8 -*-
"""BuyOrWait Streamlit App: Purchase Decision / Bombing Alert / Ask Gemini / Player Composition.
Queries only aggregated small tables in BigQuery, never touches raw data.
The Purchase Decision tab overlays a Live check pulled from the public Steam
Web API (appreviews/storesearch) so any game — even post-snapshot releases —
can be compared against the snapshot scores.
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

PROJECT = os.environ.get("GCP_PROJECT", "buyorwait-2026")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title="BuyOrWait — Steam Review Intelligence Platform", page_icon="🎮", layout="wide")

# Modern Glassmorphism Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background & Sleek Atmosphere */
.stApp {
    background: radial-gradient(circle at 50% 0%, #162033 0%, #0d121c 60%, #080b10 100%);
    color: #e2e8f0;
}

/* Top Navigation Bar & Headers */
header[data-testid="stHeader"] {
    background: rgba(13, 18, 28, 0.7) !important;
    backdrop-filter: blur(16px);
}

/* Hero Title Card */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #66c0f4 0%, #a5b4fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Metric Cards Styling */
[data-testid="stMetric"] {
    background: rgba(22, 32, 50, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(12px);
    transition: all 0.2s ease-in-out;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(102, 192, 244, 0.4);
    box-shadow: 0 6px 24px rgba(102, 192, 244, 0.15);
}
[data-testid="stMetricValue"] {
    font-weight: 800;
    color: #f8fafc;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.85rem;
}

/* Styled Tabs */
[data-testid="stTab"] {
    font-weight: 600;
    font-size: 15px;
    padding: 10px 22px;
    border-radius: 8px;
    color: #94a3b8;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid transparent;
    transition: all 0.2s ease;
    margin-right: 6px;
}
[data-testid="stTab"]:hover {
    color: #e2e8f0;
    border-color: rgba(102, 192, 244, 0.3);
}
[data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102, 192, 244, 0.22) 0%, rgba(56, 189, 248, 0.12) 100%);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background-color: rgba(13, 19, 31, 0.9);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Streamlit Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #f1f5f9;
    padding: 8px 16px;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    transform: translateY(-1px);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


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
    base = "RECOMMENDED (Buy)" if score >= 70 else ("PROCEED WITH CAUTION (Wait)" if score >= 40 else "NOT RECOMMENDED (Skip)")
    if recent is not None and not pd.isna(recent) and recent + 15 < score:
        base += " (Recent sentiment decline detected)"
    return base


# ---- Live check: today's sentiment straight from the public Steam Web API ----
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
              "—" if live["sample_pos"] is None else f"{live['sample_pos']:.0f}% Positive",
              None if (live["sample_pos"] is None or snap_recent is None)
              else f"{live['sample_pos'] - snap_recent:+.0f}% vs snapshot recent 90d")
    c4.metric("Newest review", live["newest"] or "—")
    st.caption("Fetched seconds ago from the public Steam appreviews API — the same feed an "
               "incremental ingestion job would stream into BigQuery to keep scores current.")


# ---- Semantic layer: query embedding + BigQuery VECTOR_SEARCH ----------------
@st.cache_resource
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")


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
    "Monetization & MTX": "microtransactions pay to win overpriced cash grab battle pass",
    "Low-End PC Performance": "fps drops stuttering lag poor optimization low end pc",
    "Bugs & Stability": "bugs crashes broken glitches corrupted save unplayable",
    "Server & Network Stability": "servers down disconnect lag matchmaking dead online",
    "Content Depth & Length": "too short lacking content finished in a few hours",
    "Repetitive Grind": "grindy repetitive boring farming time gated chores",
    "Steam Deck & Controller": "steam deck controller support broken keyboard only",
}


def radar_level(hits: pd.DataFrame) -> tuple[str, int]:
    close = hits[hits["distance"] < 0.45] if len(hits) else hits
    n = len(close)
    return ("High Risk", n) if n >= 12 else ("Moderate Risk", n) if n >= 5 else ("Low Risk", n)


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
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.25); padding: 6px 14px; border-radius: 20px; width: fit-content; margin-bottom: 14px;">
            <span style="color: #4ade80; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;">LIVE SYNC ACTIVE</span>
            <span style="color: #94a3b8; font-size: 13px;">| Steam Sync <strong>{hrs}h ago</strong> · Ingested <strong>{int(f.tracked_games):,} games</strong> · <strong>+{int(f.delta_reviews):,}</strong> reviews synced</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(234, 179, 8, 0.1); border: 1px solid rgba(234, 179, 8, 0.25); padding: 6px 14px; border-radius: 20px; width: fit-content; margin-bottom: 14px;">
            <span style="color: #facc15; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;">SNAPSHOT ACTIVE</span>
            <span style="color: #94a3b8; font-size: 13px;">| 114M Review Corpus Loaded · Nightly Sync Ingesting</span>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<div class="hero-title">BuyOrWait — Steam Review Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">114M Steam Review Corpus · 90-Day Exponential Half-Life Model · Gemini AI Agent</div>', unsafe_allow_html=True)
freshness_badge()

# ---- My Radar: personal dealbreaker profile (session-only, no login) ---------
with st.sidebar:
    st.header("Personal Risk Radar")
    st.caption("Customize your hardware, time budget, and dealbreaker dimensions to generate personalized purchase recommendations.")
    my_dims = st.multiselect("Personal Dealbreakers", list(RADAR_DIMS), default=[])
    my_device = st.selectbox("Hardware Platform", ["High-end PC", "Low-end PC", "Steam Deck"], index=0)
    my_hours = st.slider("Gaming Hours / Week", 1, 40, 8)
    if my_device == "Low-end PC" and "Low-End PC Performance" not in my_dims:
        my_dims.append("Low-End PC Performance")
    if my_device == "Steam Deck" and "Steam Deck & Controller" not in my_dims:
        my_dims.append("Steam Deck & Controller")

tab_buy, tab_alert, tab_ask, tab_comp = st.tabs(
    ["Purchase Decision", "Review Bombing Alerts", "Ask Gemini AI", "Player Composition"])

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
                st.subheader("Live Verification — Steam Web API")
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
            c4.metric("Reviews Analyzed", f"{int(row.n_reviews_total):,}")
        with st.expander("Score Rationale & Decay Model"):
            st.markdown(
                f"- **All-time positive rate:** {row.raw_pos_rate:.0f}% — baseline store page ratio\n"
                f"- **Confidence score:** {row.score_live:.0f} — weighted by "
                f"`log(1+playtime) × exp(−age/90d)`, anchored to **today**\n"
                f"- **Recent 90 days:** "
                f"{'no recent reviews' if pd.isna(row.recent_pos_rate_live) else f'{row.recent_pos_rate_live:.0f}% positive over {int(row.recent_n_live):,} reviews'}\n"
                f"- **Latest review date:** {row.last_review_day}\n\n"
                "A large divergence between confidence score and all-time positive rate indicates recent sentiment shifts (patch updates or review bombing)."
            )

        # Reviewer composition X-ray
        try:
            comp = q(f"""SELECT n, purchase_pct, free_pct, ea_pct
                         FROM {T('game_composition')} WHERE appid = @a""", a=appid)
            if not comp.empty:
                c0 = comp.iloc[0]
                st.caption(f"Reviewer Acquisition Mix ({int(c0.n):,} reviews): "
                           f"{c0.purchase_pct:.0f}% Direct Purchase · "
                           f"{c0.free_pct:.0f}% Free/Gift Key · "
                           f"{c0.ea_pct:.0f}% Early Access Reviewers")
        except Exception:
            pass

        # ---------------- For YOU: personal collision card ----------------
        try:
            extra = q(f"""SELECT refund_zone_pct, pos_median_hours
                          FROM {T('game_scores')} WHERE appid = @a""", a=appid)
        except Exception:
            extra = pd.DataFrame()
        st.subheader("Personalized Dealbreaker Radar")
        red_flags = []
        if my_dims:
            cols = st.columns(min(len(my_dims), 4))
            for i, dim in enumerate(my_dims):
                hits = vsearch(appid, RADAR_DIMS[dim], k=25, polarity=False)
                if hits.empty:
                    cols[i % 4].metric(dim, "—", "not in indexed set", delta_color="off")
                    continue
                level, n = radar_level(hits)
                if level.startswith("High"):
                    red_flags.append(dim)
                cols[i % 4].metric(dim, level, f"{n} relevant complaints", delta_color="off")
                with cols[i % 4].popover("View Evidence Quotes"):
                    for t in hits["text"].head(3):
                        st.caption(f"“{t[:220]}…”")
        if not extra.empty and not pd.isna(extra.iloc[0].pos_median_hours):
            med_h = float(extra.iloc[0].pos_median_hours)
            weeks = med_h / max(my_hours, 1)
            rz = extra.iloc[0].refund_zone_pct
            line = (f"Optimal Immersion Curve: Happy players reach core satisfaction at **{med_h:.0f}h** — "
                    f"at your {my_hours}h/week budget, that represents **~{weeks:.1f} weeks** to full value.")
            if not pd.isna(rz):
                line += (f" | Steam Refund Window: {rz:.0f}% of dissatisfied players requested a refund "
                         f"within the 2-hour window.")
            st.markdown(line)
        if red_flags:
            st.error(f"Personal Verdict: PROCEED WITH CAUTION — Your selected dealbreakers "
                     f"({', '.join(red_flags)}) show elevated complaint density.")
        elif my_dims:
            st.success("Personal Verdict: NO ELEVATED RISK detected for your profile.")
        else:
            st.caption("Select your dealbreaker dimensions in the sidebar to view personalized risk matching.")

        # ---------------- Courtroom mode: forced adversarial verdict ----
        if st.button("Courtroom Analysis — AI Adversarial Arguments", key=f"court_{appid}"):
            pros = vsearch(appid, "amazing experience totally worth it best game recommended", 10, polarity=True)
            cons = vsearch(appid, "broken disappointed waste of money refund problems", 10, polarity=False)
            if pros.empty and cons.empty:
                st.info("This game is not in the semantic vector index.")
            else:
                ev_p = "\n".join(f"- {t[:200]}" for t in pros["text"].head(8))
                ev_c = "\n".join(f"- {t[:200]}" for t in cons["text"].head(8))
                try:
                    from google import genai as _genai_mod
                    key = os.environ.get("GEMINI_API_KEY")
                    _gc = (_genai_mod.Client(api_key=key) if key else
                           _genai_mod.Client(vertexai=True, project=PROJECT,
                                             location=os.environ.get("VERTEX_LOCATION", "global")))
                    verdictmd = _gc.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=(f"You are a courtroom analyst evaluating '{row.game}'. Real player evidence:\n"
                                  f"DEFENSE EVIDENCE (Positive Reviews):\n{ev_p}\n"
                                  f"PROSECUTION EVIDENCE (Negative Reviews):\n{ev_c}\n"
                                  "Write in markdown: '#### Defense Argument (Case for Purchase)' (3 bullets), "
                                  "'#### Prosecution Argument (Case Against)' (3 bullets), "
                                  "'#### Final Judicial Ruling' (2 sentences, decisive advice). Ground every claim in evidence.")).text
                    st.markdown(verdictmd)
                    st.caption("Adversarial Analysis: Forces the LLM to argue both defense and prosecution positions using real retrieved player reviews.")
                except Exception as e:
                    st.error(f"Gemini service unavailable: {e}")

        daily = q(f"""
            SELECT day, n, SAFE_DIVIDE(pos, n) AS pos_rate
            FROM {T('v_daily_all')} WHERE appid = @a ORDER BY day""", a=appid)
        if not daily.empty:
            daily["day"] = pd.to_datetime(daily["day"])
            daily = daily.set_index("day")
            daily["Positive Rate (7d rolling avg)"] = daily["pos_rate"].rolling(7, min_periods=1).mean() * 100
            st.line_chart(daily["Positive Rate (7d rolling avg)"], height=260)
            st.bar_chart(daily["n"].rename("Daily Review Volume"), height=160)

        st.divider()
        st.subheader("Live Verification — Steam API")
        live_panel(appid,
                   None if pd.isna(row.recent_pos_rate_live) else float(row.recent_pos_rate_live))

# ---------------------------------------------------------------- Bombing Alert
with tab_alert:
    c1, c2 = st.columns(2)
    zmin = c1.slider("Minimum Z-Score Severity", 3.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Min Daily Reviews Threshold", 3, 200, 30, 1)
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
    st.caption(f"Identified {len(alerts)} review bombing episodes. Criterion: Negative review rate Z > {zmin} AND daily volume > 2x baseline. AI Attribution uses distinctive TF-IDF keyword extraction summarized by Gemini.")
    st.dataframe(alerts, use_container_width=True, height=480,
                 column_config={"why_bombed_ai": st.column_config.TextColumn(
                     "AI Attribution Summary", width="large")})

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
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=1024 ** 3)
    return _client().query(sql, job_config=cfg).to_dataframe()


with tab_ask:
    ask_mode = st.radio("Mode", ["SQL Analytics", "RAG Semantic Q&A"],
                        horizontal=True, label_visibility="collapsed")
    if ask_mode == "RAG Semantic Q&A":
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
                numbered = "\n".join(f"[{i+1}] ({'Positive' if v else 'Negative'}, {h:.0f}h played) {t[:250]}"
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
                    st.caption("Grounded RAG: Query embedded in-process → BigQuery VECTOR_SEARCH → Gemini answers only from evidence.")
                except Exception as e:
                    st.error(f"Gemini unavailable: {e}")
        elif rag_pick and rag_q:
            st.warning("Ask-Gemini session limit (5) reached — refresh for a new session.")
    if ask_mode == "SQL Analytics":
        st.caption(f"Ask in natural English — Gemini ({GEMINI_MODEL}) generates read-only BigQuery SQL over aggregated tables:")
        examples = [
            "Top 10 games by purchase confidence with at least 100k reviews",
            "Which 5 games had the most review-bombing days, and when was the latest?",
            "Show games with recent positive rate higher than all-time average",
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

# ---------------------------------------------------------------- Player Composition
with tab_comp:
    st.subheader("Reviewer Composition & Astroturf Screening Radar")
    st.caption("Acquisition channel X-ray across 32,793 games in the snapshot — Direct Steam Purchase %, Free Key / Gift %, Early Access Review %.")
    try:
        comp_df = q(f"""
            SELECT s.game, s.appid, s.n_reviews_total AS n_reviews,
                   c.purchase_pct, c.free_pct, c.ea_pct
            FROM {T('game_composition')} c
            JOIN {T('v_scores_live')} s ON c.appid = s.appid
            WHERE s.n_reviews_total > 500
            ORDER BY s.n_reviews_total DESC
            LIMIT 500
        """)
        st.dataframe(
            comp_df,
            use_container_width=True,
            height=420,
            column_config={
                "game": st.column_config.TextColumn("Game Title", width="medium"),
                "purchase_pct": st.column_config.NumberColumn("Direct Steam Purchase %", format="%.1f%%"),
                "free_pct": st.column_config.NumberColumn("Free Key / Gift % (Astroturf Risk)", format="%.1f%%"),
                "ea_pct": st.column_config.NumberColumn("Early Access Review %", format="%.1f%%"),
                "n_reviews": st.column_config.NumberColumn("Total Reviews", format="%d"),
            }
        )
    except Exception as e:
        st.info(f"Player composition data unavailable ({e}).")

st.divider()
st.caption("Data: 114M-review Kaggle snapshot + nightly Steam Web API sync | "
           "Architecture: GCS + BigQuery + Cloud Run + Cloud Scheduler/Jobs + RAPIDS on L4 + Gemini (Vertex AI) | "
           "App only queries aggregated tables, latency < 2s")
