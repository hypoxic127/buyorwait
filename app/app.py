# -*- coding: utf-8 -*-
"""BuyOrWait Streamlit App: Person-to-Game Resonance Engine.
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

st.set_page_config(page_title="BuyOrWait — Person-to-Game Resonance Engine", page_icon="🎮", layout="wide")

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

/* Styled Tabs — target the stable BaseWeb tab hooks (data-testid="stTab" is not reliable) */
[data-testid="stTabs"] button[data-baseweb="tab"] {
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
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #e2e8f0;
    border-color: rgba(102, 192, 244, 0.3);
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102, 192, 244, 0.22) 0%, rgba(56, 189, 248, 0.12) 100%);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);
}
/* Hide the default red BaseWeb tab underline highlight so our pill styling reads cleanly */
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background-color: rgba(13, 19, 31, 0.9);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Fix Streamlit Selectbox and Multiselect Text Truncation & Popover Alignment */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    border-radius: 8px;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    white-space: normal !important;
    word-wrap: break-word !important;
    font-size: 13px !important;
}

/* Expand Dropdown Popover Menu Width to Avoid Horizontal Truncation */
div[role="listbox"] {
    min-width: 320px !important;
    max-width: 420px !important;
    background: #0f172a !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
}
div[role="option"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
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


def fit_verdict(score: float, red_flags: list) -> tuple[str, str, str]:
    """Personal-fit badge as (title, description, accent_color).
    Thresholds mirror the documented purchase spec: >=70 strong, >=40 moderate."""
    if red_flags:
        return ("POOR MATCH", f"Collides with your dealbreakers: {', '.join(red_flags)}", "#f87171")
    if score >= 70:
        return ("EXCELLENT MATCH", "High alignment between your life rhythm and this game.", "#4ade80")
    if score >= 40:
        return ("MODERATE MATCH", "Playable, but may require adjusting your time expectations.", "#facc15")
    return ("LOW MATCH", "May not fit your current gaming routine or time budget.", "#f87171")


def verdict_badge(title: str, desc: str, color: str) -> str:
    """A color-coded verdict banner (used instead of st.metric, which truncates long text)."""
    return (f'<div style="background:{color}1a;border:1px solid {color}55;border-radius:12px;'
            f'padding:12px 18px;margin-bottom:12px;">'
            f'<span style="color:{color};font-weight:800;font-size:1.05rem;letter-spacing:0.3px;">{title}</span>'
            f'<div style="color:#cbd5e1;font-size:0.9rem;margin-top:3px;">{desc}</div></div>')


_RISK_COLOR = {"High Risk": "#f87171", "Moderate Risk": "#facc15", "Low Risk": "#4ade80"}


def risk_chip(dim: str, level: str, highlighted: bool) -> str:
    """A compact risk pill; highlighted dims (the user's dealbreakers) get a ring and star."""
    c = _RISK_COLOR.get(level, "#94a3b8")
    ring = f"box-shadow:0 0 0 1px {c};" if highlighted else ""
    star = " ★" if highlighted else ""
    return (f'<span style="display:inline-block;background:{c}1a;border:1px solid {c}55;{ring}'
            f'color:{c};border-radius:20px;padding:5px 12px;margin:3px;font-size:0.82rem;'
            f'font-weight:600;">{dim}: {level}{star}</span>')


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
    c1.metric("Overall Positive (Live)", f"{live['total_pos']:.0f}%")
    c2.metric("Total Reviews (Live)", f"{live['total']:,}")
    c3.metric(f"Newest {live['sample_n']} Reviews",
              "—" if live["sample_pos"] is None else f"{live['sample_pos']:.0f}%",
              None if (live["sample_pos"] is None or snap_recent is None)
              else f"{live['sample_pos'] - snap_recent:+.0f}% vs recent snapshot")
    c4.metric("Newest Review Date", live["newest"] or "—")
    st.caption(f"Steam summary: **{live['desc']}** — live data fetched directly from the Steam API.")


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
    # use_brute_force is required: the table has an ANN vector index, but we pre-filter
    # to one appid inside the base subquery. ANN probing would miss that game's rows and
    # return nothing — a full scan of one game's ~hundreds of vectors is cheap and exact.
    sql = f"""
        SELECT base.text, base.voted_up, base.votes_up,
               base.playtime_h, base.day, distance
        FROM VECTOR_SEARCH(
          (SELECT * FROM {T('review_vectors')} WHERE {where}),
          'embedding', (SELECT @qv AS embedding),
          top_k => {int(k)}, distance_type => 'COSINE',
          options => '{{"use_brute_force": true}}')
        ORDER BY distance"""
    try:
        return _client().query(sql, job_config=cfg).to_dataframe()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def game_vec_total(appid: int) -> int:
    """Total indexed review vectors for a game — the denominator for friction prevalence."""
    try:
        return int(q(f"SELECT COUNT(*) AS c FROM {T('review_vectors')} WHERE appid = @a",
                     a=int(appid)).iloc[0].c)
    except Exception:
        return 0


RADAR_DIMS = {
    "Monetization & MTX": "microtransactions pay to win overpriced cash grab battle pass",
    "Low-End PC Performance": "fps drops stuttering lag poor optimization low end pc",
    "Bugs & Stability": "bugs crashes broken glitches corrupted save unplayable",
    "Server & Disconnects": "servers down disconnect lag matchmaking dead online",
    "Short Content": "too short lacking content finished in a few hours",
    "Repetitive Grind": "grindy repetitive boring farming time gated chores",
    "Steam Deck & Controller": "steam deck controller support broken keyboard only",
}


def radar_level(hits: pd.DataFrame, total_reviews: int) -> tuple[str, int]:
    """Grade a friction dimension by PREVALENCE rather than a raw match count.

    We take the share of the game's sampled reviews that are *strongly* on-topic
    negatives (cosine distance < 0.30 — a genuine match, not a loose one), normalized
    by the game's total review sample. A loose <0.45 count saturates (almost every game
    hits it for every topic); prevalence-at-0.30 separates a game's real problems from
    generic noise. Thresholds calibrated against high- vs low-friction reference games.
    """
    if not len(hits) or total_reviews <= 0:
        return ("Low Risk", 0)
    strong = int((hits["distance"] < 0.30).sum())
    share = strong / total_reviews
    if share >= 0.03:
        return ("High Risk", strong)
    if share >= 0.01:
        return ("Moderate Risk", strong)
    return ("Low Risk", strong)


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
            <span style="color: #94a3b8; font-size: 13px;">| 114 Million Steam Reviews Loaded · Nightly Sync Ingesting</span>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<div class="hero-title">BuyOrWait — Person-to-Game Resonance Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Connecting Your Life Rhythm, Time Budget & Hardware to 114 Million Steam Player Experiences</div>', unsafe_allow_html=True)
freshness_badge()

# ---- Sidebar: Your Life Profile & Gaming Persona -----------------------------
with st.sidebar:
    st.header("Your Gaming Profile")
    st.caption("Define your life rhythm, gaming time budget, and dealbreakers to calculate personal fit.")
    
    my_rhythm = st.selectbox("Life Rhythm & Time Budget", [
        "Busy (30m sessions)",
        "Weekend (2-3h chunks)",
        "Hardcore (10+ hrs/wk)"
    ], index=0)
    
    my_goal = st.selectbox("Emotional Objective", [
        "Decompress (Low Stress)",
        "Challenge (Soulslike)",
        "Story & Narrative"
    ], index=0)

    my_device = st.selectbox("Hardware Platform", ["High-end PC", "Low-end PC", "Steam Deck"], index=0)
    my_hours = st.slider("Weekly Gaming Hours Budget", 1, 40, 6)
    
    my_dims = st.multiselect("Personal Dealbreaker Filters", list(RADAR_DIMS), default=[])
    if my_device == "Low-end PC" and "Low-End PC Performance" not in my_dims:
        my_dims.append("Low-End PC Performance")
    if my_device == "Steam Deck" and "Steam Deck & Controller" not in my_dims:
        my_dims.append("Steam Deck & Controller")

tab_buy, tab_alert, tab_ask, tab_comp = st.tabs(
    ["Person-Game Fit", "Review Bombing & Crowd Noise", "Ask Gemini AI", "Player Ownership"])

# ---------------------------------------------------------------- Person-Game Fit
with tab_buy:
    names_df = q(f"""SELECT appid, game FROM {T('game_scores')}
                     WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 20000""")
    labels = (names_df["game"] + "  (#" + names_df["appid"].astype(str) + ")").tolist()
    pick = st.selectbox("Select a game to test your personal fit (top 20,000 games)",
                        labels, index=None,
                        placeholder="e.g., Cyberpunk 2077 / Overwatch 2 / ELDEN RING")

    # Inline live-search fallback — visible immediately (no accordion), hidden once a game is picked.
    if not pick:
        st.caption("Not in the list, or released after our 2023 snapshot? Search Steam directly:")
        kw_live = st.text_input("Steam live search", placeholder="e.g., Black Myth: Wukong",
                                label_visibility="collapsed")
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
                st.subheader("Live Verification — Steam API")
                live_panel(opts[pick_live], None)
                st.caption("This game post-dates the main dataset, so its score is fetched directly from live Steam ratings.")

    if pick:
        appid = int(pick.rsplit("#", 1)[1].rstrip(")"))
        hit = q(f"SELECT * FROM {T('v_scores_live')} WHERE appid = @a", a=appid)
        row = hit.iloc[0]
        _log_usage("search", str(row.game), appid)

        # ---- Friction Radar: one vector search per dimension, computed ONCE.
        #      Red flags are derived from this same pass (no duplicate searches). ----
        with st.spinner("Profiling player sentiment across friction dimensions..."):
            total_vec = game_vec_total(appid)
            radar = {}  # dim -> {"level", "n", "hits"} or None when no indexed reviews
            for dim, query in RADAR_DIMS.items():
                # k wide enough that the strong-match (<0.30) count isn't truncated
                hits = vsearch(appid, query, k=80, polarity=False)
                if hits.empty:
                    radar[dim] = None
                else:
                    level, strong = radar_level(hits, total_vec)
                    radar[dim] = {"level": level, "n": strong, "hits": hits}
        red_flags = [d for d in my_dims
                     if radar.get(d) and radar[d]["level"].startswith("High")]

        fit_title, fit_desc, fit_color = fit_verdict(row.score_live, red_flags)

        c_img, c_m = st.columns([1, 3])
        c_img.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg",
                    use_container_width=True)
        with c_m:
            st.markdown(verdict_badge(fit_title, fit_desc, fit_color), unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rating Score", f"{row.score_live:.0f}/100")
            m2.metric("Recent 90-Day Rating",
                      "—" if pd.isna(row.recent_pos_rate_live) else f"{row.recent_pos_rate_live:.0f}%",
                      delta=None if (pd.isna(row.recent_pos_rate_live) or pd.isna(row.raw_pos_rate))
                      else f"{row.recent_pos_rate_live - row.raw_pos_rate:+.0f}% vs overall")
            m3.metric("Reviews Analyzed", f"{int(row.n_reviews_total):,}")
            if (not pd.isna(row.recent_pos_rate_live)
                    and row.score_live - row.recent_pos_rate_live > 15):
                st.warning("Recent 90-day sentiment is running more than 15 points below the "
                           "overall score — momentum may be cooling.")

        # ---- How this game fits your time (real data: median hours + refund risk) ----
        try:
            extra = q(f"""SELECT refund_zone_pct, pos_median_hours
                          FROM {T('game_scores')} WHERE appid = @a""", a=appid)
        except Exception:
            extra = pd.DataFrame()

        if not extra.empty and not pd.isna(extra.iloc[0].pos_median_hours):
            med_h = float(extra.iloc[0].pos_median_hours)
            weeks = med_h / max(my_hours, 1)
            rz = extra.iloc[0].refund_zone_pct
            st.subheader("How This Game Fits Your Time")
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"**Time-to-Joy Horizon**\n\n"
                        f"Happy players hit their stride around **{med_h:.0f} hours** — "
                        f"about **{weeks:.1f} weeks** at your **{my_hours}h/week** budget.")
            k2.markdown(f"**Your Session Rhythm**\n\n"
                        f"Profile: **{my_rhythm}**, seeking **{my_goal}**. "
                        f"Weigh the pacing on the left against this.")
            k3.markdown("**2-Hour Refund Window**\n\n"
                        + (f"**{rz:.0f}%** of dissatisfied players bail inside Steam's refund "
                           f"window — early pacing decides whether it sticks."
                           if not pd.isna(rz) else "Refund-window data unavailable for this game."))

        # ---- Player Friction Radar: real per-game signal (replaces static filler) ----
        st.subheader("Player Friction Radar")
        if all(v is None for v in radar.values()):
            st.caption("Not enough indexed reviews to profile player friction for this game.")
        else:
            ranked = sorted(
                [(d, v) for d, v in radar.items() if v is not None],
                key=lambda kv: {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}[kv[1]["level"]])
            chips = "".join(risk_chip(d, v["level"], d in my_dims) for d, v in ranked)
            st.markdown(f'<div style="margin-bottom:8px;">{chips}</div>', unsafe_allow_html=True)
            if my_dims:
                st.caption("★ marks the dealbreakers from your profile.")

            col_love, col_quit = st.columns(2)
            praise = vsearch(appid, "amazing experience worth it best game highly recommend",
                             k=6, polarity=True)
            with col_love:
                st.markdown("**What fans praise**")
                if praise.empty:
                    st.caption("No positive review evidence indexed.")
                else:
                    for t in praise["text"].head(3):
                        st.markdown(f"> {' '.join(str(t).split())[:180]}")
            with col_quit:
                st.markdown("**What critics hit hardest**")
                top_neg = [v for _, v in ranked if v["level"] != "Low Risk"][:1]
                neg_texts = top_neg[0]["hits"]["text"].head(3).tolist() if top_neg else []
                if not neg_texts:
                    st.caption("No significant friction found in indexed reviews.")
                else:
                    for t in neg_texts:
                        st.markdown(f"> {' '.join(str(t).split())[:180]}")

        # ---------------- AI Life-Context Persona Analysis ----
        st.write("")
        if st.button("Generate AI Life-Context Fit Analysis", key=f"court_{appid}",
                     type="primary", use_container_width=True):
            pros = vsearch(appid, "amazing experience totally worth it best game recommended", 10, polarity=True)
            cons = vsearch(appid, "broken disappointed waste of money refund problems", 10, polarity=False)
            if pros.empty and cons.empty:
                st.info("No indexed review evidence found for this game.")
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
                        contents=(f"Analyze player reviews for '{row.game}' specifically considering a user with profile:\n"
                                  f"- Life Rhythm: {my_rhythm}\n"
                                  f"- Emotional Goal: {my_goal}\n"
                                  f"- Hardware: {my_device}\n"
                                  f"Evidence:\n"
                                  f"POSITIVE REVIEWS:\n{ev_p}\n"
                                  f"NEGATIVE REVIEWS:\n{ev_c}\n"
                                  "Write in simple markdown:\n"
                                  "#### Reasons This Fits Your Life (Pros)\n(3 bullet points)\n"
                                  "#### Potential Friction & Life Conflicts (Cons)\n(3 bullet points)\n"
                                  "#### Personal Life Verdict\n(2 decisive sentences on whether this game respects the user's time and life rhythm).")).text
                    st.markdown(verdictmd)
                    st.caption("AI evaluates how this game interacts with your specific time budget and emotional goal.")
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
    zmin = c1.slider("Alert Sensitivity Level", 1.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Minimum Daily Reviews", 1, 200, 30, 1)
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
              GROUP BY appid
            ),
            causes_agg AS (
              SELECT appid,
                     ANY_VALUE(summary) AS summary,
                     ANY_VALUE(terms) AS terms
              FROM {T('alert_causes')}
              WHERE summary IS NOT NULL OR terms IS NOT NULL
              GROUP BY appid
            )
            SELECT ep.game,
                   COALESCE(
                     c.summary,
                     IF(c.terms IS NOT NULL AND c.terms != '', CONCAT('Top complaint terms: ', c.terms), NULL),
                     CONCAT('Elevated negative review surge (', ep.peak_neg_pct, '% neg vs ', ep.baseline_neg_pct, '% baseline)')
                   ) AS why_bombed_ai,
                   ep.first_day, ep.latest_day,
                   ep.alert_days, ep.peak_daily_reviews, ep.peak_neg_pct,
                   ep.baseline_neg_pct, ep.peak_z, c.terms AS top_terms, ep.appid
            FROM ep LEFT JOIN causes_agg c ON c.appid = ep.appid
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
                   ROUND(MAX(LEAST(z, 99.9)),1)    AS peak_z,
                   CONCAT('Elevated negative review surge (', ROUND(MAX(neg_rate)*100,1), '% neg)') AS why_bombed_ai
            FROM {T('alerts')}
            WHERE z >= @z AND n >= @minn
            GROUP BY appid
            ORDER BY latest_day DESC, peak_daily_reviews DESC
            LIMIT 500""", z=float(zmin), minn=int(minn))
    st.caption(f"Found {len(alerts)} review bombing events. Distinguishes crowd noise from actual game quality drops.")
    st.dataframe(alerts, use_container_width=True, height=480,
                 column_config={"why_bombed_ai": st.column_config.TextColumn(
                     "Why Players Are Upset (AI Summary)", width="large")})

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
4. {T('v_daily_all')} — PREFERRED for any date logic: one row per game per day,
   2023 snapshot MERGED with the nightly Steam sync. appid INT64, day DATE,
   n INT64, pos INT64, w_sum FLOAT64, wv_sum FLOAT64.
5. {T('v_scores_live')} — evergreen scores anchored to CURRENT_DATE():
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
    ask_mode = st.radio("Mode", ["Explore Data Insights", "Ask About Reviews"],
                        horizontal=True, label_visibility="collapsed")
    if ask_mode == "Ask About Reviews":
        names_rag = q(f"""SELECT appid, game FROM {T('game_scores')}
                          WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 300""")
        rag_pick = st.selectbox("Game", (names_rag["game"] + "  (#" +
                                         names_rag["appid"].astype(str) + ")").tolist(),
                                index=None, placeholder="Pick an indexed game (top 300)")
        rag_q = st.text_input("Your question about this game",
                              placeholder="e.g., I have 30 mins a night. Will this feel like a second job?")
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
                    with st.expander("Evidence Reviews"):
                        st.text(numbered)
                    st.caption("AI searches real player reviews and answers using direct player feedback.")
                except Exception as e:
                    st.error(f"Gemini unavailable: {e}")
        elif rag_pick and rag_q:
            st.warning("Ask-Gemini session limit (5) reached — refresh for a new session.")
    if ask_mode == "Explore Data Insights":
        st.caption(f"Ask questions in plain English — Gemini ({GEMINI_MODEL}) writes BigQuery queries to answer:")
        examples = [
            "Top 10 games by recommendation score with at least 100k reviews",
            "Which 5 games had the most review-bombing days, and when was the latest?",
            "Show games with recent rating higher than overall average",
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
                with st.expander("Generated SQL Query", expanded=False):
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
                        st.caption(f"{len(out)} rows · query generated by {GEMINI_MODEL}.")

# ---------------------------------------------------------------- Player Composition
with tab_comp:
    st.subheader("Player Ownership & Review Quality")
    st.caption("Breakdown of player acquisition channels across 32,793 games — Direct Purchase %, Free / Gift Keys %, and Early Access Reviews %.")
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
                "purchase_pct": st.column_config.NumberColumn("Direct Purchase %", format="%.1f%%"),
                "free_pct": st.column_config.NumberColumn("Free / Gift Keys %", format="%.1f%%"),
                "ea_pct": st.column_config.NumberColumn("Early Access %", format="%.1f%%"),
                "n_reviews": st.column_config.NumberColumn("Total Reviews", format="%d"),
            }
        )
    except Exception as e:
        st.info(f"Player composition data unavailable ({e}).")

st.divider()
st.caption("Data: 114M Steam Review Dataset + Live Steam Web API Sync | Powered by GCP BigQuery, Cloud Run & Gemini AI")
