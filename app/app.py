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
from contextlib import contextmanager
from datetime import datetime, timezone

import altair as alt          # bundled with Streamlit — no extra dependency
import pandas as pd
import plotly.graph_objects as go
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
    /* Slightly smaller than default so 4-up KPI rows fit values like a full date
       without ellipsizing. */
    font-size: 1.75rem;
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
/* Nested (sub-)tabs read as secondary to the main nav: smaller, tighter, quieter. */
[data-testid="stTabs"] [data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 13px;
    padding: 6px 15px;
    margin-right: 4px;
    background: rgba(15, 23, 42, 0.28);
}
[data-testid="stTabs"] [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.13);
    box-shadow: none;
}

/* Sub-tab panels: keep hidden ones in normal flow, just collapsed and invisible.
   Streamlit hides panels with display:none, so a chart inside mounts measuring 0px
   wide; revealing the tab then makes Vega re-fit 0 -> full width, which is the
   visible "collapse then expand". In flow at height 0 the width is already correct,
   so showing the panel changes nothing to re-measure. Scoped to a tab-panel nested
   inside another (i.e. sub-tabs) so the heavy main tabs still skip layout entirely. */
[data-baseweb="tab-panel"] [data-baseweb="tab-panel"][hidden] {
    display: block !important;
    height: 0 !important;
    /* padding survives height:0 and would leave dead space under the sub-tabs */
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
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

/* Primary actions carry the accent and real weight. The prefix matches both
   stBaseButton-primary and stBaseButton-primaryFormSubmit, so the "Ask" submit —
   previously a tiny grey chip next to a full-width input — reads as the main action. */
button[data-testid^="stBaseButton-primary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
    border: none !important;
    color: #06131f !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 11px 34px !important;
    min-width: 148px;
    border-radius: 9px;
    box-shadow: 0 2px 12px rgba(56, 189, 248, 0.22);
}
button[data-testid^="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #7dd3fc 0%, #38bdf8 100%) !important;
    box-shadow: 0 5px 20px rgba(56, 189, 248, 0.4);
    color: #06131f !important;
    transform: translateY(-1px);
}
/* Let the submit sit tight under its input instead of drifting down the form. */
[data-testid="stFormSubmitButton"] { margin-top: 2px; }

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* ---- Panels -------------------------------------------------------------
   In Streamlit 1.58 a bordered container IS the stVerticalBlock (there is no
   separate border wrapper), and stVerticalBlock is also used for every other
   vertical block. panel() drops a hidden marker inside and the child-combinator
   chain below matches ONLY that block — a bare :has(.panel-mark) would also
   match every ancestor block containing the panel. Verified in-browser. */
[data-testid="stVerticalBlock"]:has(
    > [data-testid="stElementContainer"] > [data-testid="stMarkdown"] .panel-mark) {
    background: rgba(20, 29, 46, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 18px 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 18px rgba(0, 0, 0, 0.18);
}
[data-testid="stElementContainer"]:has(.panel-mark),
[data-testid="stMarkdown"]:has(.panel-mark) { display: none !important; }

/* ---- Personal-fit adjustment chips -------------------------------------- */
.fit-factors { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; }
.fit-factor {
    font-size: 0.75rem; padding: 4px 10px; border-radius: 7px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.07);
    color: #94a3b8; line-height: 1.35;
}
.fit-factor b { font-weight: 700; }

/* ---- Section headers ---------------------------------------------------- */
.sec { margin: 6px 0 14px; }
.sec-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 5px;
}
.sec-title {
    font-size: 1.28rem; font-weight: 700; color: #f1f5f9;
    letter-spacing: -0.2px; line-height: 1.25;
}
.sec-sub { font-size: 0.87rem; color: #94a3b8; margin-top: 4px; line-height: 1.5; }

/* ---- Review quotes: default blockquote is a thin grey rule, off-language -- */
[data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid rgba(56, 189, 248, 0.45);
    background: rgba(255, 255, 255, 0.03);
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin: 9px 0;
    color: #cbd5e1;
    font-size: 0.89rem;
    line-height: 1.55;
}

/* ---- Hero band ---------------------------------------------------------- */
.hero {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 24px; flex-wrap: wrap;
    padding-bottom: 16px; margin-bottom: 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
/* Cap the text column so the stats sit on the right instead of wrapping under it. */
.hero > div:first-child { flex: 1 1 420px; max-width: 660px; }
.hero-stats { display: flex; gap: 26px; padding-bottom: 2px; }

/* The cover art's fullscreen button floats outside the image once the panel adds
   padding, and it buys nothing on a header image. Charts/tables keep theirs. */
[data-testid="stElementContainer"]:has([data-testid="stImage"]) [data-testid="stElementToolbar"],
[data-testid="stElementContainer"]:has([data-testid="stImage"]) [data-testid="stElementToolbarButton"] {
    display: none !important;
}
.hero-stat b {
    display: block; font-size: 1.18rem; font-weight: 800; color: #e2e8f0;
    line-height: 1.15; letter-spacing: -0.3px;
}
.hero-stat span {
    font-size: 0.7rem; color: #7c8ba1; text-transform: uppercase;
    letter-spacing: 0.09em; font-weight: 600;
}
/* System status reads blue; green/amber/red stay reserved for risk levels. */
.sync-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(56, 189, 248, 0.09);
    border: 1px solid rgba(56, 189, 248, 0.22);
    padding: 5px 13px; border-radius: 20px; margin-bottom: 4px;
}
.sync-dot { width: 6px; height: 6px; border-radius: 50%; background: #38bdf8; }
.sync-text { font-size: 12px; color: #94a3b8; }

/* ---- Sidebar grouping --------------------------------------------------- */
.side-group {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #64748b;
    margin: 16px 0 2px; padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


@st.cache_resource
def _genai_client():
    """Long-lived Gemini client, defined up here because the tab bodies below run at
    import time and call it. Must be cached rather than built per call: a throwaway
    `_genai_client().models.generate_content(...)` leaves the Client unreferenced, it is
    finalized mid-request, and the call dies with 'the client has been closed'."""
    key = os.environ.get("GEMINI_API_KEY")
    from google import genai as _gm
    return (_gm.Client(api_key=key) if key else
            _gm.Client(vertexai=True, project=PROJECT,
                       location=os.environ.get("VERTEX_LOCATION", "global")))


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


@contextmanager
def panel():
    """A bordered surface. The hidden marker lets CSS style only our panels via :has(),
    rather than every vertical block Streamlit wraps."""
    box = st.container(border=True)
    with box:
        st.markdown('<span class="panel-mark"></span>', unsafe_allow_html=True)
        yield box


def section(title: str, eyebrow: str | None = None, sub: str | None = None):
    """Consistent section header — replaces st.subheader, which renders every section
    at identical weight and leaves the page with no scannable hierarchy."""
    html = '<div class="sec">'
    if eyebrow:
        html += f'<div class="sec-eyebrow">{eyebrow}</div>'
    html += f'<div class="sec-title">{title}</div>'
    if sub:
        html += f'<div class="sec-sub">{sub}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


FIT_DEALBREAKER_CAP = 35


def personal_fit(score, radar, med_hours, refund_pct, rhythm, goal, device, hours, dims):
    """Score how well a game fits THIS player, not how good the game is.

    The old fit_verdict() just thresholded score_live — a general purchase-confidence
    number — while the badge claimed "high alignment between your life rhythm and this
    game". It could call a game an EXCELLENT MATCH for a 30-minute-session player while
    the AI analysis of the same reviews concluded the opposite.

    Here the game's own score is the starting point and the profile applies adjustments,
    each computed from real data. The weights are transparent judgement calls rather than
    fitted parameters, which is exactly why every one is returned and shown to the user.
    Returns (fit, title, description, colour, factors).
    """
    factors = []
    fit = float(score)

    # How long before the game pays off, against the weekly budget the user set.
    if med_hours is not None and not pd.isna(med_hours) and hours > 0:
        weeks = float(med_hours) / hours
        if weeks > 20:
            d = -25
        elif weeks > 10:
            d = -14
        elif weeks > 4:
            d = -5
        else:
            d = 4
        factors.append(("Time to value", d,
                        f"~{weeks:.0f} weeks at {hours}h/week"))

    # Bailing early hurts short-session players most.
    if refund_pct is not None and not pd.isna(refund_pct):
        short = rhythm.startswith("Busy")
        d = (-12 if short else -6) if refund_pct >= 25 else \
            (-6 if short else -3) if refund_pct >= 15 else 0
        if d:
            factors.append(("Early drop-off", d,
                            f"{refund_pct:.0f}% quit inside the refund window"))

    # Hardware: use measured complaint prevalence on the matching dimension.
    hw_dim = {"Low-end PC": "Low-End PC Performance",
              "Steam Deck": "Steam Deck & Controller"}.get(device)
    if hw_dim and hw_dim in radar:
        lvl = radar[hw_dim]["level"]
        d = {"High Risk": -22, "Moderate Risk": -9}.get(lvl, 3)
        factors.append((device, d, f"{lvl.replace(' Risk', '').lower()} complaints on this hardware"))

    # Goal — only mapped where the review data actually supports it.
    if goal.startswith("Decompress") and "Repetitive Grind" in radar:
        lvl = radar["Repetitive Grind"]["level"]
        d = {"High Risk": -14, "Moderate Risk": -6}.get(lvl, 0)
        if d:
            factors.append(("Low-stress goal", d, "players call it grindy"))

    fit = max(0.0, min(100.0, fit + sum(d for _, d, _ in factors)))

    hard = [d for d in dims if radar.get(d) and radar[d]["level"].startswith("High")]
    if hard:
        fit = min(fit, FIT_DEALBREAKER_CAP)
        factors.append(("Dealbreaker", None, "capped by " + ", ".join(hard)))
        return (fit, "POOR MATCH",
                f"Collides with your dealbreakers: {', '.join(hard)}", "#f87171", factors)

    if fit >= 70:
        return (fit, "EXCELLENT MATCH",
                "Suits your time budget, hardware and what you want out of it.",
                "#4ade80", factors)
    if fit >= 40:
        return (fit, "MODERATE MATCH",
                "Workable, but expect friction against your profile.", "#facc15", factors)
    return (fit, "LOW MATCH",
            "Poorly aligned with how much you play and what you want from it.",
            "#f87171", factors)


def fit_factors_html(factors: list) -> str:
    """Show the adjustments behind the fit number so it can be audited, not just trusted."""
    if not factors:
        return ""
    out = []
    for label, d, why in factors:
        if d is None:
            col, val = "#f87171", "cap"
        elif d > 0:
            col, val = "#4ade80", f"+{d}"
        elif d <= -10:
            col, val = "#f87171", str(d)
        else:
            col, val = "#facc15", str(d)
        out.append(f'<span class="fit-factor">{label} '
                   f'<b style="color:{col}">{val}</b> · {why}</span>')
    return f'<div class="fit-factors">{"".join(out)}</div>'


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


RADAR_DIMS = {
    "Monetization & MTX": "microtransactions pay to win overpriced cash grab battle pass",
    "Low-End PC Performance": "fps drops stuttering lag poor optimization low end pc",
    "Bugs & Stability": "bugs crashes broken glitches corrupted save unplayable",
    "Server & Disconnects": "servers down disconnect lag matchmaking dead online",
    "Short Content": "too short lacking content finished in a few hours",
    "Repetitive Grind": "grindy repetitive boring farming time gated chores",
    "Steam Deck & Controller": "steam deck controller support broken keyboard only",
}

# Canonical polarity probes — shared by the radar snippets and the AI analysis so both
# hit the same embedding + query cache instead of paying for near-identical variants.
PRAISE_Q = "amazing experience totally worth it best game highly recommend"
CRITIC_Q = "broken disappointed waste of money refund problems"


def clean_text(text: str) -> str:
    """Strip Kaomoji, ASCII art, repetitive symbols, and garbled characters from review text."""
    if not text:
        return ""
    t = str(text)
    # Remove BBCode tags like [h1], [/h1], [b], [/b], [i], [/i]
    t = re.sub(r'\[/?[a-zA-Z0-9]+\]', '', t)
    # Strip common Kaomoji / ASCII art characters & symbols: e.g. (╯°□°)╯, ಠ_ಠ, (◕‿◕), •, ﹏, 乛, ฅ, ¯
    t = re.sub(r'[°□•﹏乛ฅ¯ノ凸ಠ‿◕≡⊙∀Δω▼▲★☆♪♫♥♠♣♦►◄═║╚╝╗╔╩╦╠═╬\(\)\\\/\^]', ' ', t)
    # Remove repeated non-alphanumeric noise symbols
    t = re.sub(r'[^\w\s,\.\?!\'"\-–—:;]', ' ', t)
    # Normalize multiple whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def snippet(text: str, limit: int = 180) -> str:
    """Clean review text and collapse whitespace for display."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


# Chart tokens. Each chart carries a single series, so identity never rides on colour
# alone and no legend is needed — the title names the series. Text keeps ink tokens.
CHART_ACCENT = "#38bdf8"   # primary signal (sentiment)
CHART_MUTED = "#64748b"    # supporting context (volume)
CHART_INK = "#94a3b8"      # axis/label ink, matches the app's muted text


def chart_theme(chart):
    """Recessive axes and grid over the app's dark surface (dark mode is chosen here,
    not an automatic flip of a light theme)."""
    return (chart
            .properties(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=CHART_INK, titleColor=CHART_INK,
                            labelFontSize=11, titleFontSize=11,
                            gridColor="rgba(255,255,255,0.06)",
                            domainColor="rgba(255,255,255,0.15)",
                            tickColor="rgba(255,255,255,0.15)"))


_DIM_SHORT = {
    "Monetization & MTX": "Monetisation", "Low-End PC Performance": "Low-end PC",
    "Bugs & Stability": "Bugs", "Server & Disconnects": "Servers",
    "Short Content": "Short content", "Repetitive Grind": "Grind",
    "Steam Deck & Controller": "Steam Deck",
}


def radar_figure(radar: dict, my_dims: list):
    """Polar view of complaint prevalence. Dotted rings mark the Moderate/High thresholds,
    so the shape is read against a scale instead of by area alone."""
    dims = [d for d in RADAR_DIMS if d in radar]
    if not dims:
        return None
    vals = [radar[d]["share"] for d in dims]
    theta = [_DIM_SHORT.get(d, d) + (" ★" if d in my_dims else "") for d in dims]
    top = max(max(vals) * 1.3, HIGH_SHARE * 100 * 1.6)

    fig = go.Figure()
    for lvl, col in ((MODERATE_SHARE * 100, "#facc15"), (HIGH_SHARE * 100, "#f87171")):
        fig.add_trace(go.Scatterpolar(
            r=[lvl] * (len(dims) + 1), theta=theta + theta[:1], mode="lines",
            line=dict(color=col, width=1, dash="dot"), hoverinfo="skip"))
    ring_colors = [_RISK_COLOR[radar[d]["level"]] for d in dims]
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1], theta=theta + theta[:1], mode="lines+markers", fill="toself",
        fillcolor="rgba(56,189,248,0.16)", line=dict(color=CHART_ACCENT, width=2),
        marker=dict(size=9, color=ring_colors + ring_colors[:1]),
        hovertemplate="%{theta}<br>%{r:.1f}% of reviews<extra></extra>"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            # nticks keeps the radial labels sparse; the default ladder of ~9 values
            # stacks diagonally across the middle and collides with the polygon.
            radialaxis=dict(range=[0, top], ticksuffix="%", showline=False, nticks=4,
                            angle=90, tickangle=0,
                            gridcolor="rgba(255,255,255,0.09)",
                            tickfont=dict(size=9, color="#64748b")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.09)",
                             tickfont=dict(size=11, color="#cbd5e1"))),
        showlegend=False, height=340, margin=dict(l=70, r=70, t=26, b=26),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
    return fig


def score_gauge(score: float, color: str):
    """Bands mirror the documented verdict thresholds (>=70 strong, >=40 moderate)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(score),
        number=dict(valueformat=".0f", font=dict(size=34, color="#f8fafc")),
        gauge=dict(
            # Tick labels clip against the arc ends; the bands plus the number carry it.
            axis=dict(range=[0, 100], tickwidth=0, showticklabels=False),
            bar=dict(color=color, thickness=0.3), bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[dict(range=[0, 40], color="rgba(248,113,113,0.10)"),
                   dict(range=[40, 70], color="rgba(250,204,21,0.10)"),
                   dict(range=[70, 100], color="rgba(74,222,128,0.10)")])))
    fig.update_layout(height=165, margin=dict(l=12, r=12, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
    return fig
# Prevalence thresholds — calibrated against high- vs low-friction reference games.
STRONG_DIST = 0.30      # cosine distance below which a review is genuinely on-topic
HIGH_SHARE = 0.03       # >=3% of a game's reviews on-topic-negative -> High Risk
MODERATE_SHARE = 0.01   # >=1% -> Moderate Risk


def radar_level(strong: int, total_reviews: int) -> str:
    """Grade a friction dimension by PREVALENCE rather than a raw match count.

    `strong` is the number of the game's negative reviews that are genuinely on-topic
    (cosine distance < STRONG_DIST), normalized by the game's total review sample. A
    loose distance threshold saturates — almost every game hits it for every topic —
    whereas prevalence separates a game's real problems from generic noise.
    """
    if strong <= 0 or total_reviews <= 0:
        return "Low Risk"
    share = strong / total_reviews
    if share >= HIGH_SHARE:
        return "High Risk"
    if share >= MODERATE_SHARE:
        return "Moderate Risk"
    return "Low Risk"


_PRAISE_KEY = "__praise__"


@st.cache_data(ttl=600, show_spinner=False)
def friction_radar(appid: int) -> tuple[dict, list]:
    """Score every friction dimension AND fetch praise quotes in ONE BigQuery round trip.

    Cross-joins the game's reviews against all query vectors at once, giving an exact
    COUNTIF over the *full* negative set (no top-k truncation) plus the three closest
    snippets per theme. Each probe carries the polarity it wants, so the positive praise
    lookup rides along instead of costing a second query.
    Returns ({dim: {...}}, praise_texts); ({}, []) when nothing is indexed.
    """
    probes = [(d, text, False) for d, text in RADAR_DIMS.items()]
    probes.append((_PRAISE_KEY, PRAISE_Q, True))
    struct_params = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("dim", "STRING", d),
            bigquery.ArrayQueryParameter("qv", "FLOAT64", embed_query(text)),
            bigquery.ScalarQueryParameter("want_pos", "BOOL", want_pos),
        )
        for d, text, want_pos in probes
    ]
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("dims", "STRUCT", struct_params),
                          bigquery.ScalarQueryParameter("a", "INT64", int(appid))],
        maximum_bytes_billed=3 * 1024 ** 3)
    sql = f"""
        WITH v AS (
          SELECT text, embedding, voted_up FROM {T('review_vectors')} WHERE appid = @a
        ),
        scored AS (
          SELECT q.dim AS dim, v.text AS text,
                 ML.DISTANCE(v.embedding, q.qv, 'COSINE') AS dist
          FROM v CROSS JOIN UNNEST(@dims) AS q
          WHERE v.voted_up = q.want_pos
        )
        SELECT dim,
               COUNTIF(dist < {STRONG_DIST}) AS strong,
               ARRAY_AGG(text ORDER BY dist LIMIT 3) AS texts,
               (SELECT COUNT(*) FROM {T('review_vectors')} WHERE appid = @a) AS total_vec
        FROM scored GROUP BY dim"""
    try:
        df = _client().query(sql, job_config=cfg).to_dataframe()
    except Exception:
        return {}, []
    if df.empty:
        return {}, []
    total = int(df.iloc[0].total_vec)
    rows = {r.dim: {"level": radar_level(int(r.strong), total),
                    "n": int(r.strong),
                    "share": 100.0 * int(r.strong) / total if total else 0.0,
                    "texts": [str(t) for t in r.texts]}
            for r in df.itertuples()}
    praise = rows.pop(_PRAISE_KEY, {}).get("texts", [])
    return rows, praise


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


def hero():
    """Header band: identity on the left, live scale on the right — the stats fill what
    used to be dead space and give the page a top edge to hang the hierarchy from."""
    games = reviews = None
    status = "Snapshot loaded · nightly Steam sync running"
    try:
        f = q(f"SELECT last_sync, tracked_games, delta_reviews FROM {T('v_freshness')}").iloc[0]
        if pd.isna(f.last_sync):
            raise ValueError("no sync yet")
        hrs = max(0, int((pd.Timestamp.now(tz="UTC")
                          - pd.Timestamp(f.last_sync)).total_seconds() // 3600))
        games, reviews = int(f.tracked_games), int(f.delta_reviews)
        status = f"Live sync · {hrs}h ago"
    except Exception:
        pass

    stats = '<div class="hero-stat"><b>114M</b><span>reviews analysed</span></div>'
    if games:
        stats += f'<div class="hero-stat"><b>{games:,}</b><span>games synced</span></div>'
    if reviews:
        stats += f'<div class="hero-stat"><b>+{reviews:,}</b><span>new reviews</span></div>'

    st.markdown(f"""
    <div class="hero">
      <div>
        <div class="sync-pill"><span class="sync-dot"></span>
          <span class="sync-text">{status}</span></div>
        <div class="hero-title">BuyOrWait</div>
        <div class="hero-subtitle">Person-to-Game Resonance Engine — matching your life
          rhythm, time budget and hardware against real Steam player experience</div>
      </div>
      <div class="hero-stats">{stats}</div>
    </div>
    """, unsafe_allow_html=True)


hero()

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

    st.markdown('<div class="side-group">Setup</div>', unsafe_allow_html=True)
    my_device = st.selectbox("Hardware Platform", ["High-end PC", "Low-end PC", "Steam Deck"], index=0)
    my_hours = st.slider("Weekly Gaming Hours Budget", 1, 40, 6)

    st.markdown('<div class="side-group">Dealbreakers</div>', unsafe_allow_html=True)
    my_dims = st.multiselect("Personal Dealbreaker Filters", list(RADAR_DIMS), default=[],
                             label_visibility="collapsed")
    auto_dims = []
    if my_device == "Low-end PC" and "Low-End PC Performance" not in my_dims:
        auto_dims.append("Low-End PC Performance")
    if my_device == "Steam Deck" and "Steam Deck & Controller" not in my_dims:
        auto_dims.append("Steam Deck & Controller")
    my_dims += auto_dims
    if auto_dims:
        st.caption(f"Added from your hardware choice: {', '.join(auto_dims)}.")

tab_buy, tab_alert, tab_ask, tab_comp = st.tabs(
    ["Person-Game Fit", "Review Bombing & Crowd Noise", "Ask Gemini AI", "Player Ownership"])

@st.cache_data(ttl=3600, show_spinner=False)
def featured_games(n: int = 5) -> pd.DataFrame:
    """Popular games that actually have indexed review vectors, so the one-click demos
    land on a fully-populated radar instead of an empty state."""
    try:
        return q(f"""SELECT s.game, s.appid
                     FROM {T('game_scores')} s
                     WHERE s.game IS NOT NULL
                       AND s.appid IN (SELECT DISTINCT appid FROM {T('review_vectors')})
                     ORDER BY s.n_reviews DESC LIMIT {int(n)}""")
    except Exception:
        return pd.DataFrame()


def _pick_game(label: str):
    """on_click callback — runs before the rerun, so writing the selectbox's own state
    key here is safe (assigning it after the widget is created would raise)."""
    st.session_state["game_pick"] = label


@st.cache_data(ttl=600, show_spinner="Gemini is weighing this game against your profile...")
def life_fit_analysis(appid: int, game: str, rhythm: str, goal: str, device: str) -> str:
    """Cached so the verdict survives reruns. It previously ran inline inside
    `if st.button(...)`, which is only true on the click itself — so any later interaction
    erased the analysis, and because the call was uncached, regenerating cost another
    Gemini request."""
    pros = vsearch(int(appid), PRAISE_Q, 10, polarity=True)
    cons = vsearch(int(appid), CRITIC_Q, 10, polarity=False)
    if pros.empty and cons.empty:
        return ""
    ev_p = "\n".join(f"- {t[:200]}" for t in pros["text"].head(8))
    ev_c = "\n".join(f"- {t[:200]}" for t in cons["text"].head(8))
    client = _genai_client()
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(f"Analyze player reviews for '{game}' specifically considering a user "
                  f"with profile:\n- Life Rhythm: {rhythm}\n- Emotional Goal: {goal}\n"
                  f"- Hardware: {device}\n"
                  f"Evidence:\nPOSITIVE REVIEWS:\n{ev_p}\nNEGATIVE REVIEWS:\n{ev_c}\n"
                  "Write in simple markdown:\n"
                  "#### Reasons This Fits Your Life (Pros)\n(3 bullet points)\n"
                  "#### Potential Friction & Life Conflicts (Cons)\n(3 bullet points)\n"
                  "#### Personal Life Verdict\n(2 decisive sentences on whether this game "
                  "respects the user's time and life rhythm).")).text


def render_ai_analysis(appid: int, game: str, rhythm: str, goal: str, device: str):
    section("AI Life-Context Fit", eyebrow="Your profile vs. real reviews",
            sub="Gemini reads this game's reviews through your rhythm, goal and hardware.")
    key = (int(appid), rhythm, goal, device)
    if st.button("Generate analysis", key=f"court_{appid}", type="primary"):
        st.session_state["fit_ai_key"] = key
    if st.session_state.get("fit_ai_key") != key:
        st.caption("Runs on demand — one Gemini call, then kept for the rest of the session.")
        return
    try:
        md = life_fit_analysis(int(appid), game, rhythm, goal, device)
    except Exception as e:
        st.error("Gemini is unavailable right now — everything else on this page still works.")
        with st.expander("Technical details"):
            st.write(str(e))
        return
    if not md:
        st.info("No indexed review evidence found for this game.")
    else:
        st.markdown(md)
        st.caption("Written from this game's actual reviews, weighed against your profile.")


# ---------------------------------------------------------------- Person-Game Fit
@st.fragment
def person_game_fit(my_rhythm, my_goal, my_device, my_hours, my_dims):
    """Rendered as a fragment: changing the game reruns ONLY this block.

    Streamlit re-executes every tab body on each interaction, so picking a game used to
    re-serialise the 10k-row ownership table and the alerts table as well. Parameters are
    named after the sidebar globals so the body reads identically either way.
    """
    names_df = q(f"""SELECT appid, game FROM {T('game_scores')}
                     WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 20000""")
    all_labels = names_df["game"] + "  (#" + names_df["appid"].astype(str) + ")"
    # Handing the widget all 20,000 options costs ~709 KB of JSON per rerun and was the
    # main source of the lag. Filter server-side and send a short list instead.
    term = st.text_input("Find a game", key="game_search",
                         placeholder="Type a game name — e.g. Elden Ring, Hades, Rust")
    pool = all_labels
    if term.strip():
        pool = all_labels[names_df["game"].str.contains(
            re.escape(term.strip()), case=False, na=False)]
    opts = pool.head(50).tolist()
    current = st.session_state.get("game_pick")
    if current and current not in opts:
        # A selected value must stay among the options or Streamlit raises — but append
        # it, never prepend: putting the previous pick above fresh search results makes
        # the top match the game you already had.
        opts = opts + [current]
    labels = all_labels           # featured-button validity is checked against the full set
    pick = st.selectbox("Select a game to test your personal fit", opts, index=None,
                        key="game_pick", placeholder="Pick one of the matches")
    if term.strip() and len(pool) > 50:
        st.caption(f"Showing the 50 most-reviewed of {len(pool):,} matches — refine the search to narrow it.")

    # Inline live-search fallback — visible immediately (no accordion), hidden once a game is picked.
    if not pick:
        st.markdown("Pick a game and BuyOrWait weighs **your** life rhythm, weekly time budget "
                    "and hardware against 114M real Steam reviews — then surfaces the friction "
                    "points that would actually land on you.")
        feat = featured_games()
        if not feat.empty:
            # Only offer games the selectbox actually contains, or setting its state would raise.
            label_set = set(labels)
            shots = [(r.game, f"{r.game}  (#{r.appid})", r.appid) for r in feat.itertuples()]
            shots = [s for s in shots if s[1] in label_set]
            if shots:
                st.caption("Fully analysed examples — one click:")
                for col, (name, label, aid) in zip(st.columns(len(shots)), shots):
                    # Truncate so a long title can't wrap and break the row's alignment.
                    col.button(name if len(name) <= 20 else name[:19] + "…",
                               key=f"feat_{aid}", use_container_width=True,
                               help=name, on_click=_pick_game, args=(label,))
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
                section("Live Verification", eyebrow="Straight from Steam")
                live_panel(opts[pick_live], None)
                st.caption("This game post-dates the main dataset, so its score is fetched directly from live Steam ratings.")

    hit = pd.DataFrame()
    if pick:
        appid = int(pick.rsplit("#", 1)[1].rstrip(")"))
        hit = q(f"SELECT * FROM {T('v_scores_live')} WHERE appid = @a", a=appid)
        if hit.empty:
            st.warning("No scored data for this game yet — it may be newer than the scored "
                       "set. Clear the selection and use the live Steam search instead.")

    if pick and not hit.empty:
        row = hit.iloc[0]
        _log_usage("search", str(row.game), appid)

        # ---- Friction Radar: all dimensions in ONE round trip.
        #      Red flags are derived from the same pass (no duplicate searches). ----
        with st.spinner("Profiling player sentiment across friction dimensions..."):
            radar, praise_texts = friction_radar(appid)
        # Needed before the card now: the fit score is derived from these.
        try:
            extra = q(f"""SELECT refund_zone_pct, pos_median_hours
                          FROM {T('game_scores')} WHERE appid = @a""", a=appid)
        except Exception:
            extra = pd.DataFrame()
        _med = extra.iloc[0].pos_median_hours if not extra.empty else None
        _rz = extra.iloc[0].refund_zone_pct if not extra.empty else None

        fit_score, fit_title, fit_desc, fit_color, fit_factors = personal_fit(
            row.score_live, radar, _med, _rz,
            my_rhythm, my_goal, my_device, my_hours, my_dims)

        with panel():
            c_img, c_m, c_g = st.columns([1.1, 2.2, 1.1])
            c_img.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg",
                        use_container_width=True)
            with c_m:
                st.markdown(verdict_badge(fit_title, fit_desc, fit_color), unsafe_allow_html=True)
                st.markdown(fit_factors_html(fit_factors), unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                # Game quality and personal fit are separate answers — showing both is
                # what lets "great game, wrong game for you" actually be visible.
                m1.metric("Game Rating", f"{row.score_live:.0f}/100")
                m2.metric("Recent 90-Day",
                          "—" if pd.isna(row.recent_pos_rate_live) else f"{row.recent_pos_rate_live:.0f}%",
                          delta=None if (pd.isna(row.recent_pos_rate_live) or pd.isna(row.raw_pos_rate))
                          else f"{row.recent_pos_rate_live - row.raw_pos_rate:+.0f}% vs overall")
                m3.metric("Reviews", f"{int(row.n_reviews_total):,}")
            with c_g:
                st.plotly_chart(score_gauge(fit_score, fit_color),
                                use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown('<div style="text-align:center;margin-top:-14px;color:#7c8ba1;'
                            'font-size:0.7rem;letter-spacing:0.09em;text-transform:uppercase;'
                            'font-weight:600;">Your personal fit</div>',
                            unsafe_allow_html=True)
            if (not pd.isna(row.recent_pos_rate_live)
                    and row.score_live - row.recent_pos_rate_live > 15):
                st.warning("Recent 90-day sentiment is running more than 15 points below the "
                           "overall score — momentum may be cooling.")

        # The detail below used to be one long vertical stack: you had to scroll past
        # three panels to reach the AI button, and its output then pushed everything
        # further down. Sub-tabs keep the decision card in view and put each block one
        # click away. Sections stay in their original code order — the tab objects are
        # entered as extra context managers, which is also why nothing needed reindenting.
        t_ai, t_radar, t_time, t_live = st.tabs(
            ["AI Fit Analysis", "Friction Radar", "Time & Trend", "Live Check"])

        with t_ai, panel():
            render_ai_analysis(appid, str(row.game), my_rhythm, my_goal, my_device)

        # ---- How this game fits your time (extra was fetched above for the fit score) ----
        if not extra.empty and not pd.isna(extra.iloc[0].pos_median_hours):
            med_h = float(extra.iloc[0].pos_median_hours)
            weeks = med_h / max(my_hours, 1)
            rz = extra.iloc[0].refund_zone_pct
            with t_time, panel():
                section("How This Game Fits Your Time", eyebrow="Time budget")
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
        with t_radar, panel():
            section("Player Friction Radar", eyebrow="What actually goes wrong",
                    sub="Share of this game's reviews that complain about each theme.")
            if not radar:
                st.caption("Not enough indexed reviews to profile player friction for this game.")
            else:
                ranked = sorted(
                    radar.items(),
                    key=lambda kv: {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}[kv[1]["level"]])
                c_plot, c_list = st.columns([1.15, 1])
                with c_plot:
                    fig = radar_figure(radar, my_dims)
                    if fig is not None:
                        # Keep the mode bar (it appears on hover) so the radar can be
                        # zoomed and reset; displayModeBar:False left it with no zoom
                        # affordance at all.
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displaylogo": False,
                                                "displayModeBar": True,
                                                "modeBarButtonsToRemove": [
                                                    "select2d", "lasso2d", "toggleSpikelines"]})
                with c_list:
                    chips = "".join(risk_chip(d, v["level"], d in my_dims) for d, v in ranked)
                    st.markdown(f'<div style="margin:6px 0 4px;">{chips}</div>',
                                unsafe_allow_html=True)
                    st.caption("Dotted rings on the chart are the Moderate (1%) and High (3%) "
                               "thresholds."
                               + (" ★ marks your dealbreakers." if my_dims else ""))

                col_love, col_quit = st.columns(2)
                with col_love:
                    st.markdown("**What fans praise**")
                    if not praise_texts:
                        st.caption("No positive review evidence indexed.")
                    else:
                        for t in praise_texts[:3]:
                            st.markdown(f"> {snippet(t)}")
                with col_quit:
                    st.markdown("**What critics hit hardest**")
                    top_neg = [v for _, v in ranked if v["level"] != "Low Risk"][:1]
                    neg_texts = top_neg[0]["texts"][:3] if top_neg else []
                    if not neg_texts:
                        st.caption("No significant friction found in indexed reviews.")
                    else:
                        for t in neg_texts:
                            st.markdown(f"> {snippet(t)}")

        daily = q(f"""
            SELECT day, n, SAFE_DIVIDE(pos, n) AS pos_rate
            FROM {T('v_daily_all')} WHERE appid = @a ORDER BY day""", a=appid)
        if not daily.empty:
            daily["day"] = pd.to_datetime(daily["day"])
            # Reindex to a continuous daily range. The snapshot ends 2023-10-30 while the
            # nightly sync only covers recent days, so most games have a real multi-year
            # gap. Without this the line interpolates straight across it and implies data
            # we don't have; NaN makes Altair break the line instead. Volume is a true 0.
            daily = daily.set_index("day").sort_index()
            daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
            daily.index.name = "day"
            # A decade of daily points is ~4,700 rows embedded in the Vega spec, twice.
            # Roll long histories up to weeks: same shape, ~7x less to ship and draw.
            if len(daily) > 900:
                daily = daily.resample("W").agg({"n": "sum", "pos_rate": "mean"})
                smoothing = "weekly average"
            else:
                daily["pos_rate"] = daily["pos_rate"].rolling(7, min_periods=1).mean()
                smoothing = "7-day average"
            daily["pos_pct"] = daily["pos_rate"] * 100
            daily["n"] = daily["n"].fillna(0)
            daily = daily.reset_index()

            with t_time, panel():
                section("Sentiment & Volume Over Time", eyebrow="History")
                # Two measures on different scales -> two charts, never a dual y-axis.
                x_enc = alt.X("day:T", axis=alt.Axis(title=None, format="%Y", tickCount=6))
                hover = alt.selection_point(fields=["day"], nearest=True,
                                            on="mouseover", empty=False)
                tips = [alt.Tooltip("day:T", title="Date"),
                        alt.Tooltip("pos_pct:Q", title="Positive rate %", format=".1f"),
                        alt.Tooltip("n:Q", title="Reviews", format=",")]
                # Scroll to zoom, drag to pan. A decade on one axis is unreadable without
                # it. Bound per chart: Vega-Lite scale binding only works on a unit spec,
                # and vconcat (which would share one zoom) does not support autosize fit,
                # which is what keeps these charts responsive.
                zoom_line = alt.selection_interval(bind="scales", encodings=["x"])
                zoom_vol = alt.selection_interval(bind="scales", encodings=["x"])

                base = alt.Chart(daily).encode(x=x_enc)
                line = base.mark_line(color=CHART_ACCENT, strokeWidth=2).encode(
                    y=alt.Y("pos_pct:Q", scale=alt.Scale(zero=False),
                            axis=alt.Axis(title=f"Positive rate (%) · {smoothing}"))
                ).add_params(zoom_line)
                # Invisible until hovered: a crosshair-style readout without ink noise.
                marks = base.mark_point(color=CHART_ACCENT, size=70, filled=True).encode(
                    y=alt.Y("pos_pct:Q"), tooltip=tips,
                    opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
                st.altair_chart(chart_theme(alt.layer(line, marks).properties(height=240)),
                                use_container_width=True, theme=None)

                vol = alt.Chart(daily).mark_bar(color=CHART_MUTED, cornerRadiusEnd=2).encode(
                    x=x_enc, y=alt.Y("n:Q", axis=alt.Axis(
                        title="Reviews per week" if smoothing.startswith("weekly")
                        else "Reviews per day")),
                    tooltip=tips).add_params(zoom_vol)
                st.altair_chart(chart_theme(vol.properties(height=130)),
                                use_container_width=True, theme=None)
                st.caption("Scroll to zoom, drag to pan, double-click to reset · hover a "
                           "chart for the expand button. Reviews up to 2023-10-30 come from "
                           "the snapshot; later days come from the nightly Steam sync.")

        with t_live, panel():
            section("Live Verification", eyebrow="Straight from Steam",
                    sub="Compare this game's mood right now against the snapshot above.")
            # Three blocking HTTP calls (~2s, up to 30s if Steam is slow) used to run on
            # every game switch — the intermittent "stuck switching". Now opt-in.
            if st.button("Check Steam right now", key=f"live_{appid}",
                         use_container_width=True):
                live_panel(appid, None if pd.isna(row.recent_pos_rate_live)
                           else float(row.recent_pos_rate_live))
            else:
                st.caption("Queries the public Steam API on demand — takes a second or two.")


with tab_buy:
    person_game_fit(my_rhythm, my_goal, my_device, my_hours, my_dims)

# ---------------------------------------------------------------- Bombing Alert
with tab_alert:
    c1, c2 = st.columns(2)
    zmin = c1.slider("Alert Sensitivity Level", 1.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Minimum Daily Reviews", 1, 200, 30, 1)
    try:
        # rf-string: the REGEXP_REPLACE pattern below contains \* and \s, which are not
        # valid Python escapes and warn (and will eventually error) in a plain f-string.
        alerts = q(rf"""
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
                   REGEXP_REPLACE(
                     COALESCE(
                       c.summary,
                       IF(c.terms IS NOT NULL AND c.terms != '', CONCAT('Top complaint terms: ', c.terms), NULL),
                       CONCAT('Elevated negative review surge (', ep.peak_neg_pct, '% neg vs ', ep.baseline_neg_pct, '% baseline)')
                     ),
                     r'^(?:\*\*|\*|#)*\s*(?:SUMMARY|Summary|summary)\s*:\s*(?:\*\*|\*)*\s*', ''
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
    if alerts.empty:
        st.info("No review-bombing events match these filters — try lowering the sensitivity.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Games Affected", f"{alerts['appid'].nunique():,}")
        b2.metric("Total Alert Days", f"{int(alerts['alert_days'].sum()):,}")
        b3.metric("Worst Severity (z)", f"{alerts['peak_z'].max():.1f}")
        b4.metric("Most Recent Event", str(alerts['latest_day'].max()))
        st.caption("Sorted by most recent. Severity z compares a day's negative rate against "
                   "that game's own 30-day baseline, so crowd noise is separated from a real "
                   "quality drop.")
        st.dataframe(
            alerts, use_container_width=True, height=460, hide_index=True,
            column_order=("game", "why_bombed_ai", "latest_day", "alert_days",
                          "peak_daily_reviews", "peak_neg_pct", "baseline_neg_pct", "peak_z"),
            column_config={
                "game": st.column_config.TextColumn("Game", width="medium"),
                "why_bombed_ai": st.column_config.TextColumn(
                    "Why Players Are Upset (AI Summary)", width="large"),
                "latest_day": st.column_config.DateColumn("Latest Day"),
                "alert_days": st.column_config.NumberColumn("Alert Days", format="%d"),
                "peak_daily_reviews": st.column_config.NumberColumn("Peak Daily Reviews", format="%d"),
                "peak_neg_pct": st.column_config.ProgressColumn(
                    "Peak Negative", format="%.1f%%", min_value=0, max_value=100),
                "baseline_neg_pct": st.column_config.ProgressColumn(
                    "Baseline Negative", format="%.1f%%", min_value=0, max_value=100),
                "peak_z": st.column_config.NumberColumn("Severity (z)", format="%.1f"),
            })

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
6. {T('game_composition')} — how players acquired each game (~33k games).
   appid INT64, n FLOAT64 (reviews considered), purchase_pct FLOAT64 (0-100, bought
   directly), free_pct FLOAT64 (0-100, free or gift key), ea_pct FLOAT64 (0-100,
   reviewed during Early Access).
   IMPORTANT: this table has NO game name — always JOIN to {T('v_scores_live')}
   or {T('game_scores')} on appid to get the title.

Rules:
- Output exactly ONE BigQuery Standard SQL SELECT (or WITH ... SELECT) statement — no markdown, no comments, no explanation.
- Read-only. Never generate INSERT/UPDATE/DELETE/DDL.
- Match game names case-insensitively: LOWER(game) LIKE '%...%'.
- End with LIMIT 100 unless the question implies a different limit.
- Results are shown straight to Steam players, so alias EVERY output column with a
  descriptive name: AS game_name, AS total_reviews, AS positive_pct — never leave raw
  names like raw_pos_rate or n exposed. Do not select appid unless explicitly asked.
- Aliases MUST be plain BigQuery identifiers: letters, digits and underscores only.
  NEVER put a space, %, /, or any other punctuation in an alias — BigQuery rejects the
  query outright. Use positive_pct, not `Positive %`. The app prettifies headers itself.
- Round percentages and scores to one decimal.
- Data window: the review snapshot ends 2023-10-30 and only ~1,900 games get the nightly
  Steam sync, so date filters near "today" return little for most games. Prefer the
  all-time and 90-day fields on {T('v_scores_live')} over hand-rolled CURRENT_DATE()
  windows unless the user explicitly asks about recent activity.
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
    client = _genai_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=f"{SCHEMA_PROMPT}\nQuestion: {question}\nSQL:")
    sql = resp.text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip().rstrip(";")
    return sql


@st.cache_data(ttl=600, show_spinner="Querying BigQuery...")
def run_sql(sql: str) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=1024 ** 3)
    return _client().query(sql, job_config=cfg).to_dataframe()


@st.cache_data(ttl=600, show_spinner="Gemini is reading player reviews...")
def rag_answer(appid: int, game_label: str, question: str) -> tuple[str, str]:
    """Answer from real reviews. Cached, so a rerun triggered by ANY widget on ANY tab
    replays the stored answer instead of re-billing a Gemini call."""
    ev = vsearch(int(appid), question, k=12)
    if ev.empty:
        return ("", "")
    numbered = "\n".join(
        f"[{i + 1}] ({'Positive' if v else 'Negative'}, {h:.0f}h played) {clean_text(t)[:250]}"
        for i, (t, v, h) in enumerate(zip(ev["text"], ev["voted_up"], ev["playtime_h"]))
        if clean_text(t)
    )
    client = _genai_client()
    ans = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(f"Answer the question using ONLY these player reviews of '{game_label}'. "
                  f"Cite like [3]. If evidence is mixed, say so. Max 120 words.\n"
                  f"REVIEWS:\n{numbered}\nQUESTION: {question}\nANSWER:")).text
    return (ans, numbered)


GEMINI_BUDGET = 5


def _budget_left() -> int:
    return GEMINI_BUDGET - len(st.session_state.get("asked", []))


def _register(question: str) -> bool:
    """Charge the session budget once per *distinct question*. Re-asking something is
    free because the answer is served from cache — the old code charged once per script
    rerun, so unrelated clicks (even on other tabs) silently burned the quota."""
    asked = st.session_state.setdefault("asked", [])
    if question in asked:
        return True
    if len(asked) >= GEMINI_BUDGET:
        return False
    asked.append(question)
    return True


def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case -> Title Case headers. BigQuery aliases must be plain identifiers, so
    the model returns positive_pct and we turn it into 'Positive %' here for display."""
    def nice(col) -> str:
        s = re.sub(r"_(pct|percent)$", "_%", str(col))
        parts = [p for p in s.split("_") if p]
        return " ".join(p if p == "%" else p.capitalize() for p in parts) or str(col)
    return df.rename(columns={c: nice(c) for c in df.columns})


def auto_chart(df: pd.DataFrame):
    """Bar chart when the answer is obviously chartable — one label column, at least one
    number, few enough rows to read. Returns None when a table says it better."""
    if df.empty or len(df) > 25:
        return None
    nums = df.select_dtypes("number").columns.tolist()
    labels = [c for c in df.columns if c not in nums]
    if len(labels) != 1 or not nums:
        return None
    label, value = labels[0], nums[0]
    # Explicit field= form: aliased headers contain spaces, which break shorthand parsing.
    return chart_theme(
        alt.Chart(df).mark_bar(color=CHART_ACCENT, cornerRadiusEnd=3).encode(
            x=alt.X(field=value, type="quantitative", axis=alt.Axis(title=value)),
            y=alt.Y(field=label, type="nominal", axis=alt.Axis(title=None),
                    sort=alt.EncodingSortField(field=value, op="max", order="descending")),
            tooltip=[alt.Tooltip(field=c,
                                 type="quantitative" if c in nums else "nominal")
                     for c in df.columns],
        ).properties(height=min(30 * len(df) + 40, 520)))


def _use_example(ex: str):
    st.session_state["nl_q_input"] = ex
    st.session_state["nl_pending"] = ex


with tab_ask:
    ask_mode = st.radio("Mode", ["Explore Data Insights", "Ask About Reviews"],
                        horizontal=True, label_visibility="collapsed")
    if ask_mode == "Ask About Reviews":
        names_rag = q(f"""SELECT appid, game FROM {T('game_scores')}
                          WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 300""")
        rag_pick = st.selectbox("Game", (names_rag["game"] + "  (#" +
                                         names_rag["appid"].astype(str) + ")").tolist(),
                                index=None, placeholder="Pick a game (top 300 by reviews)")
        # A form means the question only fires on submit — not on every script rerun.
        with st.form("rag_form"):
            rag_q = st.text_input("Your question about this game",
                                  placeholder="e.g., I have 30 mins a night. Will this feel like a second job?")
            rag_go = st.form_submit_button("Ask", type="primary")
        if rag_go:
            if not rag_pick:
                st.info("Pick a game first, then ask.")
            elif not rag_q.strip():
                st.info("Type a question about this game, then press Ask.")
            elif _register(f"rag::{rag_pick}::{rag_q.strip()}"):
                st.session_state["rag_active"] = (rag_pick, rag_q.strip())
            else:
                st.warning(f"You've used all {GEMINI_BUDGET} questions this session — "
                           "refresh the page to start over.")
        rag_active = st.session_state.get("rag_active")
        if rag_active:
            g_label, g_question = rag_active
            try:
                ans, numbered = rag_answer(
                    int(g_label.rsplit("#", 1)[1].rstrip(")")), g_label, g_question)
            except Exception as e:
                st.error("Gemini is unavailable right now — the snapshot data above is unaffected.")
                with st.expander("Technical details"):
                    st.write(str(e))
            else:
                if not ans:
                    st.info("We don't have indexed reviews for this game yet.")
                else:
                    st.markdown(f"**You asked:** {g_question}")
                    st.markdown(ans)
                    with st.expander("The player reviews behind this answer"):
                        st.text(numbered)
        st.caption(f"{_budget_left()} of {GEMINI_BUDGET} questions left this session.")

    if ask_mode == "Explore Data Insights":
        st.caption("Ask anything about the review data in plain English — no SQL, no filters.")
        examples = [
            "Top 10 games by recommendation score with at least 100k reviews",
            "Which 5 games had the most review-bombing days, and when was the latest?",
            "Which games have the highest share of free or gift-key reviews?",
        ]
        for col, ex in zip(st.columns(len(examples)), examples):
            col.button(ex, use_container_width=True, on_click=_use_example, args=(ex,))
        with st.form("nl_form"):
            question = st.text_input("Your question", key="nl_q_input",
                                     placeholder="e.g., Which games recovered from a bad launch?")
            nl_go = st.form_submit_button("Ask", type="primary")
        # An example button submits directly; otherwise wait for the form's own button.
        pending = st.session_state.pop("nl_pending", None)
        asked_now = pending or (question.strip() if (nl_go and question.strip()) else None)
        if asked_now:
            if _register(f"nl::{asked_now}"):
                st.session_state["nl_active"] = asked_now
            else:
                st.warning(f"You've used all {GEMINI_BUDGET} questions this session — "
                           "refresh the page to start over.")
        active_q = st.session_state.get("nl_active")
        if active_q:
            try:
                sql = nl_to_sql(active_q)
            except Exception as e:
                st.error("Gemini is unavailable right now, so we couldn't build that answer.")
                with st.expander("Technical details"):
                    st.write(str(e))
            else:
                err = guard_sql(sql)
                if err:
                    st.error("That question produced an unsafe query, so we didn't run it. "
                             "Try rephrasing it.")
                else:
                    try:
                        out = run_sql(sql)
                    except Exception as e:
                        st.error("We couldn't answer that one from the data — try rephrasing, "
                                 "or ask about scores, review counts, or bombing events.")
                        with st.expander("Technical details"):
                            st.write(str(e))
                    else:
                        st.markdown(f"**You asked:** {active_q}")
                        if out.empty:
                            st.info("No games matched that. Remember the review snapshot ends "
                                    "2023-10-30, so very recent activity is thin.")
                        else:
                            shown = prettify_columns(out)
                            ch = auto_chart(shown)
                            if ch is not None:
                                st.altair_chart(ch, use_container_width=True, theme=None)
                            st.dataframe(shown, use_container_width=True, hide_index=True)
                            c_dl, _ = st.columns([1, 3])
                            c_dl.download_button(
                                "Download as CSV", shown.to_csv(index=False).encode("utf-8"),
                                file_name="buyorwait_answer.csv", mime="text/csv",
                                use_container_width=True)
                        # Kept for the curious, but out of the way — players want the answer.
                        with st.expander("How we worked this out (advanced)"):
                            st.code(sql, language="sql")
        st.caption(f"{_budget_left()} of {GEMINI_BUDGET} questions left this session.")

# ---------------------------------------------------------------- Player Composition
@st.cache_data(ttl=3600, show_spinner=False)
def composition_total() -> int:
    """Row count of the composition table — derived so the caption can't go stale."""
    try:
        return int(q(f"SELECT COUNT(*) AS c FROM {T('game_composition')}").iloc[0].c)
    except Exception:
        return 0


with tab_comp:
    _n_comp = composition_total()
    section("Player Ownership & Review Quality", eyebrow="Who is reviewing",
            sub="How players acquired each game — direct purchase, free / gift keys, and "
                "Early Access share of reviews"
                + (f", across {_n_comp:,} games." if _n_comp else "."))
    c_s, _ = st.columns([2, 1])
    search_kw = c_s.text_input("Filter by game title:", placeholder="e.g., Cyberpunk / Elden Ring / Counter-Strike", label_visibility="collapsed")
    try:
        if search_kw:
            comp_df = q(f"""
                SELECT s.game, s.appid, s.n_reviews_total AS n_reviews,
                       c.purchase_pct, c.free_pct, c.ea_pct
                FROM {T('game_composition')} c
                JOIN {T('v_scores_live')} s ON c.appid = s.appid
                WHERE LOWER(s.game) LIKE @kw
                ORDER BY s.n_reviews_total DESC
                LIMIT 2000
            """, kw=f"%{search_kw.lower()}%")
        else:
            comp_df = q(f"""
                SELECT s.game, s.appid, s.n_reviews_total AS n_reviews,
                       c.purchase_pct, c.free_pct, c.ea_pct
                FROM {T('game_composition')} c
                JOIN {T('v_scores_live')} s ON c.appid = s.appid
                ORDER BY s.n_reviews_total DESC
                LIMIT 1000
            """)
        st.dataframe(
            comp_df,
            use_container_width=True,
            height=480,
            hide_index=True,
            column_order=("game", "n_reviews", "purchase_pct", "free_pct", "ea_pct"),
            column_config={
                "game": st.column_config.TextColumn("Game Title", width="medium"),
                "n_reviews": st.column_config.NumberColumn("Total Reviews", format="%d"),
                "purchase_pct": st.column_config.ProgressColumn(
                    "Direct Purchase", format="%.1f%%", min_value=0, max_value=100),
                "free_pct": st.column_config.ProgressColumn(
                    "Free / Gift Keys", format="%.1f%%", min_value=0, max_value=100),
                "ea_pct": st.column_config.ProgressColumn(
                    "Early Access", format="%.1f%%", min_value=0, max_value=100),
            }
        )
    except Exception as e:
        st.info(f"Player composition data unavailable ({e}).")

st.divider()
st.caption("Data: 114M Steam Review Dataset + Live Steam Web API Sync | Powered by GCP BigQuery, Cloud Run & Gemini AI")
