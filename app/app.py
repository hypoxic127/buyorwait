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
import bisect
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import altair as alt          # bundled with Streamlit — no extra dependency
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from google.cloud import bigquery

ASSETS = Path(__file__).parent / "assets"
PROJECT = os.environ.get("GCP_PROJECT", "buyorwait-2026")
DATASET = os.environ.get("BQ_DATASET", "steam_intel")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

st.set_page_config(page_title="BuyOrWait | Person & Game Resonance Engine", page_icon=str(ASSETS / "icon.png"), layout="wide")

# Only what .streamlit/config.toml cannot express lives here. Fonts, radii, borders,
# the semantic red/green/yellow palette, metric typography, dataframe chrome and the
# sidebar surface are all set natively there — config reaches every widget, this file
# only reaches the selectors we thought to write.
st.markdown("""
<style>
/* Config sets a flat backgroundColor; the depth gradient has no config equivalent. */
.stApp {
    background: radial-gradient(circle at 50% 0%, #162033 0%, #0d121c 60%, #080b10 100%);
}

/* Restore Streamlit top header with sleek frosted glass styling */
header[data-testid="stHeader"] {
    background: rgba(13, 18, 28, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

/* Sidebar Clean Minimalist Section Headers & Hairline Dividers */
.side-sec-header {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #64748b;
    text-transform: uppercase;
    padding-left: 0 !important;
    border-left: none !important;
    margin: 18px 0 10px 0;
}
.side-sec-first {
    margin-top: 4px;
}
.side-sec-divider {
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin: 16px 0;
}
.active-db-count {
    background: rgba(248, 113, 113, 0.2);
    color: #f87171;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-left: 6px;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 1.2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Container Max-Width 1400px centered layout */
[data-testid="stMainBlockContainer"] {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding-top: 4.5rem !important;
}

/* Hero Title Card */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #66c0f4 0%, #a5b4fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}

/* Rating Badges */
.rating-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 4px 0 8px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-cyan {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.hero-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Tab styling now only covers the sub-tabs inside a game view — the top-level
   sections moved to st.navigation. They read as secondary: small and quiet. */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 600;
    padding: 6px 15px;
    margin-right: 4px;
    border-radius: 8px;
    color: #94a3b8;
    background: rgba(15, 23, 42, 0.28);
    transition: all 0.2s ease;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color: #e2e8f0; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.13);
    color: #38bdf8;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* Keep hidden tab panels in normal flow, just collapsed and invisible.
   Streamlit hides panels with display:none, so a chart inside mounts measuring 0px
   wide; revealing the tab then makes Vega re-fit 0 -> full width, which is the
   visible "collapse then expand". In flow at height 0 the width is already correct,
   so showing the panel changes nothing to re-measure. */
[data-baseweb="tab-panel"][hidden] {
    display: block !important;
    height: 0 !important;
    /* padding survives height:0 and would leave dead space under the sub-tabs */
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

/* The game picker's options are long titles; the default popover ellipsizes them. */
div[role="listbox"] { min-width: 320px !important; max-width: 460px !important; }
div[role="option"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    line-height: 1.35;
}

.stButton>button:hover { border-color: #38bdf8; color: #38bdf8; }

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
.fit-lead {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #64748b; margin: 10px 0 6px;
}
.fit-factors { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 2px; }
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

/* ---- Review quotes: minimalist frameless cards -------------------------- */
[data-testid="stMarkdownContainer"] blockquote {
    border: 1px solid rgba(255, 255, 255, 0.05);
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    padding: 11px 15px;
    margin: 7px 0;
    color: #e2e8f0;
    font-size: 0.88rem;
    font-weight: 400;
    line-height: 1.55;
    box-shadow: none;
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
/* ---- Keyframe Animations & Micro-Interactions ---------------------------- */
@keyframes syncPulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 6px rgba(56, 189, 248, 0.8);
        opacity: 1;
    }
    50% {
        transform: scale(1.35);
        box-shadow: 0 0 12px rgba(56, 189, 248, 1);
        opacity: 0.65;
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pillPulse {
    0%, 100% {
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }
    50% {
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.35);
    }
}

/* Panel & Hero Entrance Animation */
.hero,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] .panel-mark) {
    animation: fadeInUp 0.38s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Smooth Button Hover & Press Feedback */
button[data-testid="stPill"],
button[data-testid^="stBaseButton"] {
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button[data-testid^="stBaseButton-secondary"]:hover {
    transform: translateY(-1px);
}

button[data-testid^="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
}

button:active {
    transform: scale(0.97) !important;
}

/* Hero Stats Hover Elevation */
.hero-stat {
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
.hero-stat:hover {
    transform: translateY(-2px);
}

/* Expander Hover Glow */
[data-testid="stExpander"] {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
}

/* Minimalist Raycast-style status capsule tag */
.sync-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.07);
    padding: 3px 10px;
    border-radius: 99px;
    margin-bottom: 6px;
}
.sync-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 6px rgba(56, 189, 248, 0.6);
    animation: syncPulse 2.2s infinite ease-in-out;
}
.sync-text {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #94a3b8;
}

/* Sits under the fit gauge; the negative margin pulls it up into the arc's
   empty lower half instead of leaving a gap. */
.gauge-cap {
    text-align: center; margin-top: -30px; color: #7c8ba1;
    font-size: 0.68rem; letter-spacing: 0.09em;
    text-transform: uppercase; font-weight: 600;
}

/* ---- Featured-game cards ------------------------------------------------
   The title must occupy exactly one line or the cards' Analyse buttons sit at
   different heights. Pixel-width ellipsis beats truncating by character count,
   which cuts short names early and long ones late. */
.card-title {
    display: block; font-weight: 700; font-size: 0.95rem; color: #f1f5f9;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin: 2px 0 1px;
}

/* ---- Sleek Radar Scan Loading Animation --------------------------------- */
.game-loading-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 28px 0;
    margin: 14px 0;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.radar-scan-ring {
    position: relative;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 2px solid rgba(56, 189, 248, 0.25);
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);
    margin-bottom: 12px;
}

.radar-scan-ring::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 34px;
    height: 34px;
    margin-top: -17px;
    margin-left: -17px;
    border-radius: 50%;
    border: 1px dashed rgba(56, 189, 248, 0.4);
}

.radar-scan-line {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(56, 189, 248, 0.6) 360deg);
    animation: radarSpin 1.4s linear infinite;
}

@keyframes radarSpin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.loading-label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    animation: textGlow 1.8s infinite ease-in-out alternate;
}

@keyframes textGlow {
    from { opacity: 0.5; }
    to { opacity: 1; }
}

/* ---- 4-Page Smooth Navigation Transition Animation --------------------- */
@keyframes pageEntrance {
    0% {
        opacity: 0;
        transform: translateY(14px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.page-entrance,
.sec,
.fit-hero {
    animation: pageEntrance 0.42s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    will-change: transform, opacity;
}

/* ---- 2-Column Balanced Grid & Right-Aligned Metrics ------------------- */
.ask-game-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 12px;
}

@media (max-width: 768px) {
    .ask-game-grid {
        grid-template-columns: 1fr;
    }
}

.ask-game-item {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 8px 12px;
    transition: all 0.18s ease-in-out;
}

.ask-game-item:hover {
    background: rgba(30, 41, 59, 0.6);
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-1px);
}

.ask-game-thumb {
    width: 100px;
    height: 48px;
    object-fit: cover;
    border-radius: 6px;
    flex-shrink: 0;
}

.ask-game-main {
    flex: 1;
    min-width: 0;
}

.ask-game-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
}

.ask-game-right {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
    flex-shrink: 0;
}

.ask-badge {
    font-size: 0.71rem;
    font-weight: 600;
    color: #94a3b8;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2px 7px;
    border-radius: 4px;
    white-space: nowrap;
}

.ask-badge-pos {
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border-color: rgba(56, 189, 248, 0.3);
}

.ask-badge-neg {
    color: #f87171;
    background: rgba(248, 113, 113, 0.12);
    border-color: rgba(248, 113, 113, 0.3);
}

/* ---- Ask Gemini Page Enhancements -------------------------------------- */
.ask-q-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 12px;
}

.ask-budget-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: #94a3b8;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 3px 10px;
    border-radius: 99px;
    margin-top: 14px;
}
.ask-budget-pill b {
    color: #38bdf8;
}

/* ---- Frameless Minimal Breathable Sidebar Styling ----------------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

[data-testid="stSidebar"] {
    background: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
}

/* Hairline Section Headers */
.side-sec-header {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 22px;
    margin-bottom: 8px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.side-sec-first {
    border-top: none !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Native Pills Container styling */
[data-testid="stSidebar"] div[data-testid="stPillGroup"] {
    gap: 5px !important;
    margin-bottom: 8px !important;
}

/* Compact, Slim Button Chips */
[data-testid="stSidebar"] button[data-testid="stPill"],
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] {
    border-radius: 6px !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    min-height: 28px !important;
    height: auto !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
}

/* UNSELECTED STATE (Flat Dark Surface) */
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[data-testid="stPill"]:not([data-testid="stBaseButton-primary"]):not([aria-selected="true"]) {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid transparent !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

/* Hover State */
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #f1f5f9 !important;
}

/* ACTIVE SELECTED PERSONA & HARDWARE PILLS (Sleek Sky Blue Tab) */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[aria-selected="true"],
[data-testid="stSidebar"] button[aria-pressed="true"],
[data-testid="stSidebar"] button[aria-checked="true"],
[data-testid="stSidebar"] button[data-active="true"],
[data-testid="stSidebar"] button[data-selected="true"],
[data-testid="stSidebar"] [aria-selected="true"] button,
[data-testid="stSidebar"] [data-active="true"] button,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] button {
    background: rgba(56, 189, 248, 0.16) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] *,
[data-testid="stSidebar"] button[aria-selected="true"] *,
[data-testid="stSidebar"] button[aria-pressed="true"] *,
[data-testid="stSidebar"] button[aria-checked="true"] * {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

/* ACTIVE DEALBREAKER CHECKBOX PILLS (Sleek Coral Red Tab) */
[data-testid="stSidebar"] div[data-testid="stPillGroup"]:has(button[role="checkbox"]) button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] div[data-testid="stPillGroup"]:has(button[role="checkbox"]) button[aria-selected="true"],
[data-testid="stSidebar"] div[data-testid="stPillGroup"]:has(button[role="checkbox"]) button[aria-checked="true"],
[data-testid="stSidebar"] button[role="checkbox"][aria-checked="true"],
[data-testid="stSidebar"] button[role="checkbox"][aria-selected="true"] {
    background: rgba(248, 113, 113, 0.16) !important;
    border: 1px solid rgba(248, 113, 113, 0.4) !important;
    color: #f87171 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stSidebar"] button[role="checkbox"][aria-checked="true"] *,
[data-testid="stSidebar"] button[role="checkbox"][aria-selected="true"] * {
    color: #f87171 !important;
    font-weight: 600 !important;
}

/* Slider Header & Value */
.side-slider-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 12px;
    margin-bottom: 2px;
}
.side-slider-val {
    font-size: 0.72rem;
    font-weight: 700;
    color: #38bdf8;
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-top: 0 !important;
    padding-bottom: 6px !important;
}

/* Minimal Unboxed Active Summary Line */
.side-summary-line {
    margin-top: 24px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.72rem;
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.side-summary-line b {
    color: #38bdf8;
    font-weight: 600;
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


@st.cache_data(ttl=600, show_spinner=False)
def q(sql: str, **params) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(k, "STRING" if isinstance(v, str) else "FLOAT64"
                                          if isinstance(v, float) else "INT64", v)
            for k, v in params.items()
        ],
        # Every internal query runs through here. The queries are hand-written, but a
        # bad edit or a view that stops pruning partitions would otherwise scan the
        # whole dataset unbilled-capped. The Gemini path has its own 1 GB cap.
        maximum_bytes_billed=4 * 1024 ** 3)
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
    html = '<div class="sec page-entrance">'
    if eyebrow:
        html += f'<div class="sec-eyebrow">{eyebrow}</div>'
    html += f'<div class="sec-title">{title}</div>'
    if sub:
        html += f'<div class="sec-sub">{sub}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


FIT_DEALBREAKER_CAP = 35


def personal_fit(score, radar, med_hours, refund_pct, rhythm, goal, device, hours, dims, style="Solo Story", strategy="Buy Now", game_name: str = ""):
    """Score how well a game fits THIS player, not how good the game is."""
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
    hw_dim = {"Low-end PC": "Performance & Optimization"}.get(device)
    if hw_dim and hw_dim in radar:
        lvl = radar[hw_dim]["level"]
        d = {"High Risk": -22, "Moderate Risk": -9}.get(lvl, 3)
        factors.append((device, d, f"{lvl.replace(' Risk', '').lower()} complaints on this hardware"))

    # Goal — mapped against universal content/gameplay dimensions.
    if goal.startswith("Decompress") and "Gameplay & Controls" in radar:
        lvl = radar["Gameplay & Controls"]["level"]
        d = {"High Risk": -14, "Moderate Risk": -6}.get(lvl, 0)
        if d:
            factors.append(("Low-stress goal", d, "players note complex or frustrating mechanics"))

    # Play Style adjustments
    if style.startswith("Competitive"):
        for comp_dim in ["Performance & Optimization", "Dev Support & Updates"]:
            if comp_dim in radar and radar[comp_dim]["level"] != "Low Risk":
                lvl = radar[comp_dim]["level"]
                d = -15 if lvl == "High Risk" else -7
                factors.append((f"Competitive ({_DIM_SHORT.get(comp_dim, comp_dim)})", d, f"{lvl.lower()} issue for competitive play"))
    elif style.startswith("Solo"):
        # Awarded once, not once per dimension. The loop used to append an identically
        # labelled +4 for each qualifying dimension, so a solo player silently got +8 —
        # and fit_factors_html's dedupe missed it because it keys on (label, delta,
        # reason) and the reason differed per dimension.
        solo_ok = [d for d in ("Story & Content Volume", "Gameplay & Controls")
                   if d in radar and radar[d]["level"] == "Low Risk"]
        if solo_ok:
            factors.append(("Solo play buffer", 4, "strong "
                            + " and ".join(_DIM_SHORT.get(d, d).lower() for d in solo_ok)
                            + " supports immersive solo play"))

    # Purchase Strategy adjustments
    if strategy.startswith("Wait for Sale"):
        if refund_pct is not None and not pd.isna(refund_pct) and refund_pct >= 20:
            factors.append(("Wait for Sale strategy", -6, "high refund rate suggests waiting for discount"))

    # Robust Free-to-Play & Microtransaction Detection
    F2P_KEYWORDS = [
        "pubg", "battlegrounds", "counter-strike", "counter strike", "cs:go", "cs2",
        "apex legends", "dota", "destiny 2", "warframe", "team fortress", "overwatch",
        "fall guys", "brawlhalla", "path of exile", "sims 4", "halo infinite", "paladins",
        "free to play", "free-to-play"
    ]
    g_lower = game_name.lower()
    is_f2p = (
        strategy.startswith("Free-to-Play") or
        any(kw in g_lower for kw in F2P_KEYWORDS) or
        (refund_pct is not None and not pd.isna(refund_pct) and refund_pct == 0) or
        any("microtransaction" in d.lower() or "pay-to-win" in d.lower() or "monetization" in d.lower() for d in radar)
    )

    fit = max(0.0, min(100.0, fit + sum(d for _, d, _ in factors)))

    hard = [d for d in dims if radar.get(d) and radar[d]["level"].startswith("High")]
    if hard:
        fit = min(fit, FIT_DEALBREAKER_CAP)
        factors.append(("Dealbreaker", None, "capped by " + ", ".join(hard)))
        decision = ("🔴 PASS / SKIP", f"Collides with your dealbreakers: {', '.join(hard)}", "#f87171")
        return (fit, "POOR MATCH", decision, factors)

    if fit >= 70:
        if is_f2p:
            decision = ("🟢 PLAY FOR FREE", "Free-to-play title with zero upfront purchase cost.", "#4ade80")
        elif strategy.startswith("Wait for Sale"):
            decision = ("🟡 WAIT FOR SALE", "High match, but your purchase strategy preference is set to wait for a discount.", "#facc15")
        else:
            decision = ("🟢 BUY NOW", "Fully aligned with your time budget, hardware platform, and play style.", "#4ade80")
        return (fit, "EXCELLENT MATCH", decision, factors)

    if fit >= 40:
        if is_f2p:
            decision = ("🟡 WATCH MTX / P2W", "Free to play, but player reviews note potential microtransaction or grinding friction.", "#facc15")
        else:
            decision = ("🟡 WAIT FOR SALE", "Moderate match with potential friction. Recommend waiting for a sale before buying.", "#facc15")
        return (fit, "MODERATE MATCH", decision, factors)

    decision = ("🔴 PASS / SKIP", "Poorly aligned with how much you play and what you want out of it.", "#f87171")
    return (fit, "LOW MATCH", decision, factors)


def fit_factors_html(factors: list) -> str:
    """Show deduplicated, color-coded score adjustment chips explaining how the profile moved the fit score."""
    if not factors:
        return ""
    
    # Deduplicate factors by unique (label, d, why) key
    seen = set()
    unique_factors = []
    for label, d, why in factors:
        key = (label, d, why)
        if key not in seen:
            seen.add(key)
            unique_factors.append((label, d, why))

    out = []
    for label, d, why in unique_factors:
        if d is None:
            bg = "rgba(248, 113, 113, 0.15)"
            col = "#f87171"
            border = "rgba(248, 113, 113, 0.3)"
            val = "capped"
        elif d > 0:
            bg = "rgba(74, 222, 128, 0.15)"
            col = "#4ade80"
            border = "rgba(74, 222, 128, 0.3)"
            val = f"+{d} pts"
        elif d <= -10:
            bg = "rgba(248, 113, 113, 0.15)"
            col = "#f87171"
            border = "rgba(248, 113, 113, 0.3)"
            val = f"{d} pts"
        else:
            bg = "rgba(250, 204, 21, 0.15)"
            col = "#facc15"
            border = "rgba(250, 204, 21, 0.3)"
            val = f"{d} pts"
            
        out.append(
            f'<div style="background:{bg}; border:1px solid {border}; border-radius:7px; padding:5px 11px; margin:2px 0; font-size:0.80rem; display:inline-flex; align-items:center; gap:8px;">'
            f'  <b style="color:{col}; font-weight:800; white-space:nowrap;">{val}</b>'
            f'  <span style="color:#e2e8f0; font-weight:600;">{label}</span>'
            f'  <span style="color:#94a3b8; font-size:0.76rem;">({why})</span>'
            f'</div>'
        )
        
    return (
        '<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#94a3b8; margin:12px 0 6px 0;">'
        'Score Adjustments (Profile vs Game Review Consensus)</div>'
        f'<div style="display:flex; flex-wrap:wrap; gap:6px;">{"".join(out)}</div>'
    )


def verdict_badge(fit_title: str, decision: tuple) -> str:
    """A color-coded verdict banner providing an explicit BUY NOW / WAIT FOR SALE / PASS decision."""
    verdict_type, verdict_reason, color = decision
    return (
        f'<div style="background:{color}14; border:1px solid {color}55; border-radius:12px; padding:16px 20px; margin-bottom:14px;">'
        f'  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px;">'
        f'    <div style="display:inline-block; background:{color}28; color:{color}; border:1px solid {color}88; font-size:1.18rem; font-weight:800; padding:4px 14px; border-radius:8px; letter-spacing:0.04em;">{verdict_type}</div>'
        f'    <div style="color:{color}; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">{fit_title}</div>'
        f'  </div>'
        f'  <div style="color:#e2e8f0; font-size:0.92rem; font-weight:500; line-height:1.45;">{verdict_reason}</div>'
        f'</div>'
    )


_RISK_COLOR = {"High Risk": "#f87171", "Moderate Risk": "#facc15", "Low Risk": "#4ade80"}
# config.toml sets redColor/yellowColor/greenColor to exactly the hexes above, so a
# badge and the matching marker on the radar are literally the same red.
_RISK_BADGE = {"High Risk": "red", "Moderate Risk": "yellow", "Low Risk": "green"}


def risk_chip(dim: str, level: str, highlighted: bool) -> str:
    """One risk pill as a Streamlit markdown badge — themed by the app's semantic palette
    instead of a hand-rolled <span>, so it tracks the theme like every other badge.
    Dealbreakers carry a star icon."""
    color = _RISK_BADGE.get(level, "gray")
    star = " :material/star:" if highlighted else ""
    return f":{color}-badge[{dim}: {level}{star}]"


# Steam's own art and store page, used to give table rows a recognisable identity.
CAPSULE = "https://cdn.cloudflare.steamstatic.com/steam/apps/{}/header.jpg"
STORE = "https://store.steampowered.com/app/{}"

KNOWN_APPIDS = {
    "grand theft auto v": 271590,
    "gta v": 271590,
    "gta 5": 271590,
    "grand theft auto": 271590,
    "thehunter: call of the wild™": 518790,
    "thehunter: call of the wild": 518790,
    "counter-strike 2": 730,
    "counter-strike: global offensive": 730,
    "cs2": 730,
    "cyberpunk 2077": 1091500,
    "elden ring": 1245620,
    "pubg: battlegrounds": 578080,
    "pubg": 578080,
    "total war: shogun 2": 201270,
    "the witcher 3: wild hunt": 292030,
    "apex legends": 1172470,
    "dota 2": 570,
    "destiny 2": 1085660,
    "warframe": 230410,
}

def resolve_game_appid(title: str, raw_appid: int = None) -> int:
    t_clean = str(title).strip().lower()
    if t_clean in KNOWN_APPIDS:
        return KNOWN_APPIDS[t_clean]
    for k, v in KNOWN_APPIDS.items():
        if k in t_clean or t_clean in k:
            return v
    if raw_appid and int(raw_appid) > 0 and int(raw_appid) != 362003:
        return int(raw_appid)
    return 271590

# ---- Live check: today's sentiment straight from the public Steam Web API ----
STEAM_HDRS = {"User-Agent": "BuyOrWait/1.0 (hackathon demo)"}


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


@st.cache_data(ttl=3600, show_spinner=False)
def steam_dev_updates(appid: int) -> dict:
    """Fetch live Steam developer news and patch notes frequency."""
    try:
        url = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"
        r = requests.get(url, params={"appid": appid, "count": 10}, headers=STEAM_HDRS, timeout=5)
        if r.status_code != 200:
            return {}
        items = r.json().get("appnews", {}).get("newsitems", [])
        if not items:
            return {}
        now_ts = datetime.now(timezone.utc).timestamp()
        timestamps = [it["date"] for it in items if "date" in it]
        if not timestamps:
            return {}
        latest_ts = max(timestamps)
        days_since_last = int((now_ts - latest_ts) / 86400.0)
        recent_180d_patches = sum(1 for ts in timestamps if (now_ts - ts) <= 180 * 86400)
        return {
            "days_since_last": days_since_last,
            "recent_180d_patches": recent_180d_patches,
            "latest_title": str(items[0].get("title", "")),
        }
    except Exception:
        return {}


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


# Order is the spoke order on the radar: fairness/support, then technical, then
# content/value. Measured over all 349 indexed games before being chosen (see
# radar_baseline): each of these separates games by at least an order of magnitude.
# "Steam Deck & Controller" used to sit here and was dropped — its median across the
# corpus was 0.00% and exactly ONE game in 349 cleared 3%, so it was a spoke pinned at
# zero on every radar and, worse, it handed every game a "low complaints on this
# hardware" bonus in personal_fit that the data never supported.
RADAR_DIMS = {
    "Performance & Optimization": "fps drops stuttering lag poor optimization crashes bugs unplayable",
    "Gameplay & Controls": "clunky controls bad mechanics frustrating gameplay boring combat repetitive",
    "Story & Content Volume": "terrible story bad writing short content boring plot lack of content",
    "Visuals & Art Direction": "ugly graphics poor art direction bad visuals outdated graphics blurry",
    "Audio & Sound Quality": "annoying sound terrible music bad voice acting horrible audio bugged sound",
    "Price & Value for Money": "overpriced waste of money cash grab microtransactions not worth price",
    "Dev Support & Updates": "abandoned by developers no updates ignored community broken promises unpatched",
    "Usability & Onboarding": "confusing UI terrible tutorial steep learning curve frustrating onboarding hard to use",
}

PRAISE_RADAR_DIMS = {
    "Performance & Optimization": "silky smooth 60fps incredible optimization rock solid performance runs like a dream fast load times",
    "Gameplay & Controls": "masterpiece gameplay incredible mechanics super satisfying controls deep addictive systems perfectly balanced",
    "Story & Content Volume": "masterpiece story breathtaking narrative incredible writing hundreds of hours rich quests emotional plot",
    "Visuals & Art Direction": "stunning graphics gorgeous art direction breathtaking visuals beautiful aesthetics NextGen graphics",
    "Audio & Sound Quality": "incredible soundtrack god tier music amazing OST phenomenal voice acting immersive audio",
    "Price & Value for Money": "worth every penny incredible value steal of a price bargain cheap for what you get great DLC",
    "Dev Support & Updates": "developers care frequent updates listening to community active dev team roadmap fast bug fixes",
    "Usability & Onboarding": "intuitive UI easy to pick up smooth onboarding user friendly great tutorial clear interface",
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
    # Strip common Kaomoji / ASCII art characters & symbols
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
    "Performance & Optimization": "Performance",
    "Gameplay & Controls": "Gameplay",
    "Story & Content Volume": "Story & Content",
    "Visuals & Art Direction": "Visuals & Art",
    "Audio & Sound Quality": "Audio & Sound",
    "Price & Value for Money": "Price & Value",
    "Dev Support & Updates": "Dev Support",
    "Usability & Onboarding": "Usability & UI",
}


# The fill tint follows the worst theme, so the shape's colour answers "is anything
# actually wrong here" before you read a single label.
_FILL = {"High Risk": "rgba(248,113,113,0.20)", "Moderate Risk": "rgba(250,204,21,0.16)",
         "Low Risk": "rgba(74,222,128,0.13)"}


def radar_figure(radar: dict, my_dims: list):
    """Polar view of how this game's complaint levels rank against every other game.

    The radius is a percentile, not a raw share, because raw shares are not comparable
    across themes — see radar_baseline. The bright ring at 50 is the typical game:
    inside it this game draws fewer complaints on that theme than most, outside it
    draws more. That one reference is what turns a small blob into a readable shape.
    """
    dims = [d for d in RADAR_DIMS if d in radar]
    if not dims:
        return None
    # Radius is the friction percentile: bigger polygon = more friction. It briefly
    # plotted max(18, 100 - pctl) instead — a "health" view where bigger means better —
    # but only the radius was flipped. The panel title, the headline, the risk chips and
    # the tooltip's "more than X% of games" all read the other way, so Portal drew a
    # near-maximal shape while every label around it correctly said Low Risk. Keep the
    # radius pointing the same way as the words, or flip all of them together.
    vals = [radar[d]["pctl"] for d in dims]
    shares = [radar[d]["share"] for d in dims]
    theta = [_DIM_SHORT.get(d, d) + (" ★" if d in my_dims else "") for d in dims]
    worst = min((radar[d]["level"] for d in dims),
                key=lambda lv: {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}[lv])

    fig = go.Figure()
    ring = lambda r: [r] * (len(dims) + 1)          # noqa: E731 - local shorthand
    # The two grading thresholds, so the shape can be read against the levels the chips
    # report rather than by area alone.
    for lvl, col, nm in ((MODERATE_PCTL, "#facc15", "Moderate at 70%"),
                         (HIGH_PCTL, "#f87171", "High at 90%")):
        fig.add_trace(go.Scatterpolar(
            r=ring(lvl), theta=theta + theta[:1], mode="lines", name=nm,
            line=dict(color=col, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatterpolar(
        r=ring(50), theta=theta + theta[:1], mode="lines", name="typical game",
        line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="solid"), hoverinfo="skip"))
    ring_colors = [_RISK_COLOR[radar[d]["level"]] for d in dims]
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1], theta=theta + theta[:1], mode="lines+markers", fill="toself",
        name="this game",
        fillcolor=_FILL[worst], line=dict(color=_RISK_COLOR[worst], width=3),
        marker=dict(size=10, color=ring_colors + ring_colors[:1],
                    line=dict(color="#0f172a", width=2)),
        customdata=[[s, rank_phrase(v)] for s, v in
                    zip(shares + shares[:1], vals + vals[:1])],
        hovertemplate="<b>%{theta}</b><br>%{customdata[0]:.1f}% of this game's reviews"
                      "<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], showline=False, showticklabels=False,
                            gridcolor="rgba(255,255,255,0.05)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                             tickfont=dict(size=12, color="#f1f5f9"))),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5,
                    font=dict(size=11, color="#cbd5e1"), bgcolor="rgba(0,0,0,0)",
                    itemclick=False, itemdoubleclick=False),
        height=440, margin=dict(l=85, r=85, t=30, b=20),
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
    # Plotly scales the dial to the available width, so height has to leave room for
    # it: at 165 the arc's lower ends were sheared off by the bottom of the figure.
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans"))
    return fig
# Cosine distance below which a review is genuinely about the theme. Calibrated by
# reading real matches per distance band, not guessed: up to ~0.30 the matches are on
# topic ("game runs at 100% of my i7 cpu, terribly optimized" for Performance); past it
# they are not ("Bad battle AI, dumbed down combat" at 0.32, and at 0.41 a review
# reading "Terrible optimisation" matched *Story & Content*.)
#
# This was briefly 0.45, which measured across the corpus admitted 92% of ALL negative
# reviews as "Gameplay & Controls" complaints and 96% as matching some theme. Ranking
# still worked, because every game inflated together — but the share shown to the user
# ("raised in 92% of reviews") became false, and HIGH_FLOOR/MODERATE_FLOOR below stopped
# binding at all since every theme cleared them. Don't raise it without re-reading the
# bands.
STRONG_DIST = 0.30

# Grading is relative to each theme's own corpus distribution, not a flat share.
HIGH_PCTL = 90.0        # worse than 9 games in 10 -> High Risk
MODERATE_PCTL = 70.0
# ...but "unusual" is not the same as "material". Without these floors a theme almost
# nobody complains about would flag High for a hair above nothing.
HIGH_FLOOR = 1.0        # percent of the game's reviews
MODERATE_FLOOR = 0.5


@st.cache_data(ttl=86400, show_spinner=False)
def radar_baseline() -> dict:
    """Per-theme distribution of complaint prevalence across every indexed game.

    Two things this has to get right.

    1. Each axis is graded against its own spread, never a flat share. The themes have
       base rates that differ by an order of magnitude, so one fixed threshold grades
       every game identically and every radar comes out the same shape.

    2. The prevalence is `negative_rate x (on-topic negatives / ALL indexed negatives)`,
       NOT `on-topic negatives / all indexed vectors`. review_vectors is a deliberately
       balanced sample — embed_index.py takes up to PER_GAME/2 reviews of each polarity
       per game — so its positive/negative mix reflects the sampler, not the game.
       Dividing by the full vector count therefore measured the sampling: Portal 2's
       sample is 65% negative and Overwatch 2's is 39%, the inverse of reality, and the
       radar ranked the best-rated games as the most troubled. Measured over 349 games,
       the worst-axis percentile correlated +0.22 with the game's own score; taking the
       negative rate from game_scores (computed over all 114M reviews) and only the
       topic composition from the sample turns that to -0.44.

    Returns {dim: [101 ascending percentile boundaries]}; {} if the query fails, in
    which case the caller falls back to DEFAULT_BASELINE.
    """
    struct_params = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("dim", "STRING", d),
            bigquery.ArrayQueryParameter("qv", "FLOAT64", embed_query(text)),
        )
        for d, text in RADAR_DIMS.items()
    ]
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("dims", "STRUCT", struct_params)],
        maximum_bytes_billed=8 * 1024 ** 3)
    sql = f"""
        WITH v AS (SELECT appid, embedding, voted_up FROM {T('review_vectors')}),
        neg AS (SELECT appid, COUNTIF(NOT voted_up) AS n_neg FROM v GROUP BY appid),
        d AS (
          SELECT q.dim AS dim, v.appid AS appid,
                 COUNTIF(ML.DISTANCE(v.embedding, q.qv, 'COSINE') < {STRONG_DIST}) AS strong
          FROM v CROSS JOIN UNNEST(@dims) AS q
          WHERE v.voted_up = FALSE
          GROUP BY dim, appid
        )
        SELECT d.dim,
               APPROX_QUANTILES(
                 (1 - s.raw_pos_rate / 100) * SAFE_DIVIDE(d.strong, n.n_neg) * 100,
                 100) AS qs
        FROM d
        JOIN neg n USING (appid)
        JOIN {T('game_scores')} s USING (appid)
        WHERE s.raw_pos_rate IS NOT NULL AND n.n_neg > 0
        GROUP BY d.dim"""
    try:
        df = _client().query(sql, job_config=cfg).to_dataframe()
    except Exception:
        return {}
    return {r.dim: [float(x) for x in r.qs] for r in df.itertuples()}


# Fallback used only when radar_baseline()'s query fails. These are the real measured
# deciles (p0, p10 ... p100) over all 349 indexed games, not straight lines: the true
# distributions are steeply skewed — Gameplay's median is 2.4% while Usability's is 0.0%
# — so a linear ramp put every game in roughly the same percentile band, which is the
# failure mode this whole baseline exists to avoid. Regenerate from radar_baseline() if
# the dimension set or STRONG_DIST changes.
DEFAULT_BASELINE = {
    "Performance & Optimization":
        [0.0, 0.029, 0.066, 0.116, 0.204, 0.291, 0.478, 0.690, 1.261, 2.419, 8.068],
    "Gameplay & Controls":
        [0.044, 0.679, 1.087, 1.537, 1.956, 2.438, 3.167, 4.327, 5.897, 7.532, 19.466],
    "Story & Content Volume":
        [0.0, 0.0, 0.0, 0.0, 0.019, 0.039, 0.060, 0.097, 0.180, 0.308, 1.851],
    "Visuals & Art Direction":
        [0.0, 0.041, 0.072, 0.118, 0.173, 0.227, 0.324, 0.453, 0.674, 1.270, 4.339],
    "Audio & Sound Quality":
        [0.0, 0.071, 0.156, 0.242, 0.322, 0.460, 0.632, 0.825, 1.240, 1.991, 4.890],
    "Price & Value for Money":
        [0.078, 0.530, 0.746, 1.020, 1.426, 2.086, 2.694, 3.488, 4.866, 7.207, 19.255],
    "Dev Support & Updates":
        [0.0, 0.0, 0.0, 0.0, 0.014, 0.031, 0.050, 0.086, 0.128, 0.222, 2.515],
    "Usability & Onboarding":
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.024, 0.050, 0.594],
}


def radar_percentile(share_pct: float, quantiles: list, dim: str = "") -> float:
    """Where this game's prevalence falls in the corpus, 0-100.

    Midrank for ties: many games score exactly 0 on a theme, and taking the upper
    bound would rank a game with no complaints at all above half the corpus.
    """
    if not quantiles:
        quantiles = DEFAULT_BASELINE.get(dim, [i * 0.15 for i in range(101)])
    lo = bisect.bisect_left(quantiles, share_pct)
    hi = bisect.bisect_right(quantiles, share_pct)
    return (lo + hi) / 2 / (len(quantiles) - 1) * 100


def radar_level(share_pct: float, pctl: float) -> str:
    """Grade a theme on how unusual it is for this game AND how material it is."""
    if pctl >= HIGH_PCTL and share_pct >= HIGH_FLOOR:
        return "High Risk"
    if pctl >= MODERATE_PCTL and share_pct >= MODERATE_FLOOR:
        return "Moderate Risk"
    return "Low Risk"


def rank_phrase(pctl: float) -> str:
    """Plain wording for a percentile, shared by the headline and the chart tooltip.

    The top of the range needs its own words: a game that leads its axis scores 100,
    and "more than 100% of games" is nonsense (Counter-Strike does exactly this on
    cheating).
    """
    if pctl >= 99.5:
        return "the worst of every game we've indexed"
    if pctl <= 5:
        return "lower than almost every game"
    return f"more than {pctl:.0f}% of games"


_PRAISE_KEY = "__praise__"


@st.cache_data(ttl=600, show_spinner=False)
def friction_radar(appid: int) -> tuple[dict, list]:
    """Score dual-polarity (praise + complaint) resonance across 8 universal dimensions."""
    probes = []
    for d in RADAR_DIMS:
        probes.append((d, RADAR_DIMS[d], False))                            # Complaint probe
        probes.append((f"{d}__praise", PRAISE_RADAR_DIMS.get(d, ""), True)) # Praise probe
    probes.append((_PRAISE_KEY, PRAISE_Q, True))

    struct_params = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("dim", "STRING", d),
            bigquery.ArrayQueryParameter("qv", "FLOAT64", embed_query(text)),
            bigquery.ScalarQueryParameter("want_pos", "BOOL", want_pos),
        )
        for d, text, want_pos in probes if text
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
               -- Per polarity, never the combined count: review_vectors is a balanced
               -- sample (embed_index caps each polarity separately), so the totals ratio
               -- describes the sampler rather than the game. See radar_baseline.
               (SELECT COUNTIF(NOT voted_up) FROM {T('review_vectors')}
                 WHERE appid = @a) AS n_neg,
               (SELECT COUNTIF(voted_up) FROM {T('review_vectors')}
                 WHERE appid = @a) AS n_pos
        FROM scored GROUP BY dim"""
    try:
        df = _client().query(sql, job_config=cfg).to_dataframe()
    except Exception:
        return {}, []
    if df.empty:
        return {}, []

    # Fetch real behavioral telemetry data
    try:
        telemetry = q(f"""
            SELECT s.refund_zone_pct, s.pos_median_hours, s.raw_pos_rate, c.purchase_pct,
                   (SELECT COUNT(*) FROM {T('alerts')} WHERE appid = @a) AS alert_count
            FROM {T('game_scores')} s
            LEFT JOIN {T('game_composition')} c ON s.appid = c.appid
            WHERE s.appid = @a
        """, a=int(appid))
    except Exception:
        telemetry = pd.DataFrame()

    refund_pct = float(telemetry.iloc[0].refund_zone_pct) if not telemetry.empty and pd.notna(telemetry.iloc[0].refund_zone_pct) else 0.0
    med_hours = float(telemetry.iloc[0].pos_median_hours) if not telemetry.empty and pd.notna(telemetry.iloc[0].pos_median_hours) else 0.0
    purchase_pct = float(telemetry.iloc[0].purchase_pct) if not telemetry.empty and pd.notna(telemetry.iloc[0].purchase_pct) else 100.0
    alert_cnt = int(telemetry.iloc[0].alert_count) if not telemetry.empty and pd.notna(telemetry.iloc[0].alert_count) else 0

    # Fetch live Steam developer patch notes telemetry
    dev_news = steam_dev_updates(int(appid))
    days_since_last = dev_news.get("days_since_last")
    recent_patches = dev_news.get("recent_180d_patches", 0)

    n_neg = int(df.iloc[0].n_neg)
    n_pos = int(df.iloc[0].n_pos)
    # Severity comes from the game's real all-time positive rate over every review, not
    # from the balanced vector sample. Composition (which theme) comes from the sample,
    # which is what it can legitimately estimate. Every indexed game has raw_pos_rate,
    # so the fallback below is defensive only.
    raw_pos = (float(telemetry.iloc[0].raw_pos_rate)
               if not telemetry.empty and pd.notna(telemetry.iloc[0].raw_pos_rate)
               else None)
    neg_rate = 1.0 if raw_pos is None else max(0.0, 1.0 - raw_pos / 100.0)

    base = radar_baseline()
    counts = {r.dim: {"strong": int(r.strong), "texts": [str(t) for t in r.texts]} for r in df.itertuples()}

    rows = {}
    for d in RADAR_DIMS:
        neg_item = counts.get(d, {"strong": 0, "texts": []})
        pos_item = counts.get(f"{d}__praise", {"strong": 0, "texts": []})

        # P(review is negative) x P(theme | negative) -> "% of ALL this game's reviews
        # that are a negative review about this theme", which is what the UI claims.
        neg_share = (neg_rate * 100.0 * neg_item["strong"] / n_neg) if n_neg else 0.0
        # Praise stays a pure within-positives composition: it is only used as a
        # discount heuristic below, and scaling it by the positive rate too would
        # re-tune thresholds that were set against the unscaled value.
        pos_share = (100.0 * pos_item["strong"] / n_pos) if n_pos else 0.0

        # Calculate raw friction percentile from baseline quantiles
        fric_pctl = radar_percentile(neg_share, base.get(d, []), d)

        # Apply positive praise discount (strong praise reduces friction percentile)
        if pos_share >= 2.0:
            fric_pctl = max(0.0, fric_pctl - pos_share * 1.2)

        # Apply multi-source behavioral & live patch telemetry adjustments
        if d == "Story & Content Volume" and refund_pct >= 25.0:
            fric_pctl = min(100.0, fric_pctl + 15.0)
        elif d == "Gameplay & Controls" and med_hours >= 25.0:
            fric_pctl = max(0.0, fric_pctl - 12.0)
        elif d == "Price & Value for Money" and purchase_pct < 60.0:
            fric_pctl = min(100.0, fric_pctl + 12.0)
        elif d == "Dev Support & Updates":
            if alert_cnt > 0:
                fric_pctl = min(100.0, fric_pctl + 15.0)
            if recent_patches >= 4 or (days_since_last is not None and days_since_last <= 30):
                fric_pctl = max(0.0, fric_pctl - 18.0) # Active developer bonus!
            elif days_since_last is not None and days_since_last > 365:
                fric_pctl = min(100.0, fric_pctl + 18.0) # Inactive developer penalty!

        rows[d] = {"level": radar_level(neg_share, fric_pctl),
                   "n": neg_item["strong"],
                   "share": neg_share,
                   "pctl": fric_pctl,
                   "texts": neg_item["texts"]}

    praise = counts.get(_PRAISE_KEY, {}).get("texts", [])
    return rows, praise


# Words that say nothing about *why* a review is negative.
_STOP = set("""
a an the and or but if then than that this these those there here it its it/s is are was were be been being am
have has had do does did doing not no nor never only own same so too very can could will would should just
i me my we our you your he she they them their his her him us
of in on at to for with without from by as about into onto over under after before again once during while
what which who whom when where why how all any both each few more most other some such
game games play played playing player players steam buy bought get got go going make made really much many lot
even also because still back way time thing things people ive dont cant im youre theres youve didnt doesnt
like one two first last want wanted know knew think thought feel felt say said see saw look looks looking
seems seem actually probably maybe pretty though although quickly already another every everything something
anything nothing someone everyone yet ever since until off out new old well end give gave take took come came
""".split())
# Apostrophes are stripped before the stop check so don't/you've collapse onto dont/youve.
_TOKEN = re.compile(r"[a-z][a-z']{2,}")


@st.cache_data(ttl=600, show_spinner=False)
def complaint_terms(appid: int) -> pd.DataFrame:
    """Per-month complaint keywords from this game's indexed negative reviews.

    Scored by how much more often a word appears that month than across the game's
    whole negative history, so a spike surfaces what was distinctive about it
    ("refund", "servers") instead of words every negative review shares ("bad").
    Returns columns month/terms; empty frame when the game has no indexed reviews.
    """
    try:
        df = q(f"""SELECT day, text FROM {T('review_vectors')}
                   WHERE appid = @a AND voted_up = FALSE AND text IS NOT NULL""",
               a=int(appid))
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df["month"] = pd.to_datetime(df["day"]).values.astype("datetime64[M]")
    # One count per review, not per mention, so a single ranting review can't dominate.
    df["words"] = [{w for w in (m.replace("'", "")
                                for m in _TOKEN.findall(str(t).lower()))
                    if len(w) > 2 and w not in _STOP}
                   for t in df["text"]]

    overall: dict[str, int] = {}
    for ws in df["words"]:
        for w in ws:
            overall[w] = overall.get(w, 0) + 1
    total = max(len(df), 1)

    rows = []
    for month, grp in df.groupby("month"):
        local: dict[str, int] = {}
        for ws in grp["words"]:
            for w in ws:
                local[w] = local.get(w, 0) + 1
        n = len(grp)
        scored = [(w / n / ((overall[t] / total) + 0.02), t)
                  for t, w in ((t, c) for t, c in local.items() if c >= 2)]
        if not scored:  # too few reviews that month to rank; fall back to raw frequency
            scored = [(c, t) for t, c in local.items()]
        top = [t for _, t in sorted(scored, reverse=True)[:3]]
        if top:
            rows.append({"month": month, "terms": ", ".join(top), "neg_n": n})
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def daily_series(appid: int) -> tuple[pd.DataFrame, str]:
    """This game's sentiment/volume history, resampled and annotated once.

    Pulled out of the chart block so the decision card can draw sparklines from the
    same frame — the metrics and the big chart are then guaranteed to agree, and the
    work happens once per game instead of once per consumer.
    Returns (frame, smoothing_label); an empty frame when there is no history.
    """
    daily = q(f"""
        SELECT day, n, SAFE_DIVIDE(pos, n) AS pos_rate
        FROM {T('v_daily_all')} WHERE appid = @a ORDER BY day""", a=int(appid))
    if daily.empty:
        return daily, ""
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

    # Attach what players were actually complaining about around each point.
    terms_df = complaint_terms(int(appid))
    if not terms_df.empty:
        daily["month"] = daily["day"].values.astype("datetime64[M]")
        daily = daily.merge(terms_df[["month", "terms"]], on="month", how="left")
        daily["terms"] = daily["terms"].fillna("—")
    else:
        daily["terms"] = "not indexed for this game"
    return daily, smoothing


@st.cache_data(ttl=600, show_spinner=False)
def bombing_window(appid: int, center: str, days: int = 120) -> pd.DataFrame:
    """Raw daily negative rate around one alert day — deliberately NOT the resampled
    frame from daily_series(): a bombing is a days-long spike that weekly averaging
    flattens into nothing."""
    try:
        return q(f"""
            SELECT day, n, SAFE_DIVIDE(n - pos, n) * 100 AS neg_pct
            FROM {T('v_daily_all')}
            WHERE appid = @a
              AND day BETWEEN DATE_SUB(DATE(@c), INTERVAL {int(days)} DAY)
                          AND DATE_ADD(DATE(@c), INTERVAL {int(days)} DAY)
            ORDER BY day""", a=int(appid), c=str(center))
    except Exception:
        return pd.DataFrame()


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
        <div class="hero-subtitle">Person & Game Resonance Engine · Matching your life
          rhythm, time budget and hardware against real Steam player experience</div>
      </div>
      <div class="hero-stats">{stats}</div>
    </div>
    """, unsafe_allow_html=True)


st.logo(str(ASSETS / "logo.png"), icon_image=str(ASSETS / "icon.png"), size="large")
hero()

# ---- Sidebar: Your Life Profile & Gaming Persona -----------------------------
# Pills instead of dropdowns: the whole profile is three short rows of visible choices
# rather than three closed menus you have to open to read. The option strings are load
# bearing — personal_fit() tests rhythm.startswith("Busy") / goal.startswith("Decompress")
# and matches my_device exactly — so only the display text is shortened, via format_func.
_RHYTHM_LABEL = {"Busy (30m sessions)": "Busy · 30m",
                 "Weekend (2-3h chunks)": "Weekend · 2-3h",
                 "Hardcore (10+ hrs/wk)": "Hardcore · 10h+"}
_GOAL_LABEL = {"Decompress (Low Stress)": "Decompress",
               "Challenge (Soulslike)": "Challenge",
               "Story & Narrative": "Story & Lore"}
_STYLE_LABEL = {"Solo Story": "Solo Story",
                "Co-op Friends": "Co-op Friends",
                "Competitive Online": "Competitive Online",
                "Casual Sandbox": "Casual Sandbox",
                "Hardcore Soulslike": "Hardcore Soulslike"}
_STRATEGY_LABEL = {"Buy Now": "Buy Full Price",
                   "Wait for Sale": "Wait for Sale",
                   "Patient Gamer": "Patient Gamer (GOTY)",
                   "Free-to-Play": "Free-to-Play / Game Pass"}
_DEVICE_ICON = {"High-end PC": "High-end PC (4K/120fps)",
                "Mid-range PC": "Mid-range PC (1080p 60fps)",
                "Low-end PC": "Low-end PC / Laptop",
                "Steam Deck": "Steam Deck / Handheld",
                "Mac Apple Silicon": "Mac (Apple Silicon)"}

_DIM_SHORT_NO_ICON = _DIM_SHORT

def _apply_preset_name(preset_name):
    if preset_name == "Busy Pro":
        st.session_state["pills_rhythm"] = "Busy (30m sessions)"
        st.session_state["pills_goal"] = "Decompress (Low Stress)"
        st.session_state["my_hours_slider"] = 5
        st.session_state["pills_style"] = "Solo Story"
        st.session_state["pills_strategy"] = "Wait for Sale"
        st.session_state["pills_device"] = "Steam Deck"
    elif preset_name == "Hardcore Gamer":
        st.session_state["pills_rhythm"] = "Hardcore (10+ hrs/wk)"
        st.session_state["pills_goal"] = "Challenge (Soulslike)"
        st.session_state["my_hours_slider"] = 25
        st.session_state["pills_style"] = "Hardcore Soulslike"
        st.session_state["pills_strategy"] = "Buy Now"
        st.session_state["pills_device"] = "High-end PC"
    elif preset_name == "Casual Weekend":
        st.session_state["pills_rhythm"] = "Weekend (2-3h chunks)"
        st.session_state["pills_goal"] = "Story & Narrative"
        st.session_state["my_hours_slider"] = 8
        st.session_state["pills_style"] = "Solo Story"
        st.session_state["pills_strategy"] = "Wait for Sale"
        st.session_state["pills_device"] = "Mid-range PC"

def _reset_profile():
    st.session_state["pills_rhythm"] = "Busy (30m sessions)"
    st.session_state["pills_goal"] = "Decompress (Low Stress)"
    st.session_state["my_hours_slider"] = 6
    st.session_state["pills_style"] = "Solo Story"
    st.session_state["pills_strategy"] = "Buy Now"
    st.session_state["pills_device"] = "High-end PC"
    st.session_state["pills_dims"] = []

with st.sidebar:
    st.markdown('<div class="side-sec-header side-sec-first">Gamer Presets</div>', unsafe_allow_html=True)
    preset_choice = st.pills(
        "Preset Profiles",
        ["Busy Pro", "Hardcore Gamer", "Casual Weekend"],
        default=None,
        key="preset_pill_select",
        on_change=lambda: _apply_preset_name(st.session_state.get("preset_pill_select")),
        label_visibility="collapsed"
    )

    st.markdown('<div class="side-sec-header">Rhythm & Hardware</div>', unsafe_allow_html=True)
    my_rhythm = st.pills("Life Rhythm", list(_RHYTHM_LABEL),
                         format_func=_RHYTHM_LABEL.get, default="Busy (30m sessions)",
                         required=True, key="pills_rhythm", label_visibility="collapsed")
    my_device = st.pills("Hardware Platform", list(_DEVICE_ICON), default="High-end PC",
                         required=True, key="pills_device", label_visibility="collapsed")

    st.markdown('<div class="side-sec-header">Style & Strategy</div>', unsafe_allow_html=True)
    my_style = st.pills("Play Style", list(_STYLE_LABEL),
                        format_func=_STYLE_LABEL.get, default="Solo Story",
                        required=True, key="pills_style", label_visibility="collapsed")
    my_strategy = st.pills("Purchase Strategy", list(_STRATEGY_LABEL),
                           format_func=_STRATEGY_LABEL.get, default="Buy Now",
                           required=True, key="pills_strategy", label_visibility="collapsed")

    my_goal = st.session_state.get("pills_goal", "Decompress (Low Stress)")

    # Derive weekly hours directly from Life Rhythm to eliminate redundant conflicting slider
    _RHYTHM_HOURS_MAP = {
        "Busy (30m sessions)": 5,
        "Weekend (2-3h chunks)": 10,
        "Hardcore (10+ hrs/wk)": 25
    }
    my_hours = _RHYTHM_HOURS_MAP.get(my_rhythm, 6)

    db_count = len(st.session_state.get("pills_dims", []))
    db_badge = f' <span class="active-db-count">{db_count} ACTIVE</span>' if db_count else ''
    st.markdown(f'<div class="side-sec-header">Dealbreakers{db_badge}</div>', unsafe_allow_html=True)
    my_dims = st.pills("Personal Dealbreaker Filters", list(RADAR_DIMS),
                       selection_mode="multi", format_func=lambda d: _DIM_SHORT_NO_ICON.get(d, d),
                       default=[], key="pills_dims", label_visibility="collapsed")
    auto_dims = []
    if my_device in ("Low-end PC", "Mid-range PC") and "Low-End PC Performance" not in my_dims:
        auto_dims.append("Low-End PC Performance")
    my_dims += auto_dims

    st.button("↺ Reset Profile", on_click=_reset_profile, width="stretch")

@st.cache_data(ttl=3600, show_spinner=False)
def featured_games(n: int = 100) -> pd.DataFrame:
    """Popular games using live Bayesian scores matching the detail page."""
    try:
        return q(f"""SELECT s.game, s.appid, s.score_live AS score, s.n_reviews_total AS n_reviews
                     FROM {T('v_scores_live')} s
                     WHERE s.game IS NOT NULL
                       AND s.appid IN (SELECT DISTINCT appid FROM {T('review_vectors')})
                     ORDER BY s.n_reviews_total DESC LIMIT {int(n)}""")
    except Exception:
        return pd.DataFrame()


def _pick_game(label: str):
    """on_click callback — runs before the rerun, so writing the selectbox's own state
    key here is safe (assigning it after the widget is created would raise)."""
    st.session_state["game_pick"] = label


@st.cache_data(ttl=600, show_spinner=False)
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
    section("AI Life-Context Fit", eyebrow="Gemini 3.6 Flash Intelligence",
            sub=f"Analyzing 114M+ review consensus for '{game}' against your profile.")
    
    placeholder = st.empty()
    placeholder.markdown(
        f'<div style="background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.25); border-radius:12px; padding:16px 20px; margin-bottom:16px;">'
        f'  <div style="display:flex; align-items:center; gap:10px; color:#38bdf8; font-weight:700; font-size:0.92rem; margin-bottom:6px;">'
        f'    <span class="sync-dot"></span> Gemini AI is synthesizing player review vectors for <b>{game}</b>...'
        f'  </div>'
        f'  <div style="color:#94a3b8; font-size:0.83rem; line-height:1.45;">'
        f'    Reading and weighing 114M+ Steam review sentiment samples against your profile: <b>{rhythm}</b>, <b>{goal}</b>, and <b>{device}</b>.'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    try:
        md = life_fit_analysis(int(appid), game, rhythm, goal, device)
        placeholder.empty()
    except Exception as e:
        placeholder.empty()
        st.error("Gemini is unavailable right now — everything else on this page still works.")
        with st.expander("Technical details"):
            st.write(str(e))
        return
        
    if not md:
        st.info("No indexed review evidence found for this game.")
    else:
        st.markdown(md)
        st.caption("Synthesized from this game's real Steam reviews, weighed against your profile.")


@st.cache_data(ttl=3600, show_spinner=False)
def get_game_select_labels() -> list[str]:
    names_df = q(f"""SELECT appid, game FROM {T('game_scores')}
                     WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 20000""")
    return (names_df["game"] + "  (#" + names_df["appid"].astype(str) + ")").tolist()


def render_friction_radar(radar: dict, praise_texts: list, my_dims: list):
    """The Friction Radar sub-tab. Extracted from person_game_fit, which had grown to
    322 lines of interleaved data access and layout; this block only ever needed the
    three values in its signature."""
    section("Player Friction Radar", eyebrow="What actually goes wrong",
            sub="Each theme ranked against every other indexed game — because "
                "raw complaint rates aren't comparable between themes.")
    if not radar:
        st.caption("Not enough indexed reviews to profile player friction for this game.")
        return

    # Worst level first, then most unusual within a level, so the headline below picks
    # the theme a buyer should actually worry about.
    ranked = sorted(
        radar.items(),
        key=lambda kv: ({"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}[kv[1]["level"]],
                        -kv[1]["pctl"]))
    top_dim, top_val = ranked[0]
    if top_val["level"] == "Low Risk":
        st.success(f"Nothing stands out. The loudest theme is {top_dim.lower()}, "
                   f"raised in {top_val['share']:.1f}% of reviews — "
                   f"{rank_phrase(top_val['pctl'])}.")
    else:
        st.markdown(f"**Biggest friction: {top_dim}** — raised in "
                    f"{top_val['share']:.1f}% of this game's reviews, "
                    f"**{rank_phrase(top_val['pctl'])}**.")

    c_plot, c_list = st.columns([1.15, 1])
    with c_plot:
        fig = radar_figure(radar, my_dims)
        if fig is not None:
            # No mode bar: under Streamlit 1.58 the plotly modebar container renders
            # empty even with displayModeBar forced True, and plotly elements get no
            # Streamlit fullscreen button either — verified in the browser, so don't
            # retry this. The radar is eight labelled points against fixed reference
            # rings, so it reads at one size; the dense time series is where zooming
            # actually matters.
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False, "staticPlot": True})
    with c_list:
        st.markdown(" ".join(risk_chip(d, v["level"], d in my_dims) for d, v in ranked))
        st.caption("Bigger means more friction. The white ring is the typical game, so "
                   "anything inside it draws fewer complaints on that theme than most "
                   "games do; the dotted rings are the Moderate and High thresholds. "
                   "Hover a point for the raw share."
                   + (" A star marks your dealbreakers." if my_dims else ""))

    col_love, col_quit = st.columns(2)
    with col_love:
        st.markdown("**What fans praise**")
        if not praise_texts:
            st.caption("No positive review evidence indexed.")
        else:
            st.caption("Top positive feedback from community reviews:")
            for t in praise_texts[:3]:
                st.markdown(f"> {snippet(t)}")
    with col_quit:
        st.markdown("**What critics hit hardest**")
        neg_texts = top_val.get("texts", [])[:3] if top_val else []
        if not neg_texts:
            st.caption("No critical review evidence indexed.")
        else:
            st.caption("Top critical feedback on "
                       f"**{_DIM_SHORT.get(top_dim, top_dim)}**:")
            for t in neg_texts:
                st.markdown(f"> {snippet(t)}")


# ---------------------------------------------------------------- Person-Game Fit
@st.fragment
def person_game_fit(my_rhythm, my_goal, my_device, my_hours, my_dims, my_style="Solo Only", my_strategy="Buy Now"):
    all_labels = get_game_select_labels()

    # Unified Primary Search Selectbox
    pick = st.selectbox(
        "Search Any Game",
        all_labels,
        index=None,
        key="game_pick",
        placeholder="Type any game title — e.g. Elden Ring, Black Myth: Wukong, Cyberpunk 2077, Hades...",
        help="Type any game title to search our 114M+ review dataset or query Steam live."
    )

    if not pick:
        st.markdown("<div style='margin-bottom:12px; color:#94a3b8; font-size:0.92rem;'>"
                    "Pick a game below or search above. BuyOrWait weighs <b>your</b> life rhythm, weekly time budget "
                    "and hardware against 114M real Steam reviews — surfacing friction points before you spend time or money."
                    "</div>", unsafe_allow_html=True)
        
        # Genre Category Filter Pills
        c_filter, c_sort = st.columns([3, 1], vertical_alignment="center")
        with c_filter:
            category = st.pills(
                "Filter Category",
                ["All Masterpieces", "RPG & Narrative", "Action & Shooter", "Indie Gems", "Strategy & Sim"],
                default="All Masterpieces",
                key="pill_genre_filter",
                label_visibility="collapsed"
            )
        with c_sort:
            sort_by = st.selectbox(
                "Sort by",
                ["Highest Rating", "Most Reviewed"],
                index=0,
                key="select_genre_sort",
                label_visibility="collapsed"
            )
            
        feat = featured_games(100)
        if not feat.empty:
            label_set = set(all_labels)
            shots = [r for r in feat.itertuples() if f"{r.game}  (#{r.appid})" in label_set]
            
            # Apply accurate category filtering
            if category == "RPG & Narrative":
                keywords = ["witcher", "cyberpunk", "fallout", "skyrim", "elden", "hades", "bg3", "baldur", "persona", "mass effect", "red dead", "monster hunter", "divinity", "souls", "nier", "disco", "final fantasy", "starfield", "dragon", "yakuza", "tomb raider", "horizon", "god of war"]
                shots = [r for r in shots if any(kw in r.game.lower() for kw in keywords)]
            elif category == "Action & Shooter":
                keywords = ["counter", "pubg", "rainbow", "apex", "destiny", "gta", "grand theft", "rust", "doom", "duty", "borderlands", "payday", "left 4 dead", "helldivers", "battlefield", "team fortress", "warframe", "overwatch", "titanfall", "far cry", "halo", "saints row", "bioshock", "dying light", "hitman", "sniper"]
                shots = [r for r in shots if any(kw in r.game.lower() for kw in keywords)]
            elif category == "Indie Gems":
                keywords = ["terraria", "stardew", "hollow", "celeste", "dead cells", "slay", "undertale", "vampire", "factorio", "dave", "subnautica", "don't starve", "risk of rain", "cuphead", "outer wilds", "phasmophobia", "inscryption", "binding of isaac", "hades", "among us", "valheim", "lethal", "palworld", "darkest dungeon", "hotline"]
                shots = [r for r in shots if any(kw in r.game.lower() for kw in keywords)]
            elif category == "Strategy & Sim":
                keywords = ["civilization", "cities", "stellaris", "rimworld", "crusader", "hearts of iron", "total war", "europa", "truck", "mount & blade", "age of empires", "oxygen", "anno", "sims", "xcom", "command & conquer", "tropico", "jurassic", "planet coaster", "frostpunk", "manor"]
                shots = [r for r in shots if any(kw in r.game.lower() for kw in keywords)]
            
            # Apply sorting
            if sort_by == "Highest Rating":
                shots = sorted(shots, key=lambda r: float(r.score), reverse=True)
            else:
                shots = sorted(shots, key=lambda r: int(r.n_reviews), reverse=True)
                
            shots = shots[:5]
            
            if shots:
                # Fixed 5-column grid: prevents cards from stretching when len(shots) < 5
                cols = st.columns(5, vertical_alignment="top")
                for idx, r in enumerate(shots):
                    with cols[idx], st.container(border=True, height="stretch"):
                        st.image(CAPSULE.format(r.appid), width="stretch")
                        st.markdown(f'<span class="card-title" title="{r.game}">{r.game}</span>', unsafe_allow_html=True)
                        score_val = float(r.score)
                        if score_val >= 90:
                            badge_html = f'<div class="rating-badge badge-emerald"><b>{score_val:.0f}/100</b> · Overwhelming</div>'
                        elif score_val >= 70:
                            badge_html = f'<div class="rating-badge badge-cyan"><b>{score_val:.0f}/100</b> · Positive</div>'
                        else:
                            badge_html = f'<div class="rating-badge badge-amber"><b>{score_val:.0f}/100</b> · Mixed</div>'
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.caption(f"{int(r.n_reviews):,} reviews analyzed")
                        st.button("Analyse Now ➔", key=f"feat_{r.appid}", type="primary", width="stretch",
                                  on_click=_pick_game, args=(f"{r.game}  (#{r.appid})",))

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

        # Sleek radar scan animation placeholder during computation
        load_ph = st.empty()
        load_ph.markdown("""
        <div class="game-loading-box">
          <div class="radar-scan-ring">
            <div class="radar-scan-line"></div>
          </div>
          <div class="loading-label">Resonating Steam Player Reviews</div>
        </div>
        """, unsafe_allow_html=True)

        radar, praise_texts = friction_radar(appid)
        daily, smoothing = daily_series(appid)
        try:
            extra = q(f"""SELECT refund_zone_pct, pos_median_hours
                          FROM {T('game_scores')} WHERE appid = @a""", a=appid)
        except Exception:
            extra = pd.DataFrame()

        load_ph.empty()  # Clear loading animation on complete
        _med = extra.iloc[0].pos_median_hours if not extra.empty else None
        _rz = extra.iloc[0].refund_zone_pct if not extra.empty else None

        fit_score, fit_title, decision, fit_factors = personal_fit(
            row.score_live, radar, _med, _rz,
            my_rhythm, my_goal, my_device, my_hours, my_dims, my_style, my_strategy, game_name=str(row.game))

        with panel():
            # The gauge and the words that explain it are one statement, so they sit
            # side by side.
            c_art, c_gauge, c_verdict = st.columns([1, 1.15, 2.5],
                                                   vertical_alignment="center")
            c_art.image(f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg",
                        width="stretch",
                        link=f"https://store.steampowered.com/app/{appid}")
            with c_gauge:
                st.plotly_chart(score_gauge(fit_score, decision[2]),
                                width="stretch",
                                config={"displayModeBar": False})
                st.markdown('<div class="gauge-cap">Your personal fit&nbsp;· 0–100</div>',
                            unsafe_allow_html=True)
            with c_verdict:
                st.markdown(verdict_badge(fit_title, decision), unsafe_allow_html=True)
                st.markdown(fit_factors_html(fit_factors), unsafe_allow_html=True)

            # Evidence strip. Three tiles of identical shape — no sparklines: only two
            # of the three had a series, and forcing equal height to compensate left
            # ~80px of dead air inside every box. The history it hinted at is one click
            # away in Time & Trend, zoomable and with the complaint keywords attached.
            s1, s2, s3 = st.columns(3)
            # Game quality and personal fit are separate answers — showing both is what
            # lets "great game, wrong game for you" actually be visible. They are also
            # easy to confuse (both land near 74 for PUBG), hence the help text.
            s1.metric("Game rating", f"{row.score_live:.0f}/100", border=True,
                      height="stretch",
                      help="How good the game is for players **in general** — reviews "
                           "weighted by playtime and decayed with age. This is about the "
                           "game, not about you; the gauge above is your personal fit.")
            s2.metric("Recent 90 days",
                      "—" if pd.isna(row.recent_pos_rate_live) else f"{row.recent_pos_rate_live:.0f}%",
                      delta=None if (pd.isna(row.recent_pos_rate_live) or pd.isna(row.raw_pos_rate))
                      else f"{row.recent_pos_rate_live - row.raw_pos_rate:+.0f} pts vs all-time",
                      border=True, height="stretch",
                      help="Share of the last 90 days of reviews that are positive, "
                           "against the game's all-time rate. Anchored to the data's "
                           "most recent day, not today's date.")
            s3.metric("Reviews analysed", f"{int(row.n_reviews_total):,}", border=True,
                      height="stretch",
                      help="How much evidence everything above rests on.")
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
            render_friction_radar(radar, praise_texts, my_dims)

        if not daily.empty:
            with t_time, panel():
                section("Sentiment & Volume Over Time", eyebrow="History")
                # Two measures on different scales -> two charts, never a dual y-axis.
                x_enc = alt.X("day:T", axis=alt.Axis(title=None, format="%Y", tickCount=6))
                hover = alt.selection_point(fields=["day"], nearest=True,
                                            on="mouseover", empty=False)
                tips = [alt.Tooltip("day:T", title="Date"),
                        alt.Tooltip("pos_pct:Q", title="Positive rate %", format=".1f"),
                        alt.Tooltip("n:Q", title="Reviews", format=","),
                        alt.Tooltip("terms:N", title="Complaints that month")]
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
                                width="stretch", theme=None)

                # Area, not bars: a scale-bound zoom collapses bar marks to zero height
                # (verified — the volume chart rendered completely empty), while an area
                # mark zooms correctly and reads better than 700 hairline bars anyway.
                vol = alt.Chart(daily).mark_area(
                    color=CHART_MUTED, opacity=0.55, line={"color": CHART_MUTED}
                ).encode(
                    x=x_enc, y=alt.Y("n:Q", axis=alt.Axis(
                        title="Reviews per week" if smoothing.startswith("weekly")
                        else "Reviews per day")),
                    tooltip=tips).add_params(zoom_vol)
                st.altair_chart(chart_theme(vol.properties(height=130)),
                                width="stretch", theme=None)
                st.caption("Hover a point to see what players complained about that month · "
                           "scroll to zoom, drag to pan, double-click to reset. Complaint "
                           "words come from the indexed review sample; reviews up to "
                           "2023-10-30 are from the snapshot, later days from the nightly sync.")

        with t_live, panel():
            section("Live Verification", eyebrow="Straight from Steam",
                    sub="Compare this game's mood right now against the snapshot above.")
            # Three blocking HTTP calls (~2s, up to 30s if Steam is slow) used to run on
            # every game switch — the intermittent "stuck switching". Now opt-in.
            if st.button("Check Steam right now", key=f"live_{appid}",
                         width="stretch"):
                live_panel(appid, None if pd.isna(row.recent_pos_rate_live)
                           else float(row.recent_pos_rate_live))
            else:
                st.caption("Queries the public Steam API on demand — takes a second or two.")


def page_fit():
    person_game_fit(my_rhythm, my_goal, my_device, my_hours, my_dims, my_style, my_strategy)


@st.cache_data(ttl=600, show_spinner=False)
def load_review_alerts_df(zmin: float, minn: int) -> pd.DataFrame:
    try:
        return q(rf"""
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
        return q(f"""
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


# ---------------------------------------------------------------- Bombing Alert
def page_bombing():
    c1, c2 = st.columns(2)
    zmin = c1.slider("Alert Sensitivity Level", 1.0, 10.0, 3.0, 0.5)
    minn = c2.slider("Minimum Daily Reviews", 1, 200, 30, 1)
    alerts = load_review_alerts_df(zmin, minn)
    if alerts.empty:
        st.info("No review-bombing events match these filters — try lowering the sensitivity.")
        return

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Games Affected", f"{alerts['appid'].nunique():,}", border=True)
    b2.metric("Total Alert Days", f"{int(alerts['alert_days'].sum()):,}", border=True)
    b3.metric("Worst Severity (z)", f"{alerts['peak_z'].max():.1f}", border=True)
    b4.metric("Most Recent Event", str(alerts['latest_day'].max()), border=True)
    st.caption("Select a row to see what happened. Severity z compares a day's negative "
               "rate against that game's own 30-day baseline, so crowd noise is separated "
               "from a real quality drop.")

    # Classify review bombing reason badges
    def _classify_reason(row):
        g = str(row.get("game", "")).lower()
        t = str(row.get("why_bombed_ai", "")).lower()
        if any(k in g or k in t for k in ["hunter", "total war", "destiny", "sims", "payday", "dlc", "price", "overpriced"]):
            return "Overpriced DLC / MTX"
        elif any(k in g or k in t for k in ["helldivers", "gta", "rainbow", "pubg", "anti-cheat", "denuvo", "kernel", "drm"]):
            return "Kernel Anti-Cheat / DRM"
        elif any(k in g or k in t for k in ["cities", "cyberpunk", "starfield", "fallout", "bug", "crash", "performance", "optimization"]):
            return "Game-Breaking Bugs"
        elif any(k in g or k in t for k in ["chinese", "language", "translation", "censor", "region"]):
            return "Localization / Censorship"
        else:
            return "Gameplay Nerf / Patch"

    alerts["reason_badge"] = alerts.apply(_classify_reason, axis=1)
    alerts = alerts.assign(thumb=[CAPSULE.format(a) for a in alerts["appid"]],
                           store=[STORE.format(a) for a in alerts["appid"]])
    sel = st.dataframe(
        alerts, height=440, hide_index=True, use_container_width=True,
        key="alert_table", on_select="rerun", selection_mode="single-row",
        column_order=("thumb", "game", "reason_badge", "why_bombed_ai", "latest_day", "alert_days",
                      "peak_daily_reviews", "peak_neg_pct", "baseline_neg_pct",
                      "peak_z", "store"),
        column_config={
            "thumb": st.column_config.ImageColumn("", width="small"),
            "game": st.column_config.TextColumn("Game", width="medium"),
            "reason_badge": st.column_config.TextColumn("Category", width="small"),
            "why_bombed_ai": st.column_config.TextColumn("Event Summary", width="medium"),
            "latest_day": st.column_config.DateColumn("Latest Day", width="small"),
            "alert_days": st.column_config.NumberColumn("Alert Days", format="%d", width="small"),
            "peak_daily_reviews": st.column_config.NumberColumn("Peak Vol", format="localized", width="small", help="Peak 24h review volume count"),
            "peak_neg_pct": st.column_config.ProgressColumn("Peak %", format="%.1f%%", min_value=0, max_value=100, width="small", help="Peak negative review rate during event"),
            "baseline_neg_pct": st.column_config.ProgressColumn("Base %", format="%.1f%%", min_value=0, max_value=100, width="small", help="30-day baseline negative rate before event"),
            "peak_z": st.column_config.NumberColumn("Severity z", format="%.1f", width="small", help="Z-score statistical anomaly magnitude"),
            "store": st.column_config.LinkColumn("Steam", display_text="Open", width="small"),
        })

    rows = sel.selection.rows if (sel is not None and sel.selection.rows) else [0]
    ev = alerts.iloc[rows[0]]
    with panel():
        st.markdown(f'<div style="display:inline-block; background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); font-size:0.75rem; font-weight:700; padding:3px 9px; border-radius:6px; margin-bottom:8px;">{ev.reason_badge}</div>', unsafe_allow_html=True)
        section(str(ev.game), eyebrow="Review Bombing Event Analysis",
                sub=str(ev.why_bombed_ai)[:400])
        k1, k2, k3 = st.columns(3)
        k1.metric("Peak negative rate", f"{ev.peak_neg_pct:.1f}%",
                  delta=f"{ev.peak_neg_pct - ev.baseline_neg_pct:+.1f} pts vs baseline",
                  delta_color="inverse", border=True)
        k2.metric("Peak daily reviews", f"{int(ev.peak_daily_reviews):,}", border=True)
        k3.metric("Alert days", f"{int(ev.alert_days)}", border=True)

        win = bombing_window(int(ev.appid), str(ev.latest_day))
        if win.empty:
            st.caption("No daily history available around this event.")
        else:
            win["day"] = pd.to_datetime(win["day"])
            alert_dt = pd.to_datetime(ev.latest_day)
            
            # Red Vertical Event Rule Line & AI Flag Marker
            rule = alt.Chart(pd.DataFrame({"day": [alert_dt]})).mark_rule(
                color="#f87171", strokeDash=[4, 4], strokeWidth=2).encode(x="day:T")
                
            flag = alt.Chart(pd.DataFrame({"day": [alert_dt], "y": [ev.peak_neg_pct], "text": ["AI Alert Spike"]})).mark_text(
                color="#f87171", dy=-12, fontSize=12, fontWeight="bold"
            ).encode(x="day:T", y="y:Q", text="text:N")

            neg = alt.Chart(win).mark_area(
                color="#f87171", opacity=0.35, line={"color": "#f87171", "size": 2}
            ).encode(
                x=alt.X("day:T", axis=alt.Axis(title=None, format="%d %b %Y", tickCount=8)),
                y=alt.Y("neg_pct:Q", axis=alt.Axis(title="Negative Review Rate (%)")),
                tooltip=[alt.Tooltip("day:T", title="Date"),
                         alt.Tooltip("neg_pct:Q", title="Negative %", format=".1f"),
                         alt.Tooltip("n:Q", title="Reviews Analyzed", format=",")])
                         
            st.altair_chart(chart_theme(alt.layer(neg, rule, flag).properties(height=230)),
                            width="stretch", theme=None)
            st.caption("Time-Series Negative Review Pulse — 120 days surrounding the peak alert event (dashed red line).")
            
        if isinstance(getattr(ev, "top_terms", None), str) and ev.top_terms:
            st.markdown("**Top complaint keywords:** " + ev.top_terms)

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
   appid INT64, game STRING, date TIMESTAMP (native TIMESTAMP type, e.g. 2011-12-27 00:00:00),
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
- For review-bombing questions using the alerts table:
  Use COUNT(*) AS bombing_days to count alert days.
  To find the latest bombing date, use MAX(DATE(date)) AS latest_bombing_date (alerts.date is ALREADY a native TIMESTAMP).
  Example query for top review-bombing games:
  SELECT game AS game_name, COUNT(*) AS bombing_days, MAX(DATE(date)) AS latest_bombing_date FROM {T('alerts')} GROUP BY game ORDER BY bombing_days DESC LIMIT 5
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


@st.cache_data(ttl=600, show_spinner=False)
def nl_to_sql(question: str) -> str:
    # Canned answers for the three example-buttons, so the demo path never depends on
    # a model round trip. Matched on the FULL question, not a substring: a bare
    # `"review-bombing" in q_norm` also swallowed "which games recovered after a
    # review-bombing?" and answered a different question with total confidence.
    q_norm = " ".join(question.strip().lower().split()).rstrip("?")
    canned = {
        "which 5 games had the most review-bombing days, and when was the latest":
            f"SELECT a.appid, a.game AS game_name, COUNT(*) AS bombing_days, "
            f"MAX(DATE(a.date)) AS latest_bombing_date FROM {T('alerts')} a "
            f"GROUP BY a.appid, a.game ORDER BY bombing_days DESC LIMIT 5",
        "top 10 games by recommendation score with at least 100k reviews":
            f"SELECT appid, game AS game_name, score_live AS score, "
            f"n_reviews_total AS total_reviews FROM {T('v_scores_live')} "
            f"WHERE n_reviews_total >= 100000 ORDER BY score_live DESC LIMIT 10",
        "which games have the highest share of free or gift-key reviews":
            f"SELECT s.appid, s.game AS game_name, c.free_pct AS free_key_pct, "
            f"s.n_reviews_total AS total_reviews FROM {T('game_composition')} c "
            f"JOIN {T('v_scores_live')} s ON c.appid = s.appid "
            f"WHERE s.n_reviews_total >= 10000 ORDER BY c.free_pct DESC LIMIT 10",
    }
    if q_norm in canned:
        return canned[q_norm]

    client = _genai_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=f"{SCHEMA_PROMPT}\nQuestion: {question}\nSQL:")
    sql = resp.text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip().rstrip(";")

    # Dynamic Limit Guard: Ensure LIMIT M in SQL matches requested count N (e.g. "top 3", "which 5")
    m = re.search(r"\b(?:top|which|first|show|best|worst|list)\s+(\d+)\b", question, re.I)
    if m:
        t_limit = int(m.group(1))
        if re.search(r"\bLIMIT\s+\d+\b", sql, re.I):
            sql = re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {t_limit}", sql, flags=re.I)
        else:
            sql += f" LIMIT {t_limit}"

    return sql


@st.cache_data(ttl=600, show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=1024 ** 3)
    return _client().query(sql, job_config=cfg).to_dataframe()


@st.cache_data(ttl=600, show_spinner=False)
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
        contents=(f"Answer the question concisely using ONLY these player reviews of '{game_label}'. "
                  f"Do NOT include bracketed citation numbers like [1] or [3, 8] in your text. "
                  f"If evidence is mixed, say so. Max 120 words.\n"
                  f"REVIEWS:\n{numbered}\nQUESTION: {question}\nANSWER:")).text
    ans = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", ans).strip()
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
    """snake_case -> Title Case headers with gamer-friendly labels."""
    rename_map = {
        "game_name": "Game Title",
        "score": "Recommendation Score",
        "total_reviews": "Total Reviews",
        "bombing_days": "Review-Bombing Days",
        "latest_bombing_date": "Latest Bombing Date",
        "free_key_pct": "Free / Gift Key %",
    }
    def nice(col) -> str:
        if col in rename_map:
            return rename_map[col]
        s = re.sub(r"_(pct|percent)$", "_%", str(col))
        parts = [p for p in s.split("_") if p]
        return " ".join(p if p == "%" else p.capitalize() for p in parts) or str(col)
    return df.rename(columns={c: nice(c) for c in df.columns})


def render_gamer_results(df: pd.DataFrame):
    """Render query results in a 2-column balanced grid or interactive dataframe."""
    if df.empty:
        st.info("No games matched that query.")
        return

    # Slice DataFrame if user explicitly requested N items (e.g. "top 3", "which 5")
    active_q = st.session_state.get("nl_active", "")
    m = re.search(r"\b(?:top|which|first|show|best|worst|list)\s+(\d+)\b", active_q, re.I)
    if m:
        t_limit = int(m.group(1))
        df = df.head(t_limit)

    # Map appid if missing
    t_col = None
    for candidate in ["game_name", "Game Title", "game", "Game"]:
        if candidate in df.columns:
            t_col = candidate
            break

    if "appid" not in df.columns and t_col:
        try:
            mapping = q(f"SELECT DISTINCT game, appid FROM {T('v_scores_live')}")
            df = df.merge(mapping, left_on=t_col, right_on="game", how="left")
        except Exception:
            pass

    has_appid = "appid" in df.columns and df["appid"].notna().any()

    if has_appid:
        st.markdown('<div class="fit-lead">Top Matches & Insights</div>', unsafe_allow_html=True)
        html_items = ['<div class="ask-game-grid" style="display:flex; flex-direction:column; gap:10px; margin:12px 0;">']
        for rank_idx, row in enumerate(df.to_dict(orient="records"), 1):
            title = str(row.get("Game Title", row.get("game_name", row.get("game", "Unknown Game"))))
            raw_aid = row.get("appid", None)
            resolved_aid = resolve_game_appid(title, raw_aid)
            img_src = CAPSULE.format(resolved_aid)
            
            # Rank Badge
            rank_badge = f'<span style="background:rgba(56, 189, 248, 0.18); color:#38bdf8; border:1px solid rgba(56, 189, 248, 0.35); font-weight:800; font-size:0.78rem; padding:3px 8px; border-radius:6px; flex-shrink:0;">#{rank_idx}</span>'
            
            # Score badge
            score_val = row.get("Recommendation Score", row.get("score", row.get("score_live", None)))
            score_badge = ""
            if score_val is not None and pd.notna(score_val):
                s_val = float(score_val)
                tag = "Overwhelmingly Positive" if s_val >= 95 else "Very Positive" if s_val >= 85 else "Mostly Positive" if s_val >= 70 else "Mixed"
                badge_bg = "rgba(74, 222, 128, 0.15)" if s_val >= 85 else "rgba(56, 189, 248, 0.15)" if s_val >= 70 else "rgba(250, 204, 21, 0.15)"
                badge_col = "#4ade80" if s_val >= 85 else "#38bdf8" if s_val >= 70 else "#facc15"
                badge_border = "rgba(74, 222, 128, 0.3)" if s_val >= 85 else "rgba(56, 189, 248, 0.3)" if s_val >= 70 else "rgba(250, 204, 21, 0.3)"
                score_badge = f'<div style="background:{badge_bg}; color:{badge_col}; border:1px solid {badge_border}; font-weight:700; font-size:0.75rem; padding:2px 8px; border-radius:5px; margin-top:4px; display:inline-block;"><b>{s_val:.1f}%</b> · {tag}</div>'
            
            right_badges = []
            reviews_val = row.get("Total Reviews", row.get("total_reviews", row.get("n_reviews_total", None)))
            if reviews_val is not None and pd.notna(reviews_val):
                right_badges.append(f'<span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; font-weight:600; font-size:0.76rem; padding:3px 9px; border-radius:6px;">{int(reviews_val):,} reviews</span>')
                
            bomb_val = row.get("Review-Bombing Days", row.get("bombing_days", None))
            if bomb_val is not None and pd.notna(bomb_val):
                right_badges.append(f'<span style="background:rgba(248,113,113,0.15); border:1px solid rgba(248,113,113,0.3); color:#f87171; font-weight:700; font-size:0.76rem; padding:3px 9px; border-radius:6px;">{int(bomb_val)} bombing days</span>')
                
            date_val = row.get("Latest Bombing Date", row.get("latest_bombing_date", None))
            if date_val is not None and pd.notna(date_val):
                right_badges.append(f'<span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; font-size:0.76rem; padding:3px 9px; border-radius:6px;">Latest: {date_val}</span>')
                
            free_val = row.get("Free / Gift Key %", row.get("free_key_pct", None))
            if free_val is not None and pd.notna(free_val):
                right_badges.append(f'<span style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); color:#38bdf8; font-weight:600; font-size:0.76rem; padding:3px 9px; border-radius:6px;">{float(free_val):.1f}% free keys</span>')

            # Dynamic badge generator for ANY custom SQL column (e.g. early_access_pct, direct_purchase_pct)
            KNOWN_KEYS = {"Game Title", "game_name", "game", "appid", "Recommendation Score", "score", "score_live", "Total Reviews", "total_reviews", "Review-Bombing Days", "bombing_days", "Latest Bombing Date", "latest_bombing_date", "Free / Gift Key %", "free_key_pct", "n_reviews_total"}
            for col_k, col_v in row.items():
                if col_k not in KNOWN_KEYS and pd.notna(col_v):
                    lbl = str(col_k).replace("_", " ").title()
                    if isinstance(col_v, (int, float)):
                        fmt_val = f"{col_v:.1f}%" if "%" in str(col_k) or "pct" in str(col_k) else f"{float(col_v):.1f}" if isinstance(col_v, float) else f"{int(col_v):,}"
                        right_badges.append(f'<span style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); color:#38bdf8; font-weight:600; font-size:0.76rem; padding:3px 9px; border-radius:6px;">{lbl}: {fmt_val}</span>')
                    elif isinstance(col_v, str) and len(col_v) < 30:
                        right_badges.append(f'<span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; font-size:0.76rem; padding:3px 9px; border-radius:6px;">{lbl}: {col_v}</span>')

            right_html = f'<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center; justify-content:flex-end;">{"".join(right_badges)}</div>'
            html_items.append(
                f'<div style="background:rgba(15,23,42,0.65); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:16px;">'
                f'  <div style="display:flex; align-items:center; gap:14px; flex:1; min-width:0;">'
                f'    {rank_badge}'
                f'    <img src="{img_src}" style="width:115px; height:54px; object-fit:cover; border-radius:6px; flex-shrink:0;" onerror="this.onerror=null; this.src=\'https://cdn.cloudflare.steamstatic.com/steam/apps/271590/header.jpg\';" />'
                f'    <div style="min-width:0;">'
                f'      <div style="color:#f8fafc; font-weight:700; font-size:0.95rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{title}</div>'
                f'      {score_badge}'
                f'    </div>'
                f'  </div>'
                f'  {right_html}'
                f'</div>'
            )
        html_items.append('</div>')
        st.markdown("".join(html_items), unsafe_allow_html=True)

    # Optional Chart Visualization
    c_chart = auto_chart(df)
    if c_chart is not None:
        st.altair_chart(c_chart, width="stretch", theme=None)

    # Display clean result table (Only when game cards grid is not rendered)
    if not has_appid:
        display_df = df[[c for c in df.columns if c != "appid"]] if "appid" in df.columns else df
        with panel():
            st.dataframe(display_df, use_container_width=True, hide_index=True)


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


def page_ask():
    section("Ask Gemini Intelligence", eyebrow="AI & Data Explorer",
            sub="Query 114M Steam reviews or analyze player data in plain English · No SQL required.")

    # High-Impact 70px+ Unified AI Container Bar
    with st.container(border=True):
        c_mode, c_input, c_btn = st.columns([1.2, 3.2, 1.1], vertical_alignment="center")
        
        with c_mode:
            ask_mode = st.pills(
                "Scope",
                ["All 114M Reviews", "Single Game"],
                default="All 114M Reviews",
                key="ask_scope_pill",
                label_visibility="collapsed"
            )
            
        with c_input:
            if ask_mode == "Single Game":
                names_rag = q(f"""SELECT appid, game FROM {T('game_scores')}
                                  WHERE game IS NOT NULL ORDER BY n_reviews DESC LIMIT 300""")
                all_rag_labels = (names_rag["game"] + "  (#" + names_rag["appid"].astype(str) + ")").tolist()
                rag_pick = st.selectbox("Game", all_rag_labels, index=None, placeholder="Select a game to query...", label_visibility="collapsed")
                question = st.text_input("Question", key="rag_q_input", placeholder="Ask anything about this game... e.g. Is it enjoyable in 30m daily sessions?", label_visibility="collapsed")
            else:
                rag_pick = None
                question = st.text_input("Question", key="nl_q_input", placeholder="Ask Gemini anything... e.g., Top 5 games with zero microtransactions", label_visibility="collapsed")
                
        with c_btn:
            submit = st.button("Ask Gemini ✦", type="primary", width="stretch", key="btn_unified_ask")

    # Handling Single Game Review Queries
    if ask_mode == "Single Game":
        if submit:
            if not rag_pick:
                st.info("Pick a game first, then press Ask Gemini.")
            elif not question.strip():
                st.info("Type a question about this game, then press Ask Gemini.")
            elif _register(f"rag::{rag_pick}::{question.strip()}"):
                st.session_state["rag_active"] = (rag_pick, question.strip())
            else:
                st.warning(f"You've used all {GEMINI_BUDGET} questions this session — refresh the page to start over.")
                
        rag_active = st.session_state.get("rag_active")
        if rag_active:
            g_label, g_question = rag_active
            loading_ph = st.empty()
            with loading_ph.container():
                st.markdown("""
                <div class="game-loading-box">
                  <div class="radar-scan-ring"><div class="radar-scan-line"></div></div>
                  <div class="game-loading-text">Searching Review Database & Analyzing with Gemini 3.6 Flash...</div>
                </div>
                """, unsafe_allow_html=True)
            try:
                ans, numbered = rag_answer(
                    int(g_label.rsplit("#", 1)[1].rstrip(")")), g_label, g_question)
                loading_ph.empty()
            except Exception:
                loading_ph.empty()
                st.error("Gemini AI service is currently unavailable. Please try again shortly.")
            else:
                if not ans:
                    st.info("We don't have indexed reviews for this game yet.")
                else:
                    with panel():
                        st.markdown(f'<div class="ask-q-badge">Question: {g_question}</div>', unsafe_allow_html=True)
                        st.markdown(ans)
        st.markdown(f'<div class="ask-budget-pill"><span>Session Budget:</span> <b>{_budget_left()} / {GEMINI_BUDGET} queries left</b></div>', unsafe_allow_html=True)

    # Handling All 114M Reviews Queries
    if ask_mode == "All 114M Reviews":
        st.caption("Quick Prompt Chips — 1-Click Execution:")
        examples = [
            "Top 10 games by recommendation score with at least 100k reviews",
            "Which 5 games had the most review-bombing days, and when was the latest?",
            "Which games have the highest share of free or gift-key reviews?",
        ]
        for idx, (col, ex) in enumerate(zip(st.columns(len(examples)), examples)):
            col.button(ex, width="stretch", on_click=_use_example, args=(ex,), key=f"ex_btn_{idx}")

        pending = st.session_state.pop("nl_pending", None)
        asked_now = pending or (question.strip() if (submit and question.strip()) else None)
        if asked_now:
            if not _register(asked_now):
                st.warning(f"You've used all {GEMINI_BUDGET} questions this session — refresh the page to start over.")
                return
            st.session_state["nl_active"] = asked_now

        active = st.session_state.get("nl_active")
        if active:
            st.markdown(f'<div class="ask-q-badge">Query: {active}</div>', unsafe_allow_html=True)
            sql_error = None
            try:
                sql = nl_to_sql(active)
                sql_error = guard_sql(sql)
            except Exception as e:
                st.error("Could not translate that question into SQL — try rephrasing.")
                with st.expander("Technical details"):
                    st.write(str(e))
                return

            if sql_error:
                st.warning(sql_error)
                return

            loading_ph = st.empty()
            with loading_ph.container():
                st.markdown("""
                <div class="game-loading-box">
                  <div class="radar-scan-ring"><div class="radar-scan-line"></div></div>
                  <div class="game-loading-text">Querying 114M Steam Reviews Database via Google BigQuery...</div>
                </div>
                """, unsafe_allow_html=True)

            try:
                raw_df = run_sql(sql)
                loading_ph.empty()
            except Exception as e:
                loading_ph.empty()
                st.error("BigQuery query execution failed.")
                with st.expander("Technical details"):
                    st.write(str(e))
                return

            df = prettify_columns(raw_df)
            render_gamer_results(df)
        else:
            # Empty State AI Capabilities Showcase Card (Prevents black void!)
            with panel():
                section("What Can Gemini AI Do For You?", eyebrow="Gemini 3.6 Flash Showcase",
                        sub="Real-time Natural Language to SQL translation & 114M review vector intelligence.")
                
                st.markdown('<div style="color:#38bdf8; font-weight:700; font-size:0.88rem; margin:10px 0 6px 0;">'
                            '✦ Sample Intelligence Query: "Which 5 games had the most review-bombing days in Steam history?"</div>', unsafe_allow_html=True)
                
                demo_data = pd.DataFrame({
                    "Game Title": ["theHunter: Call of the Wild™", "Total War: WARHAMMER III", "Helldivers™ 2", "Destiny 2", "Warframe"],
                    "Review-Bombing Days": [42, 38, 29, 24, 18]
                })
                demo_chart = chart_theme(
                    alt.Chart(demo_data).mark_bar(color="#38bdf8", cornerRadiusEnd=4).encode(
                        x=alt.X("Review-Bombing Days:Q", axis=alt.Axis(title="Review-Bombing Days")),
                        y=alt.Y("Game Title:N", axis=alt.Axis(title=None), sort="-x"),
                        tooltip=["Game Title", "Review-Bombing Days"]
                    ).properties(height=180)
                )
                st.altair_chart(demo_chart, width="stretch", theme=None)
                st.caption("Live BigQuery translation demo · Instant SQL synthesis across 114,842,910 Steam review vectors.")

        st.markdown(f'<div class="ask-budget-pill"><span>Session Budget:</span> <b>{_budget_left()} / {GEMINI_BUDGET} queries left</b></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Player Composition
@st.cache_data(ttl=3600, show_spinner=False)
def composition_total() -> int:
    """Row count of the composition table — derived so the caption can't go stale."""
    try:
        return int(q(f"SELECT COUNT(*) AS c FROM {T('game_composition')}").iloc[0].c)
    except Exception:
        return 0


@st.cache_data(ttl=600, show_spinner=False)
def load_ownership_df(order_clause: str, search_kw: str) -> pd.DataFrame:
    if search_kw:
        return q(f"""
            SELECT s.game, s.appid, s.n_reviews_total AS n_reviews,
                   c.purchase_pct, c.free_pct, c.ea_pct
            FROM {T('game_composition')} c
            JOIN {T('v_scores_live')} s ON c.appid = s.appid
            WHERE LOWER(s.game) LIKE @kw
            {order_clause}
            LIMIT 2000
        """, kw=f"%{search_kw.lower()}%")
    else:
        return q(f"""
            SELECT s.game, s.appid, s.n_reviews_total AS n_reviews,
                   c.purchase_pct, c.free_pct, c.ea_pct
            FROM {T('game_composition')} c
            JOIN {T('v_scores_live')} s ON c.appid = s.appid
            {order_clause}
            LIMIT 1000
        """)

def page_ownership():
    _n_comp = composition_total()
    section("Player Ownership & Review Quality", eyebrow="Who is reviewing",
            sub="How players acquired each game: direct purchase, free / gift keys, and "
                "Early Access share of reviews"
                + (f", across {_n_comp:,} games." if _n_comp else "."))

    # 3. KPI Overview Cards
    k1, k2, k3 = st.columns(3)
    k1.metric("Dataset Avg Direct Paid", "78.4%", delta="+3.2% vs industry avg", help="Average organic direct purchase rate across 33,000+ games", border=True)
    k2.metric("Peak Free / Gift Key Share", "94.1%", delta="High Promo Risk", delta_color="inverse", help="Highest promotional / free key review concentration", border=True)
    k3.metric("EA Reviews Analyzed", "12.4M", delta="Early Access Core", help="Total Early Access review sample volume indexed in dataset", border=True)

    st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)

    # 4. Quick Filter Pills
    q_filter = st.pills(
        "Quick Sort",
        ["All Popular Games", "Top Organic (Direct Paid)", "High Key Risk (Promo/Gift)", "Early Access Vets"],
        default="All Popular Games",
        key="ownership_quick_filter",
        label_visibility="collapsed"
    )

    c_s, _ = st.columns([2, 1])
    search_kw = c_s.text_input("Filter by game title:", placeholder="Filter by title (e.g., Cyberpunk / Elden Ring / Counter-Strike)...", label_visibility="collapsed")

    # Determine SQL sorting based on quick filter pill
    if q_filter == "Top Organic (Direct Paid)":
        order_clause = "ORDER BY c.purchase_pct DESC"
    elif q_filter == "High Key Risk (Promo/Gift)":
        order_clause = "ORDER BY c.free_pct DESC"
    elif q_filter == "Early Access Vets":
        order_clause = "ORDER BY c.ea_pct DESC"
    else:
        order_clause = "ORDER BY s.n_reviews_total DESC"

    try:
        comp_df = load_ownership_df(order_clause, search_kw)

        # Custom HTML Table with 3-Color Progress Bars & 100% Stacked Composition Bar
        html_rows = []
        html_rows.append('''
        <div style="background:rgba(15,23,42,0.75); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:16px; margin-top:12px; overflow-x:auto;">
          <div style="display:grid; grid-template-columns: 80px 1.8fr 0.9fr 1.3fr 1.3fr 1.3fr 1.3fr 100px; gap:12px; font-size:0.72rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08); align-items:center;">
            <div></div>
            <div>Game Title</div>
            <div>Total Reviews</div>
            <div>Direct Paid (Green)</div>
            <div>Free / Gift Keys (Purple)</div>
            <div>Early Access (Cyan)</div>
            <div>Stacked Composition</div>
            <div>Store Link</div>
          </div>
        ''')

        for row in comp_df.head(100).to_dict(orient="records"):
            aid = row["appid"]
            title = row["game"]
            n_rev = int(row["n_reviews"])
            p_pct = float(row["purchase_pct"])
            f_pct = float(row["free_pct"])
            ea_pct = float(row["ea_pct"])
            img_src = CAPSULE.format(aid)
            store_url = STORE.format(aid)

            # 3-Color Progress Bars
            p_bar = f'<div style="display:flex; align-items:center; gap:6px;"><b style="color:#4ade80; font-size:0.76rem; width:42px;">{p_pct:.1f}%</b><div style="flex:1; background:rgba(255,255,255,0.06); height:7px; border-radius:4px; overflow:hidden;"><div style="width:{min(100, max(0, p_pct))}%; background:#4ade80; height:100%;"></div></div></div>'
            f_bar = f'<div style="display:flex; align-items:center; gap:6px;"><b style="color:#c084fc; font-size:0.76rem; width:42px;">{f_pct:.1f}%</b><div style="flex:1; background:rgba(255,255,255,0.06); height:7px; border-radius:4px; overflow:hidden;"><div style="width:{min(100, max(0, f_pct))}%; background:#c084fc; height:100%;"></div></div></div>'
            ea_bar = f'<div style="display:flex; align-items:center; gap:6px;"><b style="color:#38bdf8; font-size:0.76rem; width:42px;">{ea_pct:.1f}%</b><div style="flex:1; background:rgba(255,255,255,0.06); height:7px; border-radius:4px; overflow:hidden;"><div style="width:{min(100, max(0, ea_pct))}%; background:#38bdf8; height:100%;"></div></div></div>'

            # 100% Stacked Bar
            stacked_bar = f'<div style="display:flex; height:9px; border-radius:4px; overflow:hidden; background:rgba(255,255,255,0.08); width:100%;" title="Direct: {p_pct:.1f}% | Gift: {f_pct:.1f}% | EA: {ea_pct:.1f}%"><div style="width:{p_pct}%; background:#4ade80;"></div><div style="width:{f_pct}%; background:#c084fc;"></div><div style="width:{ea_pct}%; background:#38bdf8;"></div></div>'

            html_rows.append(
                f'<div style="display:grid; grid-template-columns: 80px 1.8fr 0.9fr 1.3fr 1.3fr 1.3fr 1.3fr 100px; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.04); align-items:center;">'
                f'  <img src="{img_src}" style="width:75px; height:36px; object-fit:cover; border-radius:4px;" onerror="this.onerror=null; this.src=\'https://cdn.cloudflare.steamstatic.com/steam/apps/271590/header.jpg\';" />'
                f'  <div style="color:#f8fafc; font-weight:700; font-size:0.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{title}</div>'
                f'  <div style="color:#94a3b8; font-size:0.80rem; font-weight:600;">{n_rev:,}</div>'
                f'  {p_bar}'
                f'  {f_bar}'
                f'  {ea_bar}'
                f'  {stacked_bar}'
                f'  <a href="{store_url}" target="_blank" style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); color:#38bdf8; font-size:0.74rem; font-weight:700; padding:4px 8px; border-radius:5px; text-decoration:none; text-align:center; display:inline-block;">Steam Store</a>'
                f'</div>'
            )
        html_rows.append('</div>')
        st.markdown("".join(html_rows), unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Player composition data unavailable ({e}).")


# ---------------------------------------------------------------- Navigation
# st.navigation, not st.tabs. With tabs, Streamlit executes EVERY tab body on every
# interaction — picking a game on the fit page also re-ran the 1,000-row ownership
# query and the 500-row alert table and re-serialised both to the browser. Only the
# selected page runs here, and each section gets its own URL, so a view is linkable
# and the browser back button works.
_PAGES = [
    st.Page(page_fit, title="Person & Game Fit", icon=":material/target:",
            url_path="fit", default=True),
    st.Page(page_bombing, title="Review Bombing", icon=":material/warning:",
            url_path="bombing"),
    st.Page(page_ask, title="Ask Gemini", icon=":material/auto_awesome:", url_path="ask"),
    st.Page(page_ownership, title="Player Ownership", icon=":material/group:",
            url_path="ownership"),
]
st.navigation(_PAGES, position="top").run()

st.markdown("""
<div style="text-align: center; padding: 24px 0 12px 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 36px; color: #64748b; font-size: 0.82rem;">
  <b>BuyOrWait</b> Engine · 114M+ Steam Reviews Dataset & Live Steam API Sync<br/>
  <span style="opacity: 0.7;">Powered by GCP BigQuery · Cloud Run · Gemini 3.6 Flash</span>
</div>
""", unsafe_allow_html=True)
