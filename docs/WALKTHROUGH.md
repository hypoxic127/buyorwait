# BuyOrWait — Complete Walkthrough

Two audiences, one document:

* **Part A — System walkthrough**: what runs where, in the order it runs. Read this to understand or rebuild the project.
* **Part B — Live demo script**: the exact click path, what to say, what to do if something is slow. Read this ten minutes before presenting.

Live app: https://buyorwait-1047454501331.asia-southeast1.run.app

---

# Part A — System walkthrough

## A0. The idea in one paragraph

Steam's rating answers "is this game good?". It cannot answer "is this game good **for me, today**?" — it blends 2015 opinions with this week's, hides games that were fixed or bombed since, and knows nothing about your schedule, hardware or taste. BuyOrWait separates the two questions: a **Purchase Confidence Score** (playtime-weighted, 90-day half-life, evergreen) for game quality, and a **Personal Fit Score** for the collision between that game and one person's life. Everything the model asserts is traceable to reviews it will quote on demand.

## A1. Data in

| Source | What it gives | How |
|---|---|---|
| Kaggle `kieranpoc/steam-reviews` | 114,381,811 reviews through 2023-10-30 (17 GB CSV) | one-off download, `aria2c -x16` |
| Steam Web API `appreviews` | reviews since the snapshot, ~1.9k tracked games | `fetch_recent.py`, nightly |
| Steam Web API `appreviews` (live) | today's rating for any appid, incl. post-snapshot releases | called from the app at view time |

`convert_to_parquet.py` drops the review text and keeps only the metric columns → 17 GB becomes ~3.4 GB of Parquet in Cloud Storage, which is what makes a 24 GB GPU enough for a 114 M-row job.

## A2. The batch pipeline (`pipeline/pipeline.py`) — where the GPU earns its place

One script, two launchers, **zero code changes**:

```bash
python  pipeline.py cpu                 # pandas baseline
python -m cudf.pandas pipeline.py gpu   # RAPIDS on the L4
```

Six timed stages: `read_parquet → clean_cast → daily_groupby → weighted_score → bomb_detect → write_outputs`, timings appended to `benchmark_results.csv` on every run. Same GCE `g2-standard-8` box for both: **38.32 s → 2.75 s, ~14× end-to-end**, with `daily_groupby` at 35.7× and `bomb_detect` at 30.7×.

Two implementation notes that are worth defending out loud:

* **RMM pool allocator is initialised before pandas is imported.** Doing it after leads to allocator mismatch and heap corruption under cuDF.
* **Everything is GPU-native arithmetic.** An earlier version computed review age with `timedelta.dt.days` and did a second `groupby` + `merge`; both fell back to CPU and dragged 114 M rows across the bus, making the GPU *slower* on that stage. Replacing them with an integer day index and a single fused aggregation is where most of the 14× came from. The lesson — measure per stage, not end-to-end — is the honest story of this benchmark.

Outputs: `out_game_daily.parquet`, `out_game_scores.parquet`, `out_alerts.parquet`.

## A3. Evergreen scores — the trick that keeps a 2023 snapshot current

The decay is exponential in age only, so `daily_groupby` stores **undecayed** per-day weight sums (`w_sum`, `wv_sum`) and the decay is applied at *query* time:

```sql
score(today) = SUM(wv_sum · EXP(-(today - day)/90)) / SUM(w_sum · EXP(-(today - day)/90)) · 100
```

Consequences: history is never rewritten, new days are pure appends, and the score changes every day without recomputation. `docs/evergreen.sql` builds `v_daily_all` (snapshot + delta, delta wins on overlap), `v_scores_live`, `v_freshness`, and ships an **alignment check** — anchored at the snapshot's own last day the SQL must reproduce the pipeline's numbers. Measured: `max_abs_diff = 0.20` on 100-point scores, i.e. float32-vs-float64 summation, not a logic difference.

## A4. Enrichment layers (each writes exactly one BigQuery table)

| Script | Table | What it does |
|---|---|---|
| `fetch_recent.py` | `daily_delta` | nightly Steam API top-up. Idempotent `MERGE` on (appid, day); resumes from each game's own high-water mark; jittered backoff on 429; staging table name carries `CLOUD_RUN_TASK_INDEX` + a run UUID so parallel tasks cannot collide; partial trailing day discarded so it can't depress a z-score |
| `embed_index.py` | `review_vectors` | top-300 games (balanced positive/negative, ranked by helpful votes) + every bombed game's negatives ≈ 300 k reviews → 384-d `bge-small-en-v1.5` embeddings **on the L4 via CUDAExecutionProvider** (~5 min; CPU is ~an hour), plus an IVF/COSINE vector index |
| `attribute.py` | `alert_causes` | for each bombing episode, distinctive-term extraction (episode negatives vs a 60-day baseline, lift × √frequency) then Gemini writes one sentence of cause. Historical text comes from one chunked pass over the raw CSV; post-snapshot episodes are pulled from the API |
| `composition.py` | `game_composition` | per game: direct-purchase / free-key / Early-Access share of reviews. Standalone scan, touches nothing else |
| `docs/alerts_live.sql` | `alerts_recent`, `v_alerts_all` | the same bombing rule re-expressed as a BigQuery window query over merged data, so post-snapshot events are detected too; scheduled daily 03:30 SGT |

## A5. Orchestration & serving

```
Cloud Scheduler 03:00 ─▶ Cloud Run Job (fetch_recent.py) ─▶ daily_delta
                                                              │
BigQuery scheduled query 03:30 ─▶ alerts_recent ──────────────┤
                                                              ▼
                          v_daily_all / v_scores_live / v_alerts_all
                                                              │
                                     Cloud Run (Streamlit) ◀──┴──▶ Looker Studio
                                     scale-to-zero, 2 GiB
```

Three caching layers, by design: `st.cache_data` (10 min) in front of BigQuery's own result cache, in front of the real one — **pre-aggregation**. The app never touches 114 M rows; it reads a few thousand aggregate rows, or one game's vectors. That is why a scale-to-zero container answers in under two seconds.

Security: no service-account JSON anywhere. The VM runs with `--scopes=cloud-platform`, Cloud Run and the Job use the runtime service account, Gemini goes through Vertex AI with no API key. Generated SQL is regex-guarded to `SELECT`/`WITH` only and capped at 1 GB scanned; Gemini calls are rate-limited server-side per client.

## A6. The app, page by page (`app/app.py`, `st.navigation`)

`st.navigation`, not `st.tabs` — with tabs, Streamlit re-runs *every* tab body on *every* interaction, so picking a game also re-ran the 1,000-row ownership query. Each page now has its own URL and only the visited page executes.

**Sidebar — Your Life Profile.** Life rhythm, hardware, play style, purchase strategy, weekly hours, and up to eight dealbreakers. Session-only, no login, no account data.

**1. Person & Game Fit** (default page)
* Purchase Confidence (game quality) and Personal Fit (for you) are shown as **two separate numbers** — "great game, wrong game for you" has to stay visible.
* Fit is Confidence adjusted by factors computed from data, each rendered with its reason: time-to-value (median hours-to-enjoyment ÷ your weekly budget), early drop-off (share of unhappy players who quit inside the 2-hour refund window, weighted harder for short-session players), hardware-specific complaint density, play-style and purchase-strategy effects. **Any selected dealbreaker sitting at High Risk caps the result at 35.** The weights are transparent judgement calls, not fitted parameters, and the card says so.
* Sub-tabs: **AI Fit Analysis** (Gemini, grounded in this game's retrieved reviews) · **Friction Radar** · **Time & Trend** · **Live Check** (Steam right now).

**2. Friction Radar** (inside Fit)
Eight dimensions — Performance, Gameplay, Story & Content, Visuals, Audio, Price & Value, Dev Support, Usability — scored by semantic search over the game's negative reviews, then graded **against each dimension's own distribution across every indexed game**. This is the part reviewers should push on: measured over the corpus, some themes draw complaints in two thirds of all games and others in under 5 %, so one flat threshold grades every game identically. High Risk = ≥ 90th percentile **and** ≥ 1 % of that game's reviews; the floor stops a rare theme flagging on noise.

**3. Review Bombing**
Episode-level rows (multi-day alerts collapsed per game), z-score, peak volume, baseline, and **"why bombed (AI)"** — the Gemini cause line. Selecting a row charts the event. Small-sample noise is filtered by a minimum-reviews slider; z is capped for display at 99.9.

**4. Ask Gemini** — two scopes
* *All 114 M reviews*: NL → BigQuery SQL over the aggregate tables. Generated SQL is shown, guarded, and cost-capped; results auto-chart.
* *Single game*: RAG. Query embedded in-process on CPU (~50 ms, same model as the corpus) → `VECTOR_SEARCH` over that game's vectors → Gemini answers **only** from retrieved reviews, with `[n]` citations and the evidence expandable underneath. It refuses questions the data cannot answer instead of guessing.

**5. Player Ownership**
Per game: direct-paid / free-gift-key / Early-Access composition, as bars and a 100 % stacked strip. Free-key share is the manipulation signal behind the noise-defense story. Every KPI on the page is queried, not typed.

## A7. Noise & manipulation defence (the part that is easy to miss)

* **Playtime weighting** — `log(1+playtime)` demotes drive-by and zero-hour reviews.
* **90-day half-life** — a 2015 brigade cannot dominate today's score.
* **Dual bombing condition** — rate anomaly *and* volume anomaly, so a quiet game with three bad days doesn't trip.
* **Percentile-relative radar grading** — prevents "every game is bad at performance".
* **Free-key share** surfaced per game — astroturf risk made visible rather than silently trusted.
* **Refund-window drop-off** — separates "hated it after 90 hours" from "bounced in 20 minutes".
* **Text sanitisation** — Kaomoji/ASCII art/BBCode stripped before anything is quoted or embedded.

## A8. Cost & scale posture

Idle cost is effectively zero: Cloud Run scales to zero, BigQuery is serverless and reads aggregates, the GPU box is off unless a batch is running (~10 min/night ≈ US$0.2/day). Growth path, in order: the batch has 10× headroom at 2.75 s; the corpus can be re-indexed in minutes; beyond that, Dataproc Serverless + Spark RAPIDS is the documented next step, and vector search moves to a managed store only when the corpus outgrows a per-game filter.

## A9. Rebuild from scratch

`README.md` → *Reproduction* has the commands in order (0-8). One correction to the order: run `docs/evergreen.sql` **before** `docs/alerts_live.sql` (the latter reads `v_daily_all`), and run both **before** deploying the app — the app degrades gracefully on a missing table, but there is no reason to show a judge a degraded page.

---

# Part B — Live demo script

**Target: 6–8 minutes** of driving, with the 3-minute video as the spine. Skip Part A's vocabulary; show, then explain in one line.

## B0. Ten minutes before

1. Open the app in an **incognito** window, browse every page once — this warms Cloud Run and fills the BigQuery/`st.cache_data` caches so nothing spins during the demo.
2. Confirm the header pill reads **"Live sync · Nh ago"**. If it says snapshot-only, the nightly Job didn't run — say "the nightly sync window is 03:30 SGT" and move on; do not debug live.
3. Have a second tab on the GitHub repo, a third on Looker Studio.
4. Sidebar preset ready: **Life rhythm = Busy Professional, Hardware = Steam Deck, Hours = 6, Dealbreakers = Performance + Price & Value**. That profile produces a visibly different verdict from the crowd score, which is the whole point.
5. Pick your two demo games in advance: one **well-reviewed but demanding** title (fit collides with the profile) and **Overwatch 2 / Cyberpunk 2077** for the bombing story. Verify both render before you present.

## B1. The hook (45 s) — Person & Game Fit

Do: sidebar profile already set → pick the demanding game.

Say: *"Steam says this game is 92 % positive. That's the answer to 'is it good'. My question is 'is it good for me' — six hours a week, on a Deck. The engine keeps those apart: quality on the left, fit on the right, and every adjustment between them is itemised with its reason — time to value, early drop-off, hardware complaints. Nothing here is a black box."*

Point at one factor line and read it out. That single line is the difference between a dashboard and a decision tool.

## B2. Evidence, not vibes (60 s) — Friction Radar → AI Fit Analysis

Do: open **Friction Radar**, hover a High-Risk dimension, then open **AI Fit Analysis**.

Say: *"Eight dimensions, scored by semantic search over this game's negative reviews — not keyword counting. And graded against each dimension's distribution across the whole corpus, because complaints aren't uniformly common: some themes appear in two thirds of all games, others in under five percent. A flat threshold would grade every game identically. High Risk means top decile **and** at least one percent of this game's reviews."*

Then, on the AI panel: *"Gemini writes this from reviews we retrieved for this game — it quotes them, and it declines when the data can't answer."*

## B3. The GPU payoff (60 s) — the number that only exists because of RAPIDS

Do: switch to the repo's benchmark table (or your PPT benchmark slide).

Say: *"Same machine, same script — the only change is the launcher, `python -m cudf.pandas`. Eight vCPUs: thirty-eight seconds. The L4: two point seven. Fourteen times end-to-end, thirty-five on the groupby. That's what makes a nightly full recompute of 114 million reviews affordable — about twenty cents a night — and that's why the header says 'live sync', not 'snapshot'."*

If asked how you got there: the per-stage story from A2 (the timedelta fallback that made the GPU slower until it was removed). Judges reward that answer.

## B4. Why, not just what (60 s) — Review Bombing

Do: Review Bombing → point at a famous row → read the **why bombed (AI)** cell → click the row to chart it.

Say: *"Detection is a z-score on the negative rate plus a volume condition. But detection alone isn't useful to a developer — so for every episode we pull that window's negative reviews, extract what's distinctive versus the game's own 60-day baseline, and have Gemini name the cause. This one: PVE cancelled, monetisation, hero locking. Automatically, from tens of thousands of reviews."*

## B5. Ask it anything (45 s) — the risky-but-worth-it part

Do: *All 114 M reviews* scope, click a **prepared example** question (never improvise here), show the generated SQL expander, then switch to *Single game* and ask a review-grounded question.

Say: *"Two retrieval paths behind one box. Analytical questions become BigQuery SQL — shown, validated read-only, cost-capped. Questions about experience go to vector search over the review text and get answered with citations."*

## B6. Close (30 s)

Say: *"Pipeline on the L4, aggregates in BigQuery, embeddings and vector search in BigQuery too, Gemini on Vertex, orchestrated by Scheduler and Cloud Run Jobs, served scale-to-zero. Idle cost is effectively nothing. The point isn't the stack — it's that a hundred and fourteen million opinions became one decision, for one person, that can defend every number it shows."*

## B7. Answers to the five questions you will get

| Question | Answer |
|---|---|
| "Where do the fit weights come from?" | Transparent judgement calls, stated as such in the UI, applied to data-derived quantities (median hours, drop-off share, complaint percentiles). Not fitted — and we show every adjustment so a user can disagree with one. |
| "Isn't the 2023 snapshot stale?" | The score is anchored to *today* by construction (A3), and the ~1.9k most-reviewed games are topped up nightly from the Steam API. Older reviews decay to near-zero weight at a 90-day half-life. |
| "How do you know the bombing detector works?" | Dual condition against each game's own rolling baseline, validated against publicly documented incidents; z is a severity, not a label, and the UI lets you raise the threshold and the minimum sample. |
| "Why RAPIDS instead of just BigQuery / more CPU?" | The weighting and rolling-window work is iterative dataframe logic, not a single SQL shape; RAPIDS runs the exact same pandas code 14× faster on one $1/hr box. Where SQL is the better tool — the nightly alert recompute — we use SQL. |
| "Why not a managed vector DB?" | 300 k vectors, already sitting in BigQuery, always filtered to one game. `VECTOR_SEARCH` costs no new service and no idle spend. Beyond a few million vectors, that trade flips. |

## B8. If something breaks on stage

* **Page slow / cold** — keep talking over it; the pre-warm in B0 makes this unlikely.
* **Gemini errors or rate-limits** — say "the AI calls are rate-limited per client on purpose" and pivot to the Friction Radar, which needs no LLM.
* **A game isn't in the semantic index** — the index is top-300 + every bombed game; switch to a prepared title.
* **Total failure** — fall back to the 3-minute video and the screenshots in `docs/`. Have both open before you start.
