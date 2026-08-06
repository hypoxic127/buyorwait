# Refinement Sprint — What Changed in the Final Week

BuyOrWait entered the final round of the Google Cloud Gen AI Academy APAC 2026
ranked 94th of 100. This is the engineering record of the week that followed:
what was decided, what shipped, what was deliberately not built, and why.

Outcome: **Performance Excellence Award — Best Use of NVIDIA Tools**.

---

## The diagnosis

The Top-100 submission was a good demo of a dead dataset. Every number came from a
2023 snapshot, the score was the crowd's opinion rather than the user's, and the
review-bombing detector could say *that* a game was bombed but never *why*.

One sentence framed the whole week:

> The previous phase measured the game. This phase measures **you and the game**.

Everything below is a consequence of that sentence.

---

## Priorities, in the order they were fixed

### P0 — Make the data live

The most damaging weakness was not a missing feature; it was that nothing moved.
A judge opening the app two weeks after submission would see identical numbers.

**Evergreen scores.** The playtime weight decays exponentially with review age, and
exponential decay in age *only* has a useful property: the decay can be applied at
query time instead of at compute time. The daily aggregation now stores **undecayed**
per-day weight sums, and BigQuery applies `EXP(-age/90)` against `CURRENT_DATE()`
when the score is read. History is never rewritten, new days are pure appends, and
the score changes daily with zero recomputation.

Verification mattered here: anchored at the snapshot's own last day, the SQL must
reproduce the original pipeline numbers. Measured drift was `max_abs_diff = 0.20`
on a 100-point scale — float32 vs float64 summation, not a logic difference.

**Incremental ingestion.** `fetch_recent.py` tops up the ~2,000 most-reviewed games
nightly from the Steam Web API, via Cloud Scheduler → Cloud Run Job. The MERGE is
idempotent on `(appid, day)`, each game resumes from its own high-water mark, 429s
back off with jitter, and the partial trailing day is discarded so it cannot depress
a z-score. A BigQuery scheduled query re-runs the bombing rule over merged data at
03:30 SGT, so events occurring after the snapshot are still detected.

**Visible proof.** A freshness badge in the app header reads `LIVE — last sync Nh ago`,
derived from the ingestion table itself. The claim "this is alive" had to be checkable
in one glance, not asserted in a slide.

### P0 — Answer "why", not just "what"

For every bombing episode, negative reviews inside the window are compared against
the game's own 60-day baseline; distinctive terms are ranked by lift × √frequency,
and Gemini turns them into one sentence of cause. Overwatch 2's August 2023 episode
resolves to *PVE cancelled, monetisation, hero locking* — extracted automatically
from tens of thousands of reviews.

Feasibility was checked before the feature was scheduled: one hour of manual term
extraction on a known event, to confirm the output was specific complaints and not
stopwords. It was, so the feature was built. Had it returned generic vocabulary, the
fallback was a plain Gemini summary.

### P0 — Measure the person, not just the crowd

Purchase Confidence (game quality) and Personal Fit (fit for one person) are computed
and displayed as **two separate numbers**, never blended, so "great game, wrong game
for you" stays visible. Fit adjusts Confidence by quantities derived from data —
time-to-value, refund-window drop-off, hardware complaint density — and every
adjustment is rendered next to the verdict with its reason. A selected dealbreaker at
High Risk caps the result outright.

The weights on top of those quantities are judgement calls, and the UI says so. That
admission is the point: a user is allowed to disagree with one line without discarding
the answer.

### P1 — Retrieval and vector search

The reviewer checklist for the round explicitly named retrieval, vector search,
orchestration, caching and scalability. Vector search was the one genuine gap.

~300k reviews — the top-300 games balanced across polarity, plus every bombed game's
negatives — were embedded with `bge-small-en-v1.5` **on the L4 via CUDAExecutionProvider**
(~5 minutes; roughly an hour on CPU) and written to BigQuery with a vector index. The
corpus was deliberately scoped: index the text that matters for decisions, not all
114M rows. One embedding pass then powered three features — the Friction Radar,
grounded RAG in Ask Gemini, and the per-dealbreaker risk lookups behind Personal Fit.

Query-side embedding runs on CPU inside Cloud Run in ~50 ms, using the same model, so
queries and corpus share one vector space without a second service.

### P1 — Grading that survives scrutiny

Flat risk thresholds were replaced with percentile-relative grading. Measured across
the corpus, some complaint themes appear in two thirds of all games and others in
under 5%; a single threshold grades every game identically. A dimension is High Risk
at ≥90th percentile **and** ≥1% of that game's reviews — the floor stops a rare theme
from flagging on noise.

### P2 — Production signals

Server-side AI rate limiting per client, a read-only regex guard plus a 1 GB scan cap
on generated SQL, keyless IAM throughout (no service-account JSON anywhere), review
text sanitised before it is quoted or embedded, Cloud Monitoring uptime and job-failure
alerts, and a CI build that runs a CodeMender audit against the project's own
conventions file.

---

## Decisions not to build

Recorded because the reasoning is the interesting part:

| Not built | Why |
|---|---|
| Managed vector database | 300k vectors already live in BigQuery and every query filters to one game. No new service, no idle spend. Past a few million vectors the trade flips. |
| GKE | One stateless container. Cloud Run's scale-to-zero is the correct shape; idle cost is effectively zero. |
| Pub/Sub streaming | Reviews are decided on at day granularity. A nightly batch plus query-time decay already meets the product need. |
| XGBoost forecasting | Cut in favour of vector search — the reviewer checklist named retrieval explicitly, forecasting only implicitly. Middling feature, no memorability. |
| Steam login / library integration | Needs an account system. Out of scope for one week; documented as roadmap instead. |
| TensorRT / Triton | Online inference is one query embedding at ~50 ms on CPU. No serving bottleneck to solve. |

---

## The benchmark, honestly

The first GPU run was **slower** than the CPU baseline. Two operations were silently
falling back to host memory and dragging 114M rows across the bus: review age computed
as a `timedelta` (`.dt.days`), and a second `groupby` + `merge` for the recent-window
statistics. End-to-end timing hid it — the win in one stage paid for the loss in another.

Timing every stage separately exposed it. Replacing the timedelta with an integer day
index and fusing the two aggregations into a single pass produced most of the final
**38.32s → 2.75s (~14x)**. The other prerequisite: RMM's pool allocator must be
initialised *before* pandas is imported, or cuDF and the default allocator contend for
the same memory and the process corrupts its own heap.

A GPU does not make code fast. Measuring per stage — and finding where the GPU *isn't
being used* — does. That finding is what the NVIDIA award recognised.

---

## Sequencing that made a one-week sprint survivable

- **Ship the dependency first.** The embedding pass gated four downstream features, so
  it went early; anything depending on it could be cut without cascading.
- **Verify before scheduling.** Attribution quality was checked in an hour before a day
  was committed to it.
- **Tag every night.** `git tag day-N` meant any evening's work could be abandoned
  without risking a submittable state.
- **Freeze features before the deadline, not at it.** The last two days were materials
  only — deck, video, verification. Judges spend more time on those than on the app.
- **Degrade, don't break.** Every new BigQuery object is wrapped so a missing table
  hides a section rather than crashing the page. This was learned the hard way: twice,
  a deploy shipped ahead of its schema.

---

## What the week produced

| | Before | After |
|---|---|---|
| Data | Static 2023 snapshot | Evergreen scores + nightly Steam sync |
| Scoring | Crowd score only | Personal Fit with itemised, auditable adjustments |
| Alerts | Detection only | Gemini root-cause attribution per episode |
| Retrieval | None | ~300k GPU-built embeddings + BigQuery `VECTOR_SEARCH` |
| Risk grading | Flat thresholds | Percentile-relative across the whole corpus |
| Interface | Prototype | Production UI, mobile pass, native theming |
| Hardening | None | Rate limits, SQL guard, idempotent MERGE, keyless IAM, alerting |
