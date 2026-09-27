# Slideshow Outline — "Learning Who Will Engage: A Deliberation-Map Recommender"

Target: research presentation walking through the problem → diagnosis → redesign → results.
Each slide lists its key visual (all screenshots come from the running app or the DB).

---

### Slide 1 — Title & system recap
- Multi-agent deliberation map: 8 specialist personas + facilitator + synthesizer build a
  tree (mission → stakeholders → goals → barriers → causes → solutions …).
- A recommender proposes (post, action) pairs; agents engage or **skip**.
- Research question: can the recommender **learn each agent** well enough that skips decrease?
- Visual: map-tree screenshot + agent avatars.

### Slide 2 — The metric: skip rate over time
- Skip rate = skipped / decided recommendations, per strategy.
- NEW: sliding-window timeline (windows 1–10, 2–11, 3–12…) — smooth, readable trends.
- Visual: `Skip Rate over Time` chart (Analytics page).

### Slide 3 — Baseline strategies degrade
- Head-to-head (ab_rotate, same tree): as the map saturates,
  **priority → 100% skip, random → 67%**; multi_agent holds ~44–50%.
- Takeaway: without learning, recommendations rot as the easy gaps fill.
- Visual: 3-line skip chart, session `8396c216`.

### Slide 4 — Diagnosis: WHY agents skip
- Mined the elicited skip reasons: ~½ "outside my expertise / not my focus",
  ~½ "already covered". Action type is NOT predictive (all actions ~40–55% skip).
- Conclusion: the signal is **which post** (its topic), not which action.
- Visual: skip-reason quotes + per-action skip-rate table.

### Slide 5 — Feedback loop v1: semantic fit
- Learn engaged-vs-skipped centroids per agent from the live clickstream; rank candidates by
  fit + own-frontier boost + saturation damping + skip penalty.
- Effect: early skip drops ~50% → ~30–42% fast… then **plateaus** (~42% flat over 50 ticks).
- Visual: before/after skip curves (sessions `7f59bc8a`, `c32d22c9`).

### Slide 6 — The ceiling: agent inconsistency
- A skip predictor is only as good as agents are consistent; free-LLM agents give the same
  post different decisions → irreducible noise floor.
- Fix: **synthetic ground-truth personas** — each agent gets hidden liked/disliked topic
  centroids; engage iff closest liked > closest disliked. Deterministic, no LLM in the decision.
- Bonus: a ground truth to score inference against.
- Visual: liked/disliked topic lists for one agent + decision rule diagram.

### Slide 7 — New user model: a learned decision tree
- Replaces the static prompt-made persona table as the predictor.
- Pipeline: log every (post embedding, engage/skip) → PCA (3072→50) →
  per-agent DecisionTreeClassifier (entropy criterion, min_samples_leaf) → retrain every 5 ticks.
- All events persisted → retraining redistributes them automatically; no legacy-table lock-in.
- Leaves = data-discovered subtopics, LLM-labeled post-hoc from member posts (readable).
- Visual: pipeline diagram + "Decision-tree user model" panel screenshot.

### Slide 8 — Closing the loop
- The tree's predicted engage-probability is now the recommender's primary ranking signal
  (tree > cosine fit > action-rate fallback), before the LLM selector chooses.
- Visual: architecture diagram (clickstream → tree → recommender → agent → clickstream).

### Slide 9 — Results: 50-tick synthetic run (session `14da4d73`)
- **Skip rate falls 40% → 17%** (slope −0.47 pts/tick, near-monotonic) — vs flat ~42% with
  inconsistent LLM agents. The system now demonstrably learns.
- **Inference accuracy avg 87.6%** vs hidden ground truth (100/100/91/90/84/60 per agent).
- Discovered leaves are readable & pure: e.g. TheEngineer — "Transit Accessibility & AI"
  (0% engage, n=21) vs "Decentralized Energy Hubs" (100% engage, n=20).
- Caveat to state: clickstream selection bias (train 98% vs GT 60% for one agent) — the
  recommender filters what the tree sees; exploration knob trades this off.
- Visual: the sliding-window skip curve + accuracy table + leaf listing.

### Slide 10 — Persona table vs decision tree (what changed conceptually)
- Before: 10 LLM-guessed headers, fixed at t=0, cosine matching, only splittable finer.
- After: behavior-learned tree, adapts every 5 ticks, measurable accuracy, readable post-hoc.
- Tradeoffs (honest): less stable across retrains; cold-start needs ~8 events; synthetic
  agents less naturalistic.
- Visual: side-by-side table.

### Slide 11 — What went right vs wrong (honest ledger)
- Right: decreasing skip curve (40→17%); measurable inference (87.6%); readable learned
  topics; baselines-vs-learner contrast; robustness (no more stalled runs).
- Wrong along the way: first learner optimized the wrong signal (action type — flat);
  LLM-agent inconsistency capped v2 at ~42%; a stalled API call wedged a run (→ timeouts);
  the same Gemini thinking-budget bug struck a third time (degenerate 1-post run).
- Standing tradeoffs: clickstream **selection bias** (train 98% vs GT 60% for one agent —
  the loop partly grades its own homework); tree shifts on retrain; synthetic agents are
  consistent but less human; cold start needs ~8 events.
- Visual: two-column right/wrong table.

### Slide 12 — Limitations & next steps
- Single-run noise; average N runs per condition for publication figures.
- Naturalistic (LLM-judgment) agents still cap prediction — quantify their consistency.
- Entropy-vs-sample-size split tuning; dimensionality-reduction sweep; stable post-hoc
  hierarchy across retrains.

### Slide 13 — Summary
- Skip diagnosis → topic-fit learning → consistency fix → measurable, adaptive user model.
- The recommender now demonstrably learns: baselines degrade, learned model holds/improves,
  and inference accuracy against ground truth is directly measurable.
