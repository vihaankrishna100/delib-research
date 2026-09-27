# Deliberation Map Sim — Changes Log & Measured Effects

Session date: 2026-07-28. Backup snapshot before the decision-tree work:
`~/deliberation-map-sim-backup-20260728-205942` (full source + DB, revert anytime).

---

## Phase 1 — Analytics & persona-table features

### 1.1 Skip-rate graph (bar → line → sliding window)
- **Change:** New endpoint `GET /analytics/sessions/{id}/skip_rate_timeline?window=W&step=S`
  (`backend/routes/analytics.py`). Buckets completed recommendations per strategy into tick
  windows; `step < window` gives an overlapping SLIDING window (1–10, 2–11, 3–12…).
  Frontend `SkipRateTimelineChart` (SVG multi-line, one line per strategy) on the Analytics page.
- **Effect:** The key paper figure. Sliding window smooths the noisy per-bucket rates so trends
  are readable. "forced" bootstrap actions are excluded.

### 1.2 Deep persona topic headers (prompt-only)
- **Change:** `generate_persona_topics` now returns ~10 top-level headers **each with 2–3
  sub-headers** in one persona+mission prompt (`simulation/persona_topics.py`).
  `persona_topic_headers.parent_id` added (self-FK). "split" reworked to **deepen** (adds
  children, keeps parent). Posts match to the nearest **leaf** header and roll up.
- **Constraint honored:** headers are created ONLY by the prompt, never from simulation activity.
- **Effect:** Finer-grained persona table; two-level tree view per agent in User Models.

### 1.3 Per-move action scores
- **Change:** `persona_topic_headers.action_scores` (JSON). LLM predicts, per header, how likely
  the agent is to add each deliberation **move**: goal, barrier, cause, solution, abstraction,
  analogy, inspiration, question, answer, pro, con. Observed per-move counts come from
  `recommendation_logs` (`ACTION_CATEGORY_MAP`). Endpoint `POST /score_persona_actions`.
  UI: `ActionScoreBars` under every header (predicted bar + observed count).
- **Effect (measured):** e.g. TheEconomist → "Land Value Capture": solution 0.90, pro 0.90,
  barrier 0.80 — sensible persona-consistent predictions.

## Phase 2 — Making the recommender learn

### 2.1 Diagnosis from data
- Pulled real skip reasons: agents skip on **post/topic fit** ("not my expertise",
  "already covered"), NOT on action type (per-action skip rates all ~40–55%, non-discriminative).
- **Conclusion:** the learning signal must be about WHICH POST is recommended.

### 2.2 Closed feedback loop (LLM agents)
- **Change:** `recommend_multi_agent` now ranks candidates by, in priority order:
  learned **semantic fit** (cosine of candidate post vs centroids of the agent's engaged
  vs skipped posts, embeddings cached in `_POST_EMB_CACHE`), **own-frontier boost** (×1.6 own
  post, ×1.3 child of own post), **saturation** damping (1/(1+existing achievers)), and a
  **skip penalty** (0.35^skips on that post). Top-12 + 1 explore go to the LLM selector, whose
  prompt also shows the per-action engagement-rate table.
- **Effect (measured, LLM agents):**
  - Early-run skip dropped ~50% → ~30–42% (cold→warm step: the loop learns fast).
  - 50-tick run: skip flat ~42% (slope ≈ +0.06 pts/tick ≈ 0) — learning plateaus.
  - Head-to-head (ab_rotate): multi_agent held ~44–50% while **priority degraded to 100%**
    and **random to 67%** as the tree saturated. The gap IS the recommender working.
  - Limit found: LLM agents skip **inconsistently**, which upper-bounds any predictor —
    motivation for Phase 3.

### 2.3 Robustness
- **Change:** `LLM_TIMEOUT_S = 90` via `asyncio.wait_for` on every Gemini/embedding call
  (`roles/base.py`, `recommender.py`, `topic_tree.py`); semantic-fit call wrapped in
  try/except → falls back instead of failing.
- **Effect:** A single stalled Gemini request previously wedged a 50-tick run at tick 1 for
  10+ minutes; now it degrades to a caught exception and the run continues.

## Phase 3 — Decision-tree user model + synthetic ground truth

### 3.1 Synthetic ground-truth persona models
- **Change:** `simulation/ground_truth.py` + table `ground_truth_personas`. Per agent, once per
  session, the LLM produces ~10 LIKED and ~10 DISLIKED subtopics (mission+persona), embedded
  into centroids. In `agent_mode="synthetic"` the agent's engage/skip decision is deterministic:
  engage iff nearest liked centroid beats nearest disliked centroid. **No LLM in the decision**;
  content on engage is still LLM-written so post embeddings are real.
- **Why:** consistency. A skip predictor is only as good as the agents are consistent; synthetic
  agents make the boundary perfectly learnable AND give ground truth to score inference against.

### 3.2 Clickstream store
- **Change:** tables `clickstream_events` (session, agent, post, label 0/1, tick — every
  recommendation outcome, kept forever) and `post_embeddings` (post → vector, deduped).
  Recording wired into every outcome branch of `SpecialistAgent.decide` (both modes).
- **Effect:** retraining re-derives the whole model from raw events — automatic event
  redistribution when the hierarchy changes; no legacy-table problem.

### 3.3 The decision-tree model (replaces the persona table as predictor)
- **Change:** `simulation/user_model_tree.py`. PCA (3072→≤50 dims) + per-agent
  `DecisionTreeClassifier` (entropy criterion, `min_samples_leaf=3` against over-chopping).
  Retrains every 5 ticks + at session end. `predict_engage_prob` is now the recommender's
  **primary** ranking signal (tree > cosine fit > action rate). Leaves = data-discovered
  subtopics; each labeled post-hoc by an LLM from its member posts; snapshot persisted in
  `user_model_snapshots` {leaves, purity/entropy, n, accuracy}.
- **Inference accuracy:** tree predictions vs ground-truth labels over all session posts —
  the headline "how good is the inference engine" metric.
  Endpoints: `POST /train_user_model`, `GET /user_model_tree`, `GET /inference_accuracy`.
- **UI:** Analytics page — "Decision-tree user model" panel (per-agent accuracy, expandable
  leaves with engage%, entropy H, n). Simulation page — "Agent decision mode" selector.

### 3.4 Bug found & fixed during verification
- First synthetic run produced only 1 post: `_gen_content` used `call_llm`, which leaves Gemini
  **thinking ON** → 400-token budget eaten → truncated JSON → empty content → no posts (the
  documented Gemini gotcha, third occurrence). Fixed with a direct call using
  `thinking_budget=0`. Verified content generates correctly.

---

## Phase 3 results — 50-tick synthetic simulation

Session `14da4d73` — multi_agent strategy, synthetic agents, 6 specialists, 50 ticks,
7.7 min wall-clock, 150 posts created, 300 clickstream events (213 engage / 87 skip).

### Skip rate DECREASES (the goal, finally achieved)
Sliding window = 10 ticks, step = 2:

| window | skip | | window | skip |
|---|---|---|---|---|
| 0–9 | 39.7% | | 26–35 | 26.7% |
| 4–13 | 40.0% | | 30–39 | 21.7% |
| 8–17 | 35.0% | | 34–43 | 18.3% |
| 12–21 | 31.7% | | 38–47 | 21.7% |
| 16–25 | 30.0% | | 48–57 | 16.7% |
| 20–29 | 30.0% | | | |

**Early 40% → late 17%, slope −0.47 pts/tick, near-monotonic.** This is the paper figure:
with consistent (synthetic) agents and the decision-tree model retraining every 5 ticks, the
recommender demonstrably learns each agent's engage/skip boundary and skips fall by more than
half over the run. (Contrast: with free-LLM agents the same recommender plateaued flat at ~42% —
agent inconsistency was the binding ceiling, exactly as hypothesized.)

### Inference accuracy (tree vs hidden ground truth) — avg 87.6%
| agent | accuracy | train acc | note |
|---|---|---|---|
| TheEconomist | 100% | 100% | GT liked everything it was shown → constant model |
| TheInnovator | 100% | 100% | same |
| ThePsychologist | 91.4% | 100% | clean boundary recovery |
| ThePolicyMaker | 90.0% | 100% | clean boundary recovery |
| TheEngineer | 84.3% | 98% | good; leaves readable |
| TheEnvironmentalist | 60.0% | 98% | overfit to a biased sample — see caveat |

### Discovered leaves are readable and pure (LLM-labeled post-hoc)
e.g. TheEngineer: "Transit Accessibility & AI" (0% engage, H=0.00, n=21) ·
"Decentralized Energy Hubs" (100% engage, H=0.00, n=20) ·
"Advanced AI & Energy" (33%, H=0.92, n=3 — correctly flagged low-confidence).

### Honest caveats
- **Selection bias in the clickstream:** the tree trains only on posts the recommender chose to
  show — and the recommender increasingly shows only predicted-likes. High train accuracy (98%)
  with lower GT accuracy (60% for TheEnvironmentalist) is this bias made visible. Fixable with
  more forced exploration (the `explore` knob) at the cost of a slightly higher skip floor.
- Two agents' GT models liked everything they were shown, making their 100% trivial; the
  informative scores are the 60–91% range.
- Late windows (42+) have fewer samples because the run ends.

---

## Phase 4 — Anti-saturation moves + 70-tick validation

### 4.1 New idea-generator moves (`part`, `failure_mode`)
Previously a solution's only generative child was another sibling solution, which
exhausts fast — the "already covered" half of all skips. Added two AIdeator moves:
- **`add_part`** → decompose a solution into a required sub-component
- **`add_failure_mode`** → stress-test a solution for a concrete failure mechanism
Also aligned `_aideator_improvement` verbatim to the AIdeator Lisp template.
Wired through: `PostType`, `_BASE_ACTIONS`, `ACTION_TO_PTYPE`, `ACTION_DESCRIPTIONS`,
`describe_context`, `PTYPE_COLORS` (backend + frontend), the 13-move persona taxonomy.

### 4.2 Retry on transient API failure
A 70-tick run **died at tick 0** on a Gemini `503 UNAVAILABLE`. Timeouts existed but
no retry. Added `call_with_retry` (`roles/base.py`): 4 attempts, exponential backoff
(1.5→3→6s), transient classification (503/429/500/overloaded) vs immediate raise on
400/auth. Wired into all 5 call sites. Unit-tested both paths.

### 4.3 Results — 70 ticks × 4 agents, synthetic (session `a14a62d8`)
161 posts, 278 decisions, ~10 min.

| metric | 50-tick BEFORE (6 agents) | 70-tick AFTER (4 agents) |
|---|---|---|
| posts | 150 | 161 |
| solutions | 60 (**40%** of tree) | 41 (**25%**) |
| part + failure_mode | 0 | **32** |
| max depth | 11 | 12 |
| skip rate | 40% → 17% | **18.4% → 5%** (overall 6.8%) |
| inference accuracy | 87.6% | **90.0%** |

- **New moves are used and welcomed:** `add_part` 22 engaged / **4% skip**,
  `add_failure_mode` 17 engaged / 19% skip. Solution share of the tree fell 40% → 25%
  — the map diversified exactly as intended.
- **Skip rate stayed low and still declined** (18.4% → 5%, slope −0.105 pts/tick).
- **Inference accuracy 90%** (Innovator 100%, Environmentalist 97%, Engineer 90%,
  Economist 73%).

### 4.4 NEW PROBLEM FOUND: breadth collapsed into depth
The anti-saturation fix worked on *siblings* but revealed a different failure mode:
- **Solution breadth is ~1**: mean **1.08 solutions per parent** (max 2) — 35 of 38
  parents have exactly one solution. The map is a chain, not a fan.
- **Depth ran to 12 levels**, with `add_improvement` (which produces another *solution*)
  creating long improvement chains. The deepest path is 7 consecutive solutions:
  `Community Infrastructure → Microgrid → Energy Mgmt → Predictive Maintenance →
   Data Governance → Blockchain Trusts → DAOs → DAO Governance Paralysis → Hybrid Governance`
  — which has **drifted far from the originating barrier** (charging infrastructure).
- **Tree is lopsided**: 41 solutions but only **1 goal** and 5 stakeholders; `create_topic`
  fired 4× spawning 5 missions.

Interpretation: low skip rate is partly *earned* (better targeting) and partly an artifact —
depth-first drift always offers a fresh, never-seen node, so nothing looks "already covered."

### 4.5 Tree-shape controls (applied) and the trade-off they revealed
Root cause of the depth collapse: `saturation = 1/(1+existing)` penalised the SECOND
sibling immediately. Replaced with three knobs in `_rank_pairs_by_learning`:
`TARGET_BREADTH=3` (siblings free until 3 exist), `DEPTH_SOFT_CAP=6`/`DEPTH_DECAY=0.75`,
`MAX_SAME_TYPE_CHAIN=2`/`CHAIN_DECAY=0.35`.

70-tick re-run (session `1c81ba56`) — **shape fixed, engagement regressed:**

| metric | run A (no controls) | run B (controls) |
|---|---|---|
| solutions/parent | 1.08 | **2.75** |
| max depth | 12 | **6** |
| goals | 1 | **9** |
| overall skip | **6.8%** | 26.4% |
| skip trend | −0.105 pts/tick | **+0.045 (flat)** |

`add_part` 4% → 100% skip; zero parts created in run B. Because synthetic skip is decided
purely by hidden topic preference, forcing breadth means recommending *outside* each agent's
preferred region — the shape controls override the learned model rather than regularising it.
### 4.6 RESOLVED — shape as a bounded tie-breaker (run C, session `caa70027`)
The trade-off was an artifact of *how* shape was applied, not the knob values. Run B
multiplied the primary signal by the shape factors, letting shape **veto** a well-matched
candidate. Replaced with a bounded blend —
`score = engagement × (1 − SHAPE_WEIGHT + SHAPE_WEIGHT × shape)`, `SHAPE_WEIGHT = 0.35` —
so the worst shape keeps 65% of predicted engagement. Knobs softened to
`TARGET_BREADTH=2, DEPTH_SOFT_CAP=8, DEPTH_DECAY=0.85, CHAIN_DECAY=0.60`.

| metric | A no control | B veto | **C tie-breaker** |
|---|---|---|---|
| solutions/parent | 1.08 | 2.75 | **2.86** |
| max depth | 12 | 6 | **7** |
| longest same-type run | 8 | 1 | **4** |
| part + failure_mode | 32 | 6 | **28** |
| total posts | 161 | 166 | **194** |
| overall skip | 6.8%* | 26.4% | 19.2% |
| skip trend | 18→5% | flat | **29→10%** |
| slope (pts/tick) | −0.105 | +0.045 | **−0.573** |
| inference accuracy | 90.0% | 88.3% | **90.6%** |

*\*inflated by depth-first drift.* **C wins on nearly every axis** — best breadth, steepest
learning slope (5× A), best accuracy, most posts, and it rehabilitated the moves B had killed
(`add_part` 100%→12% skip, `add_improvement` 100%→10%). **C is the recommended default.**
Remaining wrinkle: solutions are 41% of C's tree (heavy `add_improvement`, n=61 @ 10% skip),
but with breadth 2.86 and max chain 4 that is fanning out, not A's drifting corridor.

---

## Good & bad ledger — what each change actually did

### 1. Skip-rate chart (bar → line → sliding window)
- **Good:** clean paper figure; sliding window finally makes trends legible; per-strategy
  comparison in one plot; backward-compatible endpoint.
- **Bad:** two reworks to get there (first build was a bar chart, then a bucketed line — both
  discarded). Statistical caveat: overlapping windows are autocorrelated, so the smooth curve
  has fewer independent samples than it visually suggests — fine for trend-reading, weak for
  significance claims.

### 2. Deep persona headers + per-move scores
- **Good:** finer persona structure; per-move predictions are persona-plausible (Economist →
  solution 0.90); observed-vs-predicted comparison per header; prompt-only guarantee kept.
- **Bad:** the action taxonomy was built wrong the first time (reply/react/refine/view instead
  of the deliberation moves) — full rework, and sessions scored under the old taxonomy showed
  all-0.00 bars until re-scored (stale-key confusion you hit directly). And ultimately the
  whole persona table was demoted by Phase 3: it's now display/cold-start only, so some of this
  effort is superseded by the tree.

### 3. Learning loop v1 — per-action engagement rates
- **Good:** established the online-learning plumbing (log mining, candidate narrowing) reused
  by everything after.
- **Bad:** optimized the wrong signal — per-action skip rates are non-discriminative (~40–55%
  everywhere), so skip stayed flat. Cost: two full simulation runs to discover it. Lesson
  banked: diagnose from data (skip reasons) before building the learner.

### 4. Learning loop v2 — semantic fit + own-frontier
- **Good:** real learning appeared: early-run skip fell ~50% → ~30%; own-frontier routing
  killed the late-run blow-up; head-to-head showed baselines rotting (priority 100%, random
  67%) while multi_agent held — a genuine, presentable contrast.
- **Bad:** plateaued at ~42% and never decreased on a full run — LLM-agent inconsistency is an
  irreducible noise floor this approach cannot beat. One run also *rose* late (56%) before the
  own-frontier fix. Several expensive runs spent proving the plateau was real.

### 5. Timeout hardening
- **Good:** no run has stalled since; a hung Gemini call degrades to one skipped decision.
- **Bad:** the reason it exists — a 50-tick run silently wedged at tick 1 for 10+ minutes,
  wasting a full run + diagnosis time. 90s cap also means a genuinely slow-but-good response
  is dropped rather than awaited.

### 6. Synthetic ground-truth agents
- **Good:** perfectly consistent skip behavior — the single change that unlocked the decreasing
  curve; ground truth makes inference *measurable* for the first time (87.6% avg); runs are
  faster (no LLM in the skip decision).
- **Bad:** agents are less human — skip decisions are a geometric rule, not judgment; results
  from synthetic mode don't automatically transfer to naturalistic agents (the paper must say
  so). Two agents' GT models liked everything they were shown, making their 100% accuracy
  trivial. The liked/disliked lists are still LLM-authored, so "ground truth" quality depends
  on that one generation call.

### 7. Decision-tree user model
- **Good:** the headline — skip 40% → 17% within one run; adapts every 5 ticks instead of
  being frozen at t=0; leaves are pure, sensibly labeled, and quantified (H, n); full event
  log means retrains redistribute history automatically; accuracy is now a number, not a vibe.
- **Bad:**
  - **Selection bias, measured:** the tree only trains on posts the recommender chose to show,
    and the recommender increasingly shows predicted-likes — TheEnvironmentalist hit 98% train
    accuracy but only 60% ground-truth accuracy. The loop partly grades its own homework.
  - Model shifts on every retrain (leaf labels/boundaries move) — less stable for longitudinal
    UI reading, exactly the con flagged in planning.
  - Cold start: nothing to rank with until ~8 events/agent (falls back to cosine fit).
  - First integration run was degenerate (1 post, meaningless 100% accuracy) due to the
    thinking-budget bug — a third recurrence of the same Gemini gotcha.
  - New deps (numpy/sklearn) and an in-memory model cache that resets on backend restart
    (snapshots persist, live models retrain on demand).

### Cross-cutting costs
- ~7 full simulation runs (~1 hr wall-clock + Gemini API spend) to diagnose, fix, and verify.
- The old persona-table UI and the new tree panel now coexist — two "user model" views until
  one is retired.

---

## Files touched (whole session)
- Backend: `models.py`, `database.py`, `schemas.py`, `routes/analytics.py`,
  `simulation/{persona_topics,recommender,engine,ground_truth,user_model_tree,topic_tree}.py`,
  `simulation/roles/{base,specialist}.py`
- Frontend: `lib/api.ts`, `app/{analytics,user-models,simulation}/page.tsx`
- Deps added: `numpy 2.5.1`, `scikit-learn 1.9.0`

---

## Phase 5 — Short answers + learning-speed attempt (2026-09-05)

### 5.1 Brief answers (SHIPPED, verified)
Agents now write like real users filling a web form, not essayists. Added a `BE BRIEF`
directive at prompt-assembly level (so the AIdeator/Lisp templates stay faithful) plus a
`brief_answers` config flag (default `True`) threaded config → engine → `decide` →
both prompt builders, with matching token caps.

| | before | after |
|---|---|---|
| median words / post | 218 | **46** |
| median sentences / post | 10 | **2** |
| sec / tick | 8.1s | **7.1s** (~12% faster) |

This is a clean win: 4.7x shorter, on-spec (2-3 sentences), and cheaper per tick.

### 5.2 Learning-speed knobs (SHIPPED, effect UNPROVEN)
- `MIN_EVENTS_TO_TRAIN` 8 → 4 (tree activates sooner)
- Adaptive retrain: every 2 ticks for the first 20, then every 5
- Exploration decay: 3 → 2 → 1 → 0 extra candidates as decisions accumulate

Measured drop speed: 50% of the total drop by **tick 3-4**, vs tick 35 for run C.

**But this cannot be attributed to the knobs.** See below.

### 5.3 THE BENCHMARK IS FRAGILE — main finding of this phase
Shortening posts did not just change text length, it **changed the ground-truth labels**.
Short descriptions embed closer to the short liked-topic phrases, so nearly everything
became "engage". Six 70-tick runs, all 4 agents / synthetic:

| run | config | posts | words | skip | acc | verdict |
|---|---|---|---|---|---|---|
| A | verbose, slow | 161 | 214 | 7.1% | 90.0% | degenerate |
| C | verbose, slow, shape | 194 | 218 | 19.1% | 90.6% | **usable** |
| D | brief, fast, margin 0 | 273 | 46 | 3.2% | 84.2% | degenerate |
| E | brief, fast, margin 0.02 | 70 | 49 | 75.4% | 92.2% | too hard |
| F | brief, fast, margin 0.005 | 270 | 46 | 3.9% | 81.4% | degenerate |
| G | semi-verbose, fast | 58 | 76 | 6.1% | 62.5% | degenerate |

Two structural problems this exposes:

1. **Realized skip is bimodal, not continuous.** Base skip over the post space moves
   gently with the margin (23.6% → 38.8%), but *realized* skip jumps 3% → 74%. Cause: a
   feedback loop — a harder ground truth means fewer engagements, fewer posts, a thinner
   candidate pool, and therefore even more skips. E starved at 70 posts; D/F grew to 270.
2. **Inference accuracy is unreliable under class imbalance.** When skips fall below ~8%
   the tree has almost no negative examples, and accuracy *drops* (84%, 81%, 62%) even
   though the recommender looks better on skip rate.

`ENGAGE_MARGIN` was left at **0.0** (0.005 landed within noise of it; 0.02 broke the map).
It is a benchmark-difficulty knob, **not** a claim about real users — skip rates are only
comparable across runs when it is held fixed.

### 5.4 Honest status
- **Brevity: verified and shipped.**
- **Learning speed: unproven.** Runs A and C use nominally similar settings yet differ
  3x in skip rate (7.1% vs 19.1%), so single runs cannot resolve an effect of this size.
  Every conclusion in this phase rests on n=1.
- **Required next step: replication.** 3-5 runs per condition with the margin fixed, and
  a class-balance guard so degenerate runs are excluded rather than reported.
