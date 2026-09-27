# Learning Who Will Engage
### An Adaptive Recommender for Multi-Agent Deliberation Maps — Final Report

*Project: `~/deliberation-map-sim` · FastAPI + SQLite + Google Gemini / Next.js*
*Backup of pre-change state: `~/deliberation-map-sim-backup-20260728-205942`*

---

## 1. The system

Eight LLM agents with distinct expert personas (economist, engineer, environmentalist,
psychologist, policy-maker, innovator, sociologist, strategist) collaboratively grow a
**deliberation map** — a tree rooted at a mission and branching through stakeholders →
goals → barriers → causes → solutions → abstractions → analogies.

Each tick, a **recommender** proposes one `(post, action)` pair to each agent. The agent
either **engages** (writes a post) or **skips**. The research question:

> Can the recommender learn each agent well enough that the **skip rate decreases**?

Three strategies are compared: `random`, `priority` (gap-filling heuristic), and
`multi_agent` (the learning recommender).

---

## 2. Metric: skip rate over a sliding window

`skip rate = skipped ÷ decided recommendations`

Per-tick rates are noisy (6–12 decisions), so results are read through an overlapping
**sliding window** (ticks 1–10, 2–11, 3–12…). *Caveat: overlapping windows are
autocorrelated — good for trend-reading, weaker for significance claims.*

Endpoint: `GET /analytics/sessions/{id}/skip_rate_timeline?window=W&step=S`

---

## 3. Finding 1 — Baselines degrade as the map fills

Head-to-head run, all three strategies on the same tree (30 ticks, session `8396c216`):

| tick window | multi_agent | priority | random |
|---|---|---|---|
| 0–5 | 50% | 27% | 27% |
| 12–17 | 58% | 58% | 67% |
| 24–29 | **50%** | **100%** | **67%** |

As easy structural gaps disappear, non-learning strategies rot — `priority` ends at a
100% skip rate. The learning recommender is the only one that holds.

---

## 4. Finding 2 — Agents skip on *topic fit*, not action type

Every skip triggers a follow-up prompt requiring the agent to explain itself. Mining
those reasons showed roughly a 50/50 split:

- **topic-fit mismatch** — *"my expertise lies more in behavioral psychology"*
- **already covered** — *"the existing explanation sufficiently covers this"*

And critically: **every action's skip rate sat in the same 40–55% band**
(`add_barrier` 55%, `add_goal` 42%, `add_solution` 40%). Action type is *not*
discriminative. This redirected the entire design — the signal is **which post**.

---

## 5. Finding 3 — Semantic fit learns fast, then plateaus

v1 of the loop embedded every post, tracked per-agent centroids of engaged vs skipped
content, and ranked candidates by fit + own-frontier boost + saturation damping +
skip penalty.

- Early-run skip fell **~50% → ~30%** within ~5 ticks.
- But the full 50-tick curve was **flat at ~42%** (slope ≈ +0.06 pts/tick ≈ zero).

Learning saturated almost immediately, then stalled.

---

## 6. Finding 4 — The ceiling is *agent inconsistency*

**A skip predictor can only be as good as the agents are consistent.** Free LLM
judgment gives the same-ish post different decisions on different ticks — an
irreducible noise floor.

**Fix — synthetic ground-truth personas** (`simulation/ground_truth.py`): each agent
gets ~10 liked and ~10 disliked subtopics (LLM-generated from persona + mission,
embedded into centroids). The skip decision becomes deterministic geometry:

```
engage  ⟺  max cos(post, liked_centroids) ≥ max cos(post, disliked_centroids)
```

No LLM in the decision; content on engage is still LLM-written so embeddings stay real.
Two payoffs: **perfectly consistent behavior**, and a **ground truth** to score
inference against.

---

## 7. The model — a decision tree learned from behavior

Replaces the static "persona table" (10 LLM-guessed headers frozen at t=0) as the
*predictor*.

| stage | detail |
|---|---|
| **Clickstream** | every `(post embedding 3072-d, engage/skip, tick)` — all events kept forever |
| **PCA** | 3072 → 50 dims for fast training |
| **Tree** | per-agent `DecisionTreeClassifier`, entropy criterion, `min_samples_leaf=3` |
| **Retrain** | every 5 ticks, from scratch — history redistributes automatically |
| **Leaves** | data-discovered subtopics, LLM-named from their member posts |

The tree's predicted engage-probability is the recommender's **primary ranking signal**
(falls back to cosine fit, then per-action rate, while data is thin). It is
simultaneously the user model, the scorer, and a measurable artifact.

**Visualized in-app:** Analytics → *Decision-tree user model* → click an agent. Internal
nodes show `PC{k} ≤ threshold` with sample count and entropy; leaves are colour-coded by
engage rate.

> **Honest note on the persona table:** it was *not* deleted. It still exists as a table,
> a UI panel, and an optional cold-start prior injected into the LLM selector prompt when
> headers exist. The tree replaced it only as the **ranker**. In the headline result runs
> no persona headers existed, so those results are tree-only.

---

## 8. Finding 5 — Skip rate finally decreases

**50 ticks × 6 agents, synthetic** (session `14da4d73`) — 150 posts, 300 decisions, 7.7 min:

| window | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 |
|---|---|---|---|---|---|
| skip | 39.7% | 31.7% | 30.0% | 21.7% | **16.7%** |

**40% → 17%, slope −0.47 pts/tick, near-monotonic.** The same recommender with
inconsistent LLM agents was flat at ~42%.

**Inference accuracy vs hidden ground truth: 87.6% average**
(Economist 100%*, Innovator 100%*, Psychologist 91.4%, PolicyMaker 90.0%,
Engineer 84.3%, Environmentalist 60.0%). *\*trivial — their ground truth liked
everything they were shown.*

Example discovered leaves (TheEngineer): *"Transit Accessibility & AI"* 0% engage
(H=0.00, n=21) vs *"Decentralized Energy Hubs"* 100% engage (H=0.00, n=20) — perfectly
pure clusters found from behavior alone.

---

## 9. Finding 6 — Premature solution saturation, and the fix

A solution's only generative child was another **sibling solution**, which exhausts fast
(the "already covered" half of all skips). Added two AIdeator moves, styled on the
original Lisp templates:

- **`add_part`** — decompose a solution into a required sub-component
- **`add_failure_mode`** — stress-test a solution for a concrete failure mechanism

(`_aideator_improvement` was also aligned verbatim to the AIdeator Lisp template.)

### Result — 70 ticks × 4 agents (session `a14a62d8`)

| metric | before (50t, 6 agents) | after (70t, 4 agents) |
|---|---|---|
| posts | 150 | 161 |
| solutions | 60 (**40%** of tree) | 41 (**25%**) |
| part + failure_mode | 0 | **32** |
| skip rate | 40% → 17% | **18.4% → 5%** (overall 6.8%) |
| inference accuracy | 87.6% | **90.0%** |

`add_part` earned a **4% skip rate**, `add_failure_mode` 19% — both readily accepted.
Solution share of the tree fell from 40% to 25%: the map genuinely diversified.

### But this exposed a *new* failure mode: breadth collapsed into depth

- **Solution breadth ≈ 1** — mean **1.08 solutions per parent** (max 2); 35 of 38
  parents had exactly one. The map became a chain, not a fan.
- **Depth ran to 12 levels.** `add_improvement` produces another *solution*, so it built
  long improvement chains — the deepest path was **7 consecutive solutions**, drifting
  from "charging infrastructure" to *"DAO Governance Paralysis → Hybrid Governance"*.
- **Lopsided tree**: 41 solutions but only **1 goal**, 5 stakeholders.

**Root cause:** the saturation term `1/(1+existing)` penalised the *second* sibling
immediately, pushing every recommendation toward fresh deep nodes.

**And it inflated the headline number** — depth-first drift always offers a never-seen
node, so nothing looks "already covered." Part of the 6.8% was earned targeting; part
was the recommender running down a corridor.

---

## 10. Finding 7 — Tree-shape controls

Three knobs added to `_rank_pairs_by_learning`:

| knob | value | effect |
|---|---|---|
| `TARGET_BREADTH` | 3 | siblings are "free" until 3 exist, *then* saturation applies |
| `DEPTH_SOFT_CAP` / `DEPTH_DECAY` | 6 / 0.75 | damp candidates deeper than 6 levels |
| `MAX_SAME_TYPE_CHAIN` / `CHAIN_DECAY` | 2 / 0.35 | damp runs of >2 consecutive same-type posts |

Unit-verified: in a 5-deep solution chain, candidates at depth 3+ now rank **below**
fanning out from the goal, while shallow improvements stay unpenalised.

### Result — 70 ticks × 4 agents, shape-controlled (session `1c81ba56`)

**The tree-shape problem is fixed, decisively:**

| shape metric | run A (no controls, `a14a62d8`) | run B (controls, `1c81ba56`) |
|---|---|---|
| solutions per parent | 1.08 (max 2) | **2.75** (max 3) |
| max tree depth | 12 | **6** |
| goals | **1** | **9** |
| barriers | 18 | 38 |
| longest same-type chain | 7 solutions | none |

The degenerate corridor is gone. The deepest path is now a coherent
`mission → stakeholder → goal → barrier → solution → barrier → cause`, and 14 of 16
solution-bearing parents have exactly 3 solutions — precisely the `TARGET_BREADTH` we asked for.

### …but the skip rate regressed, and stopped learning

| skip metric | run A | run B |
|---|---|---|
| overall skip | **6.8%** | **26.4%** |
| trend | 18.4% → 5% (slope −0.105) | 25% → 25% (**slope +0.045 ≈ flat**) |

Per-action damage is stark: `add_part` went **4% → 100% skip** (0 engaged of 6),
`add_improvement` 100% skip, `add_question` 100% skip, `add_stakeholder` 85% skip.
Zero `part` posts were created in run B.

**Interpretation — this is a real trade-off, not a bug.** In synthetic mode, skip is decided
purely by the agent's hidden topic preferences. Forcing breadth means recommending posts
*outside* each agent's learned preferred region, so engagement falls. The shape controls
directly override the learned user model.

So the two runs bracket the tension:

- **Run A** optimises the metric and produces a **degenerate map** (breadth 1.08, depth 12,
  one goal) — and its excellent 6.8% is partly an artifact of depth-first drift always
  offering a fresh node.
- **Run B** produces a **well-formed map** but at 4× the skip rate, with learning flattened.

**Neither setting is correct.** The current knobs (`TARGET_BREADTH=3`, `DEPTH_SOFT_CAP=6`,
`CHAIN_DECAY=0.35`) are too aggressive — they dominate the tree's predictions instead of
gently regularising them. **Recommended next step:** soften to roughly
`TARGET_BREADTH=2`, `DEPTH_SOFT_CAP=8`, `CHAIN_DECAY=0.6`, and treat shape as a *tie-breaker*
among already-high-probability candidates rather than a multiplicative penalty on all of them —
then sweep the knobs and report skip rate against a tree-quality score.

**This tension is itself a finding worth reporting:** optimising engagement alone yields a
deep, drifting, low-information map; optimising structure alone yields a balanced map the
agents don't want to contribute to. A useful deliberation recommender must be scored on
*both* axes, not skip rate alone.

---

## 10b. Resolution — shape as a bounded tie-breaker

The problem in run B was not the knob *values* but *how* shape was applied. It multiplied
the primary signal directly, so a badly-shaped candidate was driven toward zero and shape
could **veto** what the user model wanted:

```python
score = predicted_engagement × saturation × depth × chain      # run B: veto
```

Replaced with a bounded blend, so the worst possible shape still retains 65% of the
candidate's predicted engagement:

```python
shape = saturation × depth × chain                             # combined, 0..1
score = predicted_engagement × (1 - SHAPE_WEIGHT + SHAPE_WEIGHT × shape)
```

Knobs also softened: `TARGET_BREADTH` 3→2, `DEPTH_SOFT_CAP` 6→8, `DEPTH_DECAY` 0.75→0.85,
`CHAIN_DECAY` 0.35→0.60, `SHAPE_WEIGHT` = 0.35.

Unit-verified in both directions: when the model strongly wants a deep node (0.95 vs 0.20)
that node **still ranks first**; when all candidates are equally probable, the shallow /
breadth option wins.

### Three-way result — 70 ticks × 4 agents, synthetic

| metric | A — no shape control | B — hard shape (veto) | **C — bounded tie-breaker** |
|---|---|---|---|
| **solutions per parent** | 1.08 | 2.75 | **2.86** |
| **max depth** | 12 | 6 | **7** |
| **longest same-type run** | 8 | 1 | **4** |
| goals | 1 | 9 | 6 |
| barriers | 18 | 38 | 35 |
| part + failure_mode | 32 | **6** | **28** |
| total posts | 161 | 166 | **194** |
| **overall skip rate** | **6.8%*** | 26.4% | 19.2% |
| **skip trend** | 18% → 5% | 25% → 25% (flat) | **29% → 10%** |
| **slope (pts/tick)** | −0.105 | +0.045 | **−0.573** |
| inference accuracy | 90.0% | 88.3% | **90.6%** |

*\*A's 6.8% is inflated: depth-first drift always offers a never-seen node, so nothing
looks "already covered."*

**Run C wins on nearly every axis:**
- **Best tree shape overall** — breadth 2.86 (beats even the hard-shape run), depth 7,
  longest same-type run cut from 8 → 4.
- **Steepest learning curve of any run** — slope **−0.573 pts/tick**, 5× steeper than A.
  Skip falls 29% → 10% while the map stays well-formed.
- **Best inference accuracy** (90.6%) and **most productive** (194 posts).
- **Rehabilitated the new moves that run B had killed**: `add_part` 100% → **12%** skip,
  `add_improvement` 100% → **10%** skip, and `part + failure_mode` recovered 6 → 28 posts.

**Remaining wrinkle:** solutions rose to 41% of the tree (80 of 194), higher than A (25%)
or B (27%) — driven by heavy `add_improvement` use (n=61 at 10% skip). But because
breadth is 2.86 and the longest chain is 4 (vs A's 8), this is *fanning out*, not the
drifting corridor of run A.

**Conclusion:** the engagement-vs-structure trade-off is not fundamental — it was an
artifact of applying structural preferences as a veto. Applied as a bounded tie-breaker,
both objectives improve together. **Run C's configuration is the recommended default.**

---

## 11. Engineering robustness (found the hard way)

| issue | fix |
|---|---|
| **Gemini thinking-budget truncation** — thinking is ON by default and eats the token budget, truncating JSON mid-string. Bit **three separate times**, once producing a degenerate 1-post run with a meaningless 100% accuracy. | `thinking_config=ThinkingConfig(thinking_budget=0)` on every short structured call |
| **Stalled request wedged a run** — a 50-tick run hung at tick 1 for 10+ min; the client has no built-in timeout. | `LLM_TIMEOUT_S = 90` via `asyncio.wait_for` on all call sites |
| **Transient 503 killed a run at tick 0** — Gemini returned `503 UNAVAILABLE` during ground-truth generation, with no retry. | `call_with_retry`: 4 attempts, exponential backoff (1.5→3→6s), transient (503/429/500) vs immediate-raise (400/auth) |

---

## 12. Limitations

- **Selection bias, measured.** The tree trains only on posts the recommender chose to
  show, and it increasingly shows predicted-likes — the loop partly grades its own
  homework. Evidence: TheEnvironmentalist scored **98% on its own training data but 60%
  against ground truth**. More forced exploration trades this off against a higher skip floor.
- **Depth-drift inflates skip metrics** (§9) — mitigated by §10, not eliminated.
- **Model instability** — leaf labels and boundaries shift on every 5-tick retrain.
- **Synthetic ≠ human.** Consistent, but less naturalistic; results need re-validation on
  LLM-judgment agents, whose consistency should itself be quantified.
- **Single-run evidence.** Publication figures should average 3–5 runs per condition.
- **Interpretability is approximate.** Split conditions are PCA components (`PC0 ≤ -0.04`),
  meaningless in isolation; readability comes from post-hoc leaf labelling, not from
  inverting embeddings (Gemini embeddings are not invertible without a trained decoder).
- **Cold start** — ~8 events per agent before the tree activates.

---

## 13. Summary

1. **Diagnose** — skips are about topic fit, not action type.
2. **Learn** — semantic fit gives fast gains, then a hard plateau at ~42%.
3. **Stabilize** — synthetic ground-truth agents make prediction a solvable problem.
4. **Model** — a learned decision tree takes skip from 40% → 17% at 87.6% accuracy.
5. **Diversify** — `part` / `failure_mode` cut solution dominance 40% → 25%.
6. **Shape** — applied as a *veto*, structural controls fixed the map but quadrupled skips.
   Applied as a **bounded tie-breaker**, both objectives improve together: breadth
   1.08 → 2.86, depth 12 → 7, skip 29% → 10% at the steepest learning slope of any run
   (−0.573 pts/tick) and the best inference accuracy (90.6%).

The recommender demonstrably learns its users, and we can measure exactly how well. The
apparent engagement-vs-structure trade-off turned out **not** to be fundamental — it was an
artifact of letting structural preferences override the learned model. Bounded to 35% of
the score, they regularise it instead, and the map gets both better-formed *and* more engaging.

### Final configuration (recommended default, run C)

```python
TARGET_BREADTH      = 2      # siblings free until 2 exist
DEPTH_SOFT_CAP      = 8      # damp candidates deeper than 8 levels
DEPTH_DECAY         = 0.85
MAX_SAME_TYPE_CHAIN = 2
CHAIN_DECAY         = 0.60
SHAPE_WEIGHT        = 0.35   # shape modulates at most 35% of the score
```

---

## Appendix — files changed

**Backend:** `models.py`, `database.py`, `schemas.py`, `routes/analytics.py`,
`simulation/{recommender,engine,persona_topics,ground_truth,user_model_tree,topic_tree}.py`,
`simulation/roles/{base,specialist}.py`, `simulation/prompts/{specialist,aideator_templates}.py`
**Frontend:** `lib/api.ts`, `app/{analytics,user-models,simulation}/page.tsx`,
`components/MapTree/PostCard.tsx`
**Dependencies added:** `numpy 2.5.1`, `scikit-learn 1.9.0`
**Companion docs:** `AGENTS_AND_PROMPTS.md` (every prompt verbatim), `CHANGES_LOG.md`,
`SLIDESHOW_OUTLINE.md`, `SPEAKER_SCRIPT.md`, `SLIDESHOW_PROMPT.md`
