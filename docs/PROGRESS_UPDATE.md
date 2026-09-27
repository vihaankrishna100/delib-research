# Progress Update — Deliberation Map Simulator

## 1. Decision-tree user model (replaces the persona table as the predictor)

The biggest change. Previously each agent's model was a fixed list of ~10 LLM-guessed
topic headers, authored up front and frozen. It is now a **per-agent decision tree learned
from behaviour**.

- Every recommendation outcome is logged as `(post embedding, engaged/skipped, tick)` in a
  new clickstream store — all events kept forever
- Embeddings are PCA-reduced 3072 → 50 dims, then a `DecisionTreeClassifier`
  (entropy criterion, `min_samples_leaf=3`) is trained per agent and **retrained every 5 ticks**
  from the full history, so the hierarchy re-derives itself rather than being stuck with a
  possibly-wrong initial guess
- The tree's predicted engage-probability is now the recommender's **primary ranking signal**
  (falling back to cosine fit, then per-action rates, while data is thin)
- Its leaves are data-discovered subtopics, named post-hoc by an LLM from the actual posts
  inside them, so the model stays human-readable
- **Visualised in the app**: Analytics → Decision-tree user model → click an agent to see the
  learned splits, leaf purity (entropy), and sample counts

## 2. Synthetic ground-truth agents (the change that made learning measurable)

The key insight blocking progress: **a skip predictor can only be as good as the agents are
consistent.** Free LLM judgment gave the same post different decisions on different ticks —
an irreducible noise floor that capped the recommender at a flat ~42% skip rate no matter
what we did.

Each test agent now gets a hidden ground-truth persona model — ~10 liked and ~10 disliked
subtopics, embedded into centroids. The skip decision became deterministic geometry:

```
engage  ⟺  max cos(post, liked) ≥ max cos(post, disliked)
```

No LLM in the decision (content on engage is still LLM-written, so embeddings stay realistic).
Two payoffs: perfectly consistent behaviour, **and a ground truth to score inference against.**

## 3. New metric — inference accuracy

Because the ground truth exists, we can now measure the thing we actually care about:
**how well does the recommender understand each user?**

The learned tree is asked to classify *every* post in the session and compared against what
the hidden ground truth would decide. Note this is a **generalisation test** — the tree only
trained on the ~50–70 posts its agent was shown, but is scored on all ~194.

**Result: 90.6% average, up from 87.6%.**

A second number, `train_accuracy` (fit on its own training data), is stored alongside it. The
gap between them exposes **selection bias**: the tree only ever sees posts the recommender
chose to show, and the recommender increasingly shows predicted-likes. TheEnvironmentalist
scored 100% on its own data but 82.8% against ground truth — a 17-point gap. This is a real
limitation of any closed-loop recommender, and the metric makes it visible instead of hidden.

## 4. Anti-saturation moves: `add_part` and `add_failure_mode`

Previously a solution's only generative child was another sibling solution, which exhausts
quickly — that is the "already covered" half of all skip reasons. Two new moves were added to
the **Python AIdeator templates** (`prompts/aideator_templates.py`), written in the AIdeator
house style with origin lookup and "as different as possible from these others" clauses:

- **`add_part`** — decompose a solution into a required sub-component (must be concrete enough
  to be built or procured, not a restatement of the solution)
- **`add_failure_mode`** — a specific, plausible way the solution could fail, requiring the
  mechanism and triggering conditions, explicitly rejecting generic risks like "it might be
  expensive"

`add_improvement` already existed; it was **aligned verbatim** to the AIdeator Lisp template.
Failure modes are especially generative because they accept `add_solution` and `add_cause`
as children, so the map runs *solution → failure → new solution addressing that failure*.

Measured: 28 part/failure posts in the final run; `add_part` earns only a **12% skip rate**.

## 5. Solution saturation — diagnosis and fix

Adding those moves exposed a deeper problem. The first 70-tick run produced a **degenerate map**:
1.08 solutions per parent, 12 levels deep, only **1 goal**, with 7-long `add_improvement`
chains drifting from "charging infrastructure" all the way to "DAO Governance Paralysis".

**Root cause:** the existing saturation term `1/(1+existing)` penalised the *second* sibling
immediately, so every recommendation was pushed toward fresh, deep, never-seen nodes.

Three shape controls were added, but the critical discovery was **how** to apply them:

- Applied as a **multiplier**, they fixed the shape but *quadrupled the skip rate* (6.8% → 26.4%)
  and flattened learning — shape was **vetoing** what the user model wanted
- Applied as a **bounded tie-breaker** — `score = engagement × (0.65 + 0.35 × shape)`, so the
  worst possible shape still keeps 65% of predicted engagement — **both objectives improved
  together**

Final config: `TARGET_BREADTH=2`, `DEPTH_SOFT_CAP=8`, `DEPTH_DECAY=0.85`,
`MAX_SAME_TYPE_CHAIN=2`, `CHAIN_DECAY=0.60`, `SHAPE_WEIGHT=0.35`.

**This is worth reporting as a finding:** the apparent engagement-vs-structure trade-off was
not fundamental — it was an artifact of letting structural preferences override the learned model.

## 6. Three 70-tick simulations (4 agents each)

| metric | A no shape control | B shape as veto | **C bounded tie-breaker** |
|---|---|---|---|
| solutions per parent | 1.08 | 2.75 | **2.86** |
| max depth | 12 | 6 | **7** |
| longest same-type run | 8 | 1 | **4** |
| goals | 1 | 9 | 6 |
| part + failure_mode | 32 | 6 | **28** |
| total posts | 161 | 166 | **194** |
| overall skip rate | 6.8%* | 26.4% | 19.2% |
| skip trend | 18% → 5% | flat | **29% → 10%** |
| slope (pts/tick) | −0.105 | +0.045 | **−0.573** |
| inference accuracy | 90.0% | 88.3% | **90.6%** |

*\*A's 6.8% is inflated — depth-first drift always offers a never-seen node, so nothing ever
looks "already covered."*

Run C is the best tree produced so far: best breadth, a learning slope **5× steeper** than any
previous run, best inference accuracy, most posts, and it rehabilitated the moves run B had
killed (`add_part` 100% → 12% skip).

## 7. Analytics and UI

- **Sliding-window skip-rate chart** — overlapping windows (1–10, 2–11, 3–12…) so the trend is
  readable instead of noisy; axes reworked for legibility
- **Decision-tree visualisation** per agent
- **Inference-accuracy panel**
- **Hierarchical persona headers** — ~10 top-level, each with 2–3 sub-headers, generated purely
  from persona + mission (never from simulation activity)
- **Per-move scores** — each header rated across the 13 deliberation moves (goal, barrier, cause,
  solution, abstraction, analogy, inspiration, question, answer, pro, con, part, failure_mode)
  with observed counts beside predictions

## 8. Speed

**12.3 → 9.2 s/tick at a constant 6 agents (~25% faster).** Two causes:

- Post embeddings are cached (`_POST_EMB_CACHE`), so each distinct post is embedded at most once
- Synthetic mode skips the specialist's decision LLM call entirely — the agent decides
  geometrically, and the LLM is only invoked to write content once the answer is "engage"

At 4 agents the run rate is **~8.1 s/tick**. Note that figure combines the optimisation *and*
the smaller agent count, so the honest apples-to-apples number is the 25% above.

## 9. Reliability and API cost

Three separate failures were hit and fixed:

- **Truncation bug (hit 3×)** — Gemini has thinking ON by default, which eats the token budget
  and truncates JSON mid-string. Once this silently produced a run with a single post and a
  meaningless 100% accuracy. Fixed with `thinking_budget=0` on every short structured call.
  This is primarily a **correctness** fix; the cost saving is a side benefit. Note that token
  limits were **increased** where output is legitimately long (deep headers 700 → 3000, per-move
  scores 1500 → 3500) — the truncation was caused by thinking, not by generous limits.
- **A stalled request wedged an entire run** for 10+ minutes — the client has no built-in
  timeout. Fixed with a 90s cap on all calls.
- **A transient `503 UNAVAILABLE` killed a 70-tick run at tick 0.** Fixed with retry and
  exponential backoff (4 attempts, 1.5→3→6s), classifying 503/429/500 as transient while
  400/auth errors raise immediately.

## 10. Housekeeping

- The project had **no git repository** — and the enclosing folder was a repo rooted at the
  home directory, so a naive commit would have staged `.ssh/`, credentials, and unrelated
  projects. Created a dedicated repo inside the project with a verified `.gitignore`
  (excludes `.env`, the 115M database, `venv/`, `node_modules/`, `.next/`) and committed
  63 source files.
- Rebuilt topic trees and computed priors for all headline sessions — the Prior/Drift views and
  Research Metrics panel had been returning 404 and were dead in the UI. All 14 analytics
  endpoints now return 200.

## Known limitations

- **Selection bias** (§3) — measured, not eliminated
- **Single-run evidence** — publication figures should average 3–5 runs per condition
- **Synthetic ≠ human** — consistent but less naturalistic; needs re-validation on
  LLM-judgment agents, whose consistency should itself be quantified
- **Two skip-rate definitions** currently coexist in the UI (`research_metrics` counts only
  recommendations tied to a clustered topic node; the timeline counts all decided
  recommendations) — should be reconciled before publication
- **Interpretability is approximate** — tree splits are PCA components, meaningless in
  isolation; readability comes from post-hoc leaf labelling, since Gemini embeddings are not
  invertible without a trained decoder
