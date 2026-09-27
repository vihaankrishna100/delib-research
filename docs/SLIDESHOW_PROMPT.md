# Prompt for Claude — generate the full slideshow

Copy everything below the line into Claude and ask it to build the deck (it will make a .pptx;
say "as HTML slides" instead if you prefer a web deck).

---

Create a polished 13-slide research presentation as a PowerPoint (.pptx) file titled
**"Learning Who Will Engage: An Adaptive Recommender for Multi-Agent Deliberation Maps."**

## Audience & tone
Research presentation (science-fair / lab-meeting level). Confident but honest — this deck
explicitly shows failures and limitations alongside results. No marketing fluff. Use precise
numbers everywhere; every claim on a slide is backed by a number I give you below.

## Design system
- Dark theme: background #060d1a, panels #0f172a, body text #e2e8f0, muted text #64748b.
- Accent cyan #00d4ff (titles, highlights, the multi_agent line), indigo #6366f1 (random),
  green #10b981 (priority/good), red #ef4444 (bad/skip), amber #f59e0b (warnings/caveats).
- Clean sans-serif, generous whitespace, one idea per slide, minimal bullets (≤5 per slide).
- Build real charts from the data tables I provide (native editable charts preferred).

## Project background (context for you, and for slide 1)
I built a research app that simulates a "deliberation map": ~8 LLM agents with distinct
personas (TheEconomist, TheEngineer, TheEnvironmentalist, ThePsychologist, ThePolicyMaker,
TheInnovator, …) collaboratively grow a tree of posts — mission → stakeholders → goals →
barriers → causes → solutions → abstractions → analogies → questions → pros/cons — about the
mission "Improve urban public transit accessibility." Each simulation tick, a **recommender**
proposes a (post, action) pair to each agent; the agent either **engages** (creates content)
or **skips**. The core research question: can the recommender learn each agent's interests
well enough that the **skip rate decreases over the run**? Stack: FastAPI + SQLite + Google
Gemini (LLM + embeddings) backend, Next.js frontend. Strategies compared: `random`,
`priority` (gap-filling heuristic), and `multi_agent` (the learning recommender).

## Slide-by-slide specification

**Slide 1 — Title.** Title above; subtitle "Can a recommender learn simulated experts well
enough that they stop skipping its suggestions?" Add a small schematic: mission node branching
to stakeholder/goal/barrier/solution nodes, with 3–4 colored agent dots acting on it.

**Slide 2 — The metric.** Skip rate = skipped ÷ decided recommendations, per strategy.
Introduce the sliding tick-window (windows 1–10, 2–11, 3–12, …) as the paper-grade view —
smooths noisy per-bucket rates so trends are readable. Show a tiny illustrative line.

**Slide 3 — Baselines degrade.** Head-to-head run (all three strategies on the same tree,
30 ticks). Line chart from this data (skip rate % by tick window, window=6):
- multi_agent: 0–5: 50, 6–11: 33, 12–17: 58, 18–23: 50, 24–29: 50
- priority: 0–5: 27, 6–11: 33, 12–17: 58, 18–23: 50, 24–29: **100**
- random: 0–5: 27, 6–11: 58, 12–17: 67, 18–23: 58, 24–29: 67
Takeaway callout: as the map saturates, non-learning strategies rot (priority hits 100% skip);
the learning recommender holds ~50%.

**Slide 4 — Diagnosis: why do agents skip?** We elicited a written reason for every skip and
mined them. Two dominant causes, roughly half each: (1) "outside my expertise / not my focus"
(topic-fit) and (2) "already covered." Include two short real quotes:
"I skipped adding a cause… my expertise lies more in behavioral psychology" and "the existing
explanation sufficiently covers this." Key finding as a big callout: **action type does NOT
predict skips** (every action's skip rate sat in 40–55%) — the signal is WHICH POST (its
topic), not which action. This redirected the whole design.

**Slide 5 — Learning loop v1 (semantic fit) and its plateau.** Approach: embed every post;
maintain per-agent centroids of engaged vs skipped posts; rank candidates by fit, boost the
agent's own branches, damp saturated targets, penalize re-showing skipped posts. Results:
early-run skip fell ~50% → ~30% (fast cold→warm learning) BUT full-run curve plateaued flat
at ~42% (50-tick run, slope ≈ +0.06 pts/tick ≈ zero). Simple two-line chart: "early gain"
then "plateau." Honest caption: learning saturated in ~5 ticks, then stalled.

**Slide 6 — The ceiling: agent inconsistency.** Insight (make this the intellectual pivot):
**a skip predictor can only be as good as the agents are consistent** — free LLM judgment
gives the same-ish post different decisions on different days, an irreducible noise floor.
Fix: give each agent a hidden **ground-truth persona model** — ~10 LIKED and ~10 DISLIKED
subtopic embeddings (generated once from persona+mission). Decision rule (show as a formula):
engage ⟺ max cos(post, liked) ≥ max cos(post, disliked). Deterministic, zero LLM in the
decision (content on engage is still LLM-written). Double win: consistent behavior + a ground
truth to score inference against.

**Slide 7 — The new user model: a learned decision tree.** Replaces the static prompt-made
"persona table" (10 LLM-guessed topic headers, fixed at t=0). Pipeline diagram:
clickstream log (post embedding 3072-d, engage/skip label) → PCA to 50 dims → per-agent
DecisionTreeClassifier (entropy criterion, min_samples_leaf=3 to avoid over-chopping) →
retrain every 5 ticks from ALL stored events (so history redistributes automatically when the
hierarchy changes) → leaves = data-discovered subtopics, LLM-labeled afterward from their
member posts so they're human-readable.

**Slide 8 — Closing the loop.** Architecture cycle diagram: recommender ranks candidate posts
by the tree's predicted engage-probability (fallbacks: cosine fit, then action-rate, while
data is thin) → agent decides via ground-truth model → outcome appended to clickstream →
tree retrains → better recommendations. Label the arrows.

**Slide 9 — RESULTS (the money slide).** 50-tick synthetic run: 6 agents, 150 posts created,
300 decisions, 7.7 minutes. Big line chart from this exact data (skip rate %, sliding window
of 10 ticks, step 2; x = window start tick):
0:39.7, 2:40.0, 4:40.0, 6:40.0, 8:35.0, 10:31.7, 12:31.7, 14:30.0, 16:30.0, 18:31.7,
20:30.0, 22:30.0, 24:30.0, 26:26.7, 28:23.3, 30:21.7, 32:18.3, 34:18.3, 36:20.0, 38:21.7,
40:21.7, 42:22.9, 44:22.2, 46:20.8, 48:16.7
Headline stat callouts: **skip 40% → 17%**, slope **−0.47 pts/tick**, near-monotonic. Contrast
annotation: same recommender with inconsistent LLM agents = flat ~42%.

**Slide 10 — Inference accuracy.** "The learned tree vs the hidden ground truth" — table:
TheEconomist 100% (trivial — liked everything shown), TheInnovator 100% (same),
ThePsychologist 91.4%, ThePolicyMaker 90.0%, TheEngineer 84.3%, TheEnvironmentalist 60.0%.
Average **87.6%**. Also show discovered leaves for TheEngineer as an example of
interpretability: "Transit Accessibility & AI" (0% engage, entropy 0.00, n=21),
"Decentralized Energy Hubs" (100% engage, 0.00, n=20), "Advanced AI & Energy" (33%, 0.92,
n=3 — correctly flagged low-confidence).

**Slide 11 — What went right vs wrong (honest ledger).** Two columns.
RIGHT: decreasing curve achieved; inference measurable (87.6%); readable learned topics;
baseline-vs-learner contrast; runs no longer stall (90s timeouts).
WRONG: learner v1 optimized the wrong signal (action type — flat curve, 2 wasted runs);
LLM-agent inconsistency capped v2 at ~42%; one run silently wedged at tick 1 for 10+ min
before timeouts existed; the same Gemini thinking-budget bug (thinking eats the token budget
→ truncated JSON) struck three separate times, once producing a degenerate 1-post run with a
meaningless 100% accuracy.

**Slide 12 — Limitations & next steps.** (amber-accent slide)
- **Selection bias, measured:** the tree trains only on posts the recommender chose to show —
  and it increasingly shows predicted-likes. Evidence: TheEnvironmentalist scored 98% on its
  own training data but only 60% vs ground truth. The loop partly grades its own homework;
  more forced exploration trades this off against a higher skip floor.
- Tree shifts on every retrain (labels/boundaries move) — less stable for longitudinal reading.
- Synthetic agents are consistent but less human; results must be re-validated on naturalistic
  LLM-judgment agents (whose consistency should itself be quantified).
- Single-run evidence; average 3–5 runs per condition for publication figures.
- Cold start: ~8 events per agent before the tree activates.

**Slide 13 — Summary.** Four-step arc as a visual: Diagnose (skips are topic-fit) → Learn
(semantic fit: fast gain, plateau) → Stabilize (synthetic ground-truth agents) → Model
(decision tree: 40%→17%, 87.6% accuracy). Closing line: "The recommender now demonstrably
learns its users — and we can measure exactly how well."

## Rules
- Every number above is real measured data — use them verbatim, do not invent or round further.
- Keep the failure content (slides 11–12) as prominent as the wins; this is a research deck.
- Charts must be built from the data given, with axis labels (x: tick window, y: skip rate %).
