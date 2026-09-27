# Speaker script — "Learning Who Will Engage"

Timing guide: ~45 sec/slide ≈ 10 min total. Bolded phrases are the beats to hit.

---

### Slide 1 — Title
"I built a simulation where AI agents with different expert personas — an economist, an
engineer, an environmentalist, and so on — collaboratively build a *deliberation map*: a tree
that grows from a mission down through stakeholders, goals, barriers, causes, and solutions.
Every turn, a **recommender** suggests to each agent where it should contribute next, and the
agent either **engages** or **skips**. My research question is simple to state: **can the
recommender learn each agent well enough that they stop skipping its suggestions?**"

### Slide 2 — The metric
"My core metric is the **skip rate** — of the recommendations agents actually decided on, what
fraction did they refuse? One problem: with only six to twelve decisions per tick, per-tick
rates are extremely noisy. So I measure over a **sliding ten-tick window** — ticks one through
ten, then two through eleven, and so on. Each window becomes one point on a curve, and the
trend becomes readable. One honest caveat: overlapping windows are correlated, so this is a
trend-reading tool, not a significance test."

### Slide 3 — Baselines degrade
"First, why does this need learning at all? I ran all three strategies head-to-head on the
same map. Watch what happens as the map fills up: the **priority** heuristic, which just fills
structural gaps, ends the run at a **one hundred percent skip rate** — every easy gap is gone
and it keeps pushing saturated branches. **Random** climbs to about two-thirds. The learning
recommender is the only one that holds steady, around fifty percent. Without learning,
recommendations literally rot."

### Slide 4 — Diagnosis
"Before building anything, I asked *why* agents skip — literally: every skip triggers a
follow-up where the agent must explain itself. The answers split roughly in half: '**this
isn't my expertise**' and '**this is already covered**.' And here's the finding that redirected
the whole design: **the type of action doesn't predict skips at all** — every action sat in
the same forty-to-fifty-five percent band. What matters is **which post** the agent is sent
to — its topic relative to that agent. So the recommender needs to learn topics, not actions."

### Slide 5 — Learning loop v1
"So version one learned topic fit: I embed every post, track what each agent engaged with
versus skipped, and steer recommendations toward each agent's demonstrated interests. It
worked — fast. Skip rate fell from about fifty to about thirty percent **within five ticks**.
But then it stopped. Over a full fifty-tick run the curve is **flat at forty-two percent** —
the slope is statistically zero. More data bought nothing. Something else was capping it."

### Slide 6 — The ceiling
"Here's the insight this project turns on: **a skip predictor can only be as good as the
agents are consistent.** And LLM agents just aren't — the same post can get a different
decision on a different day. That noise is an irreducible floor. My fix: give each agent a
**hidden ground-truth persona model** — about ten topics it likes and ten it dislikes, as
embedding centroids — and make the skip decision a **deterministic geometric rule**: engage if
the post is closer to a liked topic than a disliked one. No LLM in the decision. This buys me
two things at once: perfectly consistent behavior, **and** a ground truth I can score the
recommender's inference against."

### Slide 7 — The decision-tree model
"Now the user model itself. The old model was a *persona table* — ten topic headers an LLM
guessed up front, frozen at tick zero. I replaced it with a **decision tree learned from the
clickstream**. Every recommendation outcome is logged as a post embedding plus an
engage-or-skip label. I compress the embeddings with PCA, train a small per-agent decision
tree that splits to minimize label entropy — with a minimum leaf size so it doesn't chop
itself into unreliable slivers — and retrain every five ticks from the full history. The
leaves that emerge are **data-discovered subtopics**, and I have an LLM name each leaf from
the actual posts inside it, so the model stays human-readable."

### Slide 8 — Closing the loop
"And it's a closed loop: the recommender ranks candidate posts by the tree's predicted
engage-probability, the agent decides, the outcome lands in the clickstream, the tree
retrains, and the next recommendations get sharper. While an agent has too little data, the
system falls back to the simpler signals — cosine fit, then action rates."

### Slide 9 — Results (the money slide)
"Here's the result everything was building toward. Fifty ticks, six agents, three hundred
decisions. The skip rate starts at **forty percent and falls to seventeen** — a fitted slope
of about half a point per tick, and it's near-monotonic. Compare that to the grey dashed line:
the exact same recommender with inconsistent LLM agents was stuck flat at forty-two. **The
system now demonstrably learns who will engage.**"

### Slide 10 — Inference accuracy
"And because the agents have ground truth, I can grade the inference directly: on average the
learned trees recover **87.6 percent** of each agent's true like/dislike boundary. Two agents
are at a trivial hundred — their models liked everything they were shown. The interesting ones
are the ninety-ones and nineties. And look at what the tree discovered for the engineer:
'Transit Accessibility and AI' — zero percent engagement across twenty-one samples;
'Decentralized Energy Hubs' — a hundred percent across twenty. Perfectly pure leaves, with
sensible names, found entirely from behavior. The sixty percent outlier — hold that thought
for two slides."

### Slide 11 — Honest ledger
"I want to be honest about the path here, because half of it was wrong turns. Version one
optimized the wrong signal entirely and cost two full runs. The plateau took several more runs
to prove. One run silently hung for ten minutes on a single stalled API call — that's why
every call now has a timeout. And the same Gemini token-budget bug bit me **three separate
times**, once producing a degenerate run whose hundred-percent accuracy was completely
meaningless. The wins on the left are real, but so was the debugging on the right."

### Slide 12 — Limitations
"The biggest open issue is **selection bias** — and I can quantify it. The tree only trains on
posts the recommender chose to show, and the recommender increasingly shows things it predicts
the agent likes. The loop partly grades its own homework. That's exactly what happened to the
environmentalist: **ninety-eight percent** accurate on its own training data, only **sixty**
against ground truth. More forced exploration would fix it, at the cost of a higher skip
floor. Beyond that: the model shifts on every retrain, synthetic agents are consistent but
less human, and this is one run — publication figures need three to five averaged."

### Slide 13 — Summary
"So, the arc in four steps: **diagnose** — skips are about topic fit, not action type.
**Learn** — semantic fit gets you fast gains, then a hard plateau. **Stabilize** — consistent,
synthetic agents turn prediction into a solvable problem. **Model** — a learned decision tree
takes skip from forty percent to seventeen, at 87.6 percent inference accuracy. The
recommender now demonstrably learns its users — and, for the first time, we can measure
exactly how well. Thank you — happy to take questions."

---

## Likely Q&A
- **"Why not just always show agents what they like?"** → That's the selection-bias tradeoff on
  slide 12 — exploitation lowers skip but starves the model of diverse training data.
- **"Do results transfer to real humans?"** → Unknown; synthetic agents prove the mechanism.
  Next step is quantifying LLM-agent consistency and re-validating (slide 12).
- **"Why a decision tree and not a neural net?"** → Interpretability (labeled leaves with purity
  and sample counts), tiny data (50 events/agent), and fast retrains every 5 ticks.
- **"Is 17% the floor?"** → No — residual skips come from saturation ("already covered") plus
  exploration picks; longer runs on richer missions should push lower.
