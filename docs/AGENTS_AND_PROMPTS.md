# Agents & Prompts Reference

Every LLM role in the system, with its **verbatim prompt** and **where it's coded**.
Model throughout: `gemini-2.5-flash`. Embeddings: `gemini-embedding-001` (3072-d).

> **Gemini gotcha applied everywhere:** thinking is ON by default and eats the token budget,
> truncating JSON mid-string. Every short structured call sets
> `thinking_config=ThinkingConfig(thinking_budget=0)`. This bug bit three separate times.

---

# PART 1 — The recommendation system (the "multi-agent" pipeline)

`simulation/recommender.py :: RecommendationEngine.recommend_multi_agent()`

Runs **per agent, per tick**. Two LLM roles wrapped around a non-LLM scoring layer:

```
clickstream → [Agent A: Profile Analyst] → behavioral profile
                                                ↓
   candidates → [scoring layer: tree/fit/rates] → top 12 + 1 explore
                                                ↓
                          [Agent B: Content Selector] → picks 1
```

## Agent A — Profile Analyst

**Job:** convert raw clickstream events into a prose behavioral profile that persists and
evolves across ticks (stored in `agent_profiles`, keeps last 10 snapshots).

**System prompt** (`_ANALYST_SYSTEM`):
```
You are a behavioral profile analyst for a deliberation simulation. Your job is to
maintain concise, accurate contribution profiles for deliberation participants based
on their activity. Output only the updated profile text, nothing else.
```

**User prompt** (`_build_analyst_prompt`):
```
Deliberation participant: {agent_name} ({domain} expert)

EXISTING PROFILE:
{existing_profile}          ← or "(none yet)"

RECENT ACTIVITY (last 10 ticks):
{clickstream}               ← "[tick 7] add_barrier on "Fare evasion" (solution) — ptype: barrier, reason: ..."

Update the profile. Write 3-5 sentences describing: what post types they prefer
(barriers, solutions, pros/cons, questions, etc.), what branches they gravitate
toward, and how they approach deliberation. Output only the updated profile text.
```
`max_tokens=300, temperature=0.4`

## Scoring layer (no LLM — deterministic)

`_rank_pairs_by_learning()` narrows ~40 candidates to 13. Score per candidate:

```python
score = primary_signal × own_frontier × saturation × skip_penalty
```

| factor | source | meaning |
|---|---|---|
| **primary** | `tree_scores[pid]` → else `fit_scores[pid]` → else `rates[action]` | decision tree's engage-probability; falls back to cosine fit, then per-action rate |
| **own_frontier** | ×1.6 own post, ×1.3 child of own post | keep agents in their own coherent lane |
| **saturation** | `1/(1+existing achievers)` | avoid "already covered" skips |
| **skip_penalty** | `0.35 ^ skips_on_that_post` | stop re-pushing rejected branches |

Keeps top 12 + **1 random explore** pick.

## Agent B — Content Selector

**Job:** choose the single (post, action) pair to recommend, from the narrowed 13.

**System prompt** (`_SELECTOR_SYSTEM`):
```
You are a recommendation engine for a deliberation map simulation. You are given two
models of the expert: (1) a PERSONA TABLE of topics the expert said up-front they want
to contribute to, and (2) a BEHAVIORAL PROFILE learned from their recent clickstream.
You also get the reasons the expert previously SKIPPED recommendations. Combine all
three — persona interests, observed behavior, and skip rules — to select the single
post/action pair the expert is most likely to engage with. Treat skip reasons as
negative weights: avoid pairs resembling what the expert rejected. Also return a
CONFIDENCE from 0.0 to 1.0 — how sure you are the expert will engage with the selected
pair rather than skip it.
Return ONLY valid JSON: {"selected_index": 0, "confidence": 0.8, "reasoning": "one sentence"}
```

**User prompt** (`_build_selector_prompt`) — blocks appear only when data exists:
```
PERSONA TABLE (topics this expert said they want to contribute to):
{comma-separated headers}          ← omitted if no persona headers for the session

ENGAGEMENT RATES SO FAR (this expert, this session — higher = they act, lower = they skip):
- add_solution: 78%
- add_barrier: 45%
(overall 61% over 34 decisions). Strongly prefer higher-rate actions.

BEHAVIORAL PROFILE (learned from clickstream):
{profile from Agent A}

AVAILABLE POST/ACTION PAIRS:
[0] Action: add_barrier | On: (goal) Reduced commute times
[1] Action: add_solution | On: (barrier) Fare evasion
... up to 20

RECENT SKIP REASONS (avoid recommending pairs like these):
- skipped 'add_cause' — the existing explanation sufficiently covers this
- skipped 'add_goal' — outside my expertise as an economist

Which pair (by index) is this expert most likely to engage with?
```
`max_tokens=200, temperature=0.4`. On parse failure → falls back to the top-ranked pair.

## Baseline strategies (for comparison — no LLM)

| strategy | code | logic |
|---|---|---|
| `random` | `recommend_random` | `random.choice(valid_pairs)` |
| `priority` | `recommend_priority` | gap-fill: `1/(1+count of existing achievers)` |
| `ab_rotate` | `engine.AB_STRATEGIES` | rotates random/priority/multi_agent across agents |

---

# PART 2 — The deliberation agents (who receive recommendations)

## The 8 specialists

Each row is a real DB `Agent` seeded by `seed.py`. `llm_assist_rate` is the persona's
scripted willingness to lean on LLM assistance.

| agent | domain | assist | persona (verbatim) |
|---|---|---|---|
| **TheEconomist** | economics | 0.70 | "expertise in incentives, markets, and resource allocation… identifying cost structures, misaligned incentives, and barriers related to funding and economic sustainability. Prefers proposing barriers about resource constraints and solutions involving market mechanisms, subsidies, or economic restructuring." |
| **TheEngineer** | systems engineering | 0.80 | "focused on technical feasibility and implementation. Thinks in terms of components, interfaces, failure modes, and scalability… drawn to abstraction and analogy chains to find structural patterns across domains." |
| **ThePolicyMaker** | public policy | 0.65 | "thinks about governance, regulation, and institutional design. Identifies regulatory barriers, stakeholder misalignment, and political constraints… ensures all stakeholders are represented." |
| **TheInnovator** | technology innovation | 0.90 | "loves finding non-obvious solutions through abstraction and analogy… draws analogies from distant domains like biology, aviation, games, and history. Most energized by the abstraction → analogy → inspiration → solution chain." |
| **TheSociologist** | sociology | 0.70 | "focused on human behavior, social structures, and community dynamics. Identifies barriers related to social norms, power dynamics, and community trust… ensures goals reflect the needs of marginalized stakeholders." |
| **TheEnvironmentalist** | environmental science | 0.60 | "considers ecological impact and sustainability. Identifies environmental barriers and risks that others overlook… frames goals in terms of ecological health and planetary boundaries." |
| **ThePsychologist** | behavioral psychology | 0.75 | "focused on human motivation, cognition, and behavior change. Identifies psychological causes of barriers: cognitive biases, fear of change, loss aversion… asks probing questions about assumptions." |
| **TheStrategist** | urban planning / systems thinking | 0.70 | "sees systems-level patterns and long-term trajectories. Thinks about spatial constraints, coordination problems, and multi-stakeholder systems… skilled at connecting solutions across different branches." |

**System prompt template** (`seed.py :: build_system_prompt`) — same shape for every agent:
```
You are {name}, a {domain} expert participating in a structured deliberation.

{persona_description}

You will be shown a deliberation post and a list of available actions. Choose one action
and respond with ONLY valid JSON — no markdown, no explanation, no code fences.

For content-creating actions: {"action": "add_barrier", "name": "concise name", "description": "1-2 sentence explanation"}
For editing: {"action": "edit", "description": "improved description"}
For skip: {"action": "skip"}
```

## The specialist's decision prompt

`prompts/specialist.py :: build_action_choice_prompt` — this is what decides engage/skip
in **LLM mode**:
```
You are {agent_name}, a {domain} expert in a structured deliberation.

CONTEXT:
{context_str}                    ← full ancestor chain, mission → … → this post

CURRENT POST [{PTYPE}]: "{name}"
{description}

EXISTING CHILDREN:
  barrier: Fare evasion, Service gaps     ← so it doesn't duplicate
  solution: Off-board payment

AVAILABLE ACTIONS:
  "add_barrier" — Add a barrier (obstacle preventing success) ← SUGGESTED
  "add_solution" — Add a solution (concrete intervention addressing this)
  "add_pro" — Add a pro (concrete advantage or benefit of this post)
  "skip" — Skip (view but take no action)
  ...

SUGGESTED ACTION GUIDANCE (for "{recommended_action}"):
{action-specific instruction, e.g. "Propose a BARRIER that blocks success at 'X': an
 obstacle, constraint, or challenge. Name it concisely as a problem…"}

Choose the action that best fits your expertise and the deliberation needs.
You may choose the suggested action or any other available action.
...
Always include a short "reason" field (one sentence) explaining WHY you chose this
action. If you skip, the reason must explain what made this recommendation a poor
fit for you right now (off-topic, already covered, outside your expertise, etc.).

Return ONLY valid JSON — no markdown, no explanation:
For content actions: {"action": "add_barrier", "name": "...", "description": "...", "reason": "why this action"}
For edit:           {"action": "edit", "description": "improved description...", "reason": "why edit"}
For skip/view:      {"action": "skip", "reason": "why you skipped this recommendation"}
```
`max_tokens=800` (aideator) or `300`. **Note:** this call leaves thinking ON — it's fine
at 800 tokens.

**Action vocabulary** (`roles/specialist.py`): type-specific actions per post type
(`mission → add_stakeholder`, `goal → add_barrier/add_solution/add_abstraction`,
`barrier → add_cause/add_solution`, `abstraction → add_analogy`, `analogy → add_inspiration`,
`inspiration → use_inspiration`, `solution → add_improvement/add_barrier/add_question`,
`question → add_answer`) **plus** universal actions available on anything
(`add_pro`, `add_con`, `create_topic`, `edit`, `skip`).

## Skip-reason elicitation

Models reliably give a reason when engaging but omit it when skipping, so there's a
focused follow-up (`SpecialistAgent._elicit_skip_reason`). **Critical detail:** it does
*not* reuse `self.system_prompt` (that forces JSON) — it builds a plain-prose persona system:
```
You are {name}, a {domain} expert participating in a structured deliberation.
{persona_description} Answer in plain prose — full sentences, no JSON, no code fences.
```
```
CONTEXT: {context_str}

You were just recommended to '{recommended_action}' on the [{PTYPE}] post "{name}",
and you chose to SKIP it.

Explain WHY you skipped in ONE or TWO complete sentences — what made this a poor fit
for you right now (off-topic for your expertise, already covered, lower priority than
another branch, etc.). Write full sentences that finish the thought. Reply with the
explanation only, no preamble, no JSON.
```
These answers are what produced the **core diagnosis** (skips are topic-fit, not action type)
and feed back as negative weights into the Content Selector.

## Facilitator & Synthesizer

**MapFacilitator** — runs at the start of every tick, produces `focus_hints`:
```
You are the MapFacilitator, an expert in structured deliberation.
Review the current deliberation map (Tick {tick}):
{tree_summary}

Identify:
1. Branches that are underdeveloped (goals with no barriers, barriers with no solutions, etc.)
2. Post types that are underrepresented
3. The 2-3 most important gaps to fill next

Return ONLY valid JSON:
{"focus_hints": [...], "priority_ptypes": [...], "priority_post_ids": [...], "summary": "..."}
```

**MapSynthesizer** — runs every 5 ticks (`SYNTHESIZER_INTERVAL`):
```
You are the MapSynthesizer, skilled at connecting ideas across a deliberation map.
Current deliberation map state (Tick {tick}):
{tree_summary}

Identify 2-3 solutions or themes that could be synthesized into a broader solution.

Return ONLY valid JSON:
{"synthesis_name": "...", "synthesis_description": "...", "connected_post_ids": [...]}
```

---

# PART 3 — Supporting LLM roles

## Ground-truth persona generator (synthetic mode)
`simulation/ground_truth.py` — run once per session; output is embedded into centroids
and **replaces the LLM entirely** in the skip decision.
```
I am the following participant:
{persona_description}

I will be in deliberations about: {mission} — {mission_description}

Give me TWO lists of short subtopic headers (2-5 words each):
- liked: ~10 subtopics I would enthusiastically contribute to
- disliked: ~10 subtopics I would NOT engage with and would skip

Make them concrete and clearly separable. Return ONLY valid JSON (no markdown):
{"liked": ["...", "..."], "disliked": ["...", "..."]}
```
Decision rule afterward is pure geometry:
`engage ⟺ max cos(post, liked_centroids) ≥ max cos(post, disliked_centroids)`

## Decision-tree leaf labeler
`simulation/user_model_tree.py :: _label_leaves` — makes the learned tree readable:
```
Each numbered group below is a cluster of deliberation posts that a participant treated
the same way. Give each cluster a SHORT topic label (2-5 words) that captures what the
posts have in common.

0. Grid Capacity Scalability; Legacy Grid Architecture; On-site renewable generation
1. Fare equity programs; Subsidy design; ...

Return ONLY valid JSON, labels in the same order:
{"labels": ["...", "..."]}
```

## Persona topic headers (the older "persona table")
`simulation/persona_topics.py` — ~10 top-level headers each with 2-3 sub-headers:
```
I am the following participant:
{persona}

I will be engaging in deliberations about: {mission}

Give me a list of about 10 TOP-LEVEL topics that I will probably be interested in
contributing to in those deliberations. For EACH top-level topic, also break it into
2-3 finer SUB-TOPICS that capture the distinct facets I most likely care about.

Every topic and sub-topic needs: a short header (2-5 words), a one-sentence reason it
fits my interests, and a CONFIDENCE from 0.0 to 1.0…

Return ONLY valid JSON (no markdown, no code fences):
{"topics": [{"label": "...", "rationale": "...", "confidence": 0.85,
             "subtopics": [{"label": "...", "rationale": "...", "confidence": 0.8}]}]}
```

Plus a per-move scorer (`_action_prompt`) rating each header 0–1 across
goal / barrier / cause / solution / abstraction / analogy / inspiration / question /
answer / pro / con, and a post-hoc quality judge (`quality_judge.py`).

---

# Summary: how many LLM calls per tick?

Per specialist agent, per tick, in `multi_agent` mode:
1. **Profile Analyst** — 1 call (300 tok)
2. Embeddings for candidate posts — cached in `_POST_EMB_CACHE`, so ~0 after warmup
3. **Content Selector** — 1 call (200 tok)
4. **Specialist decision** — 1 call (800 tok) — *skipped entirely in synthetic mode*
5. **Skip-reason elicitation** — 1 call (200 tok), only when skipping without a reason

Plus per tick: 1 Facilitator call, and 1 Synthesizer call every 5 ticks.
Measured throughput: **~12.3 s/tick with 6 agents** (~0.5 agent-decisions/sec).
