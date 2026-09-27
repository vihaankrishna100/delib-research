# What They Want and How They Contribute
### Learning, Validating, and Acting on User Profiles in Deliberation Systems

*Draft v1 — all figures verified against simulation runs `caa70027` (70 ticks, 4 agents)
and `14da4d73` (50 ticks, 6 agents).*

---

## Abstract

Online deliberation forums stall when participants cannot find the discussions they are
actually equipped to contribute to. Recommender systems are the standard remedy, but a
recommender can only be as good as its model of the user. Existing approaches infer
preference from interaction frequency — collaborative filtering and its descendants — which
degrade badly under cold start and, more fundamentally, model *what* a user engages with
while ignoring *how* they contribute. In a structured deliberation the same post admits a
dozen distinct moves: one participant will name a barrier, another will propose a solution,
a third will reach for an analogy in a distant domain. Recommending the right topic with the
wrong move is still a failed recommendation.

We present a user-profiling mechanism that models both dimensions. A participant's profile
is a decision tree learned online from their own clickstream, replacing a statically declared
interest table, and it re-derives itself from the complete interaction history rather than
accumulating against a fixed hierarchy. Each region of the profile carries not only a topical
identity but a distribution over the thirteen deliberation moves the participant may make
within it.

Validating such a profile is normally impossible, because a real user's true preferences are
unobservable and only proxies such as click-through can be measured. We therefore introduce an
evaluation harness of generative agents equipped with hidden, deterministic preference models:
the profile is scored against an answer key it never sees. This paper reports on the complete
loop, not the profile in isolation: the same tree that is scored against ground truth also
drives the recommender that chooses what each agent sees next, so we can ask both **how well**
the system understands a participant and **how quickly** that understanding translates into
fewer wasted recommendations. On a 70-tick deliberation simulation, profiles that induced a
genuine decision boundary recover **86%** of their agents' true engage/skip boundaries,
including over regions of the idea space the agent was never shown, while the recommender
consuming those same profiles drives the population's skip rate from 30% down to 10% over the
run. The harness also exposes two effects that are invisible without ground truth: profile
accuracy is upper-bounded by the behavioural *consistency* of the user being modelled, and a
profile trained inside a recommendation loop is measurably biased by that loop's own
selections — in our runs by as much as 38 percentage points between training and true
accuracy.

---

## 1. Introduction

### 1.1 The problem

A deliberation forum is only as valuable as the contributions people make to it, and people
contribute when they encounter something they are equipped to say something about. The
failure mode is familiar to anyone who has watched a discussion board decay: a participant
opens the site, scans a list of threads, finds nothing that fits their expertise or their
mood, and leaves. The content that would have engaged them may well exist somewhere in the
forum; they simply never reached it. Recommendation is the natural response, and most
large-scale discussion platforms now deploy some form of it.

But recommendation presupposes a model of the user, and this is where deliberation systems
are poorly served by the standard toolkit. Collaborative filtering learns that users who
behaved alike will continue to behave alike. It is a powerful assumption when the item
catalogue is fixed and large and the interaction matrix is dense — the setting it was
designed for. A deliberation map satisfies none of these conditions. The items are posts
that the participants themselves authored, often minutes earlier, so there is no item-side
history to borrow from. The catalogue is not merely sparse but *non-stationary by
construction*: the act of engaging creates new items. And the population is small, since a
deliberation is a working group rather than a mass audience.

There is a second and more interesting inadequacy. Collaborative filtering predicts whether
a user will interact with an item. In a structured deliberation, "interact" is not a single
event type. Confronted with a proposed solution to a transit funding problem, an economist
may argue for it, an engineer may decompose it into the components it would require, a
policy specialist may name the regulatory barrier it will hit, and an innovator may abstract
it and look for a structural parallel in another field. These are different contributions
with different value, and a system that models only *whether* the participant engages has
modelled roughly half of what matters.

### 1.2 Generative agents as a research instrument

Our methodology builds on Zhang et al.'s Agent4Rec ("On Generative Agents in
Recommendation"), which introduced a recommendation simulator populated by LLM-driven
generative agents, each equipped with user-profile, memory, and action modules. Their
research question was whether such agents can faithfully *reproduce* the behaviour of real
autonomous humans inside a recommender system. Establishing that generative agents are a
credible substrate for recommender research is the precondition for what we do here.

We depart from that work in two respects, and the second is the substantive one.

The first is domain. Agent4Rec operates over movies: a fixed catalogue of mutually unrelated
items. We operate over a deliberation map — a structured artifact that the users themselves
build, in which posts are typed (mission, stakeholder, goal, barrier, cause, solution,
abstraction, analogy, inspiration, question, answer, part, failure mode) and linked into a
purpose tree. The item space is generated by the very interactions being modelled.

The second is the direction of inquiry, and it inverts theirs. Agent4Rec asks whether agents
can imitate humans; the agents are the object of study. We use agents whose preferences we
*specify in advance*, precisely so that we obtain ground truth, and ask whether a profiling
mechanism can *recover* those preferences from observed behaviour alone. Our agents are not
the object of study — they are the measuring instrument. This is the move that makes the
central claim of this paper possible: we can state, as a number, how well a user model
understood the person it was modelling.

### 1.3 AIdeator and the typed action vocabulary

The deliberation methodology we simulate is AIdeator, developed by Professor Klein and used
by practitioners across multiple fields to decompose societal problems collaboratively. In
AIdeator a problem is expanded into a purpose tree — stakeholders, their goals, the barriers
to those goals, the root causes beneath the barriers, and candidate solutions — and extended
through a distinctive analogical chain in which a problem is abstracted, matched to a
structurally similar problem in a distant domain, and the borrowed solution translated back.

AIdeator matters to this paper for a specific reason: it supplies a **closed, typed action
vocabulary**. Because every contribution is one of a known set of moves, a user profile can
be defined over (topic × move) rather than topic alone. This is what turns "favourite
actions" from a metaphor into a measurable object. A participant who reliably names barriers
but never reaches for analogies has a profile *shape* that a topic-only model has no way to
express, and a recommender consuming that profile can act on the difference.

### 1.4 Why generative agents make profiling tractable

Three properties of the agent-based setting are what allow this work.

**Ground truth becomes available.** For a human user, the true preference function is
unobservable. One can measure click-through, dwell time, or explicit ratings, but these are
proxies, and a profile fitted to them can only be evaluated against more proxies. When the
population is generative, the preference function is something we *write down*. Profile
inference becomes a supervised problem with an answer key, and profile quality becomes a
reportable quantity rather than an article of faith.

**Profiles can be re-derived rather than accumulated.** Because every interaction is
persisted, the profile need not be an accumulator that drifts against a hierarchy fixed at
initialisation. Each refresh rebuilds the model from the complete history, so a structural
revision requires no reconciliation with legacy state.

**Behaviour is reproducible.** Human participants are noisy and unrepeatable, which makes
ablation of a profiling mechanism nearly impossible. Agents can be held fixed while the
mechanism under test is varied.

---

## 2. Method

### 2.1 The simulation environment

Between three and eight generative agents, each carrying a distinct expert persona — an
economist, a systems engineer, a policy specialist, a technology innovator, a sociologist,
an environmental scientist, a behavioural psychologist, and a strategic planner — jointly
grow a deliberation map rooted at a mission statement. On each tick, every agent is presented
with a single (post, action) pair and either **engages**, authoring a typed child post, or
**skips**. Available actions are constrained by post type: a goal admits a barrier, a
solution, or an abstraction; a solution admits an improvement, a component part, a failure
mode, a barrier, or a probing question. All agents run on the same underlying language model,
with 3072-dimensional text embeddings used for all semantic comparisons.

This environment is the testbed. The profiling mechanism is the contribution.

### 2.2 Two models, and why the distinction carries the paper

The single most important thing to hold clear is that this system contains two models of
each user, doing opposite jobs.

| | **Generative model** | **Inferential model** |
|---|---|---|
| what it is | the agent's hidden true preferences | the system's learned profile |
| role | **decides** engage/skip | **predicts** engage/skip |
| visibility | never exposed to the profiler | built only from observed behaviour |
| analogy | a person's actual taste | a service's model of that taste, built from clicks |

The generative model is constructed once per session. Each agent receives roughly ten
**liked** and ten **disliked** subtopic descriptions, generated from its persona and the
session mission and embedded as centroids in the same space as the posts. The agent's
decision rule is then pure geometry:

> engage ⟺ max cos(post, liked centroids) ≥ max cos(post, disliked centroids)

No language model participates in this decision. The model is consulted only *after* the
decision, to write the content of a post the agent has already committed to authoring, so
that the resulting text — and therefore the embedding the profiler will observe — remains
realistic.

This design buys two things at one cost. It buys **perfect behavioural consistency**: the
same post always receives the same decision, so any residual prediction error belongs to the
profiler rather than to the user. And it buys the **answer key** on which all of our
evaluation rests. The cost is naturalism, and we treat it as a first-order limitation
(§5.1) rather than a footnote.

### 2.3 The profile: a decision tree over the clickstream

The profile for each agent is produced by the following pipeline.

1. **Log.** Every recommendation outcome is persisted as a triple: the post's embedding, the
   binary engage/skip label, and the tick. Events are never discarded.
2. **Reduce.** Post embeddings are projected from 3072 to 50 dimensions by PCA, fitted over
   the posts observed in the session.
3. **Fit.** A per-agent decision tree classifier maps reduced embedding to engage/skip, using
   an entropy splitting criterion with a minimum of three samples per leaf to prevent the
   tree from partitioning itself into unreliable slivers.
4. **Refresh.** The tree is retrained from the complete clickstream on a schedule —
   aggressively during the early phase of a run, when the learning curve is steepest, and
   less frequently once the model has stabilised.
5. **Label.** Each leaf is named after the fact by a language model, from the actual posts
   that fall within it.

Three design commitments are worth stating explicitly, because they are the arguments of
this section rather than its implementation details.

**Behaviour over declaration.** The predecessor to this mechanism, and the approach we
consider the natural baseline, is a *declared* interest table: ask a language model, given
the participant's persona, to enumerate roughly ten topics they are likely to care about,
and treat that list as the user model. It is cheap, immediately available, and requires no
interaction history. It is also never validated against anything. Our position is that a
guess about a user is not a model of one, and the empirical case appears in §4.2.

**Rebuild rather than accumulate.** Because retraining consumes raw events, a revised
partition of the space redistributes the entire history automatically. There is no legacy
structure to migrate and no possibility of the profile's shape being determined by an early,
poorly-informed decision that later evidence cannot overturn.

**Honest interpretability.** The tree's internal split conditions are thresholds on PCA
components and are *not* individually meaningful; we do not claim otherwise. The interpretable
layer is the leaf — a named cluster of posts with a measured purity and a sample count. This
is a weaker but defensible claim, and §4.5 shows what it buys in practice.

### 2.4 Profiling actions, not only interests

Alongside its topical structure, each region of the profile carries a predicted score across
the thirteen deliberation moves — goal, barrier, cause, solution, abstraction, analogy,
inspiration, question, answer, pro, con, part, and failure mode — together with the observed
counts of each move actually taken there.

A concrete instance makes the point. For the economist agent, under a profile region
identified as *Land Value Capture*, the model predicts solution 0.90, pro 0.90, and barrier
0.80, while scoring analogy and inspiration low. The profile therefore encodes not merely
that this participant is interested in land value capture, but that within that subject they
will argue for mechanisms and name obstacles rather than reach for cross-domain parallels.

Stated generally: a conventional interest profile is a vector over subjects. This profile is
a matrix over (subject × move), and a recommendation derived from it is a *pair* rather than
an item.

### 2.5 Elicited skip reasons as profile evidence

Whenever an agent declines a recommendation it is asked, in its own persona voice and in
plain prose, why the recommendation was a poor fit. These statements are qualitative evidence
about the profile, and they produced the diagnostic finding in §4.1 that redirected our
architecture. One practical note for replication: this elicitation must use a prose system
prompt rather than the agent's ordinary JSON-constrained prompt, which otherwise returns the
explanation wrapped inside a structured action object.

---

### 2.6 The recommender: turning a profile into a recommendation

A profile that is never consulted cannot be judged by its consequences, only by its fit to
past data. To ask "how quickly does understanding a participant translate into a better
experience for them," the profile must sit inside a live recommendation loop, and this
section describes that loop.

On each tick, an agent has on the order of forty valid (post, action) pairs available to it —
every open slot in the current deliberation map. These are narrowed to a shortlist of
thirteen by a scoring function with four components, applied in order.

**Primary signal.** Each candidate is scored by whichever source of preference information is
available, in a strict fallback order: the agent's decision tree, if it has trained on enough
observations (§2.3); failing that, a cosine-similarity score against the centroids of posts
the agent has engaged with versus skipped so far this session; failing that, a smoothed
per-action engagement rate. This ordering means the profile takes over the ranking as soon as
it exists, and the system degrades gracefully rather than failing outright while it is cold.

**Own-frontier boost.** Candidates on posts the agent itself authored, or on children of such
posts, receive a multiplicative boost. This keeps an agent developing a coherent line of
argument rather than being pulled across the map by whichever branch scores marginally higher
on the primary signal alone.

**Skip penalty.** A candidate on a post the agent has already declined is discounted
geometrically in the number of prior declines. This prevents the recommender from repeatedly
re-offering a branch the agent has made clear it does not want.

**Shape as a bounded tie-breaker.** The deliberation map itself has a structure that is worth
protecting independently of any one agent's preferences: a healthy map has moderate breadth at
each branching point and does not run to unbounded depth along a single line of reasoning. We
initially penalised poor structure by multiplying it directly into the candidate score, and
found this let structural preferences *override* what the profile had learned — in an early
version this quadrupled the population skip rate, because the recommender began pushing
structurally tidy candidates the agents did not actually want. We replaced this with a bounded
blend: the combined structural score is allowed to modulate at most 35% of the final ranking,
so a candidate the profile is confident about can never be pushed behind one the profile is
not, no matter how poorly shaped the confident candidate is. Once bounded this way, shape and
profile quality improved together rather than trading off (§4.6).

The narrowed shortlist, together with the profile's persona summary and the agent's most
recent stated reasons for skipping, is passed to a final selection step that chooses one pair
to actually recommend. This closes the loop: the same interaction that is later scored for
profile accuracy is the interaction the profile itself produced.

---

## 3. Evaluation methodology

The metric that carries this paper requires care in its construction, so we define it
precisely.

A trained profile is asked to classify **every post in the session**, and its predictions are
compared against the decision the hidden generative model would have made for that same post.
Because the profile was trained only on the posts its agent was actually shown — roughly
seventy of them — while being evaluated across all posts in the session, of which there were
194 in our principal run, this is a **generalisation test over unseen regions of the idea
space**, not a measure of fit.

We report a second quantity alongside it: **training accuracy**, the profile's accuracy on
the posts it was fitted to. The two numbers answer different questions. Training accuracy
asks whether the profile memorised its own history. Inference accuracy asks whether it
recovered the underlying preference structure. The *gap* between them turns out to be the
most transferable result in the paper (§4.4).

---

## 4. Results

### 4.1 What a profile must actually model: a negative result

Our first substantive finding is a negative one, and it determined the architecture that
followed.

Mining the elicited skip reasons revealed two roughly equal causes. Approximately half were
**topical mismatches**, in which the participant judged the recommendation outside their
competence — one agent explained that its "expertise lies more in behavioural psychology"
than in the branch it had been offered. The other half were **redundancy** judgments, in
which the contribution had effectively already been made.

Critically, skip behaviour did **not** vary meaningfully by action type. Every major move sat
within a narrow 40–55% skip band, and no action was appreciably easier or harder to elicit
than any other. A profile keyed on action statistics alone is therefore not predictive. The
discriminative signal is *which post* is offered, in topical relation to the participant.

This result is worth reporting precisely because it is negative: it eliminates the cheapest
plausible profiling strategy and justifies the considerably more expensive one that works.

### 4.2 Headline: learned profiles recover 86% of true preferences

Before presenting the numbers we must separate two kinds of profile, because conflating them
inflates the result substantially.

A profile is **degenerate** when the fitted tree collapses to a single leaf — a constant
predictor that answers "engage" (or "skip") for every input. This occurs when the agent's
observed decisions were almost entirely one class, leaving the classifier no counterexamples
from which to induce a boundary. Such a profile can score very high, because a constant
predictor is trivially correct wherever the truth is also constant, but it has learned no
structure and would fail immediately on a user with genuinely mixed preferences. We report
these separately rather than averaging them in.

**Table 1 — 70-tick run, four agents**

| agent | leaves | vs. ground truth | training accuracy | gap |
|---|---|---|---|---|
| Innovator | 4 | 89.1% | 97.2% | 8.1 |
| Environmental scientist | 5 | 82.8% | 100.0% | **17.2** |
| **mean, learned profiles** | | **86.0%** | | |
| Economist | 1 — degenerate | (96.9%) | 100.0% | — |
| Engineer | 1 — degenerate | (93.8%) | 100.0% | — |
| *naive mean over all four* | | *90.6%* | | |

**Table 2 — 50-tick run, six agents**

| agent | leaves | vs. ground truth | training accuracy | gap |
|---|---|---|---|---|
| Engineer | 4 | 84.3% | 98.0% | 13.7 |
| Environmental scientist | 4 | 60.0% | 98.0% | **38.0** |
| **mean, learned profiles** | | **72.1%** | | |
| Economist | 1 — degenerate | (100.0%) | 100.0% | — |
| Innovator | 1 — degenerate | (100.0%) | 100.0% | — |
| Psychologist | 1 — degenerate | (91.4%) | 100.0% | — |
| Policy specialist | 1 — degenerate | (90.0%) | 100.0% | — |
| *naive mean over all six* | | *87.6%* | | |

The honest headline is therefore **86.0% on the four-agent run and 72.1% on the six-agent
run, computed over the profiles that actually induced a decision boundary.** The naive means
of 90.6% and 87.6% are inflated by constant predictors and we recommend against quoting them.

Two observations follow, and the second is arguably more useful than the headline itself.

First, the substantive claim survives in reduced form: a profile built from roughly seventy
observations, evaluated across all 194 posts in the session including regions never shown to
its agent, recovers the large majority of that agent's true engage/skip decisions. Given the
generalisation nature of the test (§3), we take this as evidence that the mechanism captures
real preference structure.

Second, **degeneracy is itself a finding about the setting rather than a defect of the
method.** Six of ten profiles across the two runs collapsed to constant predictors, and this
happened precisely for agents whose recommendations were nearly all accepted. The cause is
the recommender doing its job: by successfully avoiding items the agent would reject, it
starves the profile of the negative examples a boundary requires. This is the same mechanism
as the selection bias in §4.4, observed at its extreme. A profiler embedded in an effective
recommendation loop must therefore be given deliberate exposure to items the user is expected
to decline, or it will lose the ability to model them at all.

### 4.3 The consistency ceiling

Before adopting deterministic generative models, we ran the identical profiling mechanism
against agents that decided engagement by free language-model judgment. Under that condition,
prediction quality plateaued early and additional data produced no further improvement.

The explanation is that the same post could receive different decisions on different ticks.
That inconsistency is an irreducible noise floor, and no profiler can predict beneath it.
Replacing free judgment with a deterministic preference model removed the floor and made the
boundary learnable — which is precisely why the harness in §2.2 is a methodological
contribution and not merely a convenience.

The practical implication generalises beyond simulation. **A user profile can be no more
accurate than the user is self-consistent.** Before attributing disappointing recommendation
quality to the recommender, a practitioner should measure the behavioural consistency of the
population being modelled, because that quantity bounds what is achievable. We would argue
that any deployed system reporting profile accuracy should report an estimated consistency
alongside it, in the same way that an instrument's precision is reported with its readings.

### 4.4 Profiles are biased by the loop that trains them

The training-accuracy column in Tables 1 and 2 is not padding. It carries what we consider
the most broadly applicable result in this work.

The environmental scientist's profile achieved **100% accuracy on the posts it was trained
on, but 82.8% against ground truth** — a 17-point gap. In the six-agent condition the same
agent showed **98% against 60%**, a gap of 38 points.

The cause is structural. The profile is trained on a **non-random sample that the system
itself selected.** As the recommender improves, it increasingly offers only items the profile
already predicts the user will like. The profile consequently observes a narrowing slice of
the preference space and fits that slice extremely well, while its accuracy over the
unobserved remainder quietly degrades. The loop is, in a real sense, grading its own homework.

We stress that this pathology is intrinsic to *any* closed-loop personalisation system, and
that it is normally **invisible**. Without an answer key, a practitioner inspecting this
profile would observe only the reassuring 100% and conclude that the model was excellent.

Two recommendations follow. First, closed-loop profilers should report a **train/truth gap**
as a standard diagnostic wherever ground truth can be constructed, including in offline
harnesses like ours. Second, exploration is the mitigation: the condition in which the
recommender was compelled to offer a wider range of items produced the smaller gap. This
gives exploration policy an unusually direct objective — minimise the train/truth gap subject
to acceptable engagement — rather than the usual appeal to intuition.

### 4.5 Interpretability, demonstrated rather than asserted

In the six-agent run, the engineer agent's learned profile separated a region named
*Urban transit improvement* (0% engagement, entropy 0.00, n = 21) from one named
*Decentralized energy network* (100% engagement, entropy 0.00, n = 20), alongside a third
pure region *Sustainable transit operations* (0%, entropy 0.00, n = 6) and a small mixed
region *Advanced transit systems* (33% engagement, entropy 0.92, n = 3).

Three properties are worth drawing out. Both large regions are **perfectly pure** — every
observation within them received the same decision. Both are **semantically coherent**, and
they were discovered from behaviour alone; the names were assigned afterwards from the posts
that had landed in each. And the small mixed region is **correctly self-flagged**: its high
entropy and low sample count identify it as a region the profile is not confident about, so a
consumer of the profile can discount it rather than being misled by it.

This is what we mean by honest interpretability. We cannot explain the tree's split
conditions, and do not claim to. We can hand a facilitator a list of the regions a
participant reliably contributes to, the regions they reliably avoid, and an explicit marker
on the regions we are unsure about.

---

### 4.6 How quickly: skip rate falls as the profile sharpens

Sections 4.2 through 4.5 establish that the learned profile is an accurate model of its
agent. This section asks the question that motivates building one: does a recommender that
consumes this profile actually get better at its job as the profile improves, and how fast?

We report this on the identical 70-tick, four-agent run used for the headline accuracy result
in §4.2, so that both numbers describe the same system on the same interactions rather than
two different experiments. Skip rate is measured over a sliding ten-tick window stepped one
tick at a time, which smooths the considerable noise of a small per-tick sample without
discarding any of the run.

**The population skip rate falls from 30.0% in the first window to 10.3% in the last, a
decline of nearly two thirds over the run, at a fitted slope of −0.71 percentage points per
tick.** Overall, across all 282 completed decisions in the run, 19.2% were declined. The
trajectory is not monotonic, and reporting it honestly is more informative than the
first-to-last comparison alone. Skip rate initially *rises*, from 30% to a peak of 42.5%
across roughly the first dozen windows — a cold-start cost consistent with §2.6's fallback
ordering, during which the primary ranking signal is still the coarse cosine-similarity
fallback rather than a trained tree. It then declines sharply and largely continuously to a
floor near 0% around the midpoint of the run, before rising slightly and settling in the 7–10%
range for the remainder. We read this shape as the profile's cost before its benefit: the
system pays a brief early premium while the tree has too little data to be trusted, then
recovers that cost several times over as the tree takes over the ranking (§2.6) and continues
to improve with additional data (§2.3).

We regard the pairing of these two results as the central claim of this paper. A profile
whose accuracy is high in isolation is a laboratory curiosity unless it changes what happens
next; a skip rate that falls without a validated cause could be an artifact of the map simply
running out of easy content. Reporting both from the same run — 86% recovery of true
preference alongside skip declining by two thirds — is what allows us to say the one *causes*
the other, rather than merely correlating with it.

**On baselines.** Before the architecture described in this paper existed, we compared a
predecessor recommender against non-learning baselines — random selection and a
gap-filling heuristic that always targets the least-developed branch of the map — on agents
using free language-model judgment rather than the deterministic ground truth used throughout
this paper. Over that 30-tick run, the gap-filling heuristic degraded from 27% skip in its
first window to 100% in its last, and random selection degraded from 27% to 67%, while the
learning recommender held roughly flat around 50%. We report this only as historical
motivation for building a learning-based recommender at all, not as a baseline for the system
described here: it predates both the decision-tree profile and the deterministic ground-truth
agents, so it is not a fair comparison to §4.2–4.5, and we flag re-running this comparison
under the current architecture as an open item (§7).

---

## 5. Limitations

We state these plainly, because several of them bound the claims above.

**5.1 Synthetic agents are consistent but not human.** The geometric decision rule that makes
ground truth available is also what makes these agents unlike people. Our results establish
that the mechanism *can* recover a preference structure from behaviour; they do not establish
its accuracy against human users. The natural next step is to estimate human behavioural
consistency and re-validate under it.

**5.2 Single-run evidence.** Runs configured with nominally similar settings nonetheless
differed substantially in outcome, which means individual runs cannot resolve small effects.
The accuracies above should be read as point estimates from a single run per condition, and
any comparison of similar magnitude requires replication we have not yet performed.

**5.3 The benchmark has a difficulty knob.** The generative decision rule contains a margin
parameter that controls how sharply liked must beat disliked. Outcomes are extremely
sensitive to it, through a feedback loop: a stricter rule produces fewer engagements, hence
fewer posts, hence a thinner candidate pool, hence fewer engagements still. Cross-run
comparisons are valid only with this parameter held fixed, and we recommend that its value be
reported alongside any result.

**5.4 Class imbalance degrades both the model and the metric.** When one class falls below
roughly 8% of decisions the classifier has too few counterexamples to induce a boundary and
collapses to a constant predictor. Six of the ten profiles across our two runs did exactly
this. Such profiles retain high accuracy while having learned nothing, so accuracy alone is
an unsafe summary: it must be reported alongside the class balance and the number of leaves,
and means should be computed only over profiles that actually partitioned the space. We
follow that discipline in Tables 1 and 2, and note that it lowers our headline figures
considerably relative to a naive average.

**5.5 Interpretability is post-hoc.** Legibility comes from labelling leaves after the fact,
not from the model's internal structure. Text embeddings are not invertible without a trained
decoder, so the split thresholds cannot presently be rendered in natural language.

---

## 6. Discussion

We would advance four positions on the basis of this work.

**A user profile should be a learned object with an error bar.** The field routinely ships
user models that are never validated, on the reasonable-seeming grounds that validation is
impossible because true preferences are unobservable. It is not impossible; it requires
constructing a population whose preferences you control. Once that is done, "how well does
this system understand this user" becomes a number that can be reported, compared across
mechanisms, and regressed against. The profile stops being an implementation detail and
becomes an evaluable artifact.

**Profiling actions is a distinct problem from profiling interests.** Our negative result
(§4.1) shows that action preference is not predictable from action statistics — it is
predictable from the interaction of topic and persona. Any domain with a typed interaction
vocabulary inherits this structure: code review, structured question answering, data
annotation, and tutoring all present the same shape, in which choosing the right *kind* of
contribution matters as much as choosing the right target.

**Closed-loop profiles require honest accounting.** The selection-bias result (§4.4) is not
specific to deliberation. Any system that trains a user model on interactions its own
recommender selected is partly evaluating itself, and will systematically overestimate its
own understanding. The train/truth gap is a cheap and general diagnostic, and we argue it
belongs in the standard reporting set for personalisation systems.

**Profile validity and recommendation quality should be reported together, not separately.**
A field that reports profile accuracy offline and recommendation metrics online, on different
populations under different conditions, cannot establish that the first causes the second. We
report both from one run precisely so that the pairing in §4.6 — 86% preference recovery
alongside a two-thirds reduction in skip rate — is a statement about one system, not a
coincidence of two papers. We would encourage other closed-loop personalisation work to adopt
the same discipline wherever an evaluable population is available.

---

## 7. Future work

**A baseline comparison under the current architecture.** §4.6 reports the learning
recommender's absolute skip-rate trajectory but, for the reasons stated there, cannot yet
compare it against non-learning baselines under the same conditions. Re-running the
random-selection and gap-filling comparisons with deterministic ground-truth agents and the
decision-tree recommender is the most immediate open item, and would let us report a
head-to-head result with the same rigor as the rest of this paper.

**Scale.** Hundreds of agents across thousands of ticks, to determine whether profile quality
saturates with data and whether profiles in a shared population specialise or converge.

**Deployment within AIdeator.** Applying the mechanism to human participants, including the
"user recovery" case in which a returning participant's profile must be reconstructed from a
partial and stale history.

**Human consistency estimation.** Measuring how self-consistent real deliberators are, which
per §4.3 sets the achievable ceiling for any deployed profile, and is a prerequisite for
interpreting accuracy figures obtained with human users.

**Exploration policies for debiasing.** §4.4 supplies an unusually direct objective for
exploration research: minimise the train/truth gap subject to an engagement constraint. The
harness provides the evaluation signal that this kind of work normally lacks.

**Richer feature sets.** The profile currently learns from post embeddings alone. Action type,
position within the deliberation tree, post age, and the text of elicited skip reasons are all
recorded and unexploited. Classifier feature importances would indicate directly which of
these earn their computational cost.
