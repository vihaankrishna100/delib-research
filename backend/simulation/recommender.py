import asyncio
import json
import random
import re
from datetime import datetime
from google import genai
from google.genai import types as genai_types
import os
from sqlalchemy.orm import Session
from models import Event, DeliberationPost, AgentProfile, RecommendationLog, PersonaTopicHeader
from simulation.user_model_tree import predict_engage_prob, has_model

FLASH_MODEL = "gemini-2.5-flash"
_client = None

# Post-text embeddings are stable, so cache them across ticks/agents to keep the
# semantic-fit signal cheap (each distinct post is embedded at most once).
_POST_EMB_CACHE: dict[str, list] = {}

# ── Tree-shape controls (breadth vs depth) ───────────────────────────────────
# History (both failure modes measured on 70-tick runs):
#   * A pure 1/(1+existing) saturation term penalised the SECOND sibling immediately,
#     driving the map depth-first: 1.08 solutions/parent but 12 levels deep, with
#     7-long `add_improvement` chains drifting away from the originating barrier.
#   * Correcting that with hard multiplicative penalties fixed the shape (2.75
#     solutions/parent, depth 6) but QUADRUPLED the skip rate (6.8% -> 26.4%) and
#     flattened learning: shape was overriding the learned preference model instead
#     of regularising it.
# Resolution: keep the knobs gentle AND apply shape as a BOUNDED TIE-BREAKER — it can
# reorder candidates of similar predicted engagement, but can never drive a
# high-probability candidate below a low-probability one.
TARGET_BREADTH = 2          # children of a type that are "free" before saturating
DEPTH_SOFT_CAP = 8          # depth beyond which candidates start being penalised
DEPTH_DECAY = 0.85          # multiplier per level past the soft cap
MAX_SAME_TYPE_CHAIN = 2     # consecutive same-ptype ancestors before penalty
CHAIN_DECAY = 0.60          # multiplier per level past the chain cap
# Fraction of the score that shape may modulate. At 0.35 the worst possible shape
# keeps 65% of the candidate's predicted engagement, so shape breaks ties rather
# than vetoing what the user model actually wants.
SHAPE_WEIGHT = 0.35

# Exploration schedule. Fast = wide early sampling that tapers; slow = the flat
# baseline. Toggled per session so the learning-speed change can be ablated.
FAST_LEARNING = True

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client

_ANALYST_SYSTEM = (
    "You are a behavioral profile analyst for a deliberation simulation. "
    "Your job is to maintain concise, accurate contribution profiles for deliberation participants "
    "based on their activity. Output only the updated profile text, nothing else."
)

_SELECTOR_SYSTEM = (
    "You are a recommendation engine for a deliberation map simulation. "
    "You are given two models of the expert: (1) a PERSONA TABLE of topics the "
    "expert said up-front they want to contribute to, and (2) a BEHAVIORAL PROFILE "
    "learned from their recent clickstream. You also get the reasons the expert "
    "previously SKIPPED recommendations. Combine all three — persona interests, "
    "observed behavior, and skip rules — to select the single post/action pair the "
    "expert is most likely to engage with. Treat skip reasons as negative weights: "
    "avoid pairs resembling what the expert rejected. "
    "Also return a CONFIDENCE from 0.0 to 1.0 — how sure you are the expert will engage "
    "with the selected pair rather than skip it. "
    'Return ONLY valid JSON: {"selected_index": 0, "confidence": 0.8, "reasoning": "one sentence"}'
)


# Cap each call so one stalled request can't hang a whole multi-tick run.
LLM_TIMEOUT_S = 90


async def _gemini(system: str, prompt: str, max_tokens: int = 300) -> str:
    from simulation.roles.base import call_with_retry
    resp = await call_with_retry(lambda: _get_client().aio.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.4,
            # Disable thinking so the whole token budget goes to the answer — otherwise
            # the selector's JSON gets truncated mid-string and silently falls back to random.
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    ))
    return (resp.text or "").strip()


class RecommendationEngine:
    # ── Simple strategies ────────────────────────────────────────────────────

    def recommend_random(self, valid_pairs: list) -> dict:
        return random.choice(valid_pairs)

    def recommend_priority(self, valid_pairs: list, all_posts: dict) -> dict:
        """
        Gap-filling: prioritize post/action pairs where the target ptype
        has fewest existing achievers (underexplored branches).
        """
        from simulation.roles.specialist import ACTION_TO_PTYPE

        def gap_score(pair):
            post = pair["post"]
            action = pair["action"]
            target_ptype = ACTION_TO_PTYPE.get(action, "solution")
            count = sum(
                1 for p in all_posts.values()
                if p.get("purpose_id") == post["id"] and p.get("ptype") == target_ptype
            )
            return 1.0 / (1.0 + count)

        scored = sorted(valid_pairs, key=gap_score, reverse=True)
        return scored[0]

    # ── Multi-agent pipeline ─────────────────────────────────────────────────

    async def recommend_multi_agent(
        self,
        valid_pairs: list,
        agent_id: str,
        agent_name: str,
        agent_domain: str,
        db: Session,
        session_id: str,
        tick: int,
        all_posts: dict | None = None,
    ) -> tuple[dict, str, float | None]:
        clickstream = self._build_clickstream(db, agent_id, session_id, tick)

        profile_row = db.query(AgentProfile).filter(AgentProfile.agent_id == agent_id).first()
        existing_profile = profile_row.profile_text if profile_row else ""

        # Profile Analyst
        analyst_prompt = self._build_analyst_prompt(agent_name, agent_domain, existing_profile, clickstream)
        try:
            updated_profile = await _gemini(_ANALYST_SYSTEM, analyst_prompt, max_tokens=300)
        except Exception:
            updated_profile = existing_profile or f"{agent_name} is a {agent_domain} expert."

        self._upsert_profile(db, agent_id, session_id, tick, updated_profile, profile_row)
        db.commit()

        # Persona table: the up-front topic headers this agent said it cares about.
        persona_topics = self._persona_topics(db, agent_id, session_id)
        # Skip-reason weighting rule: recent reasons this agent gave for skipping.
        skip_reasons = self._recent_skip_reasons(db, agent_id, session_id)

        # ── Closed feedback loop ─────────────────────────────────────────────
        # Learn, from the outcomes accumulated so far THIS session, how often this
        # agent actually engages with each recommended action vs skips it. Narrow the
        # candidate pairs to the ones the agent is statistically most likely to act on
        # (with a little exploration), so the skip rate should fall as the run proceeds
        # and the estimates sharpen. Random/priority strategies get no such loop — that
        # contrast is the point of the experiment.
        learned = self._learned_engagement(db, agent_id, session_id)
        # Primary learned signal: how well each candidate post matches what THIS agent
        # has actually engaged with vs skipped so far (semantic fit). This is what
        # predicts skips — agents reject off-expertise branches, not action types — so
        # as its engaged/skipped history grows, fit sharpens and the skip rate falls.
        try:
            fit_scores = await self._semantic_fit_scores(db, agent_id, session_id, valid_pairs)
        except Exception:
            fit_scores = {}   # degrade to the per-action fallback; never wedge the run

        # Decision-tree user model: if trained, its predicted engage-probability is the
        # primary ranking signal (replaces the cosine fit). Reuses cached post embeddings.
        tree_scores = None
        if has_model(session_id, agent_id):
            tree_scores = {}
            for p in valid_pairs:
                pid = p["post"]["id"]
                vec = _POST_EMB_CACHE.get(pid)
                if vec is not None:
                    prob = predict_engage_prob(session_id, agent_id, vec)
                    if prob is not None:
                        tree_scores[pid] = prob

        ranked_pairs = self._rank_pairs_by_learning(
            valid_pairs, learned, all_posts=all_posts, fit_scores=fit_scores,
            agent_id=agent_id, tree_scores=tree_scores,
        )

        # Content Selector — sees persona table + behavioral profile + skip rules +
        # the learned engagement rates.
        selector_prompt = self._build_selector_prompt(
            updated_profile, ranked_pairs, persona_topics, skip_reasons, learned
        )
        reasoning = ""
        confidence: float | None = None
        selected = None
        try:
            raw = await _gemini(_SELECTOR_SYSTEM, selector_prompt, max_tokens=200)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            result = json.loads(raw)
            idx = int(result.get("selected_index", 0))
            if 0 <= idx < len(ranked_pairs):
                selected = ranked_pairs[idx]
            reasoning = result.get("reasoning", "")
            try:
                confidence = max(0.0, min(1.0, float(result.get("confidence"))))
            except (TypeError, ValueError):
                confidence = None
        except Exception:
            pass

        if selected is None:
            # Fall back to the highest learned-engagement candidate, not a blind random.
            selected = ranked_pairs[0] if ranked_pairs else random.choice(valid_pairs)

        return selected, reasoning, confidence

    # ── Online learning from outcomes ────────────────────────────────────────

    def _learned_engagement(self, db: Session, agent_id: str, session_id: str) -> dict:
        """
        Signals learned from this session's completed recommendations so far:

        - rates: per recommended-action engagement rate = engaged / (engaged + skipped),
          Laplace-smoothed toward 0.5 so sparsely-seen actions stay near neutral (and thus
          still get explored).
        - skipped_posts: how many times this agent skipped a recommendation ON each post —
          used to stop re-pushing branches it keeps rejecting.
        - base / n: overall smoothed rate and number of decisions.
        """
        from collections import defaultdict

        logs = (
            db.query(RecommendationLog)
            .filter(
                RecommendationLog.agent_id == agent_id,
                RecommendationLog.session_id == session_id,
                RecommendationLog.actual_action.isnot(None),   # decided only
            )
            .all()
        )
        agg: dict[str, list] = defaultdict(lambda: [0, 0])   # action -> [engaged, total]
        skipped_posts: dict[str, int] = defaultdict(int)     # post_id -> skip count
        overall = [0, 0]
        for l in logs:
            act = l.recommended_action
            if not act:
                continue
            eng = 1 if l.outcome == "engaged" else 0
            agg[act][0] += eng
            agg[act][1] += 1
            overall[0] += eng
            overall[1] += 1
            if l.outcome == "skipped" and l.recommended_post_id:
                skipped_posts[l.recommended_post_id] += 1

        rates = {a: (e + 1) / (t + 2) for a, (e, t) in agg.items()}
        base = (overall[0] + 1) / (overall[1] + 2) if overall[1] else 0.5
        return {"rates": rates, "base": base, "n": overall[1], "skipped_posts": dict(skipped_posts)}

    async def _semantic_fit_scores(
        self, db: Session, agent_id: str, session_id: str, valid_pairs: list
    ) -> dict:
        """
        For each candidate post, a 0..1 fit score = how similar it is to the posts this
        agent has ENGAGED with, minus its similarity to posts it SKIPPED, using cached
        embeddings. Returns {} while the agent has no engaged/skipped history (cold start),
        so the recommender falls back to the per-action rate until fit is learnable.
        """
        from simulation.topic_tree import _cosine

        logs = (
            db.query(RecommendationLog)
            .filter(
                RecommendationLog.agent_id == agent_id,
                RecommendationLog.session_id == session_id,
                RecommendationLog.recommended_post_id.isnot(None),
                RecommendationLog.outcome.in_(("engaged", "skipped")),
            )
            .all()
        )
        engaged_ids = [l.recommended_post_id for l in logs if l.outcome == "engaged"]
        skipped_ids = [l.recommended_post_id for l in logs if l.outcome == "skipped"]
        if not engaged_ids and not skipped_ids:
            return {}

        cand_text = {
            p["post"]["id"]: f"{p['post']['name']}. {p['post'].get('description', '')}"
            for p in valid_pairs
        }
        # Texts for engaged/skipped posts that aren't among current candidates.
        needed = (set(engaged_ids) | set(skipped_ids)) - set(cand_text)
        texts = dict(cand_text)
        if needed:
            for p in db.query(DeliberationPost).filter(DeliberationPost.id.in_(needed)).all():
                texts[p.id] = f"{p.name}. {p.description}"

        vecs = await self._embed_posts_cached([(i, texts[i]) for i in texts])

        def _centroid(ids: list):
            vs = [vecs[i] for i in ids if i in vecs]
            if not vs:
                return None
            dim = len(vs[0])
            return [sum(v[k] for v in vs) / len(vs) for k in range(dim)]

        eng_c = _centroid(engaged_ids)
        skp_c = _centroid(skipped_ids)

        out: dict[str, float] = {}
        for pid in cand_text:
            v = vecs.get(pid)
            if v is None:
                continue
            e = _cosine(v, eng_c) if eng_c else 0.0
            s = _cosine(v, skp_c) if skp_c else 0.0
            raw = e - 0.6 * s               # reward "like engaged", penalise "like skipped"
            out[pid] = max(0.0, min(1.0, 0.5 + 0.5 * raw))
        return out

    async def _embed_posts_cached(self, id_text_pairs: list) -> dict:
        """Embed (post_id, text) pairs, reusing the module-level cache for stable posts."""
        from simulation.topic_tree import _embed_text

        out: dict[str, list] = {}
        to_embed = []
        for pid, txt in id_text_pairs:
            if pid in _POST_EMB_CACHE:
                out[pid] = _POST_EMB_CACHE[pid]
            else:
                to_embed.append((pid, txt))
        if to_embed:
            vecs = await asyncio.wait_for(
                asyncio.gather(*[_embed_text(t) for _, t in to_embed]),
                timeout=LLM_TIMEOUT_S,
            )
            for (pid, _), v in zip(to_embed, vecs):
                _POST_EMB_CACHE[pid] = v
                out[pid] = v
        return out

    def _rank_pairs_by_learning(
        self, valid_pairs: list, learned: dict, all_posts: dict | None = None,
        fit_scores: dict | None = None, agent_id: str | None = None,
        tree_scores: dict | None = None,
        keep: int = 12, explore: int | None = None,
    ) -> list:
        """
        Rank candidate pairs by predicted engagement and keep the strongest, plus a
        little random exploration. Predicted engagement folds in signals that accumulate
        over a run, so the skip rate trends down as they sharpen:

          1. SEMANTIC FIT — similarity of the candidate post to what this agent has
             engaged with vs skipped (the dominant real driver of skips); falls back to
          2. learned per-action engagement rate until fit history exists,
          3. an OWN-FRONTIER boost — prefer extending the agent's own branches (posts it
             authored, or their direct children), which stay on-fit and unsaturated even
             as the wider tree drifts away from this agent's expertise,
          4. a penalty for posts the agent has already skipped (stop re-pushing them),
          5. a gap/saturation factor — free until TARGET_BREADTH siblings exist, then
             down-weight ("already covered" skips),
          6. depth and same-type-chain penalties, so the map grows OUT as well as DOWN
             instead of collapsing into one long drifting corridor.
        """
        from simulation.roles.specialist import ACTION_TO_PTYPE

        # Exploration schedule: sample MORE while the model is cold (broader coverage
        # trains the tree faster) and taper to pure exploitation once it has sharpened.
        # This is what steepens the early part of the skip-rate curve.
        if explore is None:
            n_seen = learned.get("n", 0)
            explore = (3 if n_seen < 16 else (2 if n_seen < 32 else (1 if n_seen < 60 else 0))) \
                      if FAST_LEARNING else 1

        rates, base = learned.get("rates", {}), learned.get("base", 0.5)
        skipped_posts = learned.get("skipped_posts", {})
        fit = fit_scores or {}
        tree = tree_scores or {}
        posts = all_posts or {}

        def _saturation(pair) -> float:
            """Free until TARGET_BREADTH siblings exist, then decay — so branches fan
            out to a healthy width before the recommender abandons them."""
            if not posts:
                return 1.0
            target_ptype = ACTION_TO_PTYPE.get(pair["action"])
            if not target_ptype:
                return 1.0
            existing = sum(
                1 for p in posts.values()
                if p.get("purpose_id") == pair["post"]["id"] and p.get("ptype") == target_ptype
            )
            if existing < TARGET_BREADTH:
                return 1.0
            return 1.0 / (1.0 + (existing - TARGET_BREADTH + 1))

        def _depth(post: dict) -> int:
            d, cur, seen = 0, post, set()
            while cur is not None:
                pid = cur.get("purpose_id")
                if not pid or pid in seen or pid not in posts:
                    break
                seen.add(pid)
                cur = posts[pid]
                d += 1
            return d

        def _depth_penalty(pair) -> float:
            """Damp candidates far down the tree so the map grows out, not just down."""
            if not posts:
                return 1.0
            d = _depth(pair["post"])
            return 1.0 if d <= DEPTH_SOFT_CAP else DEPTH_DECAY ** (d - DEPTH_SOFT_CAP)

        def _chain_penalty(pair) -> float:
            """
            Damp long runs of the same post type — e.g. add_improvement on a solution
            produces another solution, which produced 7-deep solution chains that drifted
            away from the originating barrier.
            """
            if not posts:
                return 1.0
            target_ptype = ACTION_TO_PTYPE.get(pair["action"])
            if not target_ptype:
                return 1.0
            run, cur, seen = 0, pair["post"], set()
            while cur is not None and cur.get("ptype") == target_ptype:
                run += 1
                pid = cur.get("purpose_id")
                if not pid or pid in seen or pid not in posts:
                    break
                seen.add(pid)
                cur = posts[pid]
            return 1.0 if run <= MAX_SAME_TYPE_CHAIN else CHAIN_DECAY ** (run - MAX_SAME_TYPE_CHAIN)

        def _own_frontier(pair) -> float:
            # Boost acting on the agent's own post, or a child of its own post — its
            # coherent, on-expertise lane where fresh, engageable openings keep appearing.
            if not agent_id:
                return 1.0
            post = pair["post"]
            if post.get("author_agent_id") == agent_id:
                return 1.6
            parent = posts.get(post.get("purpose_id")) if posts else None
            if parent and parent.get("author_agent_id") == agent_id:
                return 1.3
            return 1.0

        def _score(pair) -> float:
            pid = pair["post"]["id"]
            # Priority of predictors: trained decision tree > semantic fit > action-rate.
            if pid in tree:
                r = tree[pid]
            elif pid in fit:
                r = fit[pid]
            else:
                r = rates.get(pair["action"], base)
            r *= _own_frontier(pair)
            skips = skipped_posts.get(pid, 0)
            if skips:
                r *= 0.35 ** skips   # strongly avoid branches this agent keeps rejecting
            # Shape as a BOUNDED tie-breaker: combine the structural factors into one
            # 0..1 term and let it modulate only SHAPE_WEIGHT of the score, so a badly
            # shaped but well-matched candidate is demoted, never vetoed.
            shape = _saturation(pair) * _depth_penalty(pair) * _chain_penalty(pair)
            return r * (1.0 - SHAPE_WEIGHT + SHAPE_WEIGHT * shape)

        if len(valid_pairs) <= keep:
            return sorted(valid_pairs, key=_score, reverse=True)

        scored = sorted(valid_pairs, key=_score, reverse=True)
        top = scored[:keep]
        rest = scored[keep:]
        if rest and explore > 0:
            top = top + random.sample(rest, min(explore, len(rest)))
        return top

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_clickstream(self, db: Session, agent_id: str, session_id: str, current_tick: int) -> str:
        min_tick = max(0, current_tick - 10)
        events = (
            db.query(Event)
            .filter(
                Event.agent_id == agent_id,
                Event.session_id == session_id,
                Event.sim_tick >= min_tick,
            )
            .order_by(Event.sim_tick.asc())
            .limit(50)
            .all()
        )

        if not events:
            return "No recent activity."

        lines = []
        for e in events:
            snippet = ""
            if e.target_id:
                post = db.query(DeliberationPost).filter(DeliberationPost.id == e.target_id).first()
                if post:
                    snippet = f'on "{post.name[:50]}" ({post.ptype})'
            meta_parts = []
            if e.metadata_:
                if "ptype" in e.metadata_:
                    meta_parts.append(f"ptype: {e.metadata_['ptype']}")
                if "new_post_name" in e.metadata_:
                    meta_parts.append(f"added: {e.metadata_['new_post_name'][:40]}")
                if "reason" in e.metadata_:
                    meta_parts.append(f"reason: {e.metadata_['reason']}")
            meta_str = f" — {', '.join(meta_parts)}" if meta_parts else ""
            lines.append(f"[tick {e.sim_tick}] {e.event_type} {snippet}{meta_str}")

        return "\n".join(lines)

    def _build_analyst_prompt(self, agent_name: str, domain: str, existing_profile: str, clickstream: str) -> str:
        existing_section = (
            f"EXISTING PROFILE:\n{existing_profile}\n\n"
            if existing_profile
            else "EXISTING PROFILE: (none yet)\n\n"
        )
        return (
            f"Deliberation participant: {agent_name} ({domain} expert)\n\n"
            f"{existing_section}"
            f"RECENT ACTIVITY (last 10 ticks):\n{clickstream}\n\n"
            "Update the profile. Write 3-5 sentences describing: "
            "what post types they prefer (barriers, solutions, pros/cons, questions, etc.), "
            "what branches they gravitate toward, and how they approach deliberation. "
            "Output only the updated profile text."
        )

    def _recent_skip_reasons(self, db: Session, agent_id: str, session_id: str, limit: int = 6) -> list[str]:
        """Most-recent non-empty skip reasons this agent gave — the negative-weight signal."""
        rows = (
            db.query(RecommendationLog)
            .filter(
                RecommendationLog.agent_id == agent_id,
                RecommendationLog.session_id == session_id,
                RecommendationLog.outcome == "skipped",
                RecommendationLog.skip_reason.isnot(None),
            )
            .order_by(RecommendationLog.sim_tick.desc())
            .limit(limit)
            .all()
        )
        seen: set[str] = set()
        reasons: list[str] = []
        for r in rows:
            txt = (r.skip_reason or "").strip()
            if txt and txt != "empty_content" and txt.lower() not in seen:
                seen.add(txt.lower())
                action = r.recommended_action or "?"
                reasons.append(f"skipped '{action}' — {txt}")
        return reasons

    def _persona_topics(self, db: Session, agent_id: str, session_id: str) -> list[str]:
        """The agent's up-front persona topic headers — the 'persona table' input."""
        rows = (
            db.query(PersonaTopicHeader)
            .filter(
                PersonaTopicHeader.agent_id == agent_id,
                PersonaTopicHeader.session_id == session_id,
            )
            .order_by(PersonaTopicHeader.rank.asc())
            .all()
        )
        return [r.label for r in rows]

    def _build_selector_prompt(
        self,
        profile: str,
        valid_pairs: list,
        persona_topics: list[str] | None = None,
        skip_reasons: list[str] | None = None,
        learned: dict | None = None,
    ) -> str:
        pair_lines = [
            f"[{i}] Action: {p['action']} | On: ({p['post']['ptype']}) {p['post']['name'][:60]}"
            for i, p in enumerate(valid_pairs[:20])
        ]
        persona_block = ""
        if persona_topics:
            persona_block = (
                "PERSONA TABLE (topics this expert said they want to contribute to):\n"
                + ", ".join(persona_topics) + "\n\n"
            )
        # Quantitative feedback signal — how often the expert has engaged with each
        # recommended action so far this session. This is what lets the recommender
        # improve (lower its skip rate) as the run progresses.
        learned_block = ""
        if learned and learned.get("n", 0) > 0:
            rate_lines = sorted(learned.get("rates", {}).items(), key=lambda kv: kv[1], reverse=True)
            learned_block = (
                "ENGAGEMENT RATES SO FAR (this expert, this session — higher = they act, "
                "lower = they skip):\n"
                + "\n".join(f"- {a}: {r:.0%}" for a, r in rate_lines)
                + f"\n(overall {learned['base']:.0%} over {learned['n']} decisions). "
                "Strongly prefer higher-rate actions.\n\n"
            )
        skip_block = ""
        if skip_reasons:
            skip_block = (
                "\n\nRECENT SKIP REASONS (avoid recommending pairs like these):\n"
                + "\n".join(f"- {s}" for s in skip_reasons)
            )
        return (
            persona_block
            + learned_block
            + f"BEHAVIORAL PROFILE (learned from clickstream):\n{profile}\n\n"
            + f"AVAILABLE POST/ACTION PAIRS:\n" + "\n".join(pair_lines)
            + skip_block + "\n\n"
            "Which pair (by index) is this expert most likely to engage with?"
        )

    def _upsert_profile(self, db, agent_id, session_id, tick, new_text, existing_row):
        snapshot = {"tick": tick, "profile": new_text}
        if existing_row:
            history = list(existing_row.tick_history or [])
            history.append(snapshot)
            existing_row.tick_history = history[-10:]
            existing_row.profile_text = new_text
            existing_row.last_updated_tick = tick
            existing_row.updated_at = datetime.utcnow()
        else:
            db.add(AgentProfile(
                agent_id=agent_id,
                session_id=session_id,
                profile_text=new_text,
                tick_history=[snapshot],
                last_updated_tick=tick,
            ))
