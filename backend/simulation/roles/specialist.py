import random
import re
from datetime import datetime
from sqlalchemy.orm import Session
from google.genai import types as genai_types
from models import DeliberationPost, RecommendationLog, Event, PostType
from simulation.roles.base import BaseAgent, _get_client
from simulation.prompts.specialist import (
    build_action_choice_prompt,
    describe_context,
    _content_instruction,
)
from simulation.ground_truth import gt_engage
from simulation.user_model_tree import record_event

_MD_HTML = re.compile(r"</?(?:p|br|div|ul|ol|li|b|i|strong|em|h[1-6])\s*/?>", re.I)
_MD_MARKS = re.compile(r"(\*\*|__|^#{1,6}\s+|```)", re.M)
_LEADING_Q = re.compile(
    r"^\s*(?:what is[^?]{0,60}\?|why is[^?]{0,60}\?|why does[^?]{0,60}\?|"
    r"how does[^?]{0,60}\?|who bears[^?]{0,60}\?)\s*", re.I)


def sanitize_description(text: str) -> str:
    """
    Strip model formatting artifacts before persisting a post description.

    Observed in real runs: `</p>` tags and `**bold**` (verbose mode, ~24% of posts) and
    rhetorical-question openers like "Why is this important? ..." inherited from the
    AIdeator paragraph directives (~30% of posts in brief mode).
    """
    t = (text or "").strip()
    t = _MD_HTML.sub(" ", t)   # space, not empty: </p><p> must not weld words together
    t = _MD_MARKS.sub("", t)
    prev = None
    while prev != t:                      # openers can stack: "What is X? Why is Y? ..."
        prev = t
        t = _LEADING_Q.sub("", t).strip()
    return " ".join(t.split())


def _ensure_complete_sentence(text: str) -> str:
    """
    Guard against a truncated tail: if the text doesn't end in terminal punctuation,
    trim back to the last complete sentence. Falls back to the original text if there
    is no earlier sentence boundary to trim to.
    """
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if cut != -1:
        return text[: cut + 1].strip()
    return text


MAX_STAKEHOLDERS = 10

# Type-specific actions per post type
MAX_MISSIONS = 5

_BASE_ACTIONS: dict[str, list[str]] = {
    "mission":      ["add_stakeholder", "create_topic"],
    "stakeholder":  ["add_goal"],
    "goal":         ["add_barrier", "add_solution", "add_abstraction"],
    "barrier":      ["add_cause", "add_solution", "add_abstraction"],
    "cause":        ["add_cause", "add_solution", "add_abstraction"],
    "abstraction":  ["add_analogy"],
    "analogy":      ["add_inspiration"],
    "inspiration":  ["use_inspiration"],
    # Anti-saturation: a solution can also be decomposed into parts or stress-tested
    # for failure modes, so agents aren't forced to invent ever-more sibling solutions.
    "solution":     ["add_improvement", "add_part", "add_failure_mode", "add_barrier", "add_question"],
    "part":         ["add_improvement", "add_barrier", "add_question"],
    "failure_mode": ["add_solution", "add_cause"],
    "question":     ["add_answer"],
    "answer":       [],
    "pro":          [],
    "con":          [],
}

# Universal: available on ANY post type
UNIVERSAL_ACTIONS = ["add_pro", "add_con", "create_topic", "edit", "skip"]

# Full action table: type-specific + universal
ALLOWED_ACTIONS: dict[str, list[str]] = {
    ptype: list(dict.fromkeys(actions + UNIVERSAL_ACTIONS))
    for ptype, actions in _BASE_ACTIONS.items()
}

# Maps action → ptype of the new post it creates (None = no new post)
ACTION_TO_PTYPE: dict[str, str | None] = {
    "add_stakeholder": "stakeholder",
    "add_goal":        "goal",
    "add_barrier":     "barrier",
    "add_solution":    "solution",
    "add_abstraction": "abstraction",
    "add_cause":       "cause",
    "add_analogy":     "analogy",
    "add_inspiration": "inspiration",
    "use_inspiration": "solution",
    "add_improvement": "solution",
    "add_part":        "part",
    "add_failure_mode": "failure_mode",
    "add_question":    "question",
    "add_answer":      "answer",
    "add_pro":         "pro",
    "add_con":         "con",
    "create_topic":    "mission",  # creates a new root mission (no purpose_id)
    "edit":            None,       # modifies existing post in-place
    "skip":            None,
    "view":            None,
}

# Actions that require name + description in LLM response
_CONTENT_ACTIONS = {a for a, t in ACTION_TO_PTYPE.items() if t is not None}


class SpecialistAgent(BaseAgent):
    async def _elicit_skip_reason(self, post: dict, recommended_action: str, context_str: str) -> str:
        """
        Explicitly ASK the agent why it skipped a recommendation (Feature 2).

        Models reliably include a reason when they engage but tend to omit it when
        they skip, so we follow up with a focused one-shot question. The answer
        feeds back into the recommender as a negative-weight rule.
        """
        # IMPORTANT: do NOT reuse self.system_prompt here — it forces the agent to reply
        # in JSON, which would wrap the reason in {"action": ...}. Use a plain-prose
        # system instruction that still carries the persona voice.
        persona_system = (
            f"You are {self.name}, a {self.domain} expert participating in a structured "
            f"deliberation. {self.agent.persona_description} "
            "Answer in plain prose — full sentences, no JSON, no code fences."
        )
        prompt = (
            f"CONTEXT:\n{context_str}\n\n"
            f"You were just recommended to '{recommended_action}' on the "
            f"[{post['ptype'].upper()}] post \"{post['name']}\", and you chose to SKIP it.\n\n"
            "Explain WHY you skipped in ONE or TWO complete sentences — what made this a poor "
            "fit for you right now (off-topic for your expertise, already covered, lower "
            "priority than another branch, etc.). Write full sentences that finish the thought. "
            "Reply with the explanation only, no preamble, no JSON."
        )
        try:
            # Direct call with thinking DISABLED so the whole token budget goes to the
            # answer — otherwise Gemini's thinking step truncates short responses mid-sentence.
            resp = await _get_client().aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=persona_system,
                    max_output_tokens=260,
                    temperature=0.5,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (resp.text or "").strip().strip('"').strip()
            return _ensure_complete_sentence(text)[:400]
        except Exception:
            return ""

    async def _post_vec(self, post: dict) -> list:
        """Embedding of a post (name + description), reusing the recommender's cache."""
        from simulation.recommender import _POST_EMB_CACHE
        from simulation.topic_tree import _embed_text
        pid = post["id"]
        if pid in _POST_EMB_CACHE:
            return _POST_EMB_CACHE[pid]
        vec = await _embed_text(f"{post['name']}. {post.get('description', '')}")
        _POST_EMB_CACHE[pid] = vec
        return vec

    async def _gen_content(self, recommended_action: str, post: dict, context_str: str,
                           existing_by_ptype: dict, all_posts: dict, use_aideator: bool,
                           brief_answers: bool = True) -> tuple:
        """Generate (name, description) for a forced action (synthetic engage). No skip option."""
        instruction = _content_instruction(
            recommended_action, post, all_posts,
            existing_by_ptype=existing_by_ptype, use_aideator=use_aideator,
        )
        prompt = (
            f"You are {self.name}, a {self.domain} expert in a structured deliberation.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"CURRENT POST [{post['ptype'].upper()}]: \"{post['name']}\"\n{post.get('description', '')}\n\n"
            f"TASK: {instruction}\n\n"
            + (("BE BRIEF: the description must be 2-3 SENTENCES TOTAL, one short paragraph — "
                "the way a real participant typing into a web form would write. Ignore any "
                "guidance above asking for multiple paragraphs; keep the substance, drop the length.\n\n")
               if brief_answers else "") +
            'Return ONLY valid JSON: {"name": "...", "description": "..."}'
        )
        try:
            # Direct call with thinking DISABLED — call_llm leaves Gemini thinking on,
            # which eats the token budget and truncates the JSON mid-string.
            from simulation.roles.base import call_with_retry
            resp = await call_with_retry(lambda: _get_client().aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=340 if brief_answers else 600,
                    temperature=0.8,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            ))
            parsed = self.parse_json((resp.text or "").strip())
            return (parsed.get("name", "").strip(),
                    sanitize_description(parsed.get("description", "")))
        except Exception:
            return "", ""

    async def decide(
        self,
        db: Session,
        session_id: str,
        tick: int,
        recommended_pair: dict,      # {"post": dict, "action": str}
        strategy: str,
        reasoning: str,
        focus_hints: list,
        all_posts: dict,             # id -> post dict for context
        use_aideator: bool = True,
        is_prelude: bool = False,
        confidence: float | None = None,
        agent_mode: str = "llm",     # "llm" (free judgment) | "synthetic" (ground-truth model)
        gt=None,                     # GroundTruthPersona for synthetic mode
        brief_answers: bool = True,  # cap descriptions at 2-3 sentences
    ):
        post = recommended_pair["post"]
        recommended_action = recommended_pair["action"]
        post_id = post["id"]

        # 1. Log RecommendationLog (actual_action filled later)
        rec_log = RecommendationLog(
            session_id=session_id,
            agent_id=self.id,
            sim_tick=tick,
            strategy=strategy,
            recommended_post_id=post_id,
            recommended_action=recommended_action,
            reasoning=reasoning,
            outcome=None,
            actual_action=None,
            is_prelude=is_prelude,
            confidence=confidence,
        )
        db.add(rec_log)
        db.flush()

        self.log_event(
            db, session_id, "recommendation_shown", tick,
            target_type="post", target_id=post_id,
            metadata={"action": recommended_action, "strategy": strategy},
            is_prelude=is_prelude,
        )

        # 2. Build available actions for this post
        available = ALLOWED_ACTIONS.get(post["ptype"], UNIVERSAL_ACTIONS)

        # 3. Build context + existing achievers (for avoiding duplicates)
        context_str = describe_context(post, all_posts)
        existing_by_ptype: dict[str, list] = {}
        for p in all_posts.values():
            if p.get("purpose_id") == post_id:
                existing_by_ptype.setdefault(p["ptype"], []).append(p)

        # Embedding of the recommended post — for the clickstream + synthetic decision.
        post_vec = await self._post_vec(post)

        # ── Synthetic mode: the hidden ground-truth model decides engage/skip (no LLM). ──
        if agent_mode == "synthetic":
            engage = gt_engage(gt, post_vec) if gt is not None else True
            if not engage:
                rec_log.outcome = "skipped"
                rec_log.actual_action = "skip"
                rec_log.skip_reason = "synthetic_dislike"
                self.log_event(db, session_id, "skip", tick, target_type="post", target_id=post_id,
                               metadata={"recommended_action": recommended_action, "mode": "synthetic"},
                               is_prelude=is_prelude)
                record_event(db, session_id, self.id, post_id, post_vec, 0, tick)
                db.commit()
                return None
            # Engaged: generate content for the recommended action (no skip option).
            name, description = await self._gen_content(
                recommended_action, post, context_str, existing_by_ptype, all_posts,
                use_aideator, brief_answers)
            target_ptype = ACTION_TO_PTYPE.get(recommended_action)
            if target_ptype is None or not name or not description:
                rec_log.outcome = "engaged"
                rec_log.actual_action = recommended_action
                record_event(db, session_id, self.id, post_id, post_vec, 1, tick)
                db.commit()
                return None
            parent_id = None if recommended_action == "create_topic" else post_id
            new_post = DeliberationPost(
                session_id=session_id, ptype=PostType(target_ptype), name=name, description=description,
                purpose_id=parent_id, author_agent_id=self.id, content_mode="llm_assisted",
                sim_tick=tick, is_prelude=is_prelude)
            db.add(new_post)
            db.flush()
            rec_log.outcome = "engaged"
            rec_log.actual_action = recommended_action
            rec_log.created_post_id = new_post.id
            self.log_event(db, session_id, recommended_action, tick, target_type="post", target_id=post_id,
                           metadata={"new_post_id": new_post.id, "new_post_name": name, "ptype": target_ptype,
                                     "recommended_action": recommended_action, "mode": "synthetic"},
                           is_prelude=is_prelude)
            record_event(db, session_id, self.id, post_id, post_vec, 1, tick)
            db.commit()
            return new_post

        # 4. Build prompt — agent chooses action + provides content
        prompt = build_action_choice_prompt(
            agent_name=self.name,
            domain=self.domain,
            post=post,
            context_str=context_str,
            available_actions=available,
            recommended_action=recommended_action,
            existing_by_ptype=existing_by_ptype,
            all_posts=all_posts,
            use_aideator=use_aideator,
            brief_answers=brief_answers,
        )

        # 5. Call LLM
        try:
            raw = await self.call_llm(prompt, max_tokens=(480 if use_aideator else 340) if brief_answers else (900 if use_aideator else 400))
            parsed = self.parse_json(raw)
            chosen_action = parsed.get("action", "skip").strip().lower().replace("-", "_")
        except Exception as e:
            chosen_action = "skip"
            parsed = {}

        # The agent's own stated reason for its decision (esp. why it skipped).
        reason = str(parsed.get("reason", "")).strip()[:300]

        # Normalise: treat unknown/invalid actions as skip
        if chosen_action not in ACTION_TO_PTYPE:
            chosen_action = "skip"

        # 6. Handle chosen action
        if chosen_action in ("skip", "view"):
            # Explicitly ask why if the agent skipped without volunteering a reason.
            if not reason and chosen_action == "skip":
                reason = await self._elicit_skip_reason(post, recommended_action, context_str)
            rec_log.outcome = "skipped"
            rec_log.actual_action = chosen_action
            rec_log.skip_reason = reason
            self.log_event(
                db, session_id, chosen_action, tick,
                target_type="post", target_id=post_id,
                metadata={"recommended_action": recommended_action, "reason": reason},
                is_prelude=is_prelude,
            )
            record_event(db, session_id, self.id, post_id, post_vec, 0, tick)
            db.commit()
            return None

        if chosen_action == "edit":
            new_desc = parsed.get("description", "").strip()
            if new_desc:
                existing_post = db.query(DeliberationPost).filter(
                    DeliberationPost.id == post_id
                ).first()
                if existing_post:
                    existing_post.description = new_desc
            rec_log.outcome = "engaged"
            rec_log.actual_action = "edit"
            rec_log.skip_reason = reason
            self.log_event(
                db, session_id, "edit", tick,
                target_type="post", target_id=post_id,
                metadata={"recommended_action": recommended_action, "reason": reason},
                is_prelude=is_prelude,
            )
            record_event(db, session_id, self.id, post_id, post_vec, 1, tick)
            db.commit()
            return None

        # Creating a new post
        target_ptype = ACTION_TO_PTYPE.get(chosen_action)
        if target_ptype is None:
            rec_log.outcome = "skipped"
            rec_log.actual_action = "skip"
            rec_log.skip_reason = reason
            record_event(db, session_id, self.id, post_id, post_vec, 0, tick)
            db.commit()
            return None

        name = parsed.get("name", "").strip()
        description = sanitize_description(parsed.get("description", ""))
        if not name or not description:
            rec_log.outcome = "skipped"
            rec_log.actual_action = "skip"
            rec_log.skip_reason = reason or "empty_content"
            self.log_event(
                db, session_id, "skip", tick,
                target_type="post", target_id=post_id,
                metadata={"reason": reason or "empty_content", "chosen_action": chosen_action},
                is_prelude=is_prelude,
            )
            record_event(db, session_id, self.id, post_id, post_vec, 0, tick)
            db.commit()
            return None

        # create_topic → new root mission (no parent)
        parent_id = None if chosen_action == "create_topic" else post_id

        new_post = DeliberationPost(
            session_id=session_id,
            ptype=PostType(target_ptype),
            name=name,
            description=description,
            purpose_id=parent_id,
            author_agent_id=self.id,
            content_mode="llm_assisted",
            sim_tick=tick,
            is_prelude=is_prelude,
        )
        db.add(new_post)
        db.flush()

        rec_log.outcome = "engaged"
        rec_log.actual_action = chosen_action
        rec_log.skip_reason = reason
        rec_log.created_post_id = new_post.id
        self.log_event(
            db, session_id, chosen_action, tick,
            target_type="post", target_id=post_id,
            metadata={
                "new_post_id": new_post.id,
                "new_post_name": name,
                "ptype": target_ptype,
                "recommended_action": recommended_action,
                "matched_recommendation": chosen_action == recommended_action,
                "reason": reason,
            },
            is_prelude=is_prelude,
        )
        record_event(db, session_id, self.id, post_id, post_vec, 1, tick)
        db.commit()
        return new_post
