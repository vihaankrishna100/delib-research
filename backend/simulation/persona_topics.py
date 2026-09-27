"""
Persona-derived topic headers (Feature 1).

For each agent in a session, ask the LLM — speaking in the agent's own voice —
for ~10 topics it expects to contribute to, given its persona and the session
mission. The headers are created ENTIRELY from this persona+mission prompt; they
are never derived from the simulation's own activity. The exact prompt idea:

    "I am a <persona>. I will be engaging in deliberations about <mission>.
     Give me a list of about 10 topics that I will probably be interested in
     contributing to in those deliberations."

Each top-level header is also broken into 2-3 finer SUB-HEADERS (deeper topic
headers, by default), and each header carries per-action scores predicting how
the agent would ACT on a recommendation within it (reply / react / refine / view).

These become the initial columns of the persona/user-model table, so the table
has meaningful structure BEFORE any posts exist — a warm start for the recommender
rather than a cold one.
"""

import json
import math
import os
import re
import asyncio
from typing import List
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import Agent, SimulationSession, DeliberationPost, PersonaTopicHeader, RecommendationLog


TOPIC_MODEL = "gemini-2.5-flash"
_client = None


# ── Deliberation-move taxonomy ───────────────────────────────────────────────
# The contribution ("move") types we score/observe per topic header — i.e. what
# KIND of post the agent would add in that header (goal, barrier, analogy, …).
ACTION_CATEGORIES = [
    "goal", "barrier", "cause", "solution", "abstraction",
    "analogy", "inspiration", "question", "answer", "pro", "con",
    "part", "failure_mode",
]

# One-line descriptions used in the scoring prompt.
ACTION_DESCRIPTIONS = {
    "goal": "a desired outcome for a stakeholder",
    "barrier": "an obstacle blocking a goal",
    "cause": "a root cause behind a barrier",
    "solution": "a concrete intervention",
    "abstraction": "the general structural pattern",
    "analogy": "the same pattern in another domain",
    "inspiration": "how that other domain solved it",
    "question": "a probing Socratic question",
    "answer": "a substantive answer to a question",
    "pro": "an advantage of a post",
    "con": "a drawback of a post",
    "part": "a sub-component a solution needs to work",
    "failure_mode": "a specific way a solution could fail",
}

# Map each concrete simulation action to its move type (its target post type). "skip"
# is a non-engagement tracked separately; actions that map to no scored move (e.g.
# add_stakeholder, create_topic, edit, view) are simply ignored in the breakdown.
ACTION_CATEGORY_MAP = {
    "add_goal": "goal", "add_barrier": "barrier", "add_cause": "cause",
    "add_solution": "solution", "use_inspiration": "solution", "add_improvement": "solution",
    "add_abstraction": "abstraction", "add_analogy": "analogy", "add_inspiration": "inspiration",
    "add_question": "question", "add_answer": "answer",
    "add_pro": "pro", "add_con": "con",
    "add_part": "part", "add_failure_mode": "failure_mode",
}


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _clamp01(v, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _mean_action_scores(children: List[dict]) -> dict:
    """Mean of a set of {reply,react,refine,view} dicts — used to roll sub-headers up."""
    out = {}
    for cat in ACTION_CATEGORIES:
        vals = [c[cat] for c in children if isinstance(c, dict) and cat in c]
        out[cat] = round(sum(vals) / len(vals), 3) if vals else 0.5
    return out


# ── Topic generation (prompt-only, never from the simulation) ─────────────────

def _build_prompt(persona: str, mission_name: str, mission_desc: str) -> str:
    mission_blurb = mission_name
    if mission_desc:
        mission_blurb += f" — {mission_desc}"
    return (
        f"I am the following participant:\n{persona}\n\n"
        f"I will be engaging in deliberations about: {mission_blurb}\n\n"
        "Give me a list of about 10 TOP-LEVEL topics that I will probably be interested in "
        "contributing to in those deliberations. For EACH top-level topic, also break it into "
        "2-3 finer SUB-TOPICS that capture the distinct facets I most likely care about.\n\n"
        "Every topic and sub-topic needs: a short header (2-5 words), a one-sentence reason it "
        "fits my interests, and a CONFIDENCE from 0.0 to 1.0 expressing how sure you are that I "
        "would actually contribute to it rather than skip it (1.0 = I will almost certainly "
        "engage; 0.3 = borderline).\n\n"
        "Return ONLY valid JSON (no markdown, no code fences):\n"
        '{"topics": [{"label": "Short Header", "rationale": "one sentence", "confidence": 0.85, '
        '"subtopics": [{"label": "Finer Header", "rationale": "one sentence", "confidence": 0.8}]}]}'
    )


def _parse_topics(raw: str) -> List[dict]:
    """Parse the generation response into a list of {label, rationale, confidence, subtopics}."""
    try:
        raw = re.sub(r"```(?:json)?", "", (raw or "").strip()).strip("`").strip()
        parsed = json.loads(raw)
        topics = parsed.get("topics", [])
    except Exception:
        return []

    out: List[dict] = []
    for t in topics[:12]:
        if isinstance(t, str):
            out.append({"label": t.strip()[:80], "rationale": "", "confidence": 0.5, "subtopics": []})
            continue
        if not isinstance(t, dict):
            continue
        label = str(t.get("label", "")).strip()[:80]
        if not label:
            continue
        subs = []
        for s in (t.get("subtopics") or [])[:4]:
            if isinstance(s, dict):
                slabel = str(s.get("label", "")).strip()[:80]
                if slabel:
                    subs.append({
                        "label": slabel,
                        "rationale": str(s.get("rationale", ""))[:200],
                        "confidence": _clamp01(s.get("confidence", 0.5)),
                    })
        out.append({
            "label": label,
            "rationale": str(t.get("rationale", ""))[:200],
            "confidence": _clamp01(t.get("confidence", 0.5)),
            "subtopics": subs,
        })
    return out


async def _topics_for_agent(agent: Agent, mission_name: str, mission_desc: str) -> List[dict]:
    prompt = _build_prompt(agent.persona_description, mission_name, mission_desc)
    try:
        resp = await _get_client().aio.models.generate_content(
            model=TOPIC_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                # Deep headers make the JSON larger — give it ample room so it never
                # truncates mid-string (thinking is off, so the whole budget is output).
                max_output_tokens=3000,
                temperature=0.5,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return _parse_topics(resp.text)
    except Exception:
        return []


async def generate_persona_topics(db: Session, session_id: str) -> dict:
    """
    Generate and persist persona topic headers for every active agent in the
    session, entirely from the persona + mission prompt (never from simulation
    activity). Each top-level header gets 2-3 sub-headers, and per-action scores
    are filled in afterward. Replaces any previously generated headers.
    """
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        return {"agents": 0, "topics_written": 0}

    mission = (
        db.query(DeliberationPost).filter(DeliberationPost.id == session.mission_id).first()
        if session.mission_id else None
    )
    mission_name = mission.name if mission else "the deliberation mission"
    mission_desc = mission.description if mission else ""

    agents = db.query(Agent).filter(Agent.is_active == True).all()
    if not agents:
        return {"agents": 0, "topics_written": 0}

    results = await asyncio.gather(
        *[_topics_for_agent(a, mission_name, mission_desc) for a in agents]
    )

    db.query(PersonaTopicHeader).filter(
        PersonaTopicHeader.session_id == session_id
    ).delete()
    db.commit()

    written = 0
    for agent, topics in zip(agents, results):
        # Order top-level headers by confidence (lowest-skip-risk first).
        ordered = sorted(topics, key=lambda t: t.get("confidence", 0.5), reverse=True)
        for rank, t in enumerate(ordered):
            parent = PersonaTopicHeader(
                session_id=session_id, agent_id=agent.id,
                label=t["label"], rank=rank, rationale=t.get("rationale", ""),
                confidence=t.get("confidence", 0.5), parent_id=None,
            )
            db.add(parent)
            db.flush()
            written += 1
            for srank, s in enumerate(sorted(t.get("subtopics", []),
                                             key=lambda x: x.get("confidence", 0.5), reverse=True)):
                db.add(PersonaTopicHeader(
                    session_id=session_id, agent_id=agent.id,
                    label=s["label"], rank=srank, rationale=s.get("rationale", ""),
                    confidence=s.get("confidence", 0.5), parent_id=parent.id,
                ))
                written += 1
    db.commit()

    # Fill predicted per-action scores for every header just written.
    await score_persona_actions(db, session_id)

    return {"agents": len(agents), "topics_written": written}


# ── Per-action scoring (reply / react / refine / view) ────────────────────────

def _action_prompt(persona: str, leaf_labels: List[str]) -> str:
    topic_lines = "\n".join(f"{i}. {lbl}" for i, lbl in enumerate(leaf_labels))
    moves = "\n".join(f"- {m}: {ACTION_DESCRIPTIONS[m]}" for m in ACTION_CATEGORIES)
    keys = ", ".join(f'"{m}": 0.5' for m in ACTION_CATEGORIES)
    return (
        "You are modeling which kinds of contribution a deliberation participant would ADD "
        "within each of their topics of interest. The contribution ('move') types are:\n"
        f"{moves}\n\n"
        f"Participant persona:\n{persona}\n\n"
        "For EACH topic below, rate from 0.0 to 1.0 how likely this participant is to add each "
        "move type when it is recommended within that topic. The ratings are INDEPENDENT (a "
        "topic can be high on several moves).\n\n"
        f"TOPICS:\n{topic_lines}\n\n"
        "Return ONLY valid JSON (no markdown), the scores in the SAME order as the topics, each "
        "object containing every move key:\n"
        f'{{"scores": [{{{keys}}}]}}'
    )


async def _score_actions_for_agent(persona: str, leaves: List[PersonaTopicHeader]) -> None:
    """Set action_scores on each leaf header from one LLM call; safe fallbacks on failure."""
    def _fallback(h: PersonaTopicHeader) -> dict:
        c = round(h.confidence if h.confidence is not None else 0.5, 3)
        # Persona-agnostic default: every move tracks the header's confidence.
        return {m: c for m in ACTION_CATEGORIES}

    scores: List[dict] = []
    try:
        resp = await _get_client().aio.models.generate_content(
            model=TOPIC_MODEL,
            contents=_action_prompt(persona, [h.label for h in leaves]),
            config=genai_types.GenerateContentConfig(
                # 11 move scores per leaf → larger JSON; give it room so it never truncates.
                max_output_tokens=3500, temperature=0.4,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = re.sub(r"```(?:json)?", "", (resp.text or "").strip()).strip("`").strip()
        scores = json.loads(raw).get("scores", [])
    except Exception:
        scores = []

    for i, h in enumerate(leaves):
        s = scores[i] if i < len(scores) and isinstance(scores[i], dict) else {}
        if s:
            h.action_scores = {cat: _clamp01(s.get(cat), _fallback(h)[cat]) for cat in ACTION_CATEGORIES}
        else:
            h.action_scores = _fallback(h)


async def score_persona_actions(db: Session, session_id: str) -> dict:
    """
    Compute predicted per-action scores for every persona header in the session.
    Leaf headers are LLM-scored; parent headers inherit the mean of their children.
    Works for both hierarchical and legacy flat headers (all treated as leaves).
    """
    headers = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id)
        .all()
    )
    if not headers:
        return {"agents_scored": 0, "headers_scored": 0}

    child_parent_ids = {h.parent_id for h in headers if h.parent_id}
    by_agent: dict = {}
    for h in headers:
        by_agent.setdefault(h.agent_id, []).append(h)

    agents = {a.id: a for a in db.query(Agent).filter(Agent.id.in_(by_agent.keys())).all()}

    scored = 0
    for agent_id, ahdrs in by_agent.items():
        persona = agents[agent_id].persona_description if agent_id in agents else ""
        leaves = [h for h in ahdrs if h.id not in child_parent_ids]
        await _score_actions_for_agent(persona, leaves)
        scored += len(leaves)

        # Parents inherit the mean of their children's scores.
        children_by_parent: dict = {}
        for h in ahdrs:
            if h.parent_id:
                children_by_parent.setdefault(h.parent_id, []).append(h)
        for h in ahdrs:
            if h.id in children_by_parent:
                h.action_scores = _mean_action_scores(
                    [c.action_scores for c in children_by_parent[h.id] if c.action_scores]
                )

    db.commit()
    return {"agents_scored": len(by_agent), "headers_scored": scored}


# ── Per-agent usage tables (hierarchy + observed actions) ─────────────────────

async def compute_persona_usage(db: Session, session_id: str) -> dict:
    """
    Per-agent persona-topic usage (one table per agent), now with a two-level tree
    and per-action scores.

    For each agent we take ONLY that agent's own persona topic headers. Each post the
    agent authored is matched to its nearest LEAF header (cosine similarity), and counts
    roll up to the parent. Each header carries its predicted per-action scores plus an
    OBSERVED action breakdown (reply/react/refine/view/skip) taken from the agent's
    recommendation log, so the UI can compare predicted vs observed behaviour.
    """
    from simulation.topic_tree import _embed_all, _cosine

    headers = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id)
        .order_by(PersonaTopicHeader.agent_id.asc(), PersonaTopicHeader.rank.asc())
        .all()
    )
    if not headers:
        return {"agents": []}

    child_parent_ids = {h.parent_id for h in headers if h.parent_id}
    headers_by_agent: dict = {}
    for h in headers:
        headers_by_agent.setdefault(h.agent_id, []).append(h)

    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .all()
    )
    posts_by_id = {p.id: p for p in posts}
    authored_by_agent: dict = {}
    for p in posts:
        if p.author_agent_id:
            authored_by_agent.setdefault(p.author_agent_id, []).append(p)

    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .all()
    )
    logs_by_agent: dict = {}
    for l in logs:
        logs_by_agent.setdefault(l.agent_id, []).append(l)

    agents = db.query(Agent).filter(Agent.is_active == True).all()
    agents = [a for a in agents if a.id in headers_by_agent]

    # A "subject post" of a log is the post the action produced or targeted — used to
    # attribute observed actions to a header.
    def _subject_id(l: RecommendationLog) -> str | None:
        return l.created_post_id or l.recommended_post_id

    out_agents = []
    for a in agents:
        ahdrs = headers_by_agent[a.id]
        leaves = [h for h in ahdrs if h.id not in child_parent_ids]
        if not leaves:
            continue

        aposts = authored_by_agent.get(a.id, [])
        alogs = logs_by_agent.get(a.id, [])

        # Posts to embed = authored posts ∪ subject posts of this agent's logs.
        relevant_ids: list = []
        seen: set = set()
        for p in aposts:
            if p.id not in seen:
                seen.add(p.id); relevant_ids.append(p.id)
        for l in alogs:
            sid = _subject_id(l)
            if sid and sid in posts_by_id and sid not in seen:
                seen.add(sid); relevant_ids.append(sid)

        leaf_vecs = await _embed_all([f"{h.label}. {h.rationale or ''}" for h in leaves])
        rel_vecs = await _embed_all(
            [f"{posts_by_id[pid].name}. {posts_by_id[pid].description}" for pid in relevant_ids]
        ) if relevant_ids else []

        # Nearest leaf header for each relevant post.
        post_to_leaf: dict = {}
        for pid, vec in zip(relevant_ids, rel_vecs):
            if not leaf_vecs:
                break
            sims = [_cosine(vec, hv) for hv in leaf_vecs]
            post_to_leaf[pid] = leaves[max(range(len(sims)), key=lambda i: sims[i])].id

        # Authored-post counts + examples per leaf.
        counts: dict = {h.id: 0 for h in leaves}
        examples: dict = {h.id: [] for h in leaves}
        for p in aposts:
            lid = post_to_leaf.get(p.id)
            if lid is None:
                continue
            counts[lid] += 1
            if len(examples[lid]) < 40:
                examples[lid].append(p.name)

        # Observed action breakdown per leaf, from the agent's recommendation log.
        observed: dict = {h.id: {c: 0 for c in ACTION_CATEGORIES + ["skip"]} for h in leaves}
        for l in alogs:
            sid = _subject_id(l)
            lid = post_to_leaf.get(sid) if sid else None
            if lid is None:
                continue
            act = (l.actual_action or "").strip()
            if l.outcome == "skipped" and act in ("skip", ""):
                observed[lid]["skip"] += 1
            else:
                cat = ACTION_CATEGORY_MAP.get(act)
                if cat:
                    observed[lid][cat] += 1

        total = len(aposts)

        def _leaf_dict(h: PersonaTopicHeader) -> dict:
            return {
                "id": h.id, "parent_id": h.parent_id, "label": h.label,
                "rationale": h.rationale or "", "rank": h.rank,
                "confidence": round(h.confidence if h.confidence is not None else 0.5, 3),
                "action_scores": h.action_scores or {},
                "observed_actions": observed.get(h.id, {c: 0 for c in ACTION_CATEGORIES + ["skip"]}),
                "posts": counts.get(h.id, 0),
                "share": round(counts.get(h.id, 0) / total, 4) if total else 0.0,
                "used": counts.get(h.id, 0) > 0,
                "examples": examples.get(h.id, []),
                "subtopics": [],
            }

        # Assemble the two-level tree (top-level headers each carrying their sub-headers).
        children_by_parent: dict = {}
        for h in ahdrs:
            if h.parent_id:
                children_by_parent.setdefault(h.parent_id, []).append(h)

        topics = []
        for h in ahdrs:
            if h.parent_id is not None:
                continue  # handled under its parent
            kids = children_by_parent.get(h.id, [])
            if kids:
                sub_dicts = sorted(
                    [_leaf_dict(k) for k in kids],
                    key=lambda d: (-d["posts"], d["rank"]),
                )
                agg_posts = sum(d["posts"] for d in sub_dicts)
                agg_obs = {c: sum(d["observed_actions"].get(c, 0) for d in sub_dicts)
                           for c in ACTION_CATEGORIES + ["skip"]}
                topics.append({
                    "id": h.id, "parent_id": None, "label": h.label,
                    "rationale": h.rationale or "", "rank": h.rank,
                    "confidence": round(h.confidence if h.confidence is not None else 0.5, 3),
                    "action_scores": h.action_scores or _mean_action_scores(
                        [d["action_scores"] for d in sub_dicts if d["action_scores"]]),
                    "observed_actions": agg_obs,
                    "posts": agg_posts,
                    "share": round(agg_posts / total, 4) if total else 0.0,
                    "used": agg_posts > 0,
                    "examples": [],
                    "subtopics": sub_dicts,
                })
            else:
                topics.append(_leaf_dict(h))

        # Entropy over the leaf distribution — variety of the agent's topic spread.
        entropy = _normalised_entropy([c for c in counts.values() if c > 0], total)

        topics.sort(key=lambda d: (-d["posts"], d["rank"]))
        out_agents.append({
            "id": a.id, "name": a.name, "domain": a.domain,
            "avatar_color": a.avatar_color, "role": a.role.value,
            "total_posts": total,
            "entropy": entropy,
            "action_categories": ACTION_CATEGORIES,
            "topics": topics,
        })

    return {"agents": out_agents}


def _normalised_entropy(counts: list, total: int) -> float:
    """Shannon entropy of a category distribution, normalised to [0,1] by log(#categories)."""
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in probs)
    return round(h / math.log(len(probs)), 4) if len(probs) > 1 else 0.0


# ── Skip-driven refinement ────────────────────────────────────────────────────

async def refine_persona_topics(db: Session, session_id: str) -> dict:
    """
    Skip-driven refinement (reduce the skip rate).

    For each agent, compare its topic headers against (a) the content it actually
    engaged with and (b) the reasons it gave for SKIPPING recommendations, using
    embedding similarity. Topics that look like what the agent engaged with get a
    confidence boost; topics that look like what it skipped get a penalty. Ranks are
    then re-ordered by the updated confidence WITHIN each sibling group (so sub-headers
    reorder among themselves and top-level headers among themselves).
    """
    from simulation.topic_tree import _embed_all, _cosine

    headers = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id)
        .order_by(PersonaTopicHeader.agent_id.asc(), PersonaTopicHeader.rank.asc())
        .all()
    )
    if not headers:
        return {"agents_refined": 0, "topics_updated": 0}

    headers_by_agent: dict = {}
    for h in headers:
        headers_by_agent.setdefault(h.agent_id, []).append(h)

    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
            DeliberationPost.author_agent_id != None,
        )
        .all()
    )
    posts_by_agent: dict = {}
    for p in posts:
        posts_by_agent.setdefault(p.author_agent_id, []).append(f"{p.name}. {p.description}")

    skip_logs = (
        db.query(RecommendationLog)
        .filter(
            RecommendationLog.session_id == session_id,
            RecommendationLog.outcome == "skipped",
            RecommendationLog.skip_reason.isnot(None),
        )
        .all()
    )
    skips_by_agent: dict = {}
    for l in skip_logs:
        txt = (l.skip_reason or "").strip()
        if txt and txt != "empty_content":
            skips_by_agent.setdefault(l.agent_id, []).append(txt)

    BOOST, PENALTY = 0.25, 0.25
    agents_refined = 0
    topics_updated = 0

    for agent_id, ahdrs in headers_by_agent.items():
        engaged_texts = posts_by_agent.get(agent_id, [])
        skip_texts = skips_by_agent.get(agent_id, [])
        if not engaged_texts and not skip_texts:
            continue

        header_vecs = await _embed_all([f"{h.label}. {h.rationale or ''}" for h in ahdrs])
        engaged_vecs = await _embed_all(engaged_texts) if engaged_texts else []
        skip_vecs = await _embed_all(skip_texts) if skip_texts else []

        for h, hv in zip(ahdrs, header_vecs):
            sim_engage = max((_cosine(hv, ev) for ev in engaged_vecs), default=0.0)
            sim_skip = max((_cosine(hv, sv) for sv in skip_vecs), default=0.0)
            base = h.confidence if h.confidence is not None else 0.5
            new_conf = base + BOOST * max(0.0, sim_engage) - PENALTY * max(0.0, sim_skip)
            h.confidence = round(max(0.05, min(0.99, new_conf)), 3)
            topics_updated += 1

        # Re-order ranks within each sibling group (same parent_id).
        siblings: dict = {}
        for h in ahdrs:
            siblings.setdefault(h.parent_id, []).append(h)
        for group in siblings.values():
            for rank, h in enumerate(sorted(group, key=lambda x: x.confidence or 0.0, reverse=True)):
                h.rank = rank
        agents_refined += 1

    db.commit()
    return {"agents_refined": agents_refined, "topics_updated": topics_updated}


async def split_persona_topic(db: Session, session_id: str, topic_id: str) -> dict:
    """
    Deepen one persona topic header by generating 2-3 finer SUB-HEADERS beneath it
    (LLM). The parent header is KEPT; the new sub-headers become its children and are
    given their own per-action scores. The parent's action_scores are recomputed as
    the mean of its children.
    """
    parent = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.id == topic_id, PersonaTopicHeader.session_id == session_id)
        .first()
    )
    if not parent:
        return {"split": False, "reason": "topic not found"}

    agent = db.query(Agent).filter(Agent.id == parent.agent_id).first()
    persona = agent.persona_description if agent else ""

    prompt = (
        f"A deliberation participant has a persona topic of interest called "
        f'"{parent.label}" ({parent.rationale or ""}).\n\n'
        f"Participant persona: {persona}\n\n"
        "Split this single topic into 2 or 3 MORE SPECIFIC sub-topics that capture the "
        "distinct facets this participant most likely cares about. Each sub-topic gets a "
        "short header (2-5 words), a one-sentence rationale, and a confidence 0.0-1.0.\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"subtopics": [{"label": "...", "rationale": "...", "confidence": 0.8}, ...]}'
    )
    try:
        resp = await _get_client().aio.models.generate_content(
            model=TOPIC_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=400, temperature=0.5,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = re.sub(r"```(?:json)?", "", (resp.text or "").strip()).strip("`").strip()
        subs = json.loads(raw).get("subtopics", [])
    except Exception:
        subs = []

    subs = [s for s in subs if isinstance(s, dict) and str(s.get("label", "")).strip()][:3]
    if len(subs) < 2:
        return {"split": False, "reason": "could not split"}

    # Where to start ranking the new children (after any existing children).
    existing_children = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.parent_id == parent.id)
        .count()
    )

    created = []
    new_rows: List[PersonaTopicHeader] = []
    for i, s in enumerate(subs):
        h = PersonaTopicHeader(
            session_id=session_id, agent_id=parent.agent_id,
            label=str(s["label"]).strip()[:80],
            rank=existing_children + i,
            rationale=str(s.get("rationale", ""))[:200],
            confidence=_clamp01(s.get("confidence", 0.5)),
            parent_id=parent.id,
        )
        db.add(h)
        new_rows.append(h)
        created.append(h.label)
    db.flush()

    # Score the new children's actions, then roll the parent up to their mean.
    await _score_actions_for_agent(persona, new_rows)
    all_children = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.parent_id == parent.id)
        .all()
    )
    parent.action_scores = _mean_action_scores(
        [c.action_scores for c in all_children if c.action_scores]
    )
    db.commit()
    return {"split": True, "subtopics": created}
