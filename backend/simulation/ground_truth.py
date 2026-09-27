"""
Synthetic ground-truth persona models (for consistency + evaluation).

In synthetic mode a test agent decides engage/skip with NO LLM: it engages a post
iff the post embedding is closer to one of the agent's hidden LIKED topic centroids
than to any DISLIKED centroid. This makes skip behaviour perfectly consistent (the
same post always gets the same decision) and gives us a ground truth to score the
inferred decision tree against.

The liked/disliked topics are produced once per session by an LLM from the agent's
persona + the mission, then embedded; the embeddings are the centroids.
"""

import json
import os
import re
import asyncio
from typing import List
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import Agent, SimulationSession, DeliberationPost, GroundTruthPersona


GT_MODEL = "gemini-2.5-flash"
# Decision margin: engage only if the best liked centroid beats the best disliked one
# by at least this much. Calibrated offline against real post embeddings so the task
# stays non-degenerate.
#
# WHY THIS EXISTS: at margin 0.0 the benchmark collapsed once post descriptions were
# shortened to 2-3 sentences. Short text embeds closer to the short liked-topic phrases,
# so ~97% of decisions became "engage" (9 skips in 280) — nothing left to predict, and
# inference accuracy FELL because the tree had almost no negative examples. Measured
# base-rate over the observed post space (102 short posts x 22 personas):
#   margin 0.000 -> 23.6% base   (REALIZED 3.2% -- degenerate, nothing to learn)
#   margin 0.005 -> 26.9% base
#   margin 0.020 -> 38.8% base   (REALIZED 74.3% -- map starved to 70 posts, death spiral)
# Realized skip is dominated by a feedback loop (harder truth -> fewer posts -> worse
# candidate pool -> higher skip), not by the base rate, so this knob is very sensitive.
# 0.005 was tried and landed within noise of 0.0 (3.9% vs 3.2% realized), so we keep 0.0.
# NOTE: this is a benchmark-difficulty knob, not a claim about real users. Skip rates
# are only comparable across runs when it is held fixed.
ENGAGE_MARGIN = 0.0

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _build_prompt(persona: str, mission_name: str, mission_desc: str) -> str:
    blurb = mission_name + (f" — {mission_desc}" if mission_desc else "")
    return (
        f"I am the following participant:\n{persona}\n\n"
        f"I will be in deliberations about: {blurb}\n\n"
        "Give me TWO lists of short subtopic headers (2-5 words each):\n"
        "- liked: ~10 subtopics I would enthusiastically contribute to\n"
        "- disliked: ~10 subtopics I would NOT engage with and would skip\n\n"
        "Make them concrete and clearly separable. Return ONLY valid JSON (no markdown):\n"
        '{"liked": ["...", "..."], "disliked": ["...", "..."]}'
    )


async def _topics_for_agent(agent: Agent, mission_name: str, mission_desc: str) -> tuple:
    from simulation.roles.base import call_with_retry
    prompt = _build_prompt(agent.persona_description, mission_name, mission_desc)
    try:
        resp = await call_with_retry(lambda: _get_client().aio.models.generate_content(
            model=GT_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=800, temperature=0.6,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        ))
        raw = re.sub(r"```(?:json)?", "", (resp.text or "").strip()).strip("`").strip()
        parsed = json.loads(raw)
        liked = [str(t).strip()[:80] for t in parsed.get("liked", []) if str(t).strip()][:12]
        disliked = [str(t).strip()[:80] for t in parsed.get("disliked", []) if str(t).strip()][:12]
    except Exception:
        liked, disliked = [], []
    return liked, disliked


async def generate_ground_truth(db: Session, session_id: str) -> dict:
    """Create (or replace) a hidden ground-truth persona model for every active agent."""
    from simulation.topic_tree import _embed_all

    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        return {"agents": 0}
    mission = (
        db.query(DeliberationPost).filter(DeliberationPost.id == session.mission_id).first()
        if session.mission_id else None
    )
    mission_name = mission.name if mission else "the deliberation mission"
    mission_desc = mission.description if mission else ""

    agents = db.query(Agent).filter(Agent.is_active == True).all()
    if not agents:
        return {"agents": 0}

    results = await asyncio.gather(
        *[_topics_for_agent(a, mission_name, mission_desc) for a in agents]
    )

    db.query(GroundTruthPersona).filter(
        GroundTruthPersona.session_id == session_id
    ).delete()
    db.commit()

    written = 0
    for agent, (liked, disliked) in zip(agents, results):
        if not liked and not disliked:
            continue
        liked_vecs = await _embed_all(liked) if liked else []
        disliked_vecs = await _embed_all(disliked) if disliked else []
        db.add(GroundTruthPersona(
            session_id=session_id, agent_id=agent.id,
            liked_topics=liked, disliked_topics=disliked,
            liked_centroids=liked_vecs, disliked_centroids=disliked_vecs,
        ))
        written += 1
    db.commit()
    return {"agents": written}


def _cosine(a: list, b: list) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def gt_engage(gt: GroundTruthPersona, vec: list) -> bool:
    """Deterministic engage/skip: closer to a liked centroid than any disliked one."""
    liked = gt.liked_centroids or []
    disliked = gt.disliked_centroids or []
    best_like = max((_cosine(vec, c) for c in liked), default=-1.0)
    best_dislike = max((_cosine(vec, c) for c in disliked), default=-1.0)
    return best_like >= best_dislike + ENGAGE_MARGIN


def load_ground_truth(db: Session, session_id: str) -> dict:
    """Return {agent_id: GroundTruthPersona} for the session."""
    rows = db.query(GroundTruthPersona).filter(
        GroundTruthPersona.session_id == session_id
    ).all()
    return {r.agent_id: r for r in rows}
