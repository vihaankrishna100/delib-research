"""
Persona-derived "base user model".

For each agent in a session, ask Gemini to score every top-level (level-1)
topic cluster on a 0-1 scale, using ONLY the agent's persona description.
The result is the prior over what the agent would engage with if their
behavior were purely persona-driven and the recommender had no influence.

The prior is compared against the activity-based observed user model to
measure how well the system learns each agent's actual profile. (The observed
model folds in the prelude warm-start at a low weight; the prior does not.)
"""

import json
import os
import re
import asyncio
from typing import List, Dict
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import Agent, TopicNode, AgentTopicPrior, DeliberationPost


SCORER_MODEL = "gemini-2.5-flash"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _topic_summary(topic: TopicNode, posts_by_id: Dict[str, DeliberationPost], max_examples: int = 4) -> str:
    """Build a short human-readable summary of a topic cluster from its member posts."""
    examples = []
    for pid in (topic.member_post_ids or [])[:max_examples]:
        p = posts_by_id.get(pid)
        if p:
            examples.append(f"{p.name}")
    if not examples:
        return topic.label
    return f"{topic.label} — e.g. {'; '.join(examples)}"


async def _score_agent_against_topics(
    agent: Agent,
    topics: List[TopicNode],
    posts_by_id: Dict[str, DeliberationPost],
) -> Dict[str, dict]:
    """One LLM call: returns {topic_id: {"score": 0..1, "rationale": "..."}}."""
    topic_lines = [
        f"  [{i}] {_topic_summary(t, posts_by_id)}"
        for i, t in enumerate(topics)
    ]
    topics_block = "\n".join(topic_lines)

    prompt = f"""You are evaluating a deliberation participant's PERSONA against a list of TOPIC AREAS.

For each topic, rate from 0.0 to 1.0 how likely this persona is to want to take action in that area:
  0.0  = persona has no interest / would skip
  0.5  = neutral or weak interest
  1.0  = persona is strongly drawn to engage here

Base your scores ONLY on what the persona says about itself. Do not assume effort or expertise outside what the persona describes.

PERSONA:
Name: {agent.name}
Domain: {agent.domain}
{agent.persona_description}

TOPIC AREAS:
{topics_block}

Return ONLY valid JSON (no markdown, no code fences):
{{
  "scores": {{
    "0": {{"score": 0.85, "rationale": "one short sentence"}},
    "1": {{"score": 0.30, "rationale": "..."}},
    ...
  }}
}}"""

    try:
        resp = await _get_client().aio.models.generate_content(
            model=SCORER_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=800,
                temperature=0.2,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = (resp.text or "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        parsed = json.loads(raw)
        scores_by_idx = parsed.get("scores", {})
    except Exception as e:
        return {t.id: {"score": 0.5, "rationale": f"fallback (LLM error)"} for t in topics}

    out: Dict[str, dict] = {}
    for i, t in enumerate(topics):
        entry = scores_by_idx.get(str(i)) or scores_by_idx.get(i) or {}
        try:
            s = float(entry.get("score", 0.5))
        except (TypeError, ValueError):
            s = 0.5
        s = max(0.0, min(1.0, s))
        out[t.id] = {"score": s, "rationale": str(entry.get("rationale", ""))[:200]}
    return out


async def compute_priors(db: Session, session_id: str) -> dict:
    """
    Compute persona-derived prior scores for every (agent, level-1 topic) cell
    in this session and persist them. Returns a summary dict.
    """
    topics = (
        db.query(TopicNode)
        .filter(TopicNode.session_id == session_id, TopicNode.level == 1)
        .order_by(TopicNode.label.asc())
        .all()
    )
    if not topics:
        return {"agents_scored": 0, "topics_scored": 0, "cells_written": 0}

    posts = (
        db.query(DeliberationPost)
        .filter(DeliberationPost.session_id == session_id, DeliberationPost.is_removed == False)
        .all()
    )
    posts_by_id = {p.id: p for p in posts}

    agents = db.query(Agent).filter(Agent.is_active == True).all()
    if not agents:
        return {"agents_scored": 0, "topics_scored": len(topics), "cells_written": 0}

    db.query(AgentTopicPrior).filter(AgentTopicPrior.session_id == session_id).delete()
    db.commit()

    results = await asyncio.gather(
        *[_score_agent_against_topics(a, topics, posts_by_id) for a in agents]
    )

    cells = 0
    for agent, scored in zip(agents, results):
        for topic_id, entry in scored.items():
            db.add(AgentTopicPrior(
                session_id=session_id,
                agent_id=agent.id,
                topic_id=topic_id,
                score=entry["score"],
                rationale=entry.get("rationale", ""),
            ))
            cells += 1
    db.commit()

    return {
        "agents_scored": len(agents),
        "topics_scored": len(topics),
        "cells_written": cells,
    }
