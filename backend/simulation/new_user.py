"""
Create a new participant from a short self-description (Feature).

The user types a couple sentences about themselves. We embed that, compare it to
every existing agent's persona, pick the most semantically similar one, and seed
the new user's persona table by copying that nearest persona's topic headers
(labels + confidences). This bootstraps a brand-new user without an LLM topic
call and without a cold start — they inherit the interests of whoever they most
resemble.
"""

import random
from sqlalchemy.orm import Session

from models import Agent, AgentRole, PersonaTopicHeader

_PALETTE = ["#10b981", "#6366f1", "#f59e0b", "#ec4899", "#8b5cf6", "#00d4ff", "#f97316", "#22c55e"]
_FLASH_MODEL = "gemini-2.5-flash"


async def create_user_from_description(db: Session, session_id: str, name: str, description: str) -> dict:
    from simulation.topic_tree import _embed_all, _cosine
    from seed import build_system_prompt

    desc = (description or "").strip()
    if not desc:
        return {"created": False, "reason": "empty description"}

    existing = (
        db.query(Agent)
        .filter(Agent.is_active == True, Agent.role == AgentRole.specialist)
        .all()
    )
    if not existing:
        return {"created": False, "reason": "no existing personas to compare against"}

    # Embed the self-description alongside every existing persona, then pick the nearest.
    vecs = await _embed_all([desc] + [a.persona_description for a in existing])
    dvec, evecs = vecs[0], vecs[1:]
    sims = [_cosine(dvec, ev) for ev in evecs]
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    similar = existing[best_i]
    similarity = round(sims[best_i], 4)

    clean_name = (name or "").strip() or f"User-{str(random.randint(1000, 9999))}"
    new_agent = Agent(
        name=clean_name,
        role=AgentRole.specialist,
        persona_description=desc,
        system_prompt=build_system_prompt(clean_name, "custom participant", desc),
        domain="custom",
        avatar_color=random.choice(_PALETTE),
        llm_assist_rate=0.7,
        model_id=similar.model_id or _FLASH_MODEL,
        is_active=True,
    )
    db.add(new_agent)
    db.flush()

    # Inherit the nearest persona's topic table for this session.
    src = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id, PersonaTopicHeader.agent_id == similar.id)
        .order_by(PersonaTopicHeader.rank.asc())
        .all()
    )
    copied = 0
    for h in src:
        db.add(PersonaTopicHeader(
            session_id=session_id,
            agent_id=new_agent.id,
            label=h.label,
            rank=h.rank,
            rationale=f"(inherited from {similar.name}) {h.rationale or ''}".strip(),
            confidence=h.confidence if h.confidence is not None else 0.5,
        ))
        copied += 1
    db.commit()

    return {
        "created": True,
        "new_agent": {"id": new_agent.id, "name": new_agent.name, "avatar_color": new_agent.avatar_color},
        "similar_to": similar.name,
        "similarity": similarity,
        "topics_copied": copied,
    }
