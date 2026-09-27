"""
LLM-judge that rates each deliberation post 0..1 on content quality.

Used to measure whether the recommendation engine — by tailoring suggestions
to each agent's profile — actually produces higher-quality contributions over
the course of the simulation.

Quality dimensions the judge considers:
  - specificity  (concrete, not vague)
  - relevance    (advances the parent post's deliberation)
  - substance    (adds new information rather than restating)
"""

import json
import os
import re
import asyncio
from typing import List, Optional
from sqlalchemy.orm import Session
from google import genai
from google.genai import types as genai_types

from models import DeliberationPost, PostQualityScore


JUDGE_MODEL = "gemini-2.5-flash"
_BATCH_LIMIT = 8                # concurrent LLM calls — keep modest to avoid rate limits
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _build_prompt(post: DeliberationPost, parent: Optional[DeliberationPost]) -> str:
    parent_block = (
        f'PARENT [{parent.ptype.value.upper()}] "{parent.name}": {parent.description}'
        if parent else "PARENT: (this is a root mission)"
    )
    return f"""You are a deliberation quality judge. Rate the POST below on a 0.0–1.0 scale.

{parent_block}

POST [{post.ptype.value.upper()}] "{post.name}":
{post.description}

Rate on three dimensions, then average to one score:
  - SPECIFICITY: concrete vs vague   (0 = generic platitude, 1 = sharply specific)
  - RELEVANCE:   advances the parent (0 = unrelated, 1 = directly responsive)
  - SUBSTANCE:   adds new content    (0 = restates parent, 1 = introduces new insight)

Return ONLY valid JSON (no markdown):
{{"score": 0.72, "rationale": "one short sentence"}}"""


async def _judge_one(post: DeliberationPost, parent: Optional[DeliberationPost]) -> dict:
    try:
        resp = await _get_client().aio.models.generate_content(
            model=JUDGE_MODEL,
            contents=_build_prompt(post, parent),
            config=genai_types.GenerateContentConfig(
                max_output_tokens=120,
                temperature=0.2,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = (resp.text or "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        return {"score": score, "rationale": str(parsed.get("rationale", ""))[:200]}
    except Exception:
        return {"score": 0.5, "rationale": "fallback (LLM error)"}


async def score_session_quality(db: Session, session_id: str) -> dict:
    """
    Score every post in this session that does not yet have a quality score.
    The mission post (no author) is skipped since it isn't an agent contribution.
    """
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
            DeliberationPost.author_agent_id != None,
        )
        .all()
    )
    if not posts:
        return {"scored": 0, "skipped_existing": 0, "total_posts": 0}

    already_scored = {
        s.post_id for s in db.query(PostQualityScore)
        .filter(PostQualityScore.session_id == session_id).all()
    }
    pending = [p for p in posts if p.id not in already_scored]

    # Look up parents once
    parent_ids = {p.purpose_id for p in pending if p.purpose_id}
    parents = {
        pp.id: pp for pp in db.query(DeliberationPost)
        .filter(DeliberationPost.id.in_(parent_ids)).all()
    } if parent_ids else {}

    # Process in batches to keep concurrent LLM calls bounded
    for i in range(0, len(pending), _BATCH_LIMIT):
        batch = pending[i : i + _BATCH_LIMIT]
        results = await asyncio.gather(*[
            _judge_one(p, parents.get(p.purpose_id) if p.purpose_id else None)
            for p in batch
        ])
        for post, r in zip(batch, results):
            db.add(PostQualityScore(
                post_id=post.id, session_id=session_id,
                score=r["score"], rationale=r.get("rationale", ""),
            ))
        db.commit()

    return {
        "scored": len(pending),
        "skipped_existing": len(already_scored),
        "total_posts": len(posts),
    }
