import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from collections import defaultdict
from database import get_db
from models import (
    RecommendationLog, AgentProfile, Agent, DeliberationPost, Event,
    TopicNode, AgentTopicPrior, PostQualityScore, PersonaTopicHeader,
    UserModelSnapshot, GroundTruthPersona, SimulationSession,
)
from schemas import (
    StrategyStats, AgentProfileOut, TreeGrowthPoint,
    TopicNodeOut, UserModelCell, UserModelMatrix,
    PriorCell, PriorMatrix,
    AgentResearchMetrics, TopicSurprise, ResearchMetricsOut,
    PersonaTopicHeaderOut, PersonaTopicsOut, PersonaUsageOut,
)
from simulation.topic_tree import rebuild_topic_tree
from simulation.prior_model import compute_priors
from simulation.quality_judge import score_session_quality
from simulation.persona_topics import (
    generate_persona_topics, compute_persona_usage,
    refine_persona_topics, split_persona_topic, score_persona_actions,
)
from simulation.new_user import create_user_from_description
from simulation.agent_reports import generate_agent_reports
from simulation.ground_truth import generate_ground_truth
from simulation.user_model_tree import train_session
from pydantic import BaseModel
from typing import Dict, List

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Prelude (warm-start) activity counts toward the persona/user-model score at a
# low weight — enough to remove the cold start without letting it dominate the
# real simulation's signal.
PRELUDE_WEIGHT = 0.25


@router.get("/sessions/{session_id}/recommendation_stats")
def recommendation_stats(session_id: str, db: Session = Depends(get_db)):
    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .all()
    )

    by_strategy: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "completed": 0, "engaged": 0, "skipped": 0, "viewed": 0, "correct": 0,
    })

    for log in logs:
        s = log.strategy
        by_strategy[s]["total"] += 1

        # Only count logs where the agent actually made a decision
        if log.actual_action is None:
            continue
        by_strategy[s]["completed"] += 1

        if log.outcome == "engaged":
            by_strategy[s]["engaged"] += 1
        elif log.outcome == "skipped":
            if log.actual_action == "view":
                by_strategy[s]["viewed"] += 1
            else:
                by_strategy[s]["skipped"] += 1

        # Accuracy: agent performed exactly the recommended action
        if log.actual_action == log.recommended_action:
            by_strategy[s]["correct"] += 1

    results = []
    for strategy, c in by_strategy.items():
        completed = c["completed"]
        total = c["total"]
        results.append(StrategyStats(
            strategy=strategy,
            total_recommendations=total,
            engaged=c["engaged"],
            skipped=c["skipped"] + c["viewed"],   # both are non-engaging
            # Rates use completed (decided) logs as denominator — excludes interrupted ticks
            skip_rate=round((c["skipped"] + c["viewed"]) / completed, 3) if completed > 0 else 0.0,
            # Ground truth accuracy: agent chose the recommended action / all completed decisions
            accuracy=round(c["correct"] / completed, 3) if completed > 0 else 0.0,
        ))
    return results


@router.get("/sessions/{session_id}/skip_rate_timeline")
def skip_rate_timeline(session_id: str, window: int = 10, step: int | None = None,
                       db: Session = Depends(get_db)):
    """
    Skip rate over time, per strategy. Each point is the skip rate over a `window`-tick
    span; the span start advances by `step` ticks. step defaults to window (non-overlapping
    buckets); step=1 gives a SLIDING window (1–10, 2–11, 3–12, …) that reads much smoother.
    """
    window = max(1, window)
    step = window if step is None else max(1, step)
    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .all()
    )

    # (strategy, tick) -> {completed, skipped}, then slide a window over ticks.
    per: dict[tuple, dict] = defaultdict(lambda: {"completed": 0, "skipped": 0})
    strategies_seen: set = set()
    max_tick = 0
    for log in logs:
        if log.strategy == "forced":        # bootstrap action, not a strategy under study
            continue
        if log.actual_action is None:       # only completed (decided) recommendations
            continue
        strategies_seen.add(log.strategy)
        max_tick = max(max_tick, log.sim_tick)
        d = per[(log.strategy, log.sim_tick)]
        d["completed"] += 1
        if log.outcome == "skipped":        # includes "view" (non-engaging)
            d["skipped"] += 1

    starts = list(range(0, max_tick + 1, step)) or [0]
    strategies = []
    for strategy in sorted(strategies_seen):
        points = []
        for s in starts:
            completed = skipped = 0
            for t in range(s, s + window):
                d = per.get((strategy, t))
                if d:
                    completed += d["completed"]
                    skipped += d["skipped"]
            if completed > 0:
                points.append({
                    "tick": s,
                    "label": f"{s}–{s + window - 1}",
                    "completed": completed,
                    "skipped": skipped,
                    "skip_rate": round(skipped / completed, 3),
                })
        strategies.append({"strategy": strategy, "points": points})

    return {"window": window, "step": step, "max_tick": max_tick, "strategies": strategies}


@router.get("/agents/{agent_id}/profile", response_model=AgentProfileOut)
def get_agent_profile(agent_id: str, db: Session = Depends(get_db)):
    profile = db.query(AgentProfile).filter(AgentProfile.agent_id == agent_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    result = AgentProfileOut.model_validate(profile)
    result.agent_name = agent.name if agent else None
    return result


@router.get("/sessions/{session_id}/profiles")
def get_session_profiles(session_id: str, db: Session = Depends(get_db)):
    profiles = db.query(AgentProfile).filter(AgentProfile.session_id == session_id).all()
    results = []
    for p in profiles:
        agent = db.query(Agent).filter(Agent.id == p.agent_id).first()
        data = AgentProfileOut.model_validate(p)
        data.agent_name = agent.name if agent else None
        results.append(data)
    return results


@router.get("/sessions/{session_id}/tree_growth")
def tree_growth(session_id: str, db: Session = Depends(get_db)):
    """Returns cumulative post count per ptype per tick."""
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .order_by(DeliberationPost.sim_tick.asc())
        .all()
    )

    # Build cumulative counts per (tick, ptype)
    running: dict[str, int] = defaultdict(int)
    tick_data: dict[int, dict[str, int]] = {}
    for post in posts:
        tick = post.sim_tick
        ptype = post.ptype.value
        running[ptype] += 1
        if tick not in tick_data:
            tick_data[tick] = {}
        tick_data[tick][ptype] = running[ptype]

    # Flatten to list of points
    results = []
    for tick, ptypes in sorted(tick_data.items()):
        for ptype, count in ptypes.items():
            results.append(TreeGrowthPoint(tick=tick, ptype=ptype, count=count))
    return results


@router.get("/sessions/{session_id}/events")
def get_events(session_id: str, limit: int = 100, db: Session = Depends(get_db)):
    events = (
        db.query(Event)
        .filter(Event.session_id == session_id)
        .order_by(Event.sim_tick.desc(), Event.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "agent_id": e.agent_id,
            "agent_role": e.agent_role,
            "event_type": e.event_type,
            "target_type": e.target_type,
            "target_id": e.target_id,
            "metadata": e.metadata_,
            "sim_tick": e.sim_tick,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in events
    ]


@router.get("/sessions/{session_id}/agent_stats")
def agent_stats(session_id: str, db: Session = Depends(get_db)):
    """Posts created per agent, broken down by ptype."""
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
            DeliberationPost.author_agent_id != None,
        )
        .all()
    )

    by_agent: dict[str, dict] = defaultdict(lambda: {"total": 0, "by_ptype": defaultdict(int)})
    for p in posts:
        aid = p.author_agent_id
        by_agent[aid]["total"] += 1
        by_agent[aid]["by_ptype"][p.ptype.value] += 1

    results = []
    for agent_id, data in by_agent.items():
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        results.append({
            "agent_id": agent_id,
            "agent_name": agent.name if agent else agent_id[:8],
            "avatar_color": agent.avatar_color if agent else "#6366f1",
            "domain": agent.domain if agent else "",
            "total": data["total"],
            "by_ptype": dict(data["by_ptype"]),
        })

    results.sort(key=lambda x: x["total"], reverse=True)
    return results


@router.post("/sessions/{session_id}/rebuild_topic_tree")
async def rebuild_topic_tree_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Recompute the hierarchical topic tree from the session's current posts."""
    summary = await rebuild_topic_tree(db, session_id)
    return summary


@router.get("/sessions/{session_id}/topic_tree", response_model=List[TopicNodeOut])
def get_topic_tree(session_id: str, db: Session = Depends(get_db)):
    nodes = (
        db.query(TopicNode)
        .filter(TopicNode.session_id == session_id)
        .order_by(TopicNode.level.asc(), TopicNode.label.asc())
        .all()
    )
    return [TopicNodeOut.model_validate(n) for n in nodes]


@router.get("/sessions/{session_id}/user_models", response_model=UserModelMatrix)
def get_user_models(session_id: str, db: Session = Depends(get_db)):
    """
    Returns the (agent × topic) interest table. Each cell's score is in [0,1]
    and represents the modelled likelihood that the agent would take action
    in that topic area, based on observed engagement and authorship.
    """
    topic_nodes = (
        db.query(TopicNode)
        .filter(TopicNode.session_id == session_id)
        .order_by(TopicNode.level.asc(), TopicNode.label.asc())
        .all()
    )
    if not topic_nodes:
        raise HTTPException(
            status_code=404,
            detail="No topic tree for this session. POST /rebuild_topic_tree first.",
        )

    agents = db.query(Agent).filter(Agent.is_active == True).all()

    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .all()
    )
    post_author: Dict[str, str] = {
        p.id: p.author_agent_id for p in posts if p.author_agent_id
    }
    post_weight: Dict[str, float] = {
        p.id: (PRELUDE_WEIGHT if getattr(p, "is_prelude", False) else 1.0) for p in posts
    }

    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .all()
    )
    # (agent_id, post_id) -> {engaged, skipped} raw counts + weighted counterparts.
    log_buckets: Dict[tuple, Dict[str, float]] = defaultdict(
        lambda: {"engaged": 0, "skipped": 0, "w_engaged": 0.0, "w_skipped": 0.0}
    )
    for log in logs:
        if not log.recommended_post_id or log.outcome not in ("engaged", "skipped"):
            continue
        w = PRELUDE_WEIGHT if getattr(log, "is_prelude", False) else 1.0
        b = log_buckets[(log.agent_id, log.recommended_post_id)]
        b[log.outcome] += 1
        b[f"w_{log.outcome}"] += w

    cells: List[UserModelCell] = []
    for agent in agents:
        for node in topic_nodes:
            member_ids = set(node.member_post_ids or [])
            posts_authored = sum(
                1 for pid in member_ids if post_author.get(pid) == agent.id
            )
            engaged = sum(
                int(log_buckets.get((agent.id, pid), {}).get("engaged", 0)) for pid in member_ids
            )
            skipped = sum(
                int(log_buckets.get((agent.id, pid), {}).get("skipped", 0)) for pid in member_ids
            )

            # Weighted versions — prelude activity counts at PRELUDE_WEIGHT.
            w_authored = sum(
                post_weight.get(pid, 1.0) for pid in member_ids if post_author.get(pid) == agent.id
            )
            w_engaged = sum(
                log_buckets.get((agent.id, pid), {}).get("w_engaged", 0.0) for pid in member_ids
            )
            w_skipped = sum(
                log_buckets.get((agent.id, pid), {}).get("w_skipped", 0.0) for pid in member_ids
            )

            raw = 1.0 * w_engaged + 0.5 * w_authored - 0.3 * w_skipped
            score = 1.0 / (1.0 + math.exp(-raw))   # sigmoid

            cells.append(UserModelCell(
                agent_id=agent.id, topic_id=node.id, score=round(score, 4),
                posts_authored=posts_authored, engaged=engaged, skipped=skipped,
            ))

    return UserModelMatrix(
        agents=[
            {
                "id": a.id, "name": a.name, "domain": a.domain,
                "avatar_color": a.avatar_color, "role": a.role.value,
            }
            for a in agents
        ],
        topics=[TopicNodeOut.model_validate(n) for n in topic_nodes],
        cells=cells,
    )


@router.post("/sessions/{session_id}/compute_priors")
async def compute_priors_endpoint(session_id: str, db: Session = Depends(get_db)):
    """LLM-derived persona priors over the session's top-level topics."""
    summary = await compute_priors(db, session_id)
    return summary


@router.get("/sessions/{session_id}/priors", response_model=PriorMatrix)
def get_priors(session_id: str, db: Session = Depends(get_db)):
    topic_nodes = (
        db.query(TopicNode)
        .filter(TopicNode.session_id == session_id, TopicNode.level == 1)
        .order_by(TopicNode.label.asc())
        .all()
    )
    priors = (
        db.query(AgentTopicPrior)
        .filter(AgentTopicPrior.session_id == session_id)
        .all()
    )
    if not priors:
        raise HTTPException(
            status_code=404,
            detail="No priors computed yet. POST /compute_priors first.",
        )
    agents = db.query(Agent).filter(Agent.is_active == True).all()

    return PriorMatrix(
        agents=[
            {"id": a.id, "name": a.name, "domain": a.domain,
             "avatar_color": a.avatar_color, "role": a.role.value}
            for a in agents
        ],
        topics=[TopicNodeOut.model_validate(n) for n in topic_nodes],
        cells=[
            PriorCell(agent_id=p.agent_id, topic_id=p.topic_id,
                      score=round(p.score, 4), rationale=p.rationale)
            for p in priors
        ],
    )


@router.post("/sessions/{session_id}/score_quality")
async def score_quality_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Run the LLM quality judge over every authored post in the session."""
    summary = await score_session_quality(db, session_id)
    return summary


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@router.get("/sessions/{session_id}/research_metrics", response_model=ResearchMetricsOut)
def get_research_metrics(session_id: str, db: Session = Depends(get_db)):
    """
    Per-agent and per-topic metrics evaluating how well the recommender modelled
    each persona, how often agents skipped recommendations, and the average
    contribution quality. Requires that topic_tree, priors, and quality scores
    have been computed.
    """
    topics = (
        db.query(TopicNode)
        .filter(TopicNode.session_id == session_id, TopicNode.level == 1)
        .order_by(TopicNode.label.asc())
        .all()
    )
    if not topics:
        raise HTTPException(status_code=404, detail="No topic tree. Build one first.")

    priors = db.query(AgentTopicPrior).filter(AgentTopicPrior.session_id == session_id).all()
    if not priors:
        raise HTTPException(status_code=404, detail="No priors computed. POST /compute_priors first.")
    prior_map: Dict[tuple, float] = {(p.agent_id, p.topic_id): p.score for p in priors}

    agents = db.query(Agent).filter(Agent.is_active == True).all()
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .all()
    )
    posts_by_id = {p.id: p for p in posts}

    # Observed scores — reuse the same sigmoid formula as /user_models
    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .all()
    )
    log_buckets: Dict[tuple, Dict[str, float]] = defaultdict(
        lambda: {"engaged": 0, "skipped": 0, "w_engaged": 0.0, "w_skipped": 0.0}
    )
    for log in logs:
        if not log.recommended_post_id or log.outcome not in ("engaged", "skipped"):
            continue
        w = PRELUDE_WEIGHT if getattr(log, "is_prelude", False) else 1.0
        b = log_buckets[(log.agent_id, log.recommended_post_id)]
        b[log.outcome] += 1
        b[f"w_{log.outcome}"] += w

    def observed_score(agent_id: str, topic: TopicNode) -> float:
        member_ids = set(topic.member_post_ids or [])
        w_authored = sum(
            (PRELUDE_WEIGHT if getattr(posts_by_id[pid], "is_prelude", False) else 1.0)
            for pid in member_ids
            if posts_by_id.get(pid) and posts_by_id[pid].author_agent_id == agent_id
        )
        w_engaged = sum(log_buckets.get((agent_id, pid), {}).get("w_engaged", 0.0) for pid in member_ids)
        w_skipped = sum(log_buckets.get((agent_id, pid), {}).get("w_skipped", 0.0) for pid in member_ids)
        raw = 1.0 * w_engaged + 0.5 * w_authored - 0.3 * w_skipped
        return 1.0 / (1.0 + math.exp(-raw))

    # Quality scores
    qrows = db.query(PostQualityScore).filter(PostQualityScore.session_id == session_id).all()
    quality_by_post: Dict[str, float] = {q.post_id: q.score for q in qrows}

    per_agent_list: List[AgentResearchMetrics] = []
    per_topic_drifts: Dict[str, List[float]] = {t.id: [] for t in topics}

    for agent in agents:
        prior_vec = [prior_map.get((agent.id, t.id), 0.5) for t in topics]
        observed_vec = [observed_score(agent.id, t) for t in topics]
        for t, o, p_ in zip(topics, observed_vec, prior_vec):
            per_topic_drifts[t.id].append(abs(o - p_))

        fidelity = _cosine_similarity(prior_vec, observed_vec)

        # Skip / engagement totals across all posts in this session
        engaged_total = sum(b["engaged"] for (aid, _pid), b in log_buckets.items() if aid == agent.id)
        skipped_total = sum(b["skipped"] for (aid, _pid), b in log_buckets.items() if aid == agent.id)
        denom = engaged_total + skipped_total
        skip_rate = (skipped_total / denom) if denom > 0 else 0.0

        authored_posts = [p for p in posts if p.author_agent_id == agent.id]
        authored_qualities = [quality_by_post[p.id] for p in authored_posts if p.id in quality_by_post]
        avg_q = sum(authored_qualities) / len(authored_qualities) if authored_qualities else None

        per_agent_list.append(AgentResearchMetrics(
            agent_id=agent.id, agent_name=agent.name, avatar_color=agent.avatar_color,
            persona_fidelity=round(fidelity, 4),
            skip_rate=round(skip_rate, 4),
            avg_post_quality=round(avg_q, 4) if avg_q is not None else None,
            posts_authored=len(authored_posts),
            engaged=engaged_total,
            skipped=skipped_total,
        ))

    per_topic = [
        TopicSurprise(
            topic_id=t.id, label=t.label,
            mean_abs_drift=round(
                sum(per_topic_drifts[t.id]) / len(per_topic_drifts[t.id]), 4
            ) if per_topic_drifts[t.id] else 0.0,
        )
        for t in topics
    ]
    per_topic.sort(key=lambda x: x.mean_abs_drift, reverse=True)

    fids = [a.persona_fidelity for a in per_agent_list]
    skips = [a.skip_rate for a in per_agent_list]
    quals = [a.avg_post_quality for a in per_agent_list if a.avg_post_quality is not None]

    return ResearchMetricsOut(
        per_agent=per_agent_list,
        per_topic_surprise=per_topic,
        avg_persona_fidelity=round(sum(fids) / len(fids), 4) if fids else 0.0,
        avg_skip_rate=round(sum(skips) / len(skips), 4) if skips else 0.0,
        avg_post_quality=round(sum(quals) / len(quals), 4) if quals else None,
        quality_scored_posts=len(quality_by_post),
        total_posts=len(posts),
    )


# ── Persona topic headers (Feature 1) ────────────────────────────────────────

@router.post("/sessions/{session_id}/generate_persona_topics")
async def generate_persona_topics_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Generate ~10 persona-derived topic headers per agent (LLM, persona + mission)."""
    return await generate_persona_topics(db, session_id)


@router.get("/sessions/{session_id}/persona_topics", response_model=PersonaTopicsOut)
def get_persona_topics(session_id: str, db: Session = Depends(get_db)):
    headers = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id)
        .order_by(PersonaTopicHeader.agent_id.asc(), PersonaTopicHeader.rank.asc())
        .all()
    )
    if not headers:
        raise HTTPException(
            status_code=404,
            detail="No persona topics yet. POST /generate_persona_topics first.",
        )
    agent_ids = {h.agent_id for h in headers}
    agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
    return PersonaTopicsOut(
        agents=[
            {"id": a.id, "name": a.name, "domain": a.domain,
             "avatar_color": a.avatar_color, "role": a.role.value}
            for a in agents
        ],
        headers=[PersonaTopicHeaderOut.model_validate(h) for h in headers],
    )


@router.get("/sessions/{session_id}/persona_user_models", response_model=PersonaUsageOut)
async def persona_user_models(session_id: str, db: Session = Depends(get_db)):
    """
    Per-agent persona-topic usage: one table per agent, showing which of that
    agent's own ~10 persona topic headers they are actually contributing to.
    """
    headers_exist = (
        db.query(PersonaTopicHeader)
        .filter(PersonaTopicHeader.session_id == session_id)
        .first()
    )
    if not headers_exist:
        raise HTTPException(
            status_code=404,
            detail="No persona topics yet. POST /generate_persona_topics first.",
        )
    return await compute_persona_usage(db, session_id)


@router.post("/sessions/{session_id}/refine_persona_topics")
async def refine_persona_topics_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Skip-driven refinement: re-score topic confidence from observed engages/skips and reorder."""
    return await refine_persona_topics(db, session_id)


@router.post("/sessions/{session_id}/score_persona_actions")
async def score_persona_actions_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Predict per-action scores (reply/react/refine/view) for each existing persona header."""
    return await score_persona_actions(db, session_id)


class SplitTopicBody(BaseModel):
    topic_id: str


@router.post("/sessions/{session_id}/split_persona_topic")
async def split_persona_topic_endpoint(session_id: str, body: SplitTopicBody, db: Session = Depends(get_db)):
    """Split one persona topic header into 2-3 finer sub-topics (LLM)."""
    return await split_persona_topic(db, session_id, body.topic_id)


class NewUserBody(BaseModel):
    name: str
    description: str


@router.post("/sessions/{session_id}/users_from_description")
async def users_from_description_endpoint(session_id: str, body: NewUserBody, db: Session = Depends(get_db)):
    """Create a new participant from a self-description, seeded from the nearest persona."""
    return await create_user_from_description(db, session_id, body.name, body.description)


# ── Per-agent action documents (Feature 2) ───────────────────────────────────

@router.post("/sessions/{session_id}/generate_agent_reports")
def generate_agent_reports_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Write a per-agent action document (recommendations, outcomes, skip reasons)."""
    result = generate_agent_reports(db, session_id)
    # Strip the heavy full text from the index response — fetch per-agent below.
    return {
        "session_id": result["session_id"],
        "directory": result["directory"],
        "agents": result["agents"],
        "reports": [
            {"agent_id": aid, "name": r["name"], "path": r["path"]}
            for aid, r in result["reports"].items()
        ],
    }


@router.get("/sessions/{session_id}/agent_reports")
def get_agent_reports(session_id: str, db: Session = Depends(get_db)):
    """Regenerate and return the full text of every agent's action document."""
    result = generate_agent_reports(db, session_id)
    return {
        "session_id": result["session_id"],
        "directory": result["directory"],
        "reports": [
            {"agent_id": aid, "name": r["name"], "text": r["text"]}
            for aid, r in result["reports"].items()
        ],
    }


# ── Decision-tree user model + ground truth ──────────────────────────────────

@router.post("/sessions/{session_id}/generate_ground_truth")
async def generate_ground_truth_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Create each agent's hidden ground-truth persona model (synthetic mode)."""
    return await generate_ground_truth(db, session_id)


@router.post("/sessions/{session_id}/train_user_model")
async def train_user_model_endpoint(session_id: str, db: Session = Depends(get_db)):
    """Retrain the per-agent decision-tree user models from the clickstream (with leaf labels)."""
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    tick = session.tick_count if session else 0
    return await train_session(db, session_id, tick, label_leaves=True)


def _agent_meta(db: Session, agent_id: str) -> dict:
    a = db.query(Agent).filter(Agent.id == agent_id).first()
    if not a:
        return {"id": agent_id, "name": agent_id[:8], "avatar_color": "#6366f1", "domain": ""}
    return {"id": a.id, "name": a.name, "avatar_color": a.avatar_color, "domain": a.domain}


@router.get("/sessions/{session_id}/user_model_tree")
def get_user_model_tree(session_id: str, db: Session = Depends(get_db)):
    """Per-agent learned decision-tree summary: leaves (subtopics) + inference accuracy."""
    snaps = (
        db.query(UserModelSnapshot)
        .filter(UserModelSnapshot.session_id == session_id)
        .all()
    )
    if not snaps:
        raise HTTPException(
            status_code=404,
            detail="No user-model trees yet. Run a synthetic session or POST /train_user_model.",
        )
    agents = []
    for s in snaps:
        meta = _agent_meta(db, s.agent_id)
        meta.update({
            "n_events": s.n_events,
            "accuracy": s.accuracy,
            "train_accuracy": s.train_accuracy,
            "updated_tick": s.updated_tick,
            "leaves": s.leaves or [],
            "tree": s.tree_json,
        })
        agents.append(meta)
    agents.sort(key=lambda a: a["name"])
    return {"agents": agents}


@router.get("/sessions/{session_id}/inference_accuracy")
def inference_accuracy(session_id: str, db: Session = Depends(get_db)):
    """Headline metric: how well each agent's decision tree recovers its ground-truth
    skip boundary (only meaningful in synthetic mode)."""
    snaps = (
        db.query(UserModelSnapshot)
        .filter(UserModelSnapshot.session_id == session_id)
        .all()
    )
    per_agent = []
    accs = []
    for s in snaps:
        meta = _agent_meta(db, s.agent_id)
        per_agent.append({
            "agent_id": s.agent_id, "agent_name": meta["name"], "avatar_color": meta["avatar_color"],
            "accuracy": s.accuracy, "train_accuracy": s.train_accuracy, "n_events": s.n_events,
        })
        if s.accuracy is not None:
            accs.append(s.accuracy)
    per_agent.sort(key=lambda a: (a["accuracy"] is None, -(a["accuracy"] or 0)))
    return {
        "per_agent": per_agent,
        "avg_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "agents_modeled": len(snaps),
        "has_ground_truth": db.query(GroundTruthPersona).filter(
            GroundTruthPersona.session_id == session_id).first() is not None,
    }


@router.get("/sessions/{session_id}/activity")
def activity_per_tick(session_id: str, db: Session = Depends(get_db)):
    """Event count per simulation tick."""
    rows = (
        db.query(Event.sim_tick, func.count(Event.id).label("count"))
        .filter(Event.session_id == session_id)
        .group_by(Event.sim_tick)
        .order_by(Event.sim_tick.asc())
        .all()
    )
    return [{"tick": r.sim_tick, "count": r.count} for r in rows]
