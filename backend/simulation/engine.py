import asyncio
from sqlalchemy.orm import Session
from models import (
    Agent, AgentRole, DeliberationPost, SimulationSession, SessionStatus,
)
from simulation.recommender import RecommendationEngine
from simulation.roles.specialist import (
    SpecialistAgent, ALLOWED_ACTIONS, MAX_STAKEHOLDERS, MAX_MISSIONS, UNIVERSAL_ACTIONS, _BASE_ACTIONS
)
from simulation.roles.facilitator import FacilitatorAgent, SynthesizerAgent
from simulation.ground_truth import generate_ground_truth, load_ground_truth
from simulation.user_model_tree import train_session

AB_STRATEGIES = ["random", "priority", "multi_agent"]
SYNTHESIZER_INTERVAL = 5
# Retrain the per-agent decision-tree user models every N ticks (and at session end).
# The learning curve is steepest at the start, so retrain aggressively early and then
# back off once the model has stabilised — this is what accelerates the skip-rate drop.
TRAIN_INTERVAL = 5
TRAIN_INTERVAL_EARLY = 2
EARLY_PHASE_TICKS = 20

_rec_engine = RecommendationEngine()


def _apply_learning_profile(fast: bool) -> None:
    """Apply the session's learning-speed profile to every knob it touches."""
    import simulation.recommender as _rec
    from simulation.user_model_tree import set_learning_profile
    set_learning_profile(fast)
    _rec.FAST_LEARNING = fast


def _build_tree_state(db: Session, session_id: str) -> dict:
    posts = (
        db.query(DeliberationPost)
        .filter(
            DeliberationPost.session_id == session_id,
            DeliberationPost.is_removed == False,
        )
        .all()
    )
    all_posts = {}
    for p in posts:
        all_posts[p.id] = {
            "id": p.id,
            "ptype": p.ptype.value,
            "name": p.name,
            "description": p.description,
            "purpose_id": p.purpose_id,
            "author_agent_id": p.author_agent_id,
            "sim_tick": p.sim_tick,
        }
    root = next((p for p in all_posts.values() if p["ptype"] == "mission"), None)
    return {"root": root, "all_posts": all_posts}


def _get_valid_pairs(tree_state: dict) -> list:
    """
    Returns recommendable (post, action) pairs.
    Uses only type-specific actions as recommendations — universal actions
    (add_pro, add_con, edit, skip) are always available to the agent themselves
    and don't need to be recommended.
    """
    all_posts = tree_state["all_posts"]

    stakeholder_count = sum(1 for p in all_posts.values() if p["ptype"] == "stakeholder")
    at_stakeholder_limit = stakeholder_count >= MAX_STAKEHOLDERS
    mission_count = sum(1 for p in all_posts.values() if p["ptype"] == "mission")
    at_mission_limit = mission_count >= MAX_MISSIONS

    pairs = []
    for post in all_posts.values():
        type_specific = _BASE_ACTIONS.get(post["ptype"], [])
        for action in type_specific:
            if action == "add_stakeholder" and at_stakeholder_limit:
                continue
            if action == "create_topic" and at_mission_limit:
                continue
            pairs.append({"post": post, "action": action})
    return pairs


def _stakeholder_pairs_only(tree_state: dict) -> list:
    """Returns only add_stakeholder pairs on the mission node."""
    return [
        {"post": post, "action": "add_stakeholder"}
        for post in tree_state["all_posts"].values()
        if post["ptype"] == "mission"
    ]


async def run_tick(db: Session, session_id: str, tick: int, is_prelude: bool = False):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session or session.status != SessionStatus.running:
        return

    config = session.config or {}
    strategy_mode = config.get("recommendation_strategy", "ab_rotate")
    use_aideator = config.get("use_aideator_prompts", True)
    # Agent decision mode: "llm" (free LLM judgment) or "synthetic" (deterministic
    # ground-truth persona model — perfectly consistent skip behaviour).
    agent_mode = config.get("agent_mode", "llm")
    brief_answers = bool(config.get("brief_answers", True))
    fast_learning = bool(config.get("fast_learning", True))
    _apply_learning_profile(fast_learning)
    gts = load_ground_truth(db, session_id) if agent_mode == "synthetic" else {}

    tree_state = _build_tree_state(db, session_id)
    valid_pairs = _get_valid_pairs(tree_state)
    if not valid_pairs:
        return

    # Count existing stakeholders to know if bootstrapping is needed
    stakeholder_count = sum(
        1 for p in tree_state["all_posts"].values() if p["ptype"] == "stakeholder"
    )
    stakeholder_pairs = _stakeholder_pairs_only(tree_state)

    all_agents = db.query(Agent).filter(Agent.is_active == True).all()
    specialists = [a for a in all_agents if a.role == AgentRole.specialist]
    facilitators = [a for a in all_agents if a.role == AgentRole.facilitator]
    synthesizers = [a for a in all_agents if a.role == AgentRole.synthesizer]

    # Experiment knob: run with only the first N specialists (stable subset by name)
    # to save time/API calls. 0 = all specialists.
    sample_size = int(config.get("agent_sample_size", 0) or 0)
    if 0 < sample_size < len(specialists):
        specialists = sorted(specialists, key=lambda a: a.name)[:sample_size]

    # Facilitator analysis
    focus_hints = []
    for f in facilitators:
        agent = FacilitatorAgent(f)
        result = await agent.analyze_tree(db, session_id, tick, tree_state)
        focus_hints.extend(result.get("focus_hints", []))
        db.expire_all()

    async def get_recommendation(agent_record, agent_idx: int):
        # First 2 agents must add a stakeholder while there are fewer than 2 stakeholders
        if agent_idx < 2 and stakeholder_count < 2 and stakeholder_pairs:
            return agent_record, stakeholder_pairs[0], "forced", "", None

        if strategy_mode == "ab_rotate":
            strategy = AB_STRATEGIES[agent_idx % len(AB_STRATEGIES)]
        else:
            strategy = strategy_mode

        confidence = None
        if strategy == "random":
            pair = _rec_engine.recommend_random(valid_pairs)
            reasoning = ""
        elif strategy == "priority":
            pair = _rec_engine.recommend_priority(valid_pairs, tree_state["all_posts"])
            reasoning = ""
        else:
            pair, reasoning, confidence = await _rec_engine.recommend_multi_agent(
                valid_pairs=valid_pairs,
                agent_id=agent_record.id,
                agent_name=agent_record.name,
                agent_domain=getattr(agent_record, "domain", ""),
                db=db,
                session_id=session_id,
                tick=tick,
                all_posts=tree_state["all_posts"],
            )

        return agent_record, pair, strategy, reasoning, confidence

    rec_tasks = [get_recommendation(a, i) for i, a in enumerate(specialists)]
    rec_results = await asyncio.gather(*rec_tasks)

    async def agent_act(agent_record, pair, strategy, reasoning, confidence):
        agent = SpecialistAgent(agent_record)
        await agent.decide(
            db=db,
            session_id=session_id,
            tick=tick,
            recommended_pair=pair,
            strategy=strategy,
            reasoning=reasoning,
            focus_hints=focus_hints,
            all_posts=tree_state["all_posts"],
            use_aideator=use_aideator,
            is_prelude=is_prelude,
            confidence=confidence,
            agent_mode=agent_mode,
            gt=gts.get(agent_record.id),
            brief_answers=brief_answers,
        )

    act_tasks = [agent_act(a, p, s, r, c) for a, p, s, r, c in rec_results]
    await asyncio.gather(*act_tasks)

    # Synthesizer every N ticks
    if tick > 0 and tick % SYNTHESIZER_INTERVAL == 0:
        db.expire_all()
        tree_state = _build_tree_state(db, session_id)
        for s in synthesizers:
            synth_agent = SynthesizerAgent(s)
            await synth_agent.synthesize(db, session_id, tick, tree_state)

    # Retrain the per-agent decision-tree user models periodically (no LLM leaf-labeling
    # mid-run) so the recommender's predictions sharpen as the clickstream grows.
    _interval = (TRAIN_INTERVAL_EARLY if tick < EARLY_PHASE_TICKS else TRAIN_INTERVAL) if fast_learning else TRAIN_INTERVAL
    if tick > 0 and tick % _interval == 0 and not is_prelude:
        try:
            await train_session(db, session_id, tick, label_leaves=False)
        except Exception:
            pass

    session.tick_count = tick + 1
    db.commit()


async def run_simulation(db: Session, session_id: str, num_ticks: int = 20):
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    if not session:
        return

    session.status = SessionStatus.running
    from datetime import datetime
    session.started_at = datetime.utcnow()

    config = session.config or {}
    # Prelude: a short, low-weight warm-start deliberation that runs ONCE, before
    # the real simulation, so the persona table and recommender start from real
    # activity instead of a cold start. Only on a fresh session (tick_count == 0).
    prelude_ticks = int(config.get("prelude_ticks", 0) or 0)
    already_ran_prelude = bool((session.config or {}).get("_prelude_done", False))
    db.commit()

    # Synthetic mode: create each agent's hidden ground-truth persona model once, before
    # any ticks, so skip decisions are consistent from tick 0.
    if config.get("agent_mode", "llm") == "synthetic" and not load_ground_truth(db, session_id):
        await generate_ground_truth(db, session_id)

    current_tick = session.tick_count or 0

    if prelude_ticks > 0 and current_tick == 0 and not already_ran_prelude:
        for tick in range(0, prelude_ticks):
            db.expire(session)
            if session.status != SessionStatus.running:
                break
            await run_tick(db, session_id, tick, is_prelude=True)
        # Mark the prelude as done so resume/restart never re-runs it.
        db.expire(session)
        new_config = dict(session.config or {})
        new_config["_prelude_done"] = True
        session.config = new_config
        db.commit()
        current_tick = session.tick_count or 0

    for tick in range(current_tick, current_tick + num_ticks):
        db.expire(session)
        if session.status != SessionStatus.running:
            break
        await run_tick(db, session_id, tick, is_prelude=False)

    # Final retrain WITH human-readable LLM leaf labels for the User Models view.
    try:
        db.expire(session)
        await train_session(db, session_id, session.tick_count or 0, label_leaves=True)
    except Exception:
        pass

    db.expire(session)
    if session.status == SessionStatus.running:
        session.status = SessionStatus.stopped
        session.ended_at = datetime.utcnow()
        db.commit()
