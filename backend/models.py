import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, Enum as SAEnum, UniqueConstraint
)
from sqlalchemy.orm import relationship
from database import Base
import enum


def gen_uuid():
    return str(uuid.uuid4())


class AgentRole(str, enum.Enum):
    specialist = "specialist"
    facilitator = "facilitator"
    synthesizer = "synthesizer"


class SessionStatus(str, enum.Enum):
    idle = "idle"
    running = "running"
    paused = "paused"
    stopped = "stopped"


class PostType(str, enum.Enum):
    mission = "mission"
    stakeholder = "stakeholder"
    goal = "goal"
    barrier = "barrier"
    cause = "cause"
    solution = "solution"
    abstraction = "abstraction"
    analogy = "analogy"
    inspiration = "inspiration"
    question = "question"
    answer = "answer"
    pro = "pro"
    con = "con"
    # Anti-saturation moves: instead of only proposing sibling solutions (which
    # exhausts quickly), agents can decompose a solution into its parts or probe
    # how it could fail — both open fresh, non-duplicative branches.
    part = "part"
    failure_mode = "failure_mode"


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    role = Column(SAEnum(AgentRole), nullable=False)
    persona_description = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=False)
    domain = Column(String, default="")
    avatar_color = Column(String, default="#6366f1")
    llm_assist_rate = Column(Float, default=0.7)
    model_id = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    events = relationship("Event", back_populates="agent")


class DeliberationPost(Base):
    __tablename__ = "deliberation_posts"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    ptype = Column(SAEnum(PostType), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    purpose_id = Column(String, ForeignKey("deliberation_posts.id"), nullable=True)
    author_agent_id = Column(String, ForeignKey("agents.id"), nullable=True)
    content_mode = Column(String, default="llm_assisted")  # "llm_assisted" | "self_written"
    is_removed = Column(Boolean, default=False)
    # Posts created during the low-weight prelude deliberation (warm-start phase).
    is_prelude = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sim_tick = Column(Integer, default=0)

    purpose = relationship("DeliberationPost", remote_side="DeliberationPost.id", backref="achievers")
    author = relationship("Agent")


class SimulationSession(Base):
    __tablename__ = "simulation_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    status = Column(SAEnum(SessionStatus), default=SessionStatus.idle)
    config = Column(JSON, default={})
    mission_id = Column(String, ForeignKey("deliberation_posts.id"), nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    tick_count = Column(Integer, default=0)

    events = relationship("Event", back_populates="session")


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, default=gen_uuid)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=True)
    agent_role = Column(String, nullable=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=True)
    event_type = Column(String, nullable=False)
    target_type = Column(String, nullable=True)
    target_id = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, default={})
    is_prelude = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sim_tick = Column(Integer, default=0)

    agent = relationship("Agent", back_populates="events")
    session = relationship("SimulationSession", back_populates="events")


class AgentProfile(Base):
    __tablename__ = "agent_profiles"

    id = Column(String, primary_key=True, default=gen_uuid)
    agent_id = Column(String, ForeignKey("agents.id"), unique=True, nullable=False)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=True)
    profile_text = Column(Text, default="")
    tick_history = Column(JSON, default=[])
    last_updated_tick = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class AgentTopicPrior(Base):
    """
    Persona-derived expected interest score per (agent, topic) — the "base user model".
    Computed once per session via an LLM call that sees only the agent's persona and
    the topic labels. Compared against the observed activity-based score to measure
    whether the simulation matched the persona, and whether the recommender learned
    each agent's true profile.
    """
    __tablename__ = "agent_topic_priors"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    topic_id = Column(String, ForeignKey("topic_nodes.id"), nullable=False)
    score = Column(Float, nullable=False)              # 0..1
    rationale = Column(Text, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("session_id", "agent_id", "topic_id"),)


class PostQualityScore(Base):
    """
    LLM-judge rating per deliberation post. Used to measure whether the recommender
    increased contribution quality over the course of the simulation.
    """
    __tablename__ = "post_quality_scores"

    id = Column(String, primary_key=True, default=gen_uuid)
    post_id = Column(String, ForeignKey("deliberation_posts.id"), unique=True, nullable=False)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    score = Column(Float, nullable=False)              # 0..1
    rationale = Column(Text, nullable=True)
    judged_at = Column(DateTime, default=datetime.utcnow)


class TopicNode(Base):
    __tablename__ = "topic_nodes"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    parent_id = Column(String, ForeignKey("topic_nodes.id"), nullable=True)
    label = Column(String, nullable=False)
    level = Column(Integer, default=0)
    is_leaf = Column(Boolean, default=False)
    member_post_ids = Column(JSON, default=[])
    centroid = Column(JSON, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    sim_tick = Column(Integer, nullable=False)
    strategy = Column(String, nullable=False)
    recommended_post_id = Column(String, nullable=True)
    recommended_action = Column(String, nullable=True)
    reasoning = Column(Text, nullable=True)
    outcome = Column(String, nullable=True)    # "engaged" | "skipped"
    actual_action = Column(String, nullable=True)  # action taken; "skip" if skipped
    created_post_id = Column(String, nullable=True)
    # Free-text reason the agent gave for its decision — especially WHY it skipped.
    # Fed back into the recommender as an LLM-based weighting rule.
    skip_reason = Column(Text, nullable=True)
    # True when this recommendation happened during the prelude warm-start phase.
    is_prelude = Column(Boolean, default=False)
    # Recommender's self-estimated confidence (0..1) that the agent will engage.
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PersonaTopicHeader(Base):
    """
    LLM-generated topic header derived from an agent's persona + the session mission.

    Produced by asking the model, in the agent's own voice:
      "I am a <persona>. I will be engaging in deliberations about <mission>.
       Give me ~10 topics I will probably be interested in contributing to."

    These seed the persona/user-model table with meaningful columns BEFORE any
    posts exist, removing the cold-start problem for the recommender.
    """
    __tablename__ = "persona_topic_headers"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    label = Column(String, nullable=False)
    rank = Column(Integer, default=0)          # ordering within the agent's list (0 = top)
    rationale = Column(Text, nullable=True)
    # LLM-estimated confidence (0..1) the agent will actually contribute here — a high
    # value flags a low-skip-risk topic. Refined later from observed skips/engagements.
    confidence = Column(Float, default=0.5)
    # Hierarchy: a top-level header has parent_id=None; deeper sub-headers point at
    # their parent header. Generation produces 2-3 sub-headers per top-level header
    # so the persona table captures interests at a finer grain by default.
    parent_id = Column(String, ForeignKey("persona_topic_headers.id"), nullable=True)
    # Predicted per-action scores (0..1) for the recommender actions the agent would
    # take under this header: {"reply": .., "react": .., "refine": .., "view": ..}.
    # Parent headers inherit the mean of their sub-headers' scores.
    action_scores = Column(JSON, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Decision-tree user model (replaces the persona table as the predictor) ────

class PostEmbedding(Base):
    """Cached Gemini embedding of a post (name + description). One row per post so
    the clickstream and the decision-tree trainer can look vectors up without
    re-embedding. Stable text → stable vector."""
    __tablename__ = "post_embeddings"

    post_id = Column(String, ForeignKey("deliberation_posts.id"), primary_key=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    vector = Column(JSON, nullable=False)          # list[float]
    created_at = Column(DateTime, default=datetime.utcnow)


class ClickstreamEvent(Base):
    """Every recommendation shown to an agent, with the resulting label. This is the
    training set for the per-agent decision tree. ALL events are kept so that a
    retrain re-derives the whole hierarchy (automatic event redistribution)."""
    __tablename__ = "clickstream_events"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    post_id = Column(String, ForeignKey("deliberation_posts.id"), nullable=False)
    label = Column(Integer, nullable=False)        # 1 = engaged, 0 = skipped
    sim_tick = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class GroundTruthPersona(Base):
    """Hidden, deterministic preference model for a test agent (synthetic mode).

    The agent engages a post iff its embedding is closer to a LIKED topic centroid
    than to any DISLIKED one. This gives perfectly consistent skip behaviour and a
    ground truth to score the inferred decision tree against."""
    __tablename__ = "ground_truth_personas"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    liked_topics = Column(JSON, default=[])            # list[str]
    disliked_topics = Column(JSON, default=[])         # list[str]
    liked_centroids = Column(JSON, default=[])         # list[list[float]]
    disliked_centroids = Column(JSON, default=[])      # list[list[float]]
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("session_id", "agent_id"),)


class UserModelSnapshot(Base):
    """Latest learned decision-tree summary per (session, agent) for the UI and for
    the inference-accuracy metric. The live sklearn model itself lives in memory in
    `user_model_tree`; this table holds the human-readable, persisted view."""
    __tablename__ = "user_model_snapshots"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("simulation_sessions.id"), nullable=False)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    updated_tick = Column(Integer, default=0)
    n_events = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)            # inference vs ground truth (synthetic)
    train_accuracy = Column(Float, nullable=True)      # fit on its own training data
    leaves = Column(JSON, default=[])                  # [{label,n,engage_prob,entropy,examples}]
    # Full tree structure for the UI: nested {kind, n, engage_prob, entropy, feature,
    # threshold, label, children[]} so the learned splits can actually be drawn.
    tree_json = Column(JSON, default=None)
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("session_id", "agent_id"),)
