from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from models import PostType, AgentRole, SessionStatus


# ── Post schemas ──────────────────────────────────────────────────────────────

class PostBase(BaseModel):
    ptype: PostType
    name: str
    description: str
    purpose_id: Optional[str] = None

class PostCreate(PostBase):
    session_id: str
    author_agent_id: Optional[str] = None

class PostOut(PostBase):
    id: str
    session_id: str
    author_agent_id: Optional[str] = None
    content_mode: str
    is_removed: bool
    created_at: datetime
    sim_tick: int

    model_config = {"from_attributes": True}


class PostWithAchievers(PostOut):
    achievers: List["PostWithAchievers"] = []

    model_config = {"from_attributes": True}

PostWithAchievers.model_rebuild()


# ── Agent schemas ─────────────────────────────────────────────────────────────

class AgentOut(BaseModel):
    id: str
    name: str
    role: AgentRole
    persona_description: str
    domain: str
    avatar_color: str
    llm_assist_rate: float
    model_id: str
    is_active: bool

    model_config = {"from_attributes": True}


# ── Session schemas ───────────────────────────────────────────────────────────

class SimulationConfig(BaseModel):
    recommendation_strategy: str = "ab_rotate"   # "ab_rotate" | "random" | "priority" | "multi_agent"
    num_ticks: int = 20
    agents_per_tick: int = 8
    use_aideator_prompts: bool = True
    # Low-weight warm-start deliberation that runs before the real simulation.
    prelude_ticks: int = 5
    # Run with only the first N specialists (by name) to save time in experiments.
    # 0 = use all specialists.
    agent_sample_size: int = 0
    # Agent decision mode: "llm" (free LLM judgment) or "synthetic" (deterministic
    # ground-truth persona model — consistent skip behaviour for training/eval).
    agent_mode: str = "llm"
    # Cap post descriptions at 2-3 sentences (realistic + faster). False = verbose mode.
    brief_answers: bool = True
    # Learning-speed profile: earlier tree activation, denser early retraining,
    # decaying exploration. False = pre-2026-09 baseline (for ablation).
    fast_learning: bool = True

class SessionCreate(BaseModel):
    config: SimulationConfig = Field(default_factory=SimulationConfig)
    mission_name: str
    mission_description: str

class SessionOut(BaseModel):
    id: str
    status: SessionStatus
    config: Any
    mission_id: Optional[str]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    tick_count: int

    model_config = {"from_attributes": True}


# ── Analytics schemas ─────────────────────────────────────────────────────────

class StrategyStats(BaseModel):
    strategy: str
    total_recommendations: int
    engaged: int
    skipped: int
    skip_rate: float
    accuracy: float   # engaged / total — how often recommendation matched actual action

class AgentProfileOut(BaseModel):
    agent_id: str
    agent_name: Optional[str] = None
    profile_text: str
    last_updated_tick: int
    tick_history: List[Any]
    updated_at: Optional[datetime]

    model_config = {"from_attributes": True}

class TreeGrowthPoint(BaseModel):
    tick: int
    ptype: str
    count: int


# ── Topic tree / user model schemas ──────────────────────────────────────────

class TopicNodeOut(BaseModel):
    id: str
    parent_id: Optional[str] = None
    label: str
    level: int
    is_leaf: bool
    member_post_ids: List[str] = []

    model_config = {"from_attributes": True}


class UserModelCell(BaseModel):
    agent_id: str
    topic_id: str
    score: float                # 0..1
    posts_authored: int
    engaged: int
    skipped: int


class UserModelMatrix(BaseModel):
    agents: List[dict]          # {id, name, domain, avatar_color, role}
    topics: List[TopicNodeOut]
    cells: List[UserModelCell]


# ── Prior / comparison schemas ───────────────────────────────────────────────

class PriorCell(BaseModel):
    agent_id: str
    topic_id: str
    score: float                # 0..1
    rationale: Optional[str] = None


class PriorMatrix(BaseModel):
    agents: List[dict]
    topics: List[TopicNodeOut]
    cells: List[PriorCell]


# ── Persona topic headers (Feature 1) ────────────────────────────────────────

class PersonaTopicHeaderOut(BaseModel):
    id: str
    agent_id: str
    label: str
    rank: int
    rationale: Optional[str] = None
    parent_id: Optional[str] = None
    confidence: float = 0.5
    action_scores: Optional[dict] = None     # {reply, react, refine, view} 0..1

    model_config = {"from_attributes": True}


class PersonaTopicsOut(BaseModel):
    agents: List[dict]                       # {id, name, domain, avatar_color, role}
    headers: List[PersonaTopicHeaderOut]


class PersonaUsageTopic(BaseModel):
    id: str
    parent_id: Optional[str] = None
    label: str
    rationale: Optional[str] = None
    rank: int
    confidence: float = 0.5                   # LLM/refined confidence the agent engages here
    action_scores: dict = {}                  # predicted {reply, react, refine, view} 0..1
    observed_actions: dict = {}               # observed counts {reply, react, refine, view, skip}
    posts: int                               # posts the agent authored that map to this topic
    share: float                             # posts / total authored, 0..1
    used: bool                               # posts > 0
    examples: List[str] = []                 # post names under this topic (tree leaves)
    subtopics: List["PersonaUsageTopic"] = []  # finer sub-headers


PersonaUsageTopic.model_rebuild()


class PersonaUsageAgent(BaseModel):
    id: str
    name: str
    domain: str
    avatar_color: str
    role: str
    total_posts: int
    entropy: float = 0.0                      # 0..1 variance/diversity of topic spread
    action_categories: List[str] = []         # ["reply","react","refine","view"]
    topics: List[PersonaUsageTopic]


class PersonaUsageOut(BaseModel):
    agents: List[PersonaUsageAgent]


class AgentResearchMetrics(BaseModel):
    agent_id: str
    agent_name: str
    avatar_color: str
    persona_fidelity: float            # cosine(prior_row, observed_row) in [-1, 1] (usually 0..1)
    skip_rate: float                   # skipped / (engaged + skipped)
    avg_post_quality: Optional[float]  # mean PostQualityScore over posts authored, or None
    posts_authored: int
    engaged: int
    skipped: int


class TopicSurprise(BaseModel):
    topic_id: str
    label: str
    mean_abs_drift: float              # mean |observed - prior| across agents


class ResearchMetricsOut(BaseModel):
    per_agent: List[AgentResearchMetrics]
    per_topic_surprise: List[TopicSurprise]
    avg_persona_fidelity: float
    avg_skip_rate: float
    avg_post_quality: Optional[float]
    quality_scored_posts: int
    total_posts: int
