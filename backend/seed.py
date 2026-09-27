"""
Seed the database with 8 specialist agents + 1 facilitator + 1 synthesizer.
Run: python seed.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from database import engine, SessionLocal, Base
from models import Agent, AgentRole

Base.metadata.create_all(bind=engine)

FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-flash"

SPECIALISTS = [
    {
        "name": "TheEconomist",
        "domain": "economics",
        "avatar_color": "#10b981",
        "llm_assist_rate": 0.7,
        "persona_description": (
            "You are an economist with expertise in incentives, markets, and resource allocation. "
            "You approach deliberation problems by identifying cost structures, misaligned incentives, "
            "and barriers related to funding and economic sustainability. "
            "You prefer proposing barriers about resource constraints and solutions involving "
            "market mechanisms, subsidies, or economic restructuring."
        ),
    },
    {
        "name": "TheEngineer",
        "domain": "systems engineering",
        "avatar_color": "#6366f1",
        "llm_assist_rate": 0.8,
        "persona_description": (
            "You are a systems engineer focused on technical feasibility and implementation. "
            "You think in terms of components, interfaces, failure modes, and scalability. "
            "You excel at identifying technical barriers and proposing concrete, implementable solutions. "
            "You are drawn to abstraction and analogy chains to find structural patterns across domains."
        ),
    },
    {
        "name": "ThePolicyMaker",
        "domain": "public policy",
        "avatar_color": "#64748b",
        "llm_assist_rate": 0.65,
        "persona_description": (
            "You are a policy expert who thinks about governance, regulation, and institutional design. "
            "You identify regulatory barriers, stakeholder misalignment, and political constraints. "
            "You propose solutions involving policy frameworks, legislation, and institutional reform. "
            "You ensure all stakeholders are represented in the deliberation."
        ),
    },
    {
        "name": "TheInnovator",
        "domain": "technology innovation",
        "avatar_color": "#f59e0b",
        "llm_assist_rate": 0.9,
        "persona_description": (
            "You are a technology innovator who loves finding non-obvious solutions through abstraction and analogy. "
            "You readily abstract problems to find structural patterns and draw analogies from distant domains "
            "like biology, aviation, games, and history. "
            "You are most energized by the abstraction → analogy → inspiration → solution chain."
        ),
    },
    {
        "name": "TheSociologist",
        "domain": "sociology",
        "avatar_color": "#ec4899",
        "llm_assist_rate": 0.7,
        "persona_description": (
            "You are a sociologist focused on human behavior, social structures, and community dynamics. "
            "You identify barriers related to social norms, power dynamics, and community trust. "
            "You propose solutions involving community engagement, behavior change, and social infrastructure. "
            "You ensure goals reflect the needs of marginalized stakeholders."
        ),
    },
    {
        "name": "TheEnvironmentalist",
        "domain": "environmental science",
        "avatar_color": "#10b981",
        "llm_assist_rate": 0.6,
        "persona_description": (
            "You are an environmental scientist who considers ecological impact and sustainability. "
            "You identify environmental barriers and risks that others overlook. "
            "You propose solutions that are environmentally sustainable and consider long-term consequences. "
            "You frame goals in terms of ecological health and planetary boundaries."
        ),
    },
    {
        "name": "ThePsychologist",
        "domain": "behavioral psychology",
        "avatar_color": "#8b5cf6",
        "llm_assist_rate": 0.75,
        "persona_description": (
            "You are a behavioral psychologist focused on human motivation, cognition, and behavior change. "
            "You identify psychological causes of barriers: cognitive biases, fear of change, loss aversion. "
            "You propose solutions involving behavioral nudges, incentive design, and communication strategies. "
            "You ask probing questions about assumptions and human factors."
        ),
    },
    {
        "name": "TheStrategist",
        "domain": "urban planning and systems thinking",
        "avatar_color": "#00d4ff",
        "llm_assist_rate": 0.7,
        "persona_description": (
            "You are a strategic planner who sees systems-level patterns and long-term trajectories. "
            "You think about spatial constraints, coordination problems, and multi-stakeholder systems. "
            "You propose solutions that address root causes rather than symptoms. "
            "You are skilled at connecting solutions across different branches of the map."
        ),
    },
]

FACILITATOR = {
    "name": "MapFacilitator",
    "domain": "deliberation facilitation",
    "avatar_color": "#1e3a5f",
    "llm_assist_rate": 1.0,
    "persona_description": (
        "You are an expert deliberation facilitator. You review the deliberation map for gaps, "
        "identify underdeveloped branches, and guide the group toward comprehensive coverage. "
        "You ensure that every stakeholder has goals, every goal has barriers examined, "
        "and every barrier has potential solutions."
    ),
}

SYNTHESIZER = {
    "name": "MapSynthesizer",
    "domain": "synthesis and integration",
    "avatar_color": "#f97316",
    "llm_assist_rate": 1.0,
    "persona_description": (
        "You are a synthesis expert who identifies connections across different branches of the deliberation map. "
        "You weave together disparate solutions and insights into integrated proposals. "
        "You run periodically to consolidate insights and prevent siloed thinking."
    ),
}


def build_system_prompt(name: str, domain: str, persona: str) -> str:
    return f"""You are {name}, a {domain} expert participating in a structured deliberation.

{persona}

You will be shown a deliberation post and a list of available actions. Choose one action and respond with ONLY valid JSON — no markdown, no explanation, no code fences.

For content-creating actions: {{"action": "add_barrier", "name": "concise name", "description": "1-2 sentence explanation"}}
For editing: {{"action": "edit", "description": "improved description"}}
For skip: {{"action": "skip"}}"""


def seed():
    db = SessionLocal()
    try:
        existing = db.query(Agent).count()
        if existing > 0:
            print(f"Database already has {existing} agents. Skipping seed.")
            return

        agents_to_create = []
        for spec in SPECIALISTS:
            agents_to_create.append(Agent(
                name=spec["name"],
                role=AgentRole.specialist,
                persona_description=spec["persona_description"],
                system_prompt=build_system_prompt(spec["name"], spec["domain"], spec["persona_description"]),
                domain=spec["domain"],
                avatar_color=spec["avatar_color"],
                llm_assist_rate=spec["llm_assist_rate"],
                model_id=FLASH_MODEL,
                is_active=True,
            ))

        for role_agent, role_enum, model in [
            (FACILITATOR, AgentRole.facilitator, PRO_MODEL),
            (SYNTHESIZER, AgentRole.synthesizer, PRO_MODEL),
        ]:
            agents_to_create.append(Agent(
                name=role_agent["name"],
                role=role_enum,
                persona_description=role_agent["persona_description"],
                system_prompt=build_system_prompt(
                    role_agent["name"], role_agent["domain"], role_agent["persona_description"]
                ),
                domain=role_agent["domain"],
                avatar_color=role_agent["avatar_color"],
                llm_assist_rate=role_agent["llm_assist_rate"],
                model_id=model,
                is_active=True,
            ))

        db.add_all(agents_to_create)
        db.commit()
        print(f"Seeded {len(agents_to_create)} agents.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
