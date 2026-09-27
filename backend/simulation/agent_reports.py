"""
Per-agent action documents (Feature 2).

For every agent in a session, write a human-readable text file explaining each
action the agent took:
  - what was recommended (post + action) and which strategy produced it,
  - HOW that recommendation was created (strategy description + recommender reasoning),
  - what the agent actually did, and
  - for skipped recommendations, WHY the agent skipped (its stated reason).

Files are written to  backend/agent_reports/<session_id>/<AgentName>.txt
and the full text is also returned so the frontend can display it.
"""

import os
from collections import Counter
from datetime import datetime
from typing import Dict, List
from sqlalchemy.orm import Session

from models import Agent, RecommendationLog, DeliberationPost, SimulationSession


REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_reports")

# How each recommendation strategy chooses a post/action pair — explained in the doc.
STRATEGY_EXPLANATION = {
    "random": "Random baseline — a post/action pair was chosen uniformly at random.",
    "priority": "Priority gap-filling — pairs targeting under-developed branches (fewest existing children) were ranked highest.",
    "multi_agent": "Multi-agent LLM pipeline — a Profile Analyst summarised the agent's clickstream, then a Content Selector picked the pair best matching that profile, down-weighting things the agent previously skipped.",
    "forced": "Bootstrap rule — early in the session agents are required to add a stakeholder so the map has a root structure to grow from.",
    "ab_rotate": "A/B rotation — the agent was assigned one of the strategies above for this tick.",
}


def _fmt_outcome(log: RecommendationLog) -> str:
    if log.outcome == "engaged":
        return f"ENGAGED — performed '{log.actual_action}'" + (
            " (matched the recommendation)" if log.actual_action == log.recommended_action
            else f" (recommended was '{log.recommended_action}')"
        )
    if log.outcome == "skipped":
        return f"SKIPPED — chose '{log.actual_action or 'skip'}' instead of '{log.recommended_action}'"
    return "NO DECISION (tick interrupted)"


def _wrap(text: str, indent: str = "                ", width: int = 88) -> str:
    """Word-wrap a long value under a labelled field, keeping the hanging indent."""
    import textwrap
    if not text:
        return ""
    wrapped = textwrap.fill(text, width=width,
                            initial_indent="", subsequent_indent=indent)
    return wrapped


def _build_report_text(
    agent: Agent,
    logs: List[RecommendationLog],
    posts_by_id: Dict[str, DeliberationPost],
    mission_name: str,
) -> str:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append(f"AGENT ACTION REPORT — {agent.name}")
    lines.append("=" * 92)
    lines.append(f"Domain   : {agent.domain}")
    lines.append(f"Role     : {agent.role.value if hasattr(agent.role, 'value') else agent.role}")
    lines.append(f"Mission  : {mission_name}")
    lines.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append("PERSONA (the model of this agent that drives its behavior):")
    lines.append("  " + _wrap(agent.persona_description, indent="  "))
    lines.append("")

    engaged = [l for l in logs if l.outcome == "engaged"]
    skipped = [l for l in logs if l.outcome == "skipped"]
    prelude_n = sum(1 for l in logs if getattr(l, "is_prelude", False))

    # Contribution breakdown by created post type.
    created_types = Counter(
        posts_by_id[l.created_post_id].ptype.value
        for l in engaged
        if l.created_post_id and posts_by_id.get(l.created_post_id)
    )
    type_str = ", ".join(f"{n} {t}" for t, n in created_types.most_common()) or "none"

    # Which recommended actions this agent most often rejected.
    skipped_actions = Counter(l.recommended_action for l in skipped if l.recommended_action)
    skip_str = ", ".join(f"{a} ×{n}" for a, n in skipped_actions.most_common(5)) or "none"

    lines.append("SUMMARY")
    lines.append(f"  Recommendations : {len(logs)}  ({prelude_n} during prelude warm-start, "
                 f"{len(logs) - prelude_n} during main run)")
    lines.append(f"  Engaged         : {len(engaged)}")
    lines.append(f"  Skipped         : {len(skipped)}")
    acceptance = f"{len(engaged) / len(logs) * 100:.0f}%" if logs else "—"
    lines.append(f"  Acceptance rate : {acceptance}")
    lines.append(f"  Contributions   : {type_str}")
    lines.append(f"  Most-skipped    : {skip_str}")
    lines.append("")
    lines.append("HOW TO READ THIS LOG")
    lines.append("  Each entry shows what the recommender suggested, HOW that suggestion was")
    lines.append("  produced (the strategy), what the agent decided, the agent's own stated")
    lines.append("  reasoning, and — for skips — exactly why the agent declined.")
    lines.append("")
    lines.append("-" * 92)
    lines.append("ACTION-BY-ACTION LOG")
    lines.append("-" * 92)

    if not logs:
        lines.append("(no recommendations recorded for this agent)")

    for idx, log in enumerate(logs, 1):
        target = posts_by_id.get(log.recommended_post_id)
        target_str = (
            f'({target.ptype.value}) "{target.name}"' if target else "(unknown post)"
        )
        phase = "PRELUDE" if getattr(log, "is_prelude", False) else "MAIN"
        lines.append("")
        lines.append(f"#{idx}  [tick {log.sim_tick}] · {phase} phase")
        lines.append(f"  Recommended : {log.recommended_action}")
        lines.append(f"  On post     : {target_str}")
        if target and target.description:
            lines.append("  Post detail : " + _wrap(target.description))
        lines.append(f"  Strategy    : {log.strategy}")
        lines.append("  How created : " + _wrap(
            STRATEGY_EXPLANATION.get(log.strategy, "custom strategy")))
        if getattr(log, "confidence", None) is not None:
            lines.append(f"  Confidence  : {log.confidence:.2f}  (recommender's estimate the agent would engage)")
        if log.reasoning:
            lines.append("  Recommender's reasoning : " + _wrap(log.reasoning))
        lines.append(f"  Outcome     : {_fmt_outcome(log)}")
        # The agent's own stated rationale lives in skip_reason for both engage & skip.
        if log.skip_reason and log.skip_reason != "empty_content":
            label = "Why skipped" if log.outcome == "skipped" else "Agent's rationale"
            lines.append(f"  {label} : " + _wrap(log.skip_reason))
        if log.created_post_id and posts_by_id.get(log.created_post_id):
            cp = posts_by_id[log.created_post_id]
            lines.append(f"  Created     : ({cp.ptype.value}) \"{cp.name}\"")
            if cp.description:
                lines.append("                " + _wrap(cp.description))

    lines.append("")
    lines.append("=" * 92)
    lines.append("END OF REPORT")
    return "\n".join(lines)


def generate_agent_reports(db: Session, session_id: str) -> dict:
    """
    Build a text report per agent for this session, write each to disk, and
    return {agent_id: {name, path, text}} plus a summary.
    """
    session = db.query(SimulationSession).filter(SimulationSession.id == session_id).first()
    mission = (
        db.query(DeliberationPost).filter(DeliberationPost.id == session.mission_id).first()
        if session and session.mission_id else None
    )
    mission_name = mission.name if mission else "(unknown mission)"

    posts = (
        db.query(DeliberationPost)
        .filter(DeliberationPost.session_id == session_id)
        .all()
    )
    posts_by_id = {p.id: p for p in posts}

    logs = (
        db.query(RecommendationLog)
        .filter(RecommendationLog.session_id == session_id)
        .order_by(RecommendationLog.sim_tick.asc(), RecommendationLog.created_at.asc())
        .all()
    )
    logs_by_agent: Dict[str, List[RecommendationLog]] = {}
    for log in logs:
        logs_by_agent.setdefault(log.agent_id, []).append(log)

    agents = db.query(Agent).filter(Agent.is_active == True).all()

    out_dir = os.path.join(REPORTS_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)

    reports: Dict[str, dict] = {}
    for agent in agents:
        agent_logs = logs_by_agent.get(agent.id, [])
        text = _build_report_text(agent, agent_logs, posts_by_id, mission_name)
        safe_name = "".join(c for c in agent.name if c.isalnum() or c in ("-", "_")) or agent.id[:8]
        path = os.path.join(out_dir, f"{safe_name}.txt")
        with open(path, "w") as f:
            f.write(text)
        reports[agent.id] = {"name": agent.name, "path": path, "text": text}

    return {
        "session_id": session_id,
        "directory": out_dir,
        "agents": len(reports),
        "reports": reports,
    }
