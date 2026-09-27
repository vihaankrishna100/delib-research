from sqlalchemy.orm import Session
from models import DeliberationPost
from simulation.roles.base import BaseAgent
from simulation.prompts.specialist import build_facilitator_prompt, build_synthesizer_prompt


class FacilitatorAgent(BaseAgent):
    async def analyze_tree(
        self,
        db: Session,
        session_id: str,
        tick: int,
        tree_state: dict,
    ) -> dict:
        """
        Reviews the deliberation tree, identifies gaps, returns focus hints.
        """
        tree_summary = _summarize_tree(tree_state)
        prompt = build_facilitator_prompt(tree_summary, tick)
        try:
            raw = await self.call_llm(prompt, max_tokens=400)
            result = self.parse_json(raw)
        except Exception:
            result = {
                "focus_hints": [],
                "priority_ptypes": [],
                "priority_post_ids": [],
                "summary": "Tree analysis unavailable.",
            }

        self.log_event(
            db, session_id, "facilitator_analysis", tick,
            metadata={"summary": result.get("summary", ""), "tick": tick},
        )
        return result


class SynthesizerAgent(BaseAgent):
    async def synthesize(
        self,
        db: Session,
        session_id: str,
        tick: int,
        tree_state: dict,
    ):
        """
        Creates a synthesis post connecting solutions across branches.
        Runs every N ticks (controlled by engine).
        """
        tree_summary = _summarize_tree(tree_state)
        prompt = build_synthesizer_prompt(tree_summary, tick)
        try:
            raw = await self.call_llm(prompt, max_tokens=400)
            result = self.parse_json(raw)
        except Exception:
            return None

        name = result.get("synthesis_name", "").strip()
        description = result.get("synthesis_description", "").strip()
        connected_ids = result.get("connected_post_ids", [])

        if not name or not description:
            return None

        # Find a solution to attach the synthesis to (first connected post)
        parent_id = connected_ids[0] if connected_ids else None

        from models import PostType
        synthesis = DeliberationPost(
            session_id=session_id,
            ptype=PostType.solution,
            name=f"[Synthesis] {name}",
            description=description,
            purpose_id=parent_id,
            author_agent_id=self.id,
            content_mode="llm_assisted",
            sim_tick=tick,
        )
        db.add(synthesis)
        db.flush()

        self.log_event(
            db, session_id, "synthesis_created", tick,
            target_type="post", target_id=synthesis.id,
            metadata={
                "synthesis_name": name,
                "connected_post_ids": connected_ids,
            },
        )
        db.commit()
        return synthesis


def _summarize_tree(tree_state: dict) -> str:
    """Builds a compact text summary of the tree for LLM prompts."""
    all_posts = tree_state.get("all_posts", {})
    if not all_posts:
        return "Empty tree."

    from collections import Counter
    ptype_counts = Counter(p["ptype"] for p in all_posts.values())
    count_str = ", ".join(f"{v} {k}s" for k, v in sorted(ptype_counts.items()))

    # List posts by ptype
    lines = [f"Tree contains: {count_str}", ""]
    for ptype in ["mission", "stakeholder", "goal", "barrier", "cause", "solution",
                  "abstraction", "analogy", "inspiration", "question"]:
        posts_of_type = [p for p in all_posts.values() if p["ptype"] == ptype]
        if posts_of_type:
            lines.append(f"{ptype.upper()}S:")
            for p in posts_of_type[:8]:
                achiever_count = len([x for x in all_posts.values() if x.get("purpose_id") == p["id"]])
                lines.append(f"  [{p['id'][:8]}] {p['name']} ({achiever_count} children)")

    return "\n".join(lines)
