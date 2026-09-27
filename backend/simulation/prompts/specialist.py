"""
Deliberation map prompt logic.
Ported from Lisp propose-achiever patterns + extended for new action types.
"""

PTYPE_COLORS = {
    "mission":     "#1e3a5f",
    "stakeholder": "#64748b",
    "goal":        "#10b981",
    "barrier":     "#ef4444",
    "cause":       "#f97316",
    "solution":    "#00d4ff",
    "abstraction": "#8b5cf6",
    "analogy":     "#6366f1",
    "inspiration": "#f59e0b",
    "question":    "#ec4899",
    "answer":      "#38bdf8",
    "pro":         "#22c55e",
    "con":         "#dc2626",
    "part":        "#14b8a6",
    "failure_mode": "#f43f5e",
}

# Human-readable descriptions for each action (shown in prompt)
ACTION_DESCRIPTIONS = {
    "add_stakeholder": "Add a stakeholder (group or role affected by the mission)",
    "add_goal":        "Add a goal (desired outcome for this stakeholder)",
    "add_barrier":     "Add a barrier (obstacle preventing success)",
    "add_cause":       "Add a cause (root factor behind this barrier)",
    "add_solution":    "Add a solution (concrete intervention addressing this)",
    "add_abstraction": "Add an abstraction (generalize the structural pattern, stripping domain details)",
    "add_analogy":     "Add an analogy (same structural problem in a completely different domain)",
    "add_inspiration": "Add an inspiration (how the analogous problem was solved in that domain)",
    "use_inspiration": "Use inspiration (translate the inspiration back to the original problem as a solution)",
    "add_improvement": "Add improvement (a refined or extended version of this solution)",
    "add_part":        "Add a part (a sub-component this solution needs to actually work)",
    "add_failure_mode": "Add a failure mode (a specific way this solution could fail in practice)",
    "add_question":    "Add a Socratic question (probing question that exposes assumptions or risks)",
    "add_answer":      "Add an answer (substantive response to this question)",
    "add_pro":         "Add a pro (concrete advantage or benefit of this post)",
    "add_con":         "Add a con (concrete risk, cost, or drawback of this post)",
    "create_topic":    "Create a new topic (start a new mission-level deliberation thread)",
    "edit":            "Edit this post (improve the description without creating a new post)",
    "skip":            "Skip (view but take no action)",
}

# Actions that create a new child post
_CONTENT_ACTIONS = {
    "add_stakeholder", "add_goal", "add_barrier", "add_cause", "add_solution",
    "add_abstraction", "add_analogy", "add_inspiration", "use_inspiration",
    "add_improvement", "add_question", "add_answer", "add_pro", "add_con",
}

# Actions that target ptype
ACTION_TO_PTYPE = {
    "add_stakeholder": "stakeholder",
    "add_goal":        "goal",
    "add_barrier":     "barrier",
    "add_cause":       "cause",
    "add_solution":    "solution",
    "add_abstraction": "abstraction",
    "add_analogy":     "analogy",
    "add_inspiration": "inspiration",
    "use_inspiration": "solution",
    "add_improvement": "solution",
    "add_part":        "part",
    "add_failure_mode": "failure_mode",
    "add_question":    "question",
    "add_answer":      "answer",
    "add_pro":         "pro",
    "add_con":         "con",
}


# ── Context builders ──────────────────────────────────────────────────────────

def describe_context(post: dict, all_posts: dict) -> str:
    """
    Walks from post up to root via purpose_id.
    Returns human-readable context chain. Mirrors Lisp describe-context.
    """
    chain = []
    current = post
    while current is not None:
        chain.append(current)
        pid = current.get("purpose_id")
        current = all_posts.get(pid) if pid else None
    chain.reverse()

    templates = {
        "mission":     "The mission is: {name}. {desc}",
        "stakeholder": "One stakeholder is: {name}. {desc}",
        "goal":        "A goal for this stakeholder is: {name}. {desc}",
        "barrier":     "A barrier is: {name}. {desc}",
        "cause":       "A root cause is: {name}. {desc}",
        "solution":    "A proposed solution is: {name}. {desc}",
        "abstraction": "The abstract pattern is: {name}. {desc}",
        "analogy":     "An analogous case: {name}. {desc}",
        "inspiration": "An inspiration: {name}. {desc}",
        "question":    "An open question: {name}. {desc}",
        "answer":      "An answer given: {name}. {desc}",
        "pro":         "A noted pro: {name}. {desc}",
        "con":         "A noted con: {name}. {desc}",
        "part":        "A required part: {name}. {desc}",
        "failure_mode": "A possible failure mode: {name}. {desc}",
    }

    parts = []
    for node in chain:
        tmpl = templates.get(node["ptype"], "{name}. {desc}")
        parts.append(tmpl.format(name=node["name"], desc=node.get("description", "")))
    return " ".join(parts)


def find_first(ptypes: list, post: dict, all_posts: dict) -> dict | None:
    """
    Walks context chain upward, returns first post with ptype in ptypes.
    Mirrors Lisp find-first.
    """
    current = post
    while current is not None:
        if current["ptype"] in ptypes:
            return current
        pid = current.get("purpose_id")
        current = all_posts.get(pid) if pid else None
    return None


# ── Content instructions per action ──────────────────────────────────────────

_PARAGRAPH_DIRECTIVES = (
    "Each of those points should appear in their own paragraph prefixed with the topic "
    "expressed as a question.",
    "You should include the following points, each in their own paragraph:",
)


def _strip_long_form_directives(text: str) -> str:
    """
    In brief mode, physically remove the AIdeator directives that demand multi-paragraph,
    question-prefixed prose. Merely telling the model to "ignore" them was not enough —
    ~30% of descriptions still came back starting "Why is this important? ...".
    """
    for d in _PARAGRAPH_DIRECTIVES:
        text = text.replace(d, "")
    return " ".join(text.split(" "))


def _content_instruction(
    action: str,
    post: dict,
    all_posts: dict,
    existing_by_ptype: dict | None = None,
    use_aideator: bool = False,
    brief_answers: bool = True,
) -> str:
    """Returns specific content instructions for the chosen action."""
    if use_aideator and existing_by_ptype is not None:
        from .aideator_templates import build_aideator_instruction
        ai = build_aideator_instruction(action, post, all_posts, existing_by_ptype)
        if ai is not None:
            return _strip_long_form_directives(ai) if brief_answers else ai

    ptype = post["ptype"]
    name = post["name"]
    desc = post.get("description", "")

    if action == "add_stakeholder":
        return (
            "Propose a STAKEHOLDER: a group or role with a key stake in the mission. "
            "Name it concisely (a role or group). "
            "Describe their relationship to the mission in 1-2 sentences."
        )

    if action == "add_goal":
        return (
            f"Propose a GOAL for the stakeholder '{name}': a desired state important for their success. "
            "Propose JUST ONE simple, specific goal. "
            "Name it as a desired state (e.g. 'Reduced commute times'). "
            "Describe what achieving it means in 1-2 sentences."
        )

    if action == "add_barrier":
        return (
            f"Propose a BARRIER that blocks success at '{name}': "
            "an obstacle, constraint, or challenge. "
            "Name it concisely as a problem. "
            "Describe how it blocks progress in 1-2 sentences."
        )

    if action == "add_cause":
        return (
            f"Propose a ROOT CAUSE of '{name}': an underlying factor that produces or sustains it. "
            "Name it as a specific contributing factor. "
            "Describe how it leads to the problem in 1-2 sentences."
        )

    if action == "add_solution":
        return (
            f"Propose a SOLUTION to '{name}': a concrete intervention or mechanism. "
            "Name it as an actionable approach. "
            "Describe how it resolves the problem in 1-2 sentences."
        )

    if action == "add_abstraction":
        return (
            f"Propose an ABSTRACTION of '{name}': generalize away domain-specific features "
            "to reveal the underlying structural pattern. "
            "Name it as a general pattern (e.g. 'coordination failure among competing stakeholders'). "
            "Describe the structural essence in 1-2 sentences."
        )

    if action == "add_analogy":
        return (
            f"Propose an ANALOGY for the abstract pattern '{name}' ({desc}): "
            "a problem from a COMPLETELY DIFFERENT domain (biology, aviation, games, history, etc.) "
            "that instantiates the same structural pattern. "
            "Name it as 'The [Domain] case: [label]'. "
            "Describe the analogous situation showing the structural parallel in 1-2 sentences."
        )

    if action == "add_inspiration":
        return (
            f"Propose an INSPIRATION for the analogy '{name}' ({desc}): "
            "how was this type of problem solved IN THAT OTHER DOMAIN? "
            "Name it as the specific mechanism used there. "
            "Describe how it resolved the analogous problem in 1-2 sentences."
        )

    if action == "use_inspiration":
        original = find_first(["goal", "barrier", "cause"], post, all_posts)
        original_ref = f"'{original['name']}'" if original else "the original problem"
        return (
            f"Translate the inspiration '{name}' back to the original domain context for {original_ref}. "
            "Propose a SOLUTION that applies the same structural idea. "
            "Name it as a concrete actionable intervention. "
            "Describe how it addresses the original problem in 1-2 sentences."
        )

    if action == "add_improvement":
        return (
            f"Propose an IMPROVEMENT to the solution '{name}': "
            "a refined or extended version addressing a weakness or adding capability. "
            "Name it as 'Improved: [brief label]'. "
            "Describe specifically what changes and why it's better in 1-2 sentences."
        )

    if action == "add_part":
        return (
            f"Propose a PART of the solution '{name}': a sub-component, capability, or piece "
            "of infrastructure that must exist for this solution to actually work. "
            "Name it as the component itself. "
            "Describe what it does and why the solution depends on it in 1-2 sentences."
        )

    if action == "add_failure_mode":
        return (
            f"Propose a FAILURE MODE of the solution '{name}': a specific, plausible way this "
            "solution could fail or backfire in practice — not a generic risk. "
            "Name it as the failure itself. "
            "Describe the mechanism by which it fails and the conditions that trigger it in 1-2 sentences."
        )

    if action == "add_question":
        return (
            f"Pose a SOCRATIC or PREGNANT question about '{name}'. "
            "A pregnant question exposes hidden assumptions, unexamined risks, or unstated prerequisites — "
            "when answered, it fundamentally advances understanding. "
            "Examples: 'Who bears the cost of X?', 'What would need to be true for this to fail?', "
            "'What does this assume about Y?' "
            "Name it as the question itself (ending with '?'). "
            "Describe why this question is critical in 1 sentence."
        )

    if action == "add_answer":
        return (
            f"Answer the question '{name}'. "
            "Provide a substantive, evidence-based answer that advances the deliberation — "
            "it may strengthen, weaken, or reframe the solution being questioned. "
            "Name it as 'Answer: [brief label]'. "
            "Give the answer in 1-2 sentences."
        )

    if action == "add_pro":
        return (
            f"Identify a concrete PRO (advantage or benefit) of '{name}'. "
            "Name it as a brief advantage statement. "
            "Describe the specific benefit in 1-2 sentences."
        )

    if action == "add_con":
        return (
            f"Identify a concrete CON (risk, cost, or drawback) of '{name}'. "
            "Name it as a brief concern statement. "
            "Describe the specific drawback in 1-2 sentences."
        )

    if action == "create_topic":
        return (
            "Create a NEW TOPIC — a mission-level deliberation thread related to or inspired by "
            f"what you've been reading about '{name}'. "
            "This is a completely new problem or opportunity worth deliberating. "
            "Name it as a concise mission statement (e.g. 'Reduce food waste in urban supply chains'). "
            "Describe the scope and importance of the new topic in 1-2 sentences."
        )

    if action == "edit":
        return (
            f"Improve the description of '{name}' based on your expertise. "
            "Keep the same name; only provide an improved description. "
            "The improved description should be clearer, more specific, or more accurate."
        )

    return f"Contribute to '{name}' based on your domain expertise."


# ── Main prompt builder ───────────────────────────────────────────────────────

def build_action_choice_prompt(
    agent_name: str,
    domain: str,
    post: dict,
    context_str: str,
    available_actions: list[str],
    recommended_action: str,
    existing_by_ptype: dict,
    all_posts: dict | None = None,
    use_aideator: bool = False,
    brief_answers: bool = True,
) -> str:
    """
    Agent sees a post and chooses which action to take from the available list.
    The recommended_action is shown as a suggestion (not forced).
    Agent responds with JSON including their chosen action and content.
    """

    # Build action menu
    action_lines = []
    for a in available_actions:
        marker = " ← SUGGESTED" if a == recommended_action else ""
        desc = ACTION_DESCRIPTIONS.get(a, a)
        action_lines.append(f'  "{a}" — {desc}{marker}')
    actions_str = "\n".join(action_lines)

    # Existing children summary (avoid duplicates)
    existing_parts = []
    for ptype, posts in existing_by_ptype.items():
        names = [p["name"] for p in posts[:5]]
        existing_parts.append(f"  {ptype}: {', '.join(names)}")
    existing_str = "\n".join(existing_parts) if existing_parts else "  (none yet)"

    brevity_block = (
        'BE BRIEF. The "description" must be 2-3 SENTENCES TOTAL — not per point, not per\n'
        "paragraph. Write it as one short paragraph, the way a real participant typing into a\n"
        "web form would. Ignore any guidance above that asks for multiple paragraphs or a\n"
        "paragraph per topic; keep the substance, drop the length.\n"
        "Do NOT open with a rhetorical question or a heading (no \"What is this?\", no\n"
        "\"Why is it important?\"). Do NOT use markdown or HTML. Start with the substance.\n\n"
    ) if brief_answers else ""

    # Content instruction for the suggested action (shown as guidance)
    suggestion_instruction = _content_instruction(
        recommended_action,
        post,
        all_posts if all_posts is not None else {},
        existing_by_ptype=existing_by_ptype,
        use_aideator=use_aideator,
    )

    return f"""You are {agent_name}, a {domain} expert in a structured deliberation.

CONTEXT:
{context_str}

CURRENT POST [{post['ptype'].upper()}]: "{post['name']}"
{post.get('description', '')}

EXISTING CHILDREN:
{existing_str}

AVAILABLE ACTIONS:
{actions_str}

SUGGESTED ACTION GUIDANCE (for "{recommended_action}"):
{suggestion_instruction}

Choose the action that best fits your expertise and the deliberation needs.
You may choose the suggested action or any other available action.

{brevity_block}If choosing a content-creating action (not skip/view/edit), provide name + description.
For "edit", provide only description (improved version of the current description).
For "skip" or "view", no content needed — but you MUST give a brief "reason".

Always include a short "reason" field (one sentence) explaining WHY you chose this
action. If you skip, the reason must explain what made this recommendation a poor
fit for you right now (off-topic, already covered, outside your expertise, etc.).

Return ONLY valid JSON — no markdown, no explanation:
For content actions: {{"action": "add_barrier", "name": "...", "description": "...", "reason": "why this action"}}
For edit:           {{"action": "edit", "description": "improved description...", "reason": "why edit"}}
For skip/view:      {{"action": "skip", "reason": "why you skipped this recommendation"}}"""


# ── Facilitator / Synthesizer prompts ────────────────────────────────────────

def build_facilitator_prompt(tree_summary: str, tick: int) -> str:
    return f"""You are the MapFacilitator, an expert in structured deliberation.

Review the current deliberation map (Tick {tick}):

{tree_summary}

Identify:
1. Branches that are underdeveloped (goals with no barriers, barriers with no solutions, etc.)
2. Post types that are underrepresented
3. The 2-3 most important gaps to fill next

Return ONLY valid JSON:
{{
  "focus_hints": ["hint1", "hint2"],
  "priority_ptypes": ["barrier", "solution"],
  "priority_post_ids": ["id1", "id2"],
  "summary": "one sentence describing the tree state"
}}"""


def build_synthesizer_prompt(tree_summary: str, tick: int) -> str:
    return f"""You are the MapSynthesizer, skilled at connecting ideas across a deliberation map.

Current deliberation map state (Tick {tick}):
{tree_summary}

Identify 2-3 solutions or themes that could be synthesized into a broader solution.

Return ONLY valid JSON:
{{
  "synthesis_name": "concise name",
  "synthesis_description": "2-3 sentences explaining how these ideas connect",
  "connected_post_ids": ["id1", "id2", "id3"]
}}"""
