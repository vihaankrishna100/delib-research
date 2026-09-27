"""
AIdeator-style content instructions for deliberation-map-sim agents.

Ported from /Users/vihaankrishna/aideator-python-minimal/aideator/prompts.py
and adapted for dict-based posts. Returns the per-action body that gets
injected at the SUGGESTED ACTION GUIDANCE slot in build_action_choice_prompt;
the outer JSON contract ({action, name, description}) is owned by the caller,
so AIdeator's own _json_template is intentionally not ported.
"""

from .specialist import find_first


def _list_existing(existing: list[dict]) -> str:
    if not existing:
        return ""
    lines = [f"\n- {e['name']} {e.get('description', '')}" for e in existing]
    return "".join(lines)


def _aideator_stakeholder(post: dict, existing: list[dict]) -> str:
    s = (
        f"\n\nA stakeholder is a class of entities whose needs should be considered "
        f"during the deliberation on {post['name']}. "
        f"You should discuss WHY the stakeholder is important for coming up with a successful "
        f"solution for this deliberation, for example whether its support is critical for the "
        f"solution to succeed. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nHere are the stakeholders that have been proposed so far:"
        s += _list_existing(existing)
    s += "\n\nPropose a stakeholder that has NOT been already proposed above."
    return s


def _aideator_goal(post: dict, existing: list[dict]) -> str:
    s = (
        f"\n\nA goal represents a state (e.g. clean air) that is important for the success "
        f'of the {post["ptype"]} "{post["name"]}". '
        f"You should describe the goal as well as why it is important. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe following goals have already been proposed:"
        s += _list_existing(existing)
    s += "\n\nMake sure your proposed goal includes JUST ONE SIMPLE GOAL."
    s += '\nFor example, DO NOT say "Secure national energy sovereignty AND affordability". These are two different goals!'
    s += '\nPick just one goal e.g. "Secure national energy sovereignty".'
    s += '\nDO NOT say "Insure stable, affordable energy supply". Again, these are two different goals!'
    s += '\nPick just one goal e.g. "Secure stable energy supply".'
    return s


def _aideator_barrier(post: dict, existing: list[dict]) -> str:
    s = (
        f"\n\nA barrier should represent something that undercuts our ability to succeed "
        f'at the {post["ptype"]} "{post["name"]}". '
        f"You should describe the barrier as well as why it is an important impediment "
        f'to the {post["ptype"]}. '
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe barrier you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_cause(post: dict, existing: list[dict]) -> str:
    s = (
        f"\n\nA cause should describe something that can lead to the "
        f'{post["ptype"]} {post["name"]}. '
        f'Describe the cause, how it leads to the {post["ptype"]}, '
        f"and WHY it can have a large impact. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe cause you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_solution(post: dict, all_posts: dict, existing: list[dict]) -> str:
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f'\n\nA solution should address the {origin_type} "{origin_name}". '
        f"You should include the following points, each in their own paragraph:"
        f"\n\nWhat is the solution? <2 to 3 sentence description of the solution>"
        f"\n\nWhy is it a good solution? <2 to 3 sentence description of the advantages of this solution>"
    )
    s += "\n\nJust give me the solution, in a succinct way, without re-iterating the problem it solves."
    if existing:
        s += "\n\nThe solution you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_abstraction(post: dict, all_posts: dict, existing: list[dict]) -> str:
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f"\n\nAn abstraction is the first step of using analogy to solve a problem. "
        f'The abstraction should represent a generalization of the {origin_type} "{origin_name}". '
        f"Try to generalize the {origin_type} a LOT so I am more likely to get out of the box "
        f"ideas from the analogy process. "
        f"When you generalize it, maintain the deep structure but abstract away domain-specific "
        f"features e.g. by replacing nouns and verbs with more general ones (hypernyms) that subsume them."
    )
    if existing:
        s += "\n\nThe abstraction you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_analogy(post: dict, all_posts: dict, existing: list[dict]) -> str:
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f"\n\nAn analogy should describe a problem that comes from a different domain "
        f'than the {origin_type} "{origin_name}", '
        f'but that instantiates the same structural pattern as the abstraction "{post["name"]}".'
    )
    if existing:
        s += "\n\nThe analogy you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_inspiration(post: dict, existing: list[dict]) -> str:
    s = (
        f"\n\nYour inspiring idea should represent a possible solution "
        f'for the {post["ptype"]} "{post["name"]}".'
    )
    if existing:
        s += "\n\nThe idea you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_use_inspiration(post: dict, all_posts: dict, existing: list[dict]) -> str:
    # Walks up the chain inspiration → analogy → abstraction → goal/barrier/cause
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f'\n\nTranslate the inspiration "{post["name"]}" back to the original domain. '
        f'A solution should address the {origin_type} "{origin_name}" '
        f"by applying the same structural idea as the inspiration. "
        f"You should include the following points, each in their own paragraph:"
        f"\n\nWhat is the solution? <2 to 3 sentence description of the solution>"
        f"\n\nWhy is it a good solution? <2 to 3 sentence description of the advantages of this solution>"
    )
    s += "\n\nJust give me the solution, in a succinct way, without re-iterating the problem it solves."
    if existing:
        s += "\n\nThe solution you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_improvement(post: dict, all_posts: dict, existing: list[dict]) -> str:
    """Ported verbatim from the AIdeator Lisp `improvement` template."""
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f"\n\nAn improvement should describe how the "
        f'{post["ptype"]} "{post["name"]}" can be made better '
        f"in terms of addressing the {origin_type} {origin_name}. "
        f'You should discuss HOW it improves upon the {post["ptype"]}, '
        f"and WHY it is a good idea to implement it."
    )
    if existing:
        s += "\n\nThe improvement you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_part(post: dict, all_posts: dict, existing: list[dict]) -> str:
    """
    Anti-saturation move: decompose a solution instead of inventing another sibling.

    Written in the AIdeator house style — origin lookup, HOW/WHY framing, and an
    explicit "be different from these" clause against already-proposed parts.
    """
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f"\n\nA part should describe a sub-component, capability, or piece of infrastructure "
        f'that the {post["ptype"]} "{post["name"]}" REQUIRES in order to actually work '
        f"in addressing the {origin_type} {origin_name}. "
        f'You should discuss WHAT the part is, HOW the {post["ptype"]} depends on it, '
        f"and WHY it is non-trivial to provide. "
        f"Prefer parts that are concrete enough to be built or procured, not restatements "
        f'of the {post["ptype"]} itself.'
    )
    if existing:
        s += "\n\nThe part you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_failure_mode(post: dict, all_posts: dict, existing: list[dict]) -> str:
    """
    Anti-saturation move: stress-test a solution so new branches open from its weaknesses.
    """
    origin = find_first(["goal", "barrier", "cause"], post, all_posts)
    origin_type = origin["ptype"] if origin else post["ptype"]
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f"\n\nA failure mode should describe a specific, plausible way that the "
        f'{post["ptype"]} "{post["name"]}" could FAIL or backfire when applied to '
        f"the {origin_type} {origin_name}. "
        f"You should discuss the MECHANISM by which it fails, the CONDITIONS that trigger "
        f"the failure, and WHY the failure would be consequential. "
        f"Give a concrete failure with a causal story — not a generic risk such as "
        f'"it might be expensive" or "it could face resistance".'
    )
    if existing:
        s += "\n\nThe failure mode you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_question(post: dict, existing: list[dict]) -> str:
    s = (
        f'\n\nYour question should be one whose reply will help make the solution "{post["name"]}" '
        f"more complete. "
        f"A good question could for example identify a possible failure mode with the current "
        f"solution and ask how to avoid that failure mode. "
        f"It could ask what sub-components are needed to make the solution into a reality. "
        f"Make sure that the question's name has the structure: how can we X?"
    )
    if existing:
        s += "\n\nThe question you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


def _aideator_answer(post: dict, all_posts: dict, existing: list[dict]) -> str:
    origin = find_first(["answer", "solution"], post, all_posts)
    origin_name = origin["name"] if origin else post["name"]
    s = (
        f'\n\nThe answer should help make the solution "{origin_name}" more robust and complete. '
        f"You should describe the answer as well as why it is a good one. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe answer you propose should be as different as possible from these others:"
        s += _list_existing(existing)
    return s


# Action → ptype that the action's existing-siblings are filtered on
_ACTION_TARGET_PTYPE = {
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
}


def build_aideator_instruction(
    action: str,
    post: dict,
    all_posts: dict,
    existing_by_ptype: dict[str, list[dict]],
) -> str | None:
    """AIdeator-style content instruction for the given action,
    or None if no AIdeator template covers it (caller falls back
    to the terse default in _content_instruction)."""

    target_ptype = _ACTION_TARGET_PTYPE.get(action)
    if target_ptype is None:
        return None
    existing = existing_by_ptype.get(target_ptype, [])

    if action == "add_stakeholder":
        return _aideator_stakeholder(post, existing)
    if action == "add_goal":
        return _aideator_goal(post, existing)
    if action == "add_barrier":
        return _aideator_barrier(post, existing)
    if action == "add_cause":
        return _aideator_cause(post, existing)
    if action == "add_solution":
        return _aideator_solution(post, all_posts, existing)
    if action == "add_abstraction":
        return _aideator_abstraction(post, all_posts, existing)
    if action == "add_analogy":
        return _aideator_analogy(post, all_posts, existing)
    if action == "add_inspiration":
        return _aideator_inspiration(post, existing)
    if action == "use_inspiration":
        return _aideator_use_inspiration(post, all_posts, existing)
    if action == "add_improvement":
        return _aideator_improvement(post, all_posts, existing)
    if action == "add_part":
        return _aideator_part(post, all_posts, existing)
    if action == "add_failure_mode":
        return _aideator_failure_mode(post, all_posts, existing)
    if action == "add_question":
        return _aideator_question(post, existing)
    if action == "add_answer":
        return _aideator_answer(post, all_posts, existing)

    return None
