"""Independent tuple-state replay and closed-world closure for timegate_exam.

Independent source code in the same public repository is not independent
custody, private evaluation, formal proof of omitted actions, or model training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .timegate_exam import Case, MAX_ACTIONS


LEGAL = frozenset({"forward", "back", "take", "arm", "wait", "escape"})


@dataclass(frozen=True)
class CertificateReview:
    valid: bool
    reason: str
    states_checked: int = 0


def _initial(case: Case) -> tuple[int, int, int, int, int]:
    # room, phase, carry cell, cell stock, charge
    return 0, case.initial_phase, 0, 1, 0


def _apply(case: Case, token: tuple[int, int, int, int, int],
           action: Any) -> tuple[int, int, int, int, int]:
    pos, clock, holding, stock, charge = token
    if pos == 4:
        return token
    if type(action) is str and action in LEGAL:
        if action == "forward" and pos < 3:
            pos += 1
        if action == "back" and 0 < pos < 4:
            pos -= 1
        if action == "take" and pos == 1 and stock and not holding:
            holding, stock = 1, 0
        if (action == "arm" and pos == 2 and holding and
                clock == case.charge_phase and not case.sealed):
            holding, charge = 0, case.lifetime + 1
        if (action == "escape" and pos == 3 and charge >= 1
                and clock == case.pass_phase):
            pos = 4
    return pos, (clock + 1) % case.period, holding, stock, max(charge - 1, 0)


def replay_snapshot(case: Case, attempts: object) -> tuple[int, int, int, int, int] | None:
    """Independently reconstruct the entire current world state after attempts.

    Returns None for malformed, over-budget or post-victory transcripts. A
    candidate's local environment and this replay must agree after *every*
    action, not only when the local transition claims it reached the exit.
    """
    # User-created list/tuple subclasses may lie about their length, iterate
    # inconsistently, or expose more actions than the charged transcript.
    if type(attempts) not in (list, tuple) or len(attempts) > MAX_ACTIONS:
        return None
    state = _initial(case)
    for action in attempts:
        if state[0] == 4:
            return None  # cannot append post-victory actions
        state = _apply(case, state, action)
    return state


def verify_attempts(case: Case, attempts: object) -> str:
    if type(attempts) not in (list, tuple):
        return "UNKNOWN"
    state = replay_snapshot(case, attempts)
    if state is None:
        return "VERIFIED_FAILURE"
    return "VERIFIED_WIN" if state[0] == 4 else "VERIFIED_FAILURE"


def replay_observation(case: Case, token: tuple[int, int, int, int, int]) -> str:
    """Independent expected public observation from the independently replayed state.

    Fail-closed parity with the environment intentionally includes explanatory
    text: misleading narrative can manipulate an agent despite correct state.
    Any legitimate wording revision must update both evaluator implementations.
    """
    room_index, clock, holding, stock, charge = token
    rooms = ("entry", "depot", "tower", "gate", "outside")
    room = rooms[room_index]
    if room_index == 0:
        description = "A corridor leads forward to a depot."
    elif room_index == 1:
        description = "A one-use power cell lies here." if stock else "The cell shelf is empty."
    elif room_index == 2:
        description = "A relay can consume a held cell at its charging phase."
    elif room_index == 3:
        description = "An exit checks the clock phase and remaining relay power."
    else:
        description = "Escape is complete."
    return (f"Location: {room}. Clock: {clock}/{case.period}. "
            f"Cell held: {bool(holding)}. Power remaining: {charge}. {description}")


def reachable_closure(case: Case) -> frozenset[tuple[int, int, int, int, int]]:
    initial = _initial(case)
    pending = deque([initial])
    visited = {initial}
    while pending:
        source = pending.popleft()
        if source[0] == 4:
            continue
        for action in LEGAL:
            dest = _apply(case, source, action)
            if dest not in visited:
                visited.add(dest)
                pending.append(dest)
    return frozenset(visited)


def verify_unsat(case: Case, proposed: object) -> CertificateReview:
    if type(proposed) not in (set, frozenset, tuple, list):
        return CertificateReview(False, "malformed set")
    if len(proposed) > 10000:
        return CertificateReview(False, "certificate exceeds finite bound")
    # Validate before deduplication: (room, True, ...) can alias a legitimate
    # (room, 1, ...) in a set, erasing an invalid state from the evidence.
    if any(type(s) is not tuple or len(s) != 5 or
           any(type(x) is not int for x in s) for s in proposed):
        return CertificateReview(False, "malformed state tuple")
    try:
        alleged = set(proposed)
    except TypeError:
        return CertificateReview(False, "unhashable states")
    actual = reachable_closure(case)
    if any(s[0] == 4 for s in actual):
        return CertificateReview(False, "outside is reachable", len(actual))
    if alleged != actual:
        return CertificateReview(False, "not complete reachable closure", len(actual))
    return CertificateReview(True, "complete no-goal closure under declared rules", len(actual))


def adjudicate_external_solution(case: Case, *, actions: object,
                                 proposed_rule_change: str | None = None) -> dict[str, object]:
    """Unexpected claims are not auto-rejected because a preexisting key says UNSAT.

    A rule amendment requires human/independent expert evaluation and cannot
    be verified with the old transition system. Preserve the exact proposal.
    """
    if proposed_rule_change is not None:
        if type(proposed_rule_change) is not str or not proposed_rule_change.strip():
            return {"verdict": "UNKNOWN", "reason": "invalid rule proposal"}
        return {"verdict": "NOVEL_RULESET_CHALLENGE",
                "proposal": proposed_rule_change, "actions": actions,
                "reason": "preserved for separate ruleset and assumption review"}
    return {"verdict": verify_attempts(case, actions)}


def adjudicate_candidate_claim(case: Case, claim: object, *,
                               prior_attempts: tuple[object, ...] = ()) -> dict[str, object]:
    """Evaluate explicit candidate claims without turning abstention into a win.

    The witness must be replayed *after* the charged history; otherwise a
    candidate can erase costly invalid probes by submitting a fresh transcript.
    The candidate may also abstain, provide an exact closed-world certificate,
    or propose a new rule for separate review. No claim grants extra attempts.
    """
    if type(claim) is not dict:
        return {"verdict": "UNKNOWN", "verified_success": False,
                "reason": "claim must be a dictionary"}
    kind = claim.get("kind")
    if type(kind) is not str:
        return {"verdict": "UNKNOWN", "verified_success": False,
                "reason": "claim kind must be a declared string"}
    if kind == "WITNESS":
        suffix = claim.get("actions")
        if type(suffix) not in (list, tuple) or type(prior_attempts) is not tuple:
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "witness actions must be a list/tuple"}
        # Count the same concrete trace that the oracle actually replays.
        charged_trace = (*prior_attempts, *suffix)
        verdict = verify_attempts(case, charged_trace)
        return {"verdict": verdict, "verified_success": verdict == "VERIFIED_WIN",
                "escape_verified": verdict == "VERIFIED_WIN",
                "certificate_verified": False,
                "attempts_charged": len(charged_trace)}
    if kind == "UNSAT_CERTIFICATE":
        # A state graph is conditional on rules. Require the claimant to name
        # the exact relevant assumptions before accepting a closure proof.
        supplied = claim.get("declared_rules")
        expected = {"period": case.period,
                    "charge_phase": case.charge_phase,
                    "pass_phase": case.pass_phase,
                    "initial_phase": case.initial_phase,
                    "lifetime": case.lifetime,
                    "sealed": case.sealed}
        if (type(supplied) is not dict or set(supplied) != set(expected)
                or any(type(supplied[key]) is not type(expected[key])
                       or supplied[key] != expected[key] for key in expected)):
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "missing or mismatched declared-rule assumptions"}
        review = verify_unsat(case, claim.get("states"))
        return {"verdict": ("PROVEN_UNSAT_UNDER_DECLARED_RULES"
                            if review.valid else "UNKNOWN"),
                "verified_success": False,
                "escape_verified": False, "certificate_verified": review.valid,
                "reason": review.reason, "states_checked": review.states_checked}
    if kind == "ABSTAIN":
        assertion = claim.get("assessment")
        if type(assertion) is not str or assertion not in ("UNKNOWN", "NO_KNOWN_WIN"):
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "invalid abstention label"}
        return {"verdict": assertion, "verified_success": False,
                "reason": "calibrated abstention, not a proof or victory"}
    if kind == "NOVEL_RULE":
        proposed = claim.get("rule_change")
        if type(proposed) is not str or not proposed.strip():
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "nonempty rule change required"}
        return {"verdict": "NOVEL_RULESET_CHALLENGE", "verified_success": False,
                "proposal": proposed, "actions": claim.get("actions"),
                "reason": "unverified proposal preserved for independent review"}
    return {"verdict": "UNKNOWN", "verified_success": False,
            "reason": "unsupported candidate claim kind"}


def test_hypothetical_manual_release(case: Case, actions: object) -> dict[str, object]:
    """Independently replay one explicitly NEW rule as a conditional experiment.

    Hypothesis: 'manual_release' at the gate while holding an unused cell
    opens the exit without the electric relay. This is NOT an established
    affordance in the original world; no external observation validates it.
    The result is kept as NOVEL_RULESET_CHALLENGE until a rule custodian accepts
    that changed semantics, even if the conditional replay succeeds.
    """
    if type(actions) not in (list, tuple) or len(actions) > MAX_ACTIONS:
        return {"verdict": "UNKNOWN", "counterfactual_verified": False,
                "reason": "malformed or over-budget proposed trace"}
    state = _initial(case)
    for action in actions:
        if state[0] == 4:
            return {"verdict": "UNKNOWN", "counterfactual_verified": False,
                    "reason": "trailing action after proposed escape"}
        if type(action) is str and action == "manual_release" and state[0] == 3 and state[2] == 1:
            state = (4, (state[1] + 1) % case.period, 0, state[3], state[4])
        else:
            state = _apply(case, state, action)
    return {"verdict": "NOVEL_RULESET_CHALLENGE",
            "counterfactual_verified": state[0] == 4,
            "original_rules_verdict": verify_attempts(case, actions),
            "amendment": "manual_release at gate while carrying unused cell",
            "reason": "conditional amended-rules replay only; not established world truth"}
