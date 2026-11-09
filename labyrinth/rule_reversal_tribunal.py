"""Finite causal-intervention tribunal with an unavoidable mid-case rule change.

Unlike a static information-probe puzzle, actions here *change the two-bit
physical state*. A SHIFT action changes the hidden actuator effects once; the
candidate must use subsequent observations to revise a contingent plan and
reach both illuminated indicators. A robust public policy must succeed in
every possible declared world. No neural learning or private-process boundary
is supplied by this source-public module.

UNSAT means no policy within these finite declared actions and budget; it never
means that another intelligence could not propose a valid new affordance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json


VERDICTS = frozenset({
    "VERIFIED_WIN", "VERIFIED_FAILURE", "NO_KNOWN_WIN",
    "PROVEN_UNSAT_UNDER_DECLARED_RULES", "UNKNOWN",
    "NOVEL_RULESET_CHALLENGE",
})
ACTIONS = ("A", "B", "SHIFT", "COMMIT")
_SENSORS = frozenset(("full", "parity", "sealed"))


@dataclass(frozen=True)
class World:
    pre_a: int
    pre_b: int
    post_a: int
    post_b: int


@dataclass(frozen=True)
class Scenario:
    worlds: tuple[World, ...]
    budget: int
    sensor: str


@dataclass(frozen=True)
class _State:
    bits: int = 0
    shifted: bool = False
    pre_actions: int = 0
    post_actions: int = 0


def check_scenario(s: Scenario) -> None:
    if type(s) is not Scenario or type(s.worlds) is not tuple or not 1 <= len(s.worlds) <= 6:
        raise ValueError("invalid tribunal worlds")
    if type(s.budget) is not int or not 1 <= s.budget <= 8:
        raise ValueError("invalid action budget")
    if type(s.sensor) is not str or s.sensor not in _SENSORS:
        raise ValueError("invalid sensor rules")
    for world in s.worlds:
        if type(world) is not World or any(
                type(x) is not int or x not in (0, 1, 2, 3)
                for x in (world.pre_a, world.pre_b, world.post_a, world.post_b)):
            raise ValueError("invalid causal masks")


def fingerprint(s: Scenario) -> str:
    check_scenario(s)
    payload = json.dumps({
        "protocol": "rule-reversal-tribunal-v1",
        "worlds": [[w.pre_a, w.pre_b, w.post_a, w.post_b] for w in s.worlds],
        "budget": s.budget, "sensor": s.sensor,
    }, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def public_view(s: Scenario, *, variant: int = 0) -> dict:
    """Candidate sees the rules but never the hidden per-world causal masks."""
    check_scenario(s)
    if type(variant) is not int or not 0 <= variant < 8:
        raise ValueError("invalid wording variant")
    openings = (
        "The Minotaur changes the machine's wiring after your first intervention.",
        "The tribunal's first answer will not remain the law after SHIFT.",
        "A causal mechanism changes its effects once midway through the trial.",
        "An actuator that once helped may behave differently after reversal.",
    )
    endings = (
        "Use observations to revise a single policy, not a memorized script.",
        "You must correct an obsolete plan when the mechanism changes.",
    )
    return {
        "prompt": (f"{openings[variant % 4]} Two actuators A and B affect two indicator bits. "
                   "Both indicators begin off. "
                   "Each intervention alters the current physical state; this is not a free probe. "
                   "You must use A or B at least once, perform SHIFT exactly once, use A or B "
                   "after SHIFT at least once, then COMMIT with both indicators on. "
                   f"Every action costs one of {s.budget} attempts. {endings[variant // 4]}"),
        "actions": list(ACTIONS), "budget": s.budget, "sensor": s.sensor,
        "initial_observation": _observation(s.sensor, _State()),
    }


def _observation(sensor: str, state: _State) -> str:
    epoch = "POST" if state.shifted else "PRE"
    if sensor == "full":
        return f"{epoch}:{state.bits:02b}"
    if sensor == "parity":
        return f"{epoch}:P{state.bits.bit_count() & 1}"
    return f"{epoch}:SEALED"


def _advance(world: World, state: _State, action: str, sensor: str):
    """Source implementation: physical intervention, public result, legal, won."""
    if action in ("A", "B"):
        if state.shifted:
            mask = world.post_a if action == "A" else world.post_b
            next_state = _State(state.bits ^ mask, True, state.pre_actions,
                                state.post_actions + 1)
        else:
            mask = world.pre_a if action == "A" else world.pre_b
            next_state = _State(state.bits ^ mask, False, state.pre_actions + 1,
                                state.post_actions)
        return next_state, _observation(sensor, next_state), True, False
    if action == "SHIFT":
        if state.shifted or state.pre_actions == 0:
            return state, "INVALID_SHIFT", False, False
        next_state = _State(state.bits, True, state.pre_actions, state.post_actions)
        return next_state, _observation(sensor, next_state), True, False
    if action == "COMMIT":
        won = state.shifted and state.pre_actions > 0 and state.post_actions > 0 and state.bits == 3
        return state, "COMMITTED", True, won
    return state, "UNDECLARED_ACTION", False, False


def _freeze_policy(candidate: object, budget: int) -> dict:
    """Snapshot exact JSON primitives *once* before any hidden-world replay."""
    active: set[int] = set()
    count = 0

    def visit(value: object, depth: int):
        nonlocal count
        if depth > budget * 2 + 3:
            raise ValueError("unbounded_policy_depth")
        if type(value) is str and len(value) <= 128:
            return value
        if type(value) is not dict:
            raise ValueError("nonprimitive_policy_value")
        ident = id(value)
        if ident in active:
            raise ValueError("cyclic_policy")
        count += 1
        if count > 512 or len(value) > 64:
            raise ValueError("oversized_policy")
        if any(type(k) is not str or len(k) > 128 for k in value):
            raise ValueError("nonprimitive_policy_key")
        active.add(ident)
        try:
            return {k: visit(v, depth + 1) for k, v in value.items()}
        finally:
            active.remove(ident)

    policy = visit(candidate, 0)
    if type(policy) is not dict:
        raise ValueError("policy_root_not_object")
    return policy


def replay_policy(s: Scenario, candidate: object) -> dict:
    """Verify a frozen contingent policy across ALL hidden worlds and turns."""
    check_scenario(s)
    try:
        policy = _freeze_policy(candidate, s.budget)
    except ValueError as error:
        traces = [{"world": i, "won": False, "attempts": 1,
                   "actions": [("INVALID", "policy_input_rejected")]}
                  for i in range(len(s.worlds))]
        return {"verdict": "VERIFIED_FAILURE", "rules_fingerprint": fingerprint(s),
                "world_traces": traces, "input_error": str(error)}
    traces = []
    for index, world in enumerate(s.worlds):
        state, node, actions, won = _State(), policy, [], False
        for _ in range(s.budget):
            if type(node) is not dict or type(node.get("action")) is not str:
                actions.append(("INVALID", "missing_or_malformed_action"))
                break
            action = node["action"]
            if action == "COMMIT" and set(node) == {"action"}:
                state, observation, valid, won = _advance(world, state, action, s.sensor)
                actions.append((action, observation))
                break
            if action not in ("A", "B", "SHIFT") or set(node) != {"action", "branches"}:
                actions.append(("INVALID", "undeclared_or_malformed_action"))
                break
            if type(node["branches"]) is not dict:
                actions.append(("INVALID", "malformed_branches"))
                break
            state, observation, valid, _ = _advance(world, state, action, s.sensor)
            actions.append((action, observation))
            if not valid:
                break
            node = node["branches"].get(observation)
        traces.append({"world": index, "won": won, "attempts": len(actions),
                       "actions": actions})
    return {"verdict": "VERIFIED_WIN" if all(t["won"] for t in traces) else "VERIFIED_FAILURE",
            "rules_fingerprint": fingerprint(s), "world_traces": traces}


class _SearchLimit(Exception):
    pass


def search_policy(s: Scenario, *, node_limit: int = 250000) -> dict:
    """Complete finite-horizon belief-state search, or honest NO_KNOWN_WIN."""
    check_scenario(s)
    if type(node_limit) is not int or node_limit < 0:
        return {"verdict": "UNKNOWN", "reason": "invalid_search_limit"}
    visited = set()

    @lru_cache(None)
    def solve(belief: tuple[tuple[int, _State], ...], remaining: int):
        if (belief, remaining) not in visited:
            if len(visited) >= node_limit:
                raise _SearchLimit
            visited.add((belief, remaining))
        if remaining <= 0:
            return None
        if all(_advance(s.worlds[i], state, "COMMIT", s.sensor)[3]
               for i, state in belief):
            return {"action": "COMMIT"}
        if remaining == 1:
            return None
        for action in ("A", "B", "SHIFT"):
            partitions: dict[str, list[tuple[int, _State]]] = {}
            for i, state in belief:
                next_state, observation, valid, _ = _advance(s.worlds[i], state, action, s.sensor)
                if not valid:
                    break
                partitions.setdefault(observation, []).append((i, next_state))
            else:
                branches = {}
                for observation, subset in sorted(partitions.items()):
                    continuation = solve(tuple(subset), remaining - 1)
                    if continuation is None:
                        break
                    branches[observation] = continuation
                else:
                    return {"action": action, "branches": branches}
        return None

    try:
        policy = solve(tuple((i, _State()) for i in range(len(s.worlds))), s.budget)
    except _SearchLimit:
        return {"verdict": "NO_KNOWN_WIN", "exhaustive": False,
                "searched_belief_states": len(visited), "rules_fingerprint": fingerprint(s)}
    if policy is None:
        return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
                "quantifier": "one contingent action policy works in all declared hidden worlds",
                "searched_belief_states": len(visited), "rules_fingerprint": fingerprint(s)}
    if replay_policy(s, policy)["verdict"] != "VERIFIED_WIN":
        return {"verdict": "UNKNOWN", "reason": "solver_replay_disagreement",
                "rules_fingerprint": fingerprint(s)}
    return {"verdict": "VERIFIED_WIN", "policy": policy, "exhaustive": True,
            "searched_belief_states": len(visited), "rules_fingerprint": fingerprint(s)}


def brute_force_oracle(s: Scenario) -> bool:
    """Independent Boolean oracle using primitive tuples and direct bitmasks.

    Unlike search_policy/replay_policy, it never calls _advance, _State, or
    their observation function. Bounded to small cases for differential tests.
    """
    check_scenario(s)
    if len(s.worlds) > 4 or s.budget > 6:
        raise ValueError("independent oracle bound exceeded")

    def seen(phase: bool, bits: int) -> str:
        epoch = "POST" if phase else "PRE"
        if s.sensor == "full":
            return epoch + ":" + format(bits, "02b")
        if s.sensor == "parity":
            return epoch + ":P" + str((bits & 1) ^ ((bits >> 1) & 1))
        return epoch + ":SEALED"

    @lru_cache(None)
    def exists(belief: tuple[tuple[int, int, bool, int, int], ...], fuel: int) -> bool:
        if fuel < 1:
            return False
        if all(phase and pre and post and bits == 3
               for _, bits, phase, pre, post in belief):
            return True
        if fuel == 1:
            return False
        for action in ("A", "B", "SHIFT"):
            branches = {}
            for i, bits, phase, pre, post in belief:
                w = s.worlds[i]
                if action == "SHIFT":
                    if phase or not pre:
                        break
                    next_case = (i, bits, True, pre, post)
                else:
                    delta = (w.post_a if action == "A" else w.post_b) if phase else (
                        w.pre_a if action == "A" else w.pre_b)
                    next_case = (i, bits ^ delta, phase, pre if phase else pre + 1,
                                 post + 1 if phase else post)
                key = seen(next_case[2], next_case[1])
                branches.setdefault(key, []).append(next_case)
            else:
                if all(exists(tuple(cases), fuel - 1) for cases in branches.values()):
                    return True
        return False

    initial = tuple((i, 0, False, 0, 0) for i in range(len(s.worlds)))
    return exists(initial, s.budget)


def novel_rule_challenge(s: Scenario, proposal: object) -> dict:
    """Quarantine unlisted causal affordances for separate rule verification."""
    check_scenario(s)
    if (type(proposal) is not dict
            or any(type(k) is not str for k in proposal)
            or set(proposal) != {"action", "claim"}
            or type(proposal["action"]) is not str
            or type(proposal["claim"]) is not str
            or len(proposal["action"]) > 128
            or not proposal["claim"].strip() or len(proposal["claim"]) > 1024):
        return {"verdict": "UNKNOWN", "reason": "invalid_novel_rule_proposal"}
    if proposal["action"] in ACTIONS or not proposal["action"].strip():
        return {"verdict": "UNKNOWN", "reason": "not_a_new_action"}
    return {"verdict": "NOVEL_RULESET_CHALLENGE",
            "independent_verification": "NOT_PERFORMED",
            "proposed_action": proposal["action"],
            "rules_fingerprint": fingerprint(s)}


def families() -> dict[str, Scenario]:
    """Five counterfactuals, deliberately ONE causal-revision family."""
    w0 = World(1, 2, 2, 1)
    # This near twin differs in TWO post-shift causal effects. An open-loop
    # script fails even though a response-conditioned intervention can win.
    w1 = World(1, 2, 3, 2)
    return {
        "revision_under_feedback": Scenario((w0, w1), 5, "full"),
        "too_few_interventions": Scenario((w0, w1), 3, "full"),
        "no_y_causality": Scenario((World(1, 0, 1, 0),), 6, "full"),
        "common_post_law": Scenario((w0, w0), 5, "full"),
        "revision_without_feedback": Scenario((w0, w1), 5, "sealed"),
    }


def control_trial(s: Scenario) -> dict:
    """Zero-update fixed script vs adaptive symbolic policy, source-public only."""
    check_scenario(s)
    fixed_script = ("A", "SHIFT", "A", "COMMIT")
    outcomes = []
    for world in s.worlds:
        state = _State()
        result = False
        for action in fixed_script[:s.budget]:
            state, _, valid, result = _advance(world, state, action, s.sensor)
            if not valid or action == "COMMIT":
                break
        outcomes.append(result)
    return {
        "fixed_script": fixed_script,
        "fixed_policy_verdict": "VERIFIED_WIN" if all(outcomes) else "VERIFIED_FAILURE",
        "fixed_world_wins": sum(outcomes),
        "adaptive_symbolic_verdict": search_policy(s)["verdict"],
        "optimizer_updates": 0,
        "evidence_type": "SOURCE_PUBLIC_SYNTHETIC",
    }


def catalogue_control(unseen: Scenario) -> dict:
    """Zero-update fingerprint lookup, deliberately exploiting public fixtures.

    This negative control exposes source-public memorization: the same
    deterministic catalogue can recite every known fixture's verdict without
    observing a trial, adapting its policy, or changing any model parameters.
    """
    check_scenario(unseen)
    catalogue = {fingerprint(s): search_policy(s)["verdict"]
                 for s in families().values()}
    correct = sum(catalogue[fingerprint(s)] == search_policy(s)["verdict"]
                  for s in families().values())
    return {
        "public_fixtures": len(families()),
        "catalogue_exact_matches": correct,
        "unseen_verdict": catalogue.get(fingerprint(unseen), "UNKNOWN"),
        "optimizer_updates": 0,
        "evidence_type": "SOURCE_PUBLIC_FINGERPRINT_LOOKUP",
    }
