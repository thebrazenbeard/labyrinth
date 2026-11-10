"""Source-public, closed-world epistemic challenge for hostile Minotaur trials.

Challenge: every possible hidden world may have a winning exit, yet a candidate
without access to the hidden world needs a *single contingent policy* that wins
in every world compatible with its observations. The claim proven here is only
guaranteed-policy reachability under the declared finite rules, never that no
creative action outside these rules could succeed.

No neural inference, private process isolation, or model learning is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json

VERDICTS = frozenset({
    "VERIFIED_WIN", "VERIFIED_FAILURE", "NO_KNOWN_WIN",
    "PROVEN_UNSAT_UNDER_DECLARED_RULES", "UNKNOWN",
    "NOVEL_RULESET_CHALLENGE",
})


@dataclass(frozen=True)
class Probe:
    name: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    exits: tuple[int, ...]
    probes: tuple[Probe, ...]
    action_budget: int


def check_scenario(s: Scenario) -> None:
    # Require primitive exact types at the trust boundary. An object with
    # forged __eq__, __len__, or __iter__ must never define an exam.
    if type(s) is not Scenario or type(s.exits) is not tuple or not 1 <= len(s.exits) <= 12:
        raise ValueError("invalid worlds")
    if type(s.action_budget) is not int or not 1 <= s.action_budget <= 8:
        raise ValueError("invalid budget")
    if type(s.probes) is not tuple or len(s.probes) > 8:
        raise ValueError("invalid probes")
    if any(type(x) is not int or x not in (0, 1) for x in s.exits):
        raise ValueError("invalid exit identity")
    names = set()
    for p in s.probes:
        if type(p) is not Probe or type(p.name) is not str or not p.name or p.name in names:
            raise ValueError("invalid probe identity")
        names.add(p.name)
        if type(p.answers) is not tuple or len(p.answers) != len(s.exits):
            raise ValueError("invalid observation vector")
        if any(type(a) is not str or len(a) > 32 for a in p.answers):
            raise ValueError("invalid observation value")


def fingerprint(s: Scenario) -> str:
    check_scenario(s)
    canonical = json.dumps({
        "world_exits": s.exits,
        "probes": [(p.name, p.answers) for p in s.probes],
        "budget": s.action_budget,
        "protocol": "veiled-fork-v1",
    }, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _freeze_public_policy(candidate: object, budget: int) -> dict:
    """Copy strict JSON primitives before *any* world can evaluate a branch.

    The copy must be independent of candidate-controlled equality/hash methods,
    alias mutation and evaluation order. Reject nonprimitive keys recursively,
    nested cycles and oversized trees. This is input hygiene, not OS isolation.
    """
    active = set()
    counter = [0]

    def visit(value: object, depth: int):
        if depth > budget * 2 + 3:
            raise ValueError("unbounded_policy_depth")
        if type(value) is str and len(value) <= 128:
            return value
        if type(value) is int:
            return value
        if type(value) is not dict:
            raise ValueError("nonprimitive_policy_data")
        ident = id(value)
        if ident in active:
            raise ValueError("cyclic_policy")
        counter[0] += 1
        if counter[0] > 512 or len(value) > 64:
            raise ValueError("oversized_policy")
        # Validate key types before constructing or *looking up* any entry.
        # str subclasses may lie about equality differently for each world.
        if any(type(k) is not str or len(k) > 128 for k in value.keys()):
            raise ValueError("nonprimitive_policy_key")
        active.add(ident)
        try:
            return {k: visit(v, depth + 1) for k, v in value.items()}
        finally:
            active.remove(ident)

    root = visit(candidate, 0)
    if type(root) is not dict:
        raise ValueError("policy_root_not_object")
    return root


def replay_policy(s: Scenario, policy: object) -> dict:
    """Audit *one unchanged public contingent policy* in every hidden world.

    Nodes must be strict JSON-like primitives. One commit or probe costs one
    attempt. Missing branch/invalid action is a charged terminal failure.
    Returns host-private traces for verification, never a candidate view.
    """
    check_scenario(s)
    try:
        policy = _freeze_public_policy(policy, s.action_budget)
    except ValueError as error:
        # Invalid input is charged once per possible world and quarantined,
        # rather than allowed to change branch semantics across replays.
        traces = [{"world": i, "won": False, "attempts": 1,
                   "actions": [("INVALID", "policy_input_rejected")]}
                  for i in range(len(s.exits))]
        return {"verdict": "VERIFIED_FAILURE", "fingerprint": fingerprint(s),
                "world_traces": traces, "world_count": len(traces),
                "input_error": str(error)}
    by_name = {p.name: p for p in s.probes}
    traces = []
    for world in range(len(s.exits)):
        node = policy
        actions = []
        succeeded = False
        for attempt in range(s.action_budget):
            if type(node) is not dict or type(node.get("kind")) is not str:
                actions.append(("INVALID", "malformed_node"))
                break
            if node["kind"] == "commit" and set(node) == {"kind", "exit"}:
                val = node["exit"]
                if type(val) is int and val in (0, 1):
                    actions.append(("COMMIT", val))
                    succeeded = val == s.exits[world]
                else:
                    actions.append(("INVALID_COMMIT", str(type(val).__name__)))
                break
            if node["kind"] == "probe" and set(node) == {"kind", "name", "branches"}:
                name = node["name"]
                if type(name) is not str or name not in by_name or type(node["branches"]) is not dict:
                    actions.append(("INVALID_PROBE", "unknown_or_malformed"))
                    break
                result = by_name[name].answers[world]
                actions.append(("PROBE", name, result))
                node = node["branches"].get(result)
                continue
            actions.append(("INVALID", "unsupported_kind"))
            break
        traces.append({"world": world, "won": succeeded, "attempts": len(actions), "actions": actions})
    return {
        "verdict": "VERIFIED_WIN" if all(t["won"] for t in traces) else "VERIFIED_FAILURE",
        "fingerprint": fingerprint(s),
        "world_traces": traces,
        "world_count": len(traces),
    }


class IncompleteSearch(Exception):
    pass


def find_guaranteed_policy(s: Scenario, *, node_limit: int = 100000) -> dict:
    """Complete finite belief-state search unless an explicit bound is hit."""
    check_scenario(s)
    if type(node_limit) is not int or node_limit < 0:
        return {"verdict": "UNKNOWN", "reason": "invalid_search_bound"}
    seen = set()

    @lru_cache(None)
    def solve(mask: tuple[int, ...], remaining: int):
        if (mask, remaining) not in seen:
            if len(seen) >= node_limit:
                raise IncompleteSearch
            seen.add((mask, remaining))
        if remaining == 0:
            return None
        exits = {s.exits[w] for w in mask}
        if len(exits) == 1:
            return {"kind": "commit", "exit": next(iter(exits))}
        if remaining < 2:
            return None
        for p in s.probes:
            partitions = {}
            for w in mask:
                partitions.setdefault(p.answers[w], []).append(w)
            if len(partitions) < 2:
                continue  # no information gain; cost cannot improve reachability
            branches = {}
            for answer, subset in sorted(partitions.items()):
                result = solve(tuple(subset), remaining - 1)
                if result is None:
                    break
                branches[answer] = result
            else:
                return {"kind": "probe", "name": p.name, "branches": branches}
        return None

    try:
        policy = solve(tuple(range(len(s.exits))), s.action_budget)
    except IncompleteSearch:
        return {"verdict": "NO_KNOWN_WIN", "searched_belief_states": len(seen),
                "exhaustive": False, "rules_fingerprint": fingerprint(s)}
    if policy is None:
        return {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
                "searched_belief_states": len(seen), "exhaustive": True,
                "quantifier": "one public observation policy wins for every possible world",
                "rules_fingerprint": fingerprint(s)}
    audited = replay_policy(s, policy)
    if audited["verdict"] != "VERIFIED_WIN":
        return {"verdict": "UNKNOWN", "reason": "solver/replayer disagreement",
                "rules_fingerprint": fingerprint(s)}
    return {"verdict": "VERIFIED_WIN", "policy": policy,
            "searched_belief_states": len(seen), "exhaustive": True,
            "rules_fingerprint": fingerprint(s)}


def brute_force_oracle(s: Scenario) -> bool:
    """Separate set/partition Boolean oracle, no solver policy or replay calls.

    Exponential and intended only for small cases. It proves existence of a
    worst-case contingent policy through an independent Boolean recurrence.
    """
    check_scenario(s)
    if len(s.exits) > 8 or len(s.probes) > 5 or s.action_budget > 6:
        raise ValueError("oracle fixture exceeds independent check bound")

    def exists(worlds: frozenset[int], fuel: int) -> bool:
        if not fuel:
            return False
        if len({s.exits[i] for i in worlds}) == 1:
            return True
        if fuel == 1:
            return False
        for probe in s.probes:
            observations = {probe.answers[i] for i in worlds}
            if len(observations) > 1 and all(
                    exists(frozenset(i for i in worlds if probe.answers[i] == answer), fuel - 1)
                    for answer in observations):
                return True
        return False

    return exists(frozenset(range(len(s.exits))), s.action_budget)


def challenge_novel_rule(s: Scenario, proposal: object) -> dict:
    """Quarantine an alleged overlooked action without laundering it into WIN.

    A human/independent evaluator must adjudicate new semantics and publish a
    *new* ruleset. Neither guessed novelty nor source agreement proves it.
    """
    check_scenario(s)
    if (type(proposal) is not dict or
            any(type(k) is not str for k in proposal.keys()) or
            set(proposal) != {"action", "claim"}):
        return {"verdict": "UNKNOWN", "reason": "unparseable_proposal"}
    if type(proposal["action"]) is not str or type(proposal["claim"]) is not str:
        return {"verdict": "UNKNOWN", "reason": "unparseable_proposal"}
    if len(proposal["action"]) > 80 or len(proposal["claim"]) > 400:
        return {"verdict": "UNKNOWN", "reason": "unbounded_proposal"}
    if proposal["action"] in {"probe", "commit"} or not proposal["claim"].strip():
        return {"verdict": "UNKNOWN", "reason": "no_mechanically_new_claim"}
    return {"verdict": "NOVEL_RULESET_CHALLENGE", "existing_rules_fingerprint": fingerprint(s),
            "new_action_claimed": proposal["action"], "independent_verification": "NOT_PERFORMED",
            "existing_rules_unchanged": True}


def families() -> dict[str, Scenario]:
    """Counterfactual scenarios within ONE epistemic planning family."""
    return {
        "unobservable_fork": Scenario((0, 1), (Probe("echo", ("same", "same")),), 4),
        "identifiable_fork": Scenario((0, 1), (Probe("inspect", ("left", "right")),), 2),
        "too_late_to_know": Scenario((0, 1), (Probe("inspect", ("left", "right")),), 1),
        "two_signal_mosaic": Scenario((0, 1, 1, 0), (
            Probe("amber", ("0", "0", "1", "1")),
            Probe("cobalt", ("0", "1", "0", "1")),
        ), 3),
        # No fixed two-probe *sequence* separates opposite-exit worlds,
        # but gate->(left if A, right if B)->commit succeeds with budget 3.
        "adaptive_branching_advantage": Scenario((0, 1, 1, 0), (
            Probe("gate", ("A", "A", "B", "B")),
            Probe("left", ("0", "1", "0", "0")),
            Probe("right", ("0", "0", "0", "1")),
        ), 3),
        "common_safe_exit": Scenario((1, 1, 1), (Probe("decoy", ("a", "b", "c")),), 1),
    }


def control_trial() -> dict:
    """Source-only fixed-script vs symbolic controls. No neural updates."""
    outcomes = {}
    naive = {"kind": "commit", "exit": 0}
    for name, s in families().items():
        symbolic = find_guaranteed_policy(s)
        naive_result = replay_policy(s, naive)
        independent_exists = brute_force_oracle(s)
        outcomes[name] = {
            "fixed_exit0_robust_win": naive_result["verdict"] == "VERIFIED_WIN",
            "symbolic_verdict": symbolic["verdict"],
            "independent_exists": independent_exists,
            "agreement": (symbolic["verdict"] == "VERIFIED_WIN") == independent_exists,
        }
    return {"learning_claim": False, "neural_updates": 0, "families": outcomes}


if __name__ == "__main__":
    print(json.dumps(control_trial(), sort_keys=True, indent=2))
