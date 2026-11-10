"""Executable *public* Minotaur challenge/zero-update control receipt.

This runner exercises two different finite-rule mechanisms; it cannot measure
neural learning, hidden evaluator custody, or out-of-distribution transfer.
No model is called and every fixture is readable in source. The independent
checks are bounded to the declared finite rules, not real-world possibility.
"""

from __future__ import annotations

from collections import Counter

from . import constant_exam as constant
from . import constant_exam_verifier as constant_verifier
from . import veiled_fork as fork


def _bounded_count(value: object) -> int | None:
    """Copy only exact, plausible integer search counters into JSON receipts."""
    return value if type(value) is int and 0 <= value <= 1_000_000 else None


def _plain_policy(node: object, remaining: int, budget: list[int]) -> dict | None:
    """Boundedly snapshot exact primitive policy nodes without custom copy hooks.

    The independent replayer receives a *different* copy. Its callbacks must
    never retain or mutate the policy later emitted as verified evidence.
    """
    if type(node) is not dict or remaining <= 0 or len(node) > 3:
        return None
    budget[0] -= 1
    if budget[0] < 0 or any(type(key) is not str for key in node):
        return None
    kind = node.get("kind")
    if type(kind) is not str:
        return None
    if kind == "commit" and set(node) == {"kind", "exit"}:
        exit_code = node["exit"]
        if type(exit_code) is int and exit_code in (0, 1):
            return {"kind": "commit", "exit": exit_code}
        return None
    if kind != "probe" or set(node) != {"kind", "name", "branches"}:
        return None
    name, branches = node["name"], node["branches"]
    if (type(name) is not str or len(name) > 128 or type(branches) is not dict
            or not 1 <= len(branches) <= 16):
        return None
    copied_branches = {}
    for answer, next_node in branches.items():
        if type(answer) is not str or len(answer) > 128:
            return None
        copied_node = _plain_policy(next_node, remaining - 1, budget)
        if copied_node is None:
            return None
        copied_branches[answer] = copied_node
    return {"kind": "probe", "name": name, "branches": copied_branches}


def _constant_case(seed: int, variant: int, kind: str) -> dict:
    exam = constant.make_exam(seed, split="dev", variant=variant, kind=kind)
    result = constant.search(exam)
    # Solver assertions must be exact primitives before comparison. A str
    # subclass may branch as WIN but serialize as UNSAT or the reverse.
    solver_status = result.status if type(result.status) is str else "INVALID_SOLVER_STATUS"
    # Search's real witness is an exact tuple. Refuse arbitrary iterables,
    # subclass comparisons, and unbounded/unserializable in-process metadata.
    raw_witness = result.witness
    valid_witness = (type(raw_witness) is tuple
                     and len(raw_witness) <= constant.MAX_ACTIONS
                     and all(type(action) is str for action in raw_witness))
    witness = list(raw_witness) if valid_witness else []
    if not valid_witness:
        solver_status = "INVALID_SOLVER_WITNESS"
    independent = "NOT_RUN"
    certified_unsat = False
    if solver_status == "VERIFIED_WIN":
        raw_independent = constant_verifier.verify_trace(
            kind=kind, actions=tuple(witness),
        ).outcome
        independent = (raw_independent if type(raw_independent) is str
                       else "INVALID_INDEPENDENT_VERDICT")
        verdict = "VERIFIED_WIN" if independent == "VERIFIED_WIN" else "UNKNOWN"
    elif solver_status == "UNSAT_IN_DECLARED_FINITE_MODEL":
        raw_certificate_valid = constant_verifier.verify_unsat_certificate(
            kind=kind, states=result.reachable,
        )
        certified_unsat = type(raw_certificate_valid) is bool and raw_certificate_valid is True
        independent = "CLOSURE_VALID" if certified_unsat else "CLOSURE_REJECTED"
        verdict = "PROVEN_UNSAT_UNDER_DECLARED_RULES" if certified_unsat else "UNKNOWN"
    else:
        # Budget-limited or interrupted search is NOT an impossibility proof.
        verdict = "NO_KNOWN_WIN" if solver_status in (
            "UNSOLVED", "NO_WIN_WITHIN_ACTION_BUDGET",
        ) else "UNKNOWN"
    raw_fixed_script = constant_verifier.verify_trace(
        kind=kind, actions=("north", "north", "leave"),
    ).outcome
    fixed_script = (raw_fixed_script if type(raw_fixed_script) is str
                    else "INVALID_FIXED_SCRIPT_VERDICT")
    return {
        "verdict": verdict,
        "solver_status": solver_status,
        "independent_verdict": independent,
        "independent_certificate_valid": certified_unsat,
        "independent_check_agrees": verdict not in ("UNKNOWN", "NO_KNOWN_WIN"),
        "semantic_family_id": constant.world_fingerprint(exam),
        "public_prompt": constant.public_view(exam)["prompt"],
        "witness_actions": witness,
        "finite_states_explored": _bounded_count(result.expansions),
        "fixed_script_verdict": fixed_script,
    }


def _fork_case(scenario: fork.Scenario) -> dict:
    solved = fork.find_guaranteed_policy(scenario)
    raw_independent_exists = fork.brute_force_oracle(scenario)
    independent_exists = (raw_independent_exists
                          if type(raw_independent_exists) is bool else None)
    # A deceptive str subclass can compare equal to UNSAT and then serialize
    # to VERIFIED_WIN; reject programmable solver-result semantics outright.
    if type(solved) is not dict:
        solved = {}
    solver_verdict = solved.get("verdict")
    if type(solver_verdict) is not str:
        solver_verdict = "INVALID_SOLVER_VERDICT"
    raw_witness = solved.get("policy")
    witness = (_plain_policy(raw_witness, scenario.action_budget, [128])
               if raw_witness is not None else None)
    if raw_witness is None:
        replay_verdict = "NOT_RUN"
    elif witness is None:
        replay_verdict = "INVALID_POLICY"
    else:
        # Verify a disposable copy; publish the untouched, separately captured
        # policy. Otherwise replay callbacks can mutate post-verification data.
        replay_input = _plain_policy(witness, scenario.action_budget, [128])
        replay_result = fork.replay_policy(scenario, replay_input)
        replay_verdict = (replay_result.get("verdict")
                          if type(replay_result) is dict else None)
        if type(replay_verdict) is not str:
            replay_verdict = "INVALID_REPLAY_VERDICT"
        elif replay_input != witness:
            replay_verdict = "INVALID_MUTATED_REPLAY_INPUT"
    if solver_verdict == "VERIFIED_WIN":
        agrees = independent_exists is True and replay_verdict == "VERIFIED_WIN"
    elif solver_verdict == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        agrees = (independent_exists is False and replay_verdict == "NOT_RUN"
                  and type(solved.get("exhaustive")) is bool and solved["exhaustive"])
    else:
        agrees = False
    fixed_result = fork.replay_policy(scenario, {"kind": "commit", "exit": 0})
    raw_fixed_script = (fixed_result.get("verdict")
                        if type(fixed_result) is dict else None)
    fixed_script = (raw_fixed_script if type(raw_fixed_script) is str
                    else "INVALID_FIXED_SCRIPT_VERDICT")
    exhaustive = (solved.get("exhaustive")
                  if type(solved.get("exhaustive")) is bool else None)
    return {
        "verdict": solver_verdict if agrees else "UNKNOWN",
        "solver_verdict": solver_verdict,
        "independent_oracle_exists": independent_exists,
        "independent_oracle_agrees": agrees,
        "independent_replay_verdict": replay_verdict,
        "rules_fingerprint": fork.fingerprint(scenario),
        "searched_belief_states": _bounded_count(solved.get("searched_belief_states")),
        "exhaustive_search": exhaustive,
        "world_count": len(scenario.exits),
        "action_budget": scenario.action_budget,
        "fixed_script_verdict": fixed_script,
        # Policies are source-public. Never emit private per-world replay traces.
        "symbolic_policy": witness if agrees else None,
    }


def run_public_exam(*, seed: int = 2026, variant: int = 0) -> dict:
    """Produce a reproducible source-public receipt for TWO challenge families.

    The seed and variant change constant-exam *wording only*. They are not
    independent structural trials or protected evaluation holdouts.
    """
    if type(seed) is not int or not -(2**31) <= seed <= 2**31 - 1:
        raise ValueError("seed must be a signed 32-bit integer")
    if type(variant) is not int or not 0 <= variant <= 31:
        raise ValueError("variant must be between 0 and 31")

    constant_cases = {
        kind: _constant_case(seed, variant, kind)
        for kind in ("apparently_sealed", "sealed_control")
    }
    veiled_cases = {
        name: _fork_case(scenario)
        for name, scenario in fork.families().items()
    }
    all_cases = [*constant_cases.values(), *veiled_cases.values()]
    counts = Counter(c["verdict"] for c in all_cases)
    return {
        "protocol": "minotaur-source-public-v1",
        "parameters": {"seed": seed, "variant": variant},
        "claim_scope": {
            "model_identity": "built-in deterministic symbolic controls",
            "neural_weight_updates": 0,
            "neural_learning_tested": False,
            "hidden_custody_tested": False,
            "structural_transfer_tested": False,
            "independent_cases_across_paraphrase_seeds": False,
            "ruleset_scope": "declared source-public finite transitions only",
        },
        "families": {
            "constant_exam": {
                "challenge_mechanics": "finite_navigation_and_offering",
                "source_fingerprint": constant.ruleset_source_fingerprint(),
                "cases": constant_cases,
            },
            "veiled_fork": {
                "challenge_mechanics": "contingent_epistemic_planning",
                "cases": veiled_cases,
            },
        },
        "summary": {
            "families_run": 2,
            "cases_run": len(all_cases),
            "verdict_counts": dict(sorted(counts.items())),
            "independent_checks_agree": all(
                case["independent_check_agrees"] for case in constant_cases.values()
            ) and all(case["independent_oracle_agrees"] for case in veiled_cases.values()),
        },
    }
