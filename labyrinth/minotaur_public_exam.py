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


def _constant_case(seed: int, variant: int, kind: str) -> dict:
    exam = constant.make_exam(seed, split="dev", variant=variant, kind=kind)
    result = constant.search(exam)
    witness = list(result.witness)
    independent = "NOT_RUN"
    certified_unsat = False
    if result.status == "VERIFIED_WIN":
        independent = constant_verifier.verify_trace(kind=kind, actions=tuple(witness)).outcome
        verdict = "VERIFIED_WIN" if independent == "VERIFIED_WIN" else "UNKNOWN"
    elif result.status == "UNSAT_IN_DECLARED_FINITE_MODEL":
        certified_unsat = constant_verifier.verify_unsat_certificate(
            kind=kind, states=result.reachable,
        )
        independent = "CLOSURE_VALID" if certified_unsat else "CLOSURE_REJECTED"
        verdict = "PROVEN_UNSAT_UNDER_DECLARED_RULES" if certified_unsat else "UNKNOWN"
    else:
        # Budget-limited or interrupted search is NOT an impossibility proof.
        verdict = "NO_KNOWN_WIN" if result.status in (
            "UNSOLVED", "NO_WIN_WITHIN_ACTION_BUDGET",
        ) else "UNKNOWN"
    return {
        "verdict": verdict,
        "solver_status": result.status,
        "independent_verdict": independent,
        "independent_certificate_valid": certified_unsat,
        "independent_check_agrees": verdict not in ("UNKNOWN", "NO_KNOWN_WIN"),
        "semantic_family_id": constant.world_fingerprint(exam),
        "public_prompt": constant.public_view(exam)["prompt"],
        "witness_actions": witness,
        "finite_states_explored": result.expansions,
        "fixed_script_verdict": constant_verifier.verify_trace(
            kind=kind, actions=("north", "north", "leave"),
        ).outcome,
    }


def _fork_case(scenario: fork.Scenario) -> dict:
    solved = fork.find_guaranteed_policy(scenario)
    independent_exists = fork.brute_force_oracle(scenario)
    solver_verdict = solved["verdict"]
    witness = solved.get("policy")
    replay_verdict = (
        fork.replay_policy(scenario, witness)["verdict"] if witness is not None
        else "NOT_RUN"
    )
    if solver_verdict == "VERIFIED_WIN":
        agrees = independent_exists and replay_verdict == "VERIFIED_WIN"
    elif solver_verdict == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        agrees = not independent_exists and replay_verdict == "NOT_RUN" and bool(solved.get("exhaustive"))
    else:
        agrees = False
    return {
        "verdict": solver_verdict if agrees else "UNKNOWN",
        "solver_verdict": solver_verdict,
        "independent_oracle_exists": independent_exists,
        "independent_oracle_agrees": agrees,
        "independent_replay_verdict": replay_verdict,
        "rules_fingerprint": fork.fingerprint(scenario),
        "searched_belief_states": solved.get("searched_belief_states"),
        "exhaustive_search": solved.get("exhaustive", False),
        "world_count": len(scenario.exits),
        "action_budget": scenario.action_budget,
        "fixed_script_verdict": fork.replay_policy(scenario, {"kind": "commit", "exit": 0})["verdict"],
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
