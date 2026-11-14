"""Opt-in source-public fifth Minotaur family, without modifying v1-v3.

This is a *symbolic control* integration: no model, prompts hidden from
source, optimizer update, or independent evaluator custody is involved.
"""

from __future__ import annotations

from collections import Counter

from . import distributed_parity_tribunal as parity
from .minotaur_public_exam import run_public_exam


def _bounded_count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000 else None


def _safe_policy(scenario: parity.Scenario, raw: object) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Snapshot exact immutable bit vectors before calling any verifier."""
    if (type(raw) is not tuple or len(raw) != 2 or
            type(raw[0]) is not tuple or type(raw[1]) is not tuple or
            len(raw[0]) != scenario.left_observations or
            len(raw[1]) != scenario.right_observations):
        return None
    if any(type(bit) is not int or bit not in (0, 1)
           for half in raw for bit in half):
        return None
    return (tuple(raw[0]), tuple(raw[1]))


def parity_case(scenario: parity.Scenario, *, variant: int) -> dict:
    """Cross-check the emitted *particular* policy before calling it a win."""
    rules = parity.fingerprint(scenario)
    raw = parity.search_policy(scenario)
    solved = raw if type(raw) is dict else {}
    claimed = solved.get("verdict")
    status = claimed if type(claimed) is str else "INVALID_SOLVER_VERDICT"
    fingerprint_claim = solved.get("rules_fingerprint")
    source_tag_agrees = type(fingerprint_claim) is str and fingerprint_claim == rules
    policy = _safe_policy(scenario, solved.get("policy"))

    verified = parity.verify_policy(scenario, policy) if policy is not None else None
    replay_verdict = verified.get("verdict") if type(verified) is dict else None
    replay_tag = verified.get("rules_fingerprint") if type(verified) is dict else None
    exact_replay = (type(replay_verdict) is str and type(replay_tag) is str
                    and replay_tag == rules and replay_verdict == "VERIFIED_WIN")
    specific = parity.independent_witness_check(scenario, policy) if policy is not None else None
    specific_valid = specific if type(specific) is bool else None
    pinned = (parity.independent_gf2_exists(scenario, fixed_policy=policy)
              if policy is not None else None)
    pinned_valid = pinned if type(pinned) is bool else None
    algebra = parity.independent_gf2_exists(scenario)
    dfs = parity.independent_dfs_exists(scenario)
    algebra_exists = algebra if type(algebra) is bool else None
    dfs_exists = dfs if type(dfs) is bool else None
    total_policies = 1 << (scenario.left_observations + scenario.right_observations)
    examined = _bounded_count(solved.get("policies_examined"))

    if status == "VERIFIED_WIN":
        agrees = (source_tag_agrees and policy is not None and exact_replay
                  and specific_valid is True and pinned_valid is True
                  and algebra_exists is True and dfs_exists is True)
    elif status == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        certificate = solved.get("certificate")
        certificate_ok = (type(certificate) is dict
                          and certificate.get("kind") == "complete-deterministic-policy-enumeration"
                          and type(certificate.get("enumerated")) is int
                          and certificate["enumerated"] == total_policies
                          and type(certificate.get("finite_space_size")) is int
                          and certificate["finite_space_size"] == total_policies
                          and type(certificate.get("valid_policies_found")) is int
                          and certificate["valid_policies_found"] == 0)
        agrees = (source_tag_agrees and solved.get("policy") is None
                  and policy is None and examined == total_policies
                  and certificate_ok and algebra_exists is False and dfs_exists is False)
    else:
        agrees = False

    if agrees:
        verdict = status
    elif status == "NO_KNOWN_WIN" and source_tag_agrees and examined is not None:
        verdict = "NO_KNOWN_WIN"
    else:
        verdict = "UNKNOWN"
    view = parity.public_view(scenario, variant=variant)
    public_prompt = view.get("prompt") if type(view) is dict else None
    mechanics = parity.mechanics_fingerprint(scenario)
    if (type(public_prompt) is not str or type(view.get("case_fingerprint")) is not str
            or view["case_fingerprint"] != rules
            or type(view.get("mechanics_fingerprint")) is not str
            or view["mechanics_fingerprint"] != mechanics):
        verdict, agrees, public_prompt = "UNKNOWN", False, "INVALID_PUBLIC_VIEW"
    return {
        "verdict": verdict,
        "solver_verdict": status,
        "rules_fingerprint": rules,
        "mechanics_fingerprint": mechanics,
        "public_prompt": public_prompt,
        "world_count": len(scenario.worlds),
        "local_observations": [scenario.left_observations, scenario.right_observations],
        "total_on_budget": scenario.total_on_budget,
        "policies_examined": examined,
        "independent_oracle_agrees": agrees,
        "independent_replay_verdict": replay_verdict if type(replay_verdict) is str else "NOT_RUN",
        "independent_specific_witness_valid": specific_valid,
        "independent_pinned_gf2_valid": pinned_valid,
        "independent_gf2_has_win": algebra_exists,
        "independent_dfs_has_win": dfs_exists,
        "symbolic_policy": [list(half) for half in policy] if verdict == "VERIFIED_WIN" else None,
    }


def run_public_exam_v4(*, seed: int, variant: int) -> dict:
    """Extend the explicitly selected v3 receipt; keep legacy paths untouched."""
    report = run_public_exam(seed=seed, variant=variant,
                             include_exact_cover=True, include_temporal_repair=True)
    cases = {name: parity_case(scenario, variant=variant)
             for name, scenario in parity.families().items()}
    report["protocol"] = "minotaur-source-public-v4"
    report["parameters"]["include_distributed_parity"] = True
    report["families"]["distributed_parity"] = {
        "challenge_mechanics": "simultaneous_local_information_parity_constraints",
        "cases": cases,
        "fixed_script_control": parity.fixed_script_control(),
    }
    counts = Counter(
        case["verdict"]
        for family in report["families"].values()
        for case in family["cases"].values()
    )
    report["summary"] = {
        "families_run": len(report["families"]),
        "cases_run": sum(len(family["cases"]) for family in report["families"].values()),
        "verdict_counts": dict(sorted(counts.items())),
        "independent_checks_agree": (report["summary"]["independent_checks_agree"]
                                     and all(c["independent_oracle_agrees"] for c in cases.values())),
    }
    return report
