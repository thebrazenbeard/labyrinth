"""Executable *public* Minotaur challenge/zero-update control receipt.

The legacy v1 runner exercises two finite-rule mechanisms. Opt-in v2 exercises
three, including a combinatorial exact-cover tribunal. Opt-in v3 adds a fourth,
time-dependent repair-scheduling mechanism. None can measure
neural learning, hidden evaluator custody, or out-of-distribution transfer.
No model is called and every fixture is readable in source. The independent
checks are bounded to the declared finite rules, not real-world possibility.
"""

from __future__ import annotations

from collections import Counter

from . import constant_exam as constant
from . import constant_exam_verifier as constant_verifier
from . import exact_cover_tribunal as cover
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


def _independent_cover_witness(scenario: cover.Scenario,
                               allocation: tuple[int, ...] | None) -> bool:
    """Check the *specific* witness via disjoint site bits, not cover replay.

    The subset-existence oracle alone cannot prove that an emitted allocation
    wins. Keep this direct check independent from both source search and the
    set-based `cover.verify_allocation` implementation.
    """
    if (type(allocation) is not tuple or len(allocation) > scenario.budget
            or any(type(i) is not int or not 0 <= i < len(scenario.crews)
                   for i in allocation)):
        return False
    site_bits = {site: 1 << index for index, site in enumerate(scenario.sites)}
    visited = 0
    for crew_index in allocation:
        for site in scenario.crews[crew_index]:
            bit = site_bits[site]
            if visited & bit:
                return False
            visited |= bit
    return visited == (1 << len(scenario.sites)) - 1


def _independent_cover_exists_dp(scenario: cover.Scenario) -> bool | None:
    """Third exact-cover existence check: iterative bitmask subset reachability.

    Distinct from the source DFS and combinations-based subset oracle. Process
    each declared crew once and retain sets by action count. Return None on a
    bounded work cap; never promote an incomplete search to finite UNSAT.
    """
    bit_for_site = {site: 1 << i for i, site in enumerate(scenario.sites)}
    full = (1 << len(scenario.sites)) - 1
    by_count: list[set[int]] = [{0}] + [set() for _ in range(scenario.budget)]
    for crew in scenario.crews:
        crew_bits = 0
        for site in crew:
            crew_bits |= bit_for_site[site]
        for count in range(scenario.budget - 1, -1, -1):
            for occupied in by_count[count]:
                if not occupied & crew_bits:
                    next_mask = occupied | crew_bits
                    if next_mask == full:
                        return True
                    by_count[count + 1].add(next_mask)
        if sum(map(len, by_count)) > 250_000:
            return None
    return False


def _exact_cover_case(scenario: cover.Scenario, *, variant: int) -> dict:
    """Independently recheck a declared exact-cover win or finite UNSAT claim.

    A source-public symbolic oracle is an independently implemented control,
    not a model grader or proof of neural generalization.
    """
    raw = cover.search_allocation(scenario)
    solved = raw if type(raw) is dict else {}
    raw_status = solved.get("verdict")
    status = raw_status if type(raw_status) is str else "INVALID_SOLVER_VERDICT"
    raw_allocation = solved.get("allocation")
    valid_allocation = (type(raw_allocation) is tuple
                        and len(raw_allocation) <= scenario.budget
                        and all(type(index) is int for index in raw_allocation))
    allocation = raw_allocation if valid_allocation else None
    raw_independent_exists = cover.independent_subset_oracle(scenario)
    independent_exists = (raw_independent_exists
                          if type(raw_independent_exists) is bool else None)
    raw_replay = (cover.verify_allocation(scenario, allocation)
                  if allocation is not None else None)
    raw_replay_verdict = (raw_replay.get("verdict")
                          if type(raw_replay) is dict else None)
    replay_verdict = (raw_replay_verdict
                      if type(raw_replay_verdict) is str else "NOT_RUN")
    independent_witness_valid = _independent_cover_witness(scenario, allocation)
    dp_exists = (_independent_cover_exists_dp(scenario)
                 if status == "PROVEN_UNSAT_UNDER_DECLARED_RULES" else None)
    if status == "VERIFIED_WIN":
        agrees = (independent_exists is True
                  and replay_verdict == "VERIFIED_WIN"
                  and independent_witness_valid)
    elif status == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        agrees = (independent_exists is False and dp_exists is False
                  and allocation is None
                  and solved.get("exhaustive") is True
                  and type(solved.get("exhaustive")) is bool)
    else:
        agrees = False
    if agrees:
        verdict = status
    elif status == "NO_KNOWN_WIN" and solved.get("exhaustive") is False:
        verdict = "NO_KNOWN_WIN"
    else:
        verdict = "UNKNOWN"
    raw_fixed_result = cover.verify_allocation(scenario, (0, 1))
    raw_fixed = (raw_fixed_result.get("verdict")
                 if type(raw_fixed_result) is dict else None)
    fixed = raw_fixed if type(raw_fixed) is str else "INVALID_FIXED_SCRIPT_VERDICT"
    return {
        "verdict": verdict,
        "solver_verdict": status,
        "independent_oracle_exists": independent_exists,
        "independent_witness_valid": independent_witness_valid,
        "independent_dp_exists_on_unsat_claim": dp_exists,
        "independent_oracle_agrees": agrees,
        "independent_replay_verdict": replay_verdict,
        "exhaustive_search": (solved.get("exhaustive")
                              if type(solved.get("exhaustive")) is bool else None),
        "searched_states": _bounded_count(solved.get("searched_states")),
        "rules_fingerprint": cover.fingerprint(scenario),
        "public_prompt": cover.public_view(scenario, variant=variant % 8)["prompt"],
        "declared_sites": list(scenario.sites),
        "declared_crews": [list(crew) for crew in scenario.crews],
        "action_budget": scenario.budget,
        "fixed_script_verdict": fixed,
        "symbolic_allocation": list(allocation) if agrees and allocation is not None else None,
    }


def _temporal_repair_case(scenario: object, *, variant: int) -> dict:
    """Check each emitted schedule, not just existence of *some* win.

    This import is intentionally optional: legacy v1/v2 must not depend on a
    fourth-family implementation, a trained model, or external dependencies.
    """
    from . import temporal_repair_tribunal as temporal

    raw = temporal.search_schedule(scenario)
    solved = raw if type(raw) is dict else {}
    raw_status = solved.get("verdict")
    status = raw_status if type(raw_status) is str else "INVALID_SOLVER_VERDICT"
    raw_schedule = solved.get("schedule")
    schedule = (raw_schedule if type(raw_schedule) is tuple
                and 1 <= len(raw_schedule) <= scenario.action_budget
                and all(type(action) is int and -1 <= action < len(scenario.jobs)
                        for action in raw_schedule) else None)
    rules_tag = temporal.fingerprint(scenario)
    direct = (temporal.verify_schedule(scenario, schedule)
              if schedule is not None else None)
    direct_verdict = (direct.get("verdict") if type(direct) is dict else None)
    replay_verdict = (direct_verdict if type(direct_verdict) is str
                      else "NOT_RUN")
    replay_tag = (direct.get("rules_fingerprint")
                  if type(direct) is dict else None)
    direct_tag_agrees = (type(replay_tag) is str and replay_tag == rules_tag)
    raw_specific = (temporal.independent_witness_check(scenario, schedule)
                    if schedule is not None else None)
    specific_witness_valid = (raw_specific if type(raw_specific) is bool else None)
    raw_exists = temporal.independent_reachability(scenario)
    exists = raw_exists if type(raw_exists) is bool else None
    raw_dfs = (temporal.independent_depth_enumeration(scenario)
               if status == "PROVEN_UNSAT_UNDER_DECLARED_RULES" else None)
    dfs_exists = raw_dfs if type(raw_dfs) is bool else None
    exhaustive = (solved.get("exhaustive")
                  if type(solved.get("exhaustive")) is bool else None)
    solver_tag = solved.get("rules_fingerprint")
    source_tag_agrees = type(solver_tag) is str and solver_tag == rules_tag
    if status == "VERIFIED_WIN":
        agrees = (source_tag_agrees and direct_tag_agrees
                  and replay_verdict == "VERIFIED_WIN"
                  and specific_witness_valid is True and exists is True)
    elif status == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        agrees = (source_tag_agrees and schedule is None and exhaustive is True
                  and exists is False and dfs_exists is False)
    else:
        agrees = False
    if agrees:
        verdict = status
    elif status == "NO_KNOWN_WIN" and exhaustive is False:
        verdict = "NO_KNOWN_WIN"
    else:
        verdict = "UNKNOWN"
    public = temporal.public_view(scenario, variant=variant % 8)
    return {
        "verdict": verdict,
        "solver_verdict": status,
        "independent_oracle_agrees": agrees,
        "independent_existence_has_win": exists,
        "independent_dfs_has_win_on_unsat_claim": dfs_exists,
        "independent_specific_witness_valid": specific_witness_valid,
        "independent_replay_verdict": replay_verdict,
        "rules_fingerprint": rules_tag,
        "public_prompt": public["prompt"],
        "declared_jobs": public["jobs"],
        "machine_capacity": scenario.capacity,
        "time_horizon": scenario.horizon,
        "action_budget": scenario.action_budget,
        "searched_states": _bounded_count(solved.get("searched_states")),
        "exhaustive_search": exhaustive,
        "symbolic_schedule": list(schedule) if agrees and schedule is not None else None,
    }


def run_public_exam(*, seed: int = 2026, variant: int = 0,
                    include_exact_cover: bool = False,
                    include_temporal_repair: bool = False) -> dict:
    """Produce a source-public receipt; v1 stays byte-for-byte stable by default.

    The seed and variant alter wording/perspective only. They do not create
    independent structural trials or protected evaluation holdouts.
    """
    if type(seed) is not int or not -(2**31) <= seed <= 2**31 - 1:
        raise ValueError("seed must be a signed 32-bit integer")
    if type(variant) is not int or not 0 <= variant <= 31:
        raise ValueError("variant must be between 0 and 31")
    if type(include_exact_cover) is not bool:
        raise ValueError("include_exact_cover must be an exact bool")
    if type(include_temporal_repair) is not bool:
        raise ValueError("include_temporal_repair must be an exact bool")
    if include_temporal_repair and not include_exact_cover:
        raise ValueError("--temporal-repair requires --exact-cover")

    constant_cases = {
        kind: _constant_case(seed, variant, kind)
        for kind in ("apparently_sealed", "sealed_control")
    }
    veiled_cases = {
        name: _fork_case(scenario)
        for name, scenario in fork.families().items()
    }
    cover_cases = ({name: _exact_cover_case(scenario, variant=variant)
                    for name, scenario in cover.families().items()}
                   if include_exact_cover else {})
    temporal_cases = {}
    if include_temporal_repair:
        from . import temporal_repair_tribunal as temporal
        temporal_cases = {
            name: _temporal_repair_case(scenario, variant=variant)
            for name, scenario in temporal.families().items()
        }
    all_cases = [*constant_cases.values(), *veiled_cases.values(),
                 *cover_cases.values(), *temporal_cases.values()]
    counts = Counter(c["verdict"] for c in all_cases)
    families = {
        "constant_exam": {
            "challenge_mechanics": "finite_navigation_and_offering",
            "source_fingerprint": constant.ruleset_source_fingerprint(),
            "cases": constant_cases,
        },
        "veiled_fork": {
            "challenge_mechanics": "contingent_epistemic_planning",
            "cases": veiled_cases,
        },
    }
    if include_exact_cover:
        families["exact_cover"] = {
            "challenge_mechanics": "combinatorial_exact_cover_allocation",
            "cases": cover_cases,
            "fixed_script_control": cover.fixed_script_control(),
        }
    if include_temporal_repair:
        families["temporal_repair"] = {
            "challenge_mechanics": "temporal_deadline_energy_repair_scheduling",
            "cases": temporal_cases,
            "fixed_script_control": temporal.fixed_script_control(),
        }
    parameters = {"seed": seed, "variant": variant}
    if include_exact_cover:
        parameters["include_exact_cover"] = True
    if include_temporal_repair:
        parameters["include_temporal_repair"] = True
    return {
        "protocol": ("minotaur-source-public-v3" if include_temporal_repair
                     else "minotaur-source-public-v2" if include_exact_cover
                     else "minotaur-source-public-v1"),
        "parameters": parameters,
        "claim_scope": {
            "model_identity": "built-in deterministic symbolic controls",
            "neural_weight_updates": 0,
            "neural_learning_tested": False,
            "hidden_custody_tested": False,
            "structural_transfer_tested": False,
            "independent_cases_across_paraphrase_seeds": False,
            "ruleset_scope": "declared source-public finite transitions only",
        },
        "families": families,
        "summary": {
            "families_run": len(families),
            "cases_run": len(all_cases),
            "verdict_counts": dict(sorted(counts.items())),
            "independent_checks_agree": all(
                case["independent_check_agrees"] for case in constant_cases.values()
            ) and all(case["independent_oracle_agrees"] for case in veiled_cases.values()
            ) and all(case["independent_oracle_agrees"] for case in cover_cases.values()
            ) and all(case["independent_oracle_agrees"] for case in temporal_cases.values()),
        },
    }
