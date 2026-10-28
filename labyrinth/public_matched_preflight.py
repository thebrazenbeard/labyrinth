"""Public development-only preflight for matched Minotaur evaluations.

This module checks a *declared* separation of training and evaluation task
families and the shape of four-arm scores.  It has no private oracle, evaluator
key, executable model, trusted process observer, or graduation authority.
Its inputs are self-reported and must never be presented as an independent
qualification receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA = "labyrinth.public-matched-preflight.v1"
ARMS = ("candidate", "baseline", "zero_update", "corrupted")
SLICES = ("heldout", "transfer")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CASE_KEYS = frozenset({"case_id", "family_id", "prompt_sha256"})
_TRIAL_KEYS = frozenset({
    "case_id", "prompt_sha256", "semantic_correct", "format_valid", "response_sha256",
})
_RUN_KEYS = frozenset({"checkpoint_sha256", "fresh_process_reload_reported", "trials"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _case_list(cases: Any, name: str) -> list[dict[str, str]]:
    if type(cases) is not list or not 1 <= len(cases) <= 500:
        raise ValueError(f"{name} requires 1–500 declared cases")
    for case in cases:
        if type(case) is not dict or case.keys() != _CASE_KEYS:
            raise ValueError(f"{name}: unexpected case schema")
        if (any(type(case[field]) is not str or not _ID.fullmatch(case[field])
                for field in ("case_id", "family_id"))
                or type(case["prompt_sha256"]) is not str
                or not _DIGEST.fullmatch(case["prompt_sha256"])):
            raise ValueError(f"{name}: invalid identity or prompt commitment")
    return [{field: case[field] for field in sorted(_CASE_KEYS)} for case in cases]


def preregister_public_split(training: Any, slices: Any) -> dict[str, Any]:
    """Return a content-bound *public* plan after declared family separation.

    Family labels are supplied by the author and **not independently verified**.
    Only an external examiner with private case custody could establish that
    distinct labels really designate independent structures.
    """
    train = _case_list(training, "training")
    if type(slices) is not dict or set(slices) != set(SLICES):
        raise ValueError("exactly heldout and transfer slices are required")
    eval_slices = {name: _case_list(slices[name], name) for name in SLICES}
    all_cases = train + [case for name in SLICES for case in eval_slices[name]]
    ids = [case["case_id"] for case in all_cases]
    prompts = [case["prompt_sha256"] for case in all_cases]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("duplicate case or exact prompt across training/evaluation")
    families = [set(case["family_id"] for case in cases)
                for cases in (train, *(eval_slices[name] for name in SLICES))]
    for name in SLICES:
        declared = [case["family_id"] for case in eval_slices[name]]
        if len(set(declared)) != len(declared):
            raise ValueError(f"{name}: repeated family inflates independent case count")
    if any(families[i] & families[j] for i in range(len(families))
           for j in range(i + 1, len(families))):
        raise ValueError("task family overlaps training, heldout, or transfer")
    subject = {"schema": SCHEMA, "training": train, "slices": eval_slices}
    return {**subject, "plan_sha256": hashlib.sha256(
        b"labyrinth.public-matched-preflight.v1\0" + _canonical(subject)
    ).hexdigest()}


def score_public_matched(plan: Any, runs: Any) -> dict[str, Any]:
    """Check matched self-reported arms; *never* return graduation.

    ``fresh_process_reload_reported`` is deliberately named as a claim. This
    function cannot observe model execution, checkpoints, or process isolation.
    Semantic and format validity stay separate so format repair does not look
    like substantive learning.
    """
    if (type(plan) is not dict or set(plan) !=
            {"schema", "training", "slices", "plan_sha256"}
            or type(plan["schema"]) is not str or plan["schema"] != SCHEMA
            or type(plan["plan_sha256"]) is not str
            or not _DIGEST.fullmatch(plan["plan_sha256"])):
        raise ValueError("invalid public pre-registration plan")
    expected = preregister_public_split(plan["training"], plan["slices"])
    if plan != expected:
        raise ValueError("public plan changed after pre-registration")
    if type(runs) is not dict or set(runs) != set(ARMS):
        raise ValueError("all four matched arms are required")
    expected_ids = [case["case_id"] for name in SLICES for case in plan["slices"][name]]
    expected_prompts = {
        case["case_id"]: case["prompt_sha256"]
        for name in SLICES for case in plan["slices"][name]
    }
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    reload_claims = []
    checkpoints = []
    for arm in ARMS:
        run = runs[arm]
        if type(run) is not dict or run.keys() != _RUN_KEYS:
            raise ValueError(f"{arm}: invalid run schema")
        checkpoint = run["checkpoint_sha256"]
        if type(checkpoint) is not str or not _DIGEST.fullmatch(checkpoint):
            raise ValueError(f"{arm}: invalid checkpoint identity")
        checkpoints.append(checkpoint)
        if type(run["fresh_process_reload_reported"]) is not bool:
            raise ValueError(f"{arm}: reload report must be boolean")
        reload_claims.append(run["fresh_process_reload_reported"])
        trials = run["trials"]
        if type(trials) is not list or len(trials) != len(expected_ids):
            raise ValueError(f"{arm}: unmatched case count")
        by_id = {}
        for trial in trials:
            if type(trial) is not dict or trial.keys() != _TRIAL_KEYS:
                raise ValueError(f"{arm}: invalid scored trial")
            case_id = trial["case_id"]
            if (type(case_id) is not str or case_id in by_id
                    or type(trial["prompt_sha256"]) is not str
                    or trial["prompt_sha256"] != expected_prompts.get(case_id)
                    or type(trial["semantic_correct"]) is not bool
                    or type(trial["format_valid"]) is not bool
                    or type(trial["response_sha256"]) is not str
                    or not _DIGEST.fullmatch(trial["response_sha256"])):
                raise ValueError(f"{arm}: invalid trial values or repeated case")
            by_id[case_id] = trial
        if set(by_id) != set(expected_ids):
            raise ValueError(f"{arm}: cases are not exactly matched")
        predictions[arm] = by_id
    if checkpoints[0] in checkpoints[1:]:
        raise ValueError("candidate checkpoint duplicates a control")
    for case_id in expected_ids:
        seen_responses = {}
        for arm in ARMS:
            trial = predictions[arm][case_id]
            digest = trial["response_sha256"]
            scored = (trial["semantic_correct"], trial["format_valid"])
            if digest in seen_responses and seen_responses[digest] != scored:
                raise ValueError("identical response has contradictory paired scores")
            seen_responses[digest] = scored
    result_slices = {}
    for name in SLICES:
        ids = [case["case_id"] for case in plan["slices"][name]]
        arm_results = {}
        for arm in ARMS:
            trials = [predictions[arm][case_id] for case_id in ids]
            arm_results[arm] = {
                "semantic_accuracy": sum(t["semantic_correct"] for t in trials) / len(ids),
                "format_rate": sum(t["format_valid"] for t in trials) / len(ids),
                "strict_accuracy": sum(t["semantic_correct"] and t["format_valid"]
                                       for t in trials) / len(ids),
            }
        paired = {}
        for arm in ARMS[1:]:
            paired[arm] = {
                "candidate_only_correct": sum(
                    predictions["candidate"][case_id]["semantic_correct"]
                    and not predictions[arm][case_id]["semantic_correct"] for case_id in ids),
                "control_only_correct": sum(
                    predictions[arm][case_id]["semantic_correct"]
                    and not predictions["candidate"][case_id]["semantic_correct"] for case_id in ids),
            }
        result_slices[name] = {"cases": len(ids), "arms": arm_results, "paired": paired}
    return {
        "schema": SCHEMA,
        "status": "INTERNAL_CHECKS_PASS" if all(reload_claims) else "INCOMPLETE",
        "plan_sha256": plan["plan_sha256"],
        "slices": result_slices,
        "qualification": "NOT_ESTABLISHED",
        "promotion_allowed": False,
        "authenticated_model_execution": False,
        "independent_private_evaluator": False,
        "limitations": [
            "Case-family labels and reloads are self-reported, not independently established.",
            "Public score checks neither seal an oracle nor authenticate actual inference.",
            "No score or self-reported result from this helper can graduate a model.",
        ],
    }
