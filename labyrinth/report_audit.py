"""Offline *internal-consistency* audit of saved Labyrinth reports.

This does NOT authenticate model inference: all keys, answers and digests are
publicly reproducible. Truncated responses cannot be independently rescored.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .evaluation import score_response
from .maze import generate_maze
from .runner import (
    CONTROL_DESCRIPTIONS, REPORT_SCHEMA, _challenge_commitment, _deranged_training_labels,
    _examples_context, _model_seed, _normalized_prompt, _placebo_context,
    _source_state, validate_training_artifact,
)


AUDIT_SCHEMA = "labyrinth.report-audit.v1"
MAX_PREVIEW = 4096
ARM_NAMES = ("baseline", "candidate", "placebo", "shuffled")


def _positive_int(value: Any, maximum: int = 1000) -> bool:
    return type(value) is int and 1 <= value <= maximum


def _int(value: Any) -> bool:
    return type(value) is int


def _valid_tiers(value: Any) -> bool:
    return (isinstance(value, list) and 1 <= len(value) <= 4
            and all(type(t) is int and t in (1, 2, 3, 4) for t in value)
            and len(set(value)) == len(value))


def _proportion_matches(value: Any, expected: float) -> bool:
    return (type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)
            and math.isclose(value, expected, abs_tol=1e-12, rel_tol=0))


def audit_report(report: Any, *, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recompute available public evidence; distinguish FAIL from INCOMPLETE.

    PASS means only that available report fields agree with this source tree's
    deterministic generator and scorer. It says nothing about who generated
    the model response or whether the model learned.
    """
    failures: list[str] = []
    unknown: list[str] = []
    checked_trials = 0
    unverifiable_trials = 0

    def fail(message: str) -> None:
        if len(failures) < 40:
            failures.append(message)

    def incomplete(message: str) -> None:
        if len(unknown) < 40 and message not in unknown:
            unknown.append(message)

    def result() -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEMA,
            "status": "FAIL" if failures else "INCOMPLETE" if unknown else "PASS",
            "meaning": "internal_consistency_only",
            "authenticated_model_execution": False,
            "checked_trials": checked_trials,
            "unverifiable_trials": unverifiable_trials,
            "failures": failures,
            "unchecked": unknown,
            "limitations": [
                "Report origin and model execution are not authenticated by any digest.",
                "The public challenge generator permits fully self-consistent forged reports.",
                "PASS is not evidence of model learning, unseen evaluation, or causal training gains.",
            ],
        }

    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        fail("invalid report schema")
        return result()
    source = report.get("source")
    if not isinstance(source, dict) or source.get("source_tree_sha256") != _source_state()["source_tree_sha256"]:
        fail("source revision hash differs from local auditor source")
    mode = report.get("mode")
    if mode not in ("maze", "controlled-few-shot"):
        fail("unsupported report mode")
        return result()
    params = report.get("parameters")
    if not isinstance(params, dict) or not _valid_tiers(params.get("tiers")):
        fail("invalid report tiers/parameters")
        return result()
    tiers = tuple(params["tiers"])
    if mode == "maze":
        seed, count, split = params.get("seed"), params.get("count"), params.get("split")
        if not _int(seed) or not _positive_int(count) or not isinstance(split, str):
            fail("invalid maze seed/count/split")
            return result()
        arms = {"baseline": report.get("evaluation")}
        prefixes: dict[str, str] = {"baseline": ""}
        evaluation_seed = seed
    else:
        seed = params.get("original_seed")
        evaluation_seed = params.get("eval_seed")
        count = params.get("eval_count")
        split = "eval"
        if not _int(seed) or not _int(evaluation_seed) or evaluation_seed != seed ^ 0x5C18A7 or not _positive_int(count):
            fail("invalid experiment seed/eval_count")
            return result()
        arms = report.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
            fail("experiment must contain four prescribed control arms")
            return result()
        prefixes = {"baseline": ""}
        if artifact is None:
            incomplete("training artifact absent: nonbaseline prompt hashes cannot be reconstructed")
        else:
            try:
                examples = validate_training_artifact(artifact)
                if artifact["integrity_sha256"] != report.get("training_artifact_sha256"):
                    fail("training artifact integrity digest differs from report")
                if report.get("training_count") != len(examples):
                    fail("training count does not match supplied artifact")
                if report.get("training_model_id") != artifact.get("training_model_id"):
                    fail("training model identity differs from supplied artifact")
                if report.get("adaptation_kind") != artifact.get("adaptation_kind"):
                    fail("adaptation kind differs from supplied artifact")
                if report.get("shuffled_label_collisions") != 0 or not _int(report.get("shuffled_label_collisions")):
                    fail("shuffled control has invalid collision count")
                if report.get("controls") != CONTROL_DESCRIPTIONS:
                    fail("shuffled/control descriptions differ from canonical control contract")
                candidate = _examples_context(examples)
                prefixes.update({
                    "candidate": candidate,
                    "placebo": _placebo_context(len(candidate)),
                    "shuffled": _examples_context(_deranged_training_labels(examples)),
                })
            except (ValueError, TypeError, KeyError) as exc:
                fail(f"invalid training artifact: {exc}")
                return result()

    try:
        challenges = generate_maze(seed=evaluation_seed, count=count, tiers=tiers, split=split)
    except (ValueError, TypeError) as exc:
        fail(f"invalid challenge regeneration parameters: {exc}")
        return result()
    if report.get("challenge_commitment_sha256") != _challenge_commitment(challenges):
        fail("challenge answer commitment mismatch")
    if mode == "controlled-few-shot" and artifact is not None:
        train_ids = {ex["challenge_id"] for ex in examples}
        train_prompts = {_normalized_prompt(ex["prompt"]) for ex in examples}
        if any(c.id in train_ids or _normalized_prompt(c.prompt) in train_prompts for c in challenges):
            fail("training/evaluation split overlap")
    adapter = report.get("adapter")
    model_id = report.get("model_id")
    if (type(model_id) is not str or not model_id or not isinstance(adapter, dict)
            or type(adapter.get("id")) is not str or adapter.get("id") != model_id):
        fail("adapter identity differs from report model identity")

    for name, arm in arms.items():
        if not isinstance(arm, dict) or arm.get("arm") != name:
            fail(f"{name}: invalid arm identity")
            continue
        trials = arm.get("trials")
        if not isinstance(trials, list) or len(trials) != len(challenges):
            fail(f"{name}: wrong trial membership or count")
            continue
        if arm.get("total") != len(challenges) or not _int(arm.get("total")):
            fail(f"{name}: invalid total count")
        claimed_points = 0
        claimed_format = 0
        all_claims_numeric = True
        for index, (trial, challenge) in enumerate(zip(trials, challenges)):
            tag = f"{name}[{index}]"
            if not isinstance(trial, dict):
                fail(f"{tag}: trial not an object")
                all_claims_numeric = False
                continue
            for field, expected in (
                ("challenge_id", challenge.id),
                ("model_seed", _model_seed(evaluation_seed, challenge.id)),
                ("tier", challenge.tier),
                ("kind", challenge.kind),
            ):
                actual = trial.get(field)
                if actual != expected or (field in ("model_seed", "tier") and not _int(actual)):
                    fail(f"{tag}: {field} mismatch")
            if name in prefixes:
                prefix = prefixes[name]
                prompt = f"{prefix}{challenge.prompt}" if prefix else challenge.prompt
                expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if trial.get("model_prompt_sha256") != expected_hash:
                    fail(f"{tag}: prompt hash mismatch")
            preview, length = trial.get("response_preview"), trial.get("response_length")
            if not isinstance(preview, str) or not _int(length) or length < 0:
                fail(f"{tag}: invalid response preview/length")
            elif length <= MAX_PREVIEW:
                if len(preview) != length:
                    fail(f"{tag}: response length mismatch")
                if trial.get("response_sha256") != hashlib.sha256(preview.encode("utf-8")).hexdigest():
                    fail(f"{tag}: response hash mismatch")
                expected = score_response(challenge, preview)
                if (type(trial.get("correct")) is not bool or trial.get("correct") != expected.correct
                        or not _int(trial.get("points")) or trial.get("points") != expected.points
                        or trial.get("scoring_reason") != expected.reason):
                    fail(f"{tag}: score claim differs from regenerated scorer")
                checked_trials += 1
            else:
                if len(preview) != MAX_PREVIEW:
                    fail(f"{tag}: truncated response preview size invalid")
                unverifiable_trials += 1
                incomplete(f"{name}: responses over 4096 characters cannot be independently rescored")
            points = trial.get("points")
            if not _int(points) or points not in (0, 1):
                fail(f"{tag}: invalid claimed points")
                all_claims_numeric = False
            else:
                claimed_points += points
            if trial.get("scoring_reason") in ("correct", "incorrect"):
                claimed_format += 1
        if all_claims_numeric and (not _int(arm.get("points")) or arm.get("points") != claimed_points):
            fail(f"{name}: aggregate points mismatch")
        if not _proportion_matches(arm.get("accuracy"), claimed_points / len(challenges)):
            fail(f"{name}: aggregate accuracy mismatch")
        if not _int(arm.get("format_valid")) or arm.get("format_valid") != claimed_format:
            fail(f"{name}: aggregate format_valid mismatch")
        if not _proportion_matches(arm.get("format_adherence"), claimed_format / len(challenges)):
            fail(f"{name}: aggregate format adherence mismatch")
    return result()
