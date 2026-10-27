"""Dependency-free, paired *descriptive* analysis of HF LoRA pilot reports.

This does not run a model or authenticate the reported trial outcomes. An
internally consistent generated report can be fabricated. Paired comparisons
are exploratory until a separately held, independently run evaluation is
recorded against an immutable Minotaur adapter/backbone composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from .hf_pilot import EVAL_SEED_XOR
from .hf_report_audit import audit_hf_report
from .maze import generate_maze
from .runner import load_json, save_json_new


SCHEMA = "labyrinth.hf-paired-effect-analysis.v1"
CONTROLS = ("baseline", "sham", "corrupted")
VALID_REASONS = frozenset({"correct", "incorrect", "invalid_format", "invalid_response"})


def _validate_trials(trials: Any) -> tuple[str, ...]:
    if not isinstance(trials, list) or not 1 <= len(trials) <= 128:
        raise ValueError("trials must be a bounded nonempty list")
    identifiers = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("trial must be an object")
        identifier = trial.get("id")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 256:
            raise ValueError("trial id must be a bounded string")
        identifiers.append(identifier)
        reason = trial.get("score_reason")
        correct = trial.get("correct")
        if (type(correct) is not bool or reason not in VALID_REASONS
                or (reason == "correct") is not correct):
            raise ValueError("inconsistent trial correctness or score reason")
        response_digest = trial.get("response_sha256")
        if not isinstance(response_digest, str) or not response_digest:
            raise ValueError("missing response digest")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate challenge id in evaluation trials")
    return tuple(identifiers)


def _exact_sign_p(candidate_wins: int, reference_wins: int) -> float:
    """One-sided exact paired sign-test tail under exchangeable discordances."""
    discordant = candidate_wins + reference_wins
    return sum(math.comb(discordant, index)
               for index in range(candidate_wins, discordant + 1)) / (2 ** discordant)


def paired_binary_effect(candidate: list[dict[str, Any]],
                         reference: list[dict[str, Any]], *,
                         multiplicity: int = 3) -> dict[str, Any]:
    """Use identical ordered task IDs, not two independent accuracy estimates."""
    if type(multiplicity) is not int or not 1 <= multiplicity <= 20:
        raise ValueError("multiplicity must be an integer in [1, 20]")
    candidate_ids = _validate_trials(candidate)
    reference_ids = _validate_trials(reference)
    if candidate_ids != reference_ids:
        raise ValueError("candidate and reference must use the same unique ordered challenge IDs")

    wins = losses = both_correct = both_wrong = 0
    format_wins = valid_wrong_wins = shared_format_count = 0
    shared_wins = shared_losses = response_changes = 0
    for cand, ref in zip(candidate, reference):
        cand_correct = cand["correct"]
        ref_correct = ref["correct"]
        cand_format = cand["score_reason"] in ("correct", "incorrect")
        ref_format = ref["score_reason"] in ("correct", "incorrect")
        response_changes += cand["response_sha256"] != ref["response_sha256"]
        if cand_format and ref_format:
            shared_format_count += 1
            shared_wins += cand_correct and not ref_correct
            shared_losses += ref_correct and not cand_correct
        if cand_correct and not ref_correct:
            wins += 1
            format_wins += not ref_format
            valid_wrong_wins += ref_format
        elif ref_correct and not cand_correct:
            losses += 1
        elif cand_correct:
            both_correct += 1
        else:
            both_wrong += 1
    one_sided_p = _exact_sign_p(wins, losses)
    total = len(candidate)
    return {
        "total": total, "candidate_correct": wins + both_correct,
        "reference_correct": losses + both_correct,
        "candidate_wins": wins, "reference_wins": losses,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "discordant": wins + losses,
        "paired_accuracy_delta": (wins - losses) / total,
        "one_sided_exact_p": one_sided_p,
        "bonferroni_p": min(1.0, multiplicity * one_sided_p),
        "multiplicity": multiplicity,
        "wins_from_invalid_format": format_wins,
        "wins_against_valid_wrong_answer": valid_wrong_wins,
        "shared_valid_format_total": shared_format_count,
        "shared_valid_format_candidate_wins": shared_wins,
        "shared_valid_format_reference_wins": shared_losses,
        "response_digest_changes": response_changes,
    }


def summarize_hf_effects(arms: dict[str, list[dict[str, Any]]], *,
                         challenge_kinds: dict[str, str], audit_status: str,
                         model_identity: dict[str, Any]) -> dict[str, Any]:
    """Summarize reported data while making the evidence boundary explicit."""
    if not isinstance(arms, dict) or set(arms) != set(CONTROLS) | {"candidate"}:
        raise ValueError("exactly baseline, sham, candidate and corrupted arms required")
    if audit_status not in ("PASS", "INCOMPLETE", "FAIL"):
        raise ValueError("invalid report audit status")
    if not isinstance(model_identity, dict) or not all(
        isinstance(model_identity.get(field), dict) and model_identity[field]
        for field in ("base_weights_sha256", "candidate_adapter_sha256")
    ):
        raise ValueError("base and adapter digest receipts required")
    ids = _validate_trials(arms["candidate"])
    if not isinstance(challenge_kinds, dict) or set(challenge_kinds) != set(ids):
        raise ValueError("challenge family mapping must match candidate IDs")
    if any(not isinstance(kind, str) or not kind for kind in challenge_kinds.values()):
        raise ValueError("invalid challenge family label")
    for arm in CONTROLS:
        if _validate_trials(arms[arm]) != ids:
            raise ValueError("control trial IDs/order differ from candidate")
    comparisons = {
        arm: paired_binary_effect(arms["candidate"], arms[arm], multiplicity=len(CONTROLS))
        for arm in CONTROLS
    }
    families = {}
    for kind in sorted(set(challenge_kinds.values())):
        indices = [index for index, identifier in enumerate(ids) if challenge_kinds[identifier] == kind]
        candidate_correct = sum(arms["candidate"][index]["correct"] for index in indices)
        families[kind] = {
            "total": len(indices),
            "candidate_correct": candidate_correct,
            "candidate_accuracy": candidate_correct / len(indices),
            "control_correct": {
                arm: sum(arms[arm][index]["correct"] for index in indices)
                for arm in CONTROLS
            },
        }
    # Even a spectacular value here is exploratory: compute-matched sham,
    # blinded OOD holdouts, independent model run and provenance are absent.
    all_control_deltas_positive = all(
        comparison["candidate_wins"] > comparison["reference_wins"]
        for comparison in comparisons.values()
    )
    all_controls_adjusted_p_under_005 = all(
        comparison["bonferroni_p"] < 0.05 for comparison in comparisons.values()
    )
    all_controls_have_wrong_answer_recoveries = all(
        comparison["wins_against_valid_wrong_answer"] > 0
        for comparison in comparisons.values()
    )
    # This is a deterministic *composition receipt*, not a signature or proof
    # that any named model files were genuinely read by an inference runtime.
    composition_sha256 = hashlib.sha256(
        json.dumps(model_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "artifact_role": "Minotaur candidate adapter plus pinned backbone",
        "model_identity_receipts": model_identity,
        "composition_receipt_sha256": composition_sha256,
        "evaluation_provenance": "SELF_REPORTED",
        "internal_audit_status": audit_status,
        "comparisons": comparisons,
        "families": families,
        "exploratory_three_control_pattern": (
            audit_status == "PASS" and all_control_deltas_positive
            and all_controls_adjusted_p_under_005
            and all_controls_have_wrong_answer_recoveries
        ),
        "training_mechanism": "NOT_VERIFIED_BY_THIS_ANALYSIS",
        "behavioral_change": "REPORTED_ONLY",
        "reasoning_improvement": "NOT_ESTABLISHED",
        "generalization": "NOT_ESTABLISHED",
        "statistical_method": (
            "Same-ID paired strict-correctness disagreements; one-sided exact "
            "binomial sign test with Bonferroni adjustment for 3 controls. "
            "These calculations assume exchangeable independent challenge pairs; "
            "repeated public templates compromise that assumption."
        ),
        "limitations": [
            "Report audit checks internal consistency, not independent model execution.",
            "Response digests and checkpoint hashes are self-reported, not authenticated.",
            "Zero-update sham is not compute/exposure matched.",
            "Public finite challenge templates permit training/evaluation contamination.",
            "Strict-answer gains can arise from formatting alone; substantive wrong-answer "
            "recoveries are counted separately but do not by themselves prove reasoning.",
            "A single run and no blind external task-family evaluation cannot establish transfer.",
        ],
    }


def analyze_hf_report(report: Any) -> dict[str, Any]:
    """Reconstruct challenge-family labels and attach current internal audit."""
    if not isinstance(report, dict):
        raise ValueError("pilot report must be an object")
    config = report.get("config")
    if not isinstance(config, dict):
        raise ValueError("missing pilot configuration")
    try:
        seed, count = config["seed"], config["eval_count"]
        tiers = tuple(config["tiers"])
        heldout = generate_maze(seed=seed ^ EVAL_SEED_XOR, count=count,
                                tiers=tiers, split="eval")
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("cannot reconstruct evaluation challenges") from exc
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != set(CONTROLS) | {"candidate"}:
        raise ValueError("incomplete HF evaluation arms")
    try:
        arms = {name: evaluations[name]["trials"] for name in (*CONTROLS, "candidate")}
    except (TypeError, KeyError) as exc:
        raise ValueError("missing evaluation trials") from exc
    candidate_trials = arms["candidate"]
    if not isinstance(candidate_trials, list) or any(
        not isinstance(trial, dict) for trial in candidate_trials
    ):
        raise ValueError("malformed candidate evaluation trials")
    if [challenge.id for challenge in heldout] != [trial.get("id") for trial in candidate_trials]:
        raise ValueError("candidate IDs do not match regenerated held-out split")
    identity = {
        "base_weights_sha256": report.get("base_weights_sha256"),
        "model_metadata_sha256": report.get("model_metadata_sha256"),
        "candidate_adapter_sha256": report.get("candidate_adapter_sha256"),
    }
    audit = audit_hf_report(report)
    result = summarize_hf_effects(
        arms,
        challenge_kinds={challenge.id: challenge.kind for challenge in heldout},
        audit_status=audit["status"], model_identity=identity,
    )
    result["internal_audit"] = audit
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired evidence limits for an HF Minotaur candidate")
    parser.add_argument("--input", required=True, help="Existing pilot_report.json")
    parser.add_argument("--report", help="New analysis output; never overwritten")
    args = parser.parse_args(argv)
    try:
        result = analyze_hf_report(load_json(args.input, strict=True))
        if args.report:
            save_json_new(args.report, result)
            print(f"Wrote paired analysis to {args.report}", file=sys.stderr)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["internal_audit_status"] == "PASS" else 2
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
