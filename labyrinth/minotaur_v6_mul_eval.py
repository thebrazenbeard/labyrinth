"""Fail-closed, independent evaluator for V6 modulo-97 multiplication.

This module deliberately imports neither V6 training/data code nor a model.
Every label is recomputed from public operands before a prediction is scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "labyrinth.minotaur-v6-mul-eval.v1"
MODULUS = 97
MAX_CASES = 4096
MAX_PROGRAM_STEPS = 16


@dataclass(frozen=True)
class MulEvaluationCase:
    case_id: str
    left: int
    right: int
    claimed_output: int
    withheld_group: str = "ungrouped"


@dataclass(frozen=True)
class MulProgramRolloutCase:
    """A public multiplication trace; observations are never sent to callback."""
    case_id: str
    initial_value: int
    operands: tuple[int, ...]
    observed_values: tuple[int, ...]
    claimed_first_error: int | None
    withheld_group: str = "ungrouped"


def _is_int(value: Any) -> bool:
    return type(value) is int


def _value(value: Any) -> bool:
    return _is_int(value) and 0 <= value < MODULUS


def _identifier(value: Any) -> bool:
    return type(value) is str and 0 < len(value) <= 128


def recompute_mul_label(left: int, right: int) -> int:
    """Local evaluator arithmetic, intentionally not delegated to data code."""
    if not _value(left) or not _value(right):
        raise ValueError("malformed_multiplication_operands")
    return (left * right) % MODULUS


def _metric(actual: Sequence[int], predicted: Sequence[int | None]) -> dict[str, Any]:
    total = len(actual)
    abstentions = sum(value is None for value in predicted)
    exact = sum(a == p for a, p in zip(actual, predicted) if p is not None)
    answered = total - abstentions
    return {"total": total, "non_abstentions": answered, "abstentions": abstentions,
            "coverage": answered / total if total else None,
            "exact_correct": exact, "exact_accuracy": exact / total if total else None,
            "selective_exact_accuracy": exact / answered if answered else None}


def _training_pairs(value: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("invalid_training_pairs")
    pairs: set[tuple[int, int]] = set()
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2 or not _value(pair[0]) or not _value(pair[1]):
            raise ValueError("invalid_training_pair")
        pairs.add(pair)
    return pairs


def _validate_prediction_ids(predictions: Mapping[str, Any], expected: set[str]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    keys = set(predictions.keys())
    for case_id in sorted(expected - keys):
        failures.append({"case_id": case_id, "reason": "missing_prediction_id"})
    for case_id in sorted(keys - expected, key=repr):
        failures.append({"case_id": str(case_id), "reason": "unexpected_prediction_id"})
    return failures


def evaluate_mul_predictions(cases: Sequence[MulEvaluationCase], predictions: Mapping[str, Any], *,
                             training_pairs: Sequence[tuple[int, int]] = (),
                             training_group_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Audit and score atomic predictions, including directed and reverse aliases."""
    if (not isinstance(cases, (tuple, list)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(case) is not MulEvaluationCase for case in cases)
            or not isinstance(predictions, Mapping) or not isinstance(training_group_ids, (tuple, list))
            or any(not _identifier(group) for group in training_group_ids)):
        raise ValueError("invalid_evaluation_container")
    trained = _training_pairs(training_pairs)
    failures: list[dict[str, str]] = []
    audited: list[MulEvaluationCase] = []
    seen: set[str] = set()
    for case in cases:
        if not _identifier(case.case_id) or case.case_id in seen:
            failures.append({"case_id": str(case.case_id), "reason": "invalid_or_duplicate_case_id"})
            continue
        seen.add(case.case_id)
        if not _value(case.left) or not _value(case.right):
            failures.append({"case_id": case.case_id, "reason": "malformed_multiplication_operands"})
        elif not _value(case.claimed_output):
            failures.append({"case_id": case.case_id, "reason": "malformed_claimed_output"})
        elif not _identifier(case.withheld_group):
            failures.append({"case_id": case.case_id, "reason": "invalid_withheld_group"})
        elif case.claimed_output != recompute_mul_label(case.left, case.right):
            failures.append({"case_id": case.case_id, "reason": "claimed_output_disagrees_with_recomputed_multiplication"})
        else:
            audited.append(case)
    expected = {case.case_id for case in cases if _identifier(case.case_id)}
    failures.extend(_validate_prediction_ids(predictions, expected))
    parsed: list[int | None] = []
    for case in audited:
        prediction = predictions.get(case.case_id)
        if prediction is not None and not _value(prediction):
            failures.append({"case_id": case.case_id, "reason": "invalid_prediction_value"})
        else:
            parsed.append(prediction)
    exact = sorted((case.left, case.right) for case in audited if (case.left, case.right) in trained)
    aliases = sorted((case.left, case.right) for case in audited
                     if case.left != case.right and (case.left, case.right) not in trained
                     and (case.right, case.left) in trained)
    audit = {"total_cases": len(cases), "audited_cases": len(audited), "failures": failures}
    canonical = sorted({tuple(sorted(pair)) for pair in exact + aliases})
    overlap_audit = {"exact_training_pair_overlap": [list(pair) for pair in exact],
                     "reversed_pair_alias_overlap": [list(pair) for pair in aliases],
                     "canonical_pair_overlap": [list(pair) for pair in canonical],
                     "withheld_group_overlap": sorted({c.withheld_group for c in audited} & set(training_group_ids))}
    base = {"schema": SCHEMA, "label_source": "locally_recomputed_mod97_multiplication",
            "audit": audit, "overlap_audit": overlap_audit, "qualification": "NOT_ESTABLISHED",
            "graduation_authority": "none", "confidence_intervals": "not_computed_grouped_or_paired_dependence"}
    if failures:
        return {**base, "disposition": "REFUSED"}
    actual = [case.claimed_output for case in audited]
    groups = {group: [index for index, case in enumerate(audited) if case.withheld_group == group]
              for group in sorted({case.withheld_group for case in audited})}
    withheld = {group: _metric([actual[i] for i in indices], [parsed[i] for i in indices])
                for group, indices in groups.items()}
    constants = {str(value): _metric(actual, [value] * len(actual)) for value in range(MODULUS)}
    best = max(range(MODULUS), key=lambda value: (constants[str(value)]["exact_correct"], -value))
    return {**base, "disposition": "SCORED", "metrics": _metric(actual, parsed),
            "withheld_groups": withheld,
            "negative_baselines": {"always_constant": {"best_constant": best, "metrics": constants[str(best)]},
                                   "copy_left": _metric(actual, [case.left for case in audited]),
                                   "copy_right": _metric(actual, [case.right for case in audited])},
            "limitations": ["Reverse aliases are audited but overlap does not establish training provenance.",
                            "No IID confidence claim is made for grouped or paired cases."]}


def _replay(case: MulProgramRolloutCase) -> int:
    if (not _identifier(case.case_id) or not _identifier(case.withheld_group) or not _value(case.initial_value)
            or type(case.operands) is not tuple or not 1 <= len(case.operands) <= MAX_PROGRAM_STEPS
            or type(case.observed_values) is not tuple or len(case.observed_values) != len(case.operands)
            or any(not _value(v) for v in case.operands) or any(not _value(v) for v in case.observed_values)):
        raise ValueError("malformed_program_case")
    state, first_error = case.initial_value, 0
    for position, (operand, observed) in enumerate(zip(case.operands, case.observed_values), 1):
        state = recompute_mul_label(state, operand)
        if not first_error and state != observed:
            first_error = position
    claimed = 0 if case.claimed_first_error is None else case.claimed_first_error
    if not _is_int(claimed) or not 0 <= claimed <= len(case.operands) or claimed != first_error:
        raise ValueError("claimed_first_error_disagrees_with_public_replay")
    return first_error


def _rollout(case: MulProgramRolloutCase, callback: Callable[[int, int], int | None]) -> int | None:
    state = case.initial_value
    for position, operand in enumerate(case.operands, 1):
        try:
            output = callback(state, operand)
        except Exception as exc:
            raise ValueError("predictor_raised") from exc
        if output is None:
            return None
        if not _value(output):
            raise ValueError("invalid_model_multiplication_output")
        state = output
        if state != case.observed_values[position - 1]:
            return position
    return 0


def evaluate_mul_program_rollouts(cases: Sequence[MulProgramRolloutCase],
                                  multiplication_callback: Callable[[int, int], int | None]) -> dict[str, Any]:
    """Score free-running multiplication rollouts without exposing gold answers.

    The callback gets only its current ``(state, operand)`` pair.  It never
    receives public observed values, the independently replayed answer, or a
    claimed first-error label.
    """
    if (not isinstance(cases, (tuple, list)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(case) is not MulProgramRolloutCase for case in cases) or not callable(multiplication_callback)):
        raise ValueError("invalid_rollout_container")
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    audited: list[tuple[MulProgramRolloutCase, int]] = []
    for case in cases:
        if not _identifier(case.case_id) or case.case_id in seen:
            failures.append({"case_id": str(case.case_id), "reason": "invalid_or_duplicate_case_id"})
            continue
        seen.add(case.case_id)
        try:
            actual = _replay(case)
        except (TypeError, ValueError, OverflowError) as exc:
            reason = "claimed_first_error_disagrees_with_public_replay" if str(exc) == "claimed_first_error_disagrees_with_public_replay" else "malformed_or_unsupported_program"
            failures.append({"case_id": case.case_id, "reason": reason})
        else:
            audited.append((case, actual))
    predicted: list[int | None] = []
    for case, _ in audited:
        try:
            predicted.append(_rollout(case, multiplication_callback))
        except ValueError as exc:
            failures.append({"case_id": case.case_id, "reason": str(exc)})
    base = {"schema": SCHEMA, "label_source": "independent_public_multiplication_replay",
            "audit": {"total_cases": len(cases), "audited_cases": len(audited), "failures": failures},
            "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
            "confidence_intervals": "not_computed_grouped_or_paired_dependence"}
    if failures:
        return {**base, "disposition": "REFUSED"}
    actual = [value for _, value in audited]
    metrics = _metric(actual, predicted)
    errors, clean = sum(value != 0 for value in actual), sum(value == 0 for value in actual)
    false_approvals = sum(a != 0 and p == 0 for a, p in zip(actual, predicted))
    false_rejections = sum(a == 0 and p not in (None, 0) for a, p in zip(actual, predicted))
    metrics.update({"exact_first_error_correct": metrics["exact_correct"],
                    "exact_first_error_accuracy": metrics["exact_accuracy"],
                    "false_approvals": false_approvals, "false_rejections": false_rejections,
                    "false_approval_rate": false_approvals / errors if errors else None,
                    "false_rejection_rate": false_rejections / clean if clean else None})
    return {**base, "disposition": "SCORED", "metrics": metrics,
            "trace_composition": {"clean": clean, "corrupted": errors},
            "limitations": ["The callback sees only model rollout inputs, never gold answers."]}
