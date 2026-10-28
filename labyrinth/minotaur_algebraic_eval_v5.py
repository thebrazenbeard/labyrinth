"""Independent, fail-closed evaluation for Minotaur V5 atomic arithmetic.

This module is intentionally dependency-free and imports neither training code,
dataset generation code, nor a model.  Arithmetic is recomputed directly from
the evaluated inputs.  Consequently a supplied label is only an auditable
claim, never an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "labyrinth.minotaur-algebraic-eval.v5"
MODULUS = 97
MAX_CASES = 4096
MAX_PROGRAM_STEPS = 6
_OPS = frozenset(("ADD", "SUB", "MUL"))
_BAD = object()


@dataclass(frozen=True)
class AtomicEvaluationCase:
    """One held-out atomic transition; ``claimed_output`` is audited strictly."""
    case_id: str
    op: str
    x: int
    y: int
    claimed_output: int
    group_id: str = "ungrouped"


@dataclass(frozen=True)
class ProgramRolloutCase:
    """A short public trace for evaluating a model's atomic callback.

    Instructions are ``(op, target, operand)`` triples.  ``target`` is A/B;
    operand is A/B or an integer in the finite modulo-97 domain.
    """
    case_id: str
    initial_state: tuple[int, int]
    instructions: tuple[tuple[str, str, str | int], ...]
    observed_states: tuple[tuple[int, int], ...]
    claimed_first_error: int | None
    group_id: str = "ungrouped"


def _is_int(value: Any) -> bool:
    return type(value) is int and not isinstance(value, bool)


def _valid_id(value: Any) -> bool:
    return type(value) is str and 0 < len(value) <= 128


def _valid_value(value: Any) -> bool:
    return _is_int(value) and 0 <= value < MODULUS


def _arithmetic(op: str, x: int, y: int) -> int:
    """The evaluator's local arithmetic implementation, never imported elsewhere."""
    if op == "ADD":
        return (x + y) % MODULUS
    if op == "SUB":
        return (x - y) % MODULUS
    if op == "MUL":
        return (x * y) % MODULUS
    raise ValueError("unsupported_operation")


def _atomic_case_failure(case: AtomicEvaluationCase, seen: set[str]) -> str | None:
    if not _valid_id(case.case_id) or case.case_id in seen:
        return "invalid_or_duplicate_case_id"
    seen.add(case.case_id)
    if type(case.op) is not str or case.op not in _OPS or not _valid_value(case.x) or not _valid_value(case.y):
        return "malformed_atomic_inputs"
    if not _valid_value(case.claimed_output):
        return "malformed_claimed_output"
    if type(case.group_id) is not str or not case.group_id or len(case.group_id) > 128:
        return "invalid_group_id"
    if case.claimed_output != _arithmetic(case.op, case.x, case.y):
        return "claimed_output_disagrees_with_recomputed_arithmetic"
    return None


def _metric(actual: Sequence[int], predicted: Sequence[int | None]) -> dict[str, Any]:
    total = len(actual)
    abstentions = sum(p is None for p in predicted)
    exact = sum(a == p for a, p in zip(actual, predicted) if p is not None)
    return {
        "total": total,
        "non_abstentions": total - abstentions,
        "abstentions": abstentions,
        "coverage": (total - abstentions) / total if total else None,
        "exact_correct": exact,
        "exact_accuracy": exact / total if total else None,
        "selective_exact_accuracy": exact / (total - abstentions) if total != abstentions else None,
    }


def _parse_atomic_prediction(value: Any) -> int | None | object:
    if value is None:
        return None
    return value if _valid_value(value) else _BAD


def _validate_prediction_ids(predictions: Mapping[str, Any], expected_ids: set[str]) -> list[dict[str, str]]:
    keys = set(predictions.keys())
    failures: list[dict[str, str]] = []
    if keys != expected_ids:
        for case_id in sorted(expected_ids - keys):
            failures.append({"case_id": case_id, "reason": "missing_prediction_id"})
        for case_id in sorted(keys - expected_ids, key=lambda x: repr(x)):
            failures.append({"case_id": str(case_id), "reason": "unexpected_prediction_id"})
    return failures


def _normalise_training_atoms(training_atoms: Sequence[tuple[str, int, int]]) -> set[tuple[str, int, int]]:
    if not isinstance(training_atoms, (list, tuple)) or len(training_atoms) > 100_000:
        raise ValueError("invalid_training_atoms")
    result = set()
    for item in training_atoms:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("invalid_training_atom")
        op, x, y = item
        if type(op) is not str or op not in _OPS or not _valid_value(x) or not _valid_value(y):
            raise ValueError("invalid_training_atom")
        result.add((op, x, y))
    return result


def evaluate_atomic_predictions(cases: Sequence[AtomicEvaluationCase], predictions: Mapping[str, Any], *,
                                training_atoms: Sequence[tuple[str, int, int]] = (),
                                training_group_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Score held-out atomic values after auditing every input and prediction ID.

    Confidence intervals are intentionally never calculated: cases may be
    paired or share groups, so an IID Bernoulli premise would be invented.
    """
    if (not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(c) is not AtomicEvaluationCase for c in cases)
            or not isinstance(predictions, Mapping)
            or not isinstance(training_group_ids, (list, tuple))
            or any(not _valid_id(group) for group in training_group_ids)):
        raise ValueError("invalid_evaluation_container")
    trained = _normalise_training_atoms(training_atoms)
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    audited: list[AtomicEvaluationCase] = []
    for case in cases:
        reason = _atomic_case_failure(case, seen)
        if reason:
            failures.append({"case_id": str(case.case_id), "reason": reason})
        else:
            audited.append(case)
    expected_ids = {case.case_id for case in audited}
    # A malformed case makes all results invalid, but checking IDs still gives a useful audit.
    failures.extend(_validate_prediction_ids(predictions, {case.case_id for case in cases if _valid_id(case.case_id)}))
    parsed: list[int | None] = []
    for case in audited:
        value = _parse_atomic_prediction(predictions.get(case.case_id))
        if value is _BAD:
            failures.append({"case_id": case.case_id, "reason": "invalid_prediction_value"})
        else:
            parsed.append(value)
    pair_overlap = sorted(set((c.op, c.x, c.y) for c in audited) & trained)
    audit = {
        "total_cases": len(cases), "audited_cases": len(audited), "failures": failures,
        "structural_overlap_audit": {
            "exact_training_pair_overlap": [list(x) for x in pair_overlap],
            "operation_overlap": sorted({c.op for c in audited} & {x[0] for x in trained}),
            "evaluation_group_overlap": sorted({c.group_id for c in audited} & set(training_group_ids)),
        },
    }
    base = {"schema": SCHEMA, "oracle": "local_arithmetic_recomputed_from_op_x_y",
            "audit": audit, "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
            "confidence_intervals": "not_computed_grouped_or_paired_dependence"}
    if failures:
        return {**base, "disposition": "REFUSED"}
    actual = [c.claimed_output for c in audited]
    per_operation = {op: _metric([c.claimed_output for c in audited if c.op == op],
                                  [p for c, p in zip(audited, parsed) if c.op == op])
                     for op in sorted({c.op for c in audited})}
    groups = {
        "unseen_pair": [i for i, c in enumerate(audited) if (c.op, c.x, c.y) not in trained],
        "diagonal": [i for i, c in enumerate(audited) if c.x == c.y],
        "zero_operand": [i for i, c in enumerate(audited) if c.x == 0 or c.y == 0],
    }
    subgroups = {name: _metric([actual[i] for i in indices], [parsed[i] for i in indices])
                 for name, indices in groups.items()}
    # Keep the short name for consumers that describe this slice simply as "zero".
    subgroups["zero"] = subgroups["zero_operand"]
    constants = {str(k): _metric(actual, [k] * len(actual)) for k in range(MODULUS)}
    best_constant = max(range(MODULUS), key=lambda k: (constants[str(k)]["exact_correct"], -k))
    return {**base, "disposition": "SCORED", "metrics": _metric(actual, parsed),
            "per_operation": per_operation, "subgroups": subgroups,
            "structural_overlap_audit": audit["structural_overlap_audit"],
            "baselines": {"always_constant": {"best_constant": best_constant,
                                                  "metrics": constants[str(best_constant)]},
                          "copy_x": _metric(actual, [c.x for c in audited])},
            "limitations": ["Groups and paired inputs may be dependent; no IID confidence is claimed.",
                            "This evaluator establishes neither training provenance nor broad mastery."]}


def _replay_program(case: ProgramRolloutCase) -> tuple[list[tuple[int, int]], int]:
    if (not _valid_id(case.case_id) or type(case.group_id) is not str or not case.group_id
            or len(case.group_id) > 128
            or type(case.initial_state) is not tuple or len(case.initial_state) != 2
            or any(not _valid_value(v) for v in case.initial_state)
            or type(case.instructions) is not tuple or not 1 <= len(case.instructions) <= MAX_PROGRAM_STEPS
            or type(case.observed_states) is not tuple or len(case.observed_states) != len(case.instructions)):
        raise ValueError("malformed_program_case")
    state = {"A": case.initial_state[0], "B": case.initial_state[1]}
    truth: list[tuple[int, int]] = []
    first = 0
    for position, (instruction, observed) in enumerate(zip(case.instructions, case.observed_states), 1):
        if (type(instruction) is not tuple or len(instruction) != 3 or type(observed) is not tuple
                or len(observed) != 2 or any(not _valid_value(v) for v in observed)):
            raise ValueError("malformed_instruction_or_observed_state")
        op, target, operand = instruction
        if (type(op) is not str or op not in _OPS or
                type(target) is not str or target not in ("A", "B") or
                not (operand in ("A", "B") if type(operand) is str else _valid_value(operand))):
            raise ValueError("malformed_instruction")
        y = state[operand] if type(operand) is str else operand
        state[target] = _arithmetic(op, state[target], y)
        expected = (state["A"], state["B"])
        truth.append(expected)
        if not first and observed != expected:
            first = position
    claimed = 0 if case.claimed_first_error is None else case.claimed_first_error
    if not _is_int(claimed) or not 0 <= claimed <= len(case.instructions) or claimed != first:
        raise ValueError("claimed_first_error_disagrees_with_public_replay")
    return truth, first


def _rollout(case: ProgramRolloutCase, predictor: Callable[[str, int, int], int | None]) -> int | None:
    """Run only model outputs; the public arithmetic oracle is not passed here."""
    state = {"A": case.initial_state[0], "B": case.initial_state[1]}
    for position, (op, target, operand) in enumerate(case.instructions, 1):
        y = state[operand] if type(operand) is str else operand
        try:
            output = predictor(op, state[target], y)
        except Exception as exc:  # Model failure cannot be silently converted into correctness.
            raise ValueError("predictor_raised") from exc
        if output is None:
            return None
        if not _valid_value(output):
            raise ValueError("invalid_model_atomic_output")
        state[target] = output
        if (state["A"], state["B"]) != case.observed_states[position - 1]:
            return position
    return 0


def evaluate_program_rollouts(cases: Sequence[ProgramRolloutCase],
                              neural_atomic_predict: Callable[[str, int, int], int | None]) -> dict[str, Any]:
    """Evaluate 1--6 step clean/corrupted traces using an independent replay.

    The callback receives only ``(op, x, y)`` from its own current rollout;
    it never receives symbolic gold states, observed labels, or a gold oracle.
    """
    if (not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(c) is not ProgramRolloutCase for c in cases) or not callable(neural_atomic_predict)):
        raise ValueError("invalid_rollout_container")
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    audited: list[tuple[ProgramRolloutCase, int]] = []
    for case in cases:
        if not _valid_id(case.case_id) or case.case_id in seen:
            failures.append({"case_id": str(case.case_id), "reason": "invalid_or_duplicate_case_id"})
            continue
        seen.add(case.case_id)
        try:
            _, actual = _replay_program(case)
        except (TypeError, ValueError, OverflowError) as exc:
            reason = "claimed_first_error_disagrees_with_public_replay" if str(exc) == "claimed_first_error_disagrees_with_public_replay" else "malformed_or_unsupported_program"
            failures.append({"case_id": case.case_id, "reason": reason})
            continue
        audited.append((case, actual))
    # A malformed batch must never invoke an arbitrary model callback,
    # even for its individually valid cases.
    if failures:
        return {
            "schema": SCHEMA, "oracle": "independent_public_program_replay",
            "audit": {"total_cases": len(cases),
                      "audited_cases": len(audited), "failures": failures},
            "qualification": "NOT_ESTABLISHED",
            "graduation_authority": "none",
            "confidence_intervals": "not_computed_grouped_or_paired_dependence",
            "disposition": "REFUSED",
        }
    predicted: list[int | None] = []
    for case, _ in audited:
        try:
            predicted.append(_rollout(case, neural_atomic_predict))
        except ValueError as exc:
            failures.append({"case_id": case.case_id, "reason": str(exc)})
    base = {"schema": SCHEMA, "oracle": "independent_public_program_replay",
            "audit": {"total_cases": len(cases), "audited_cases": len(audited), "failures": failures},
            "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
            "confidence_intervals": "not_computed_grouped_or_paired_dependence"}
    if failures:
        return {**base, "disposition": "REFUSED"}
    actual = [actual for _, actual in audited]
    metrics = _metric(actual, predicted)
    errors, clean = sum(x != 0 for x in actual), sum(x == 0 for x in actual)
    metrics.update({"exact_first_error_correct": metrics["exact_correct"],
                    "exact_first_error_accuracy": metrics["exact_accuracy"],
                    "false_approvals": sum(a != 0 and p == 0 for a, p in zip(actual, predicted)),
                    "false_rejections": sum(a == 0 and p not in (None, 0) for a, p in zip(actual, predicted)),
                    "false_approval_rate": (sum(a != 0 and p == 0 for a, p in zip(actual, predicted)) / errors) if errors else None,
                    "false_rejection_rate": (sum(a == 0 and p not in (None, 0) for a, p in zip(actual, predicted)) / clean) if clean else None})
    return {**base, "disposition": "SCORED", "metrics": metrics,
            "trace_composition": {"clean": clean, "corrupted": errors},
            "limitations": ["Callback output is evaluated, not trusted as an arithmetic oracle.",
                            "Groups and paired traces may be dependent; no IID confidence is claimed."]}
