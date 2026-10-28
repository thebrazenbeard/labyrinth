"""Fail-closed, dependency-free evaluation of printed register-program traces.

This module is separate from dataset generators and model code. It uses only
the program text and displayed states to derive a first-error label, then
cross-checks claimed labels before scoring predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

SCHEMA = "labyrinth.minotaur-independent-eval.v1"
MAX_CASES, MAX_TEXT, MAX_STEPS = 4096, 8192, 32
_INITIAL = re.compile(r"Initial registers: A=([0-9]{1,2}), B=([0-9]{1,2})")
_INSTRUCTION = re.compile(r"([1-9][0-9]*)\. (ADD|SUB|MUL) ([AB]) ([AB]|[0-9]{1,2})")
_STEP = re.compile(r"(?:Step |)([1-9][0-9]*): A=([0-9]{1,2}), B=([0-9]{1,2})")
_PREDICTION = re.compile(r"ERROR_STEP:([1-9][0-9]*)", re.ASCII)
_INVALID = object()


@dataclass(frozen=True)
class TraceEvaluationCase:
    """Claimed labels are audited, never used as the scoring oracle."""
    case_id: str
    question: str
    steps: tuple[str, ...]
    claimed_first_error: int | None
    slice_name: str = "unspecified"
    structure: str | None = None


def _label(value: Any, nsteps: int, *, prediction: bool) -> int | None | object:
    if prediction and value is None:
        return None
    if not prediction and value is None:
        return 0
    if type(value) is int and not isinstance(value, bool):
        return value if 0 <= value <= nsteps else _INVALID
    if prediction and type(value) is str:
        if value == "ALL_CORRECT":
            return 0
        found = _PREDICTION.fullmatch(value)
        if found is not None:
            parsed = int(found[1])
            return parsed if parsed <= nsteps else _INVALID
    return _INVALID


def _parse_and_replay(question: str, steps: tuple[str, ...]) -> tuple[int, str]:
    """Independently interpret the narrow printed language, returning label/shape."""
    if (type(question) is not str or len(question) > MAX_TEXT or type(steps) is not tuple
            or not 1 <= len(steps) <= MAX_STEPS
            or any(type(s) is not str or len(s) > 128 for s in steps)):
        raise ValueError("malformed_or_oversize_case")
    lines = question.splitlines()
    if len(lines) < 2 or any(not line for line in lines):
        raise ValueError("malformed_program_text")
    initial_at = [i for i, line in enumerate(lines) if _INITIAL.fullmatch(line)]
    if initial_at != [1]:
        raise ValueError("missing_or_ambiguous_initial_state")
    start = _INITIAL.fullmatch(lines[1])
    assert start is not None
    instructions = []
    for index, line in enumerate(lines[2:], 1):
        match = _INSTRUCTION.fullmatch(line)
        if match is None:
            if index == len(lines[2:]) and line.startswith("What is the final value of A?"):
                break
            raise ValueError("unrecognized_program_line")
        if int(match[1]) != len(instructions) + 1:
            raise ValueError("nonsequential_instruction")
        instructions.append((match[2], match[3], match[4]))
    if len(instructions) != len(steps) or not instructions:
        raise ValueError("trace_and_program_length_mismatch")
    registers = {"A": int(start[1]), "B": int(start[2])}
    if any(value >= 97 for value in registers.values()):
        raise ValueError("state_outside_mod97")
    shape, first_error = [], 0
    for position, ((operation, target, operand), observed) in enumerate(zip(instructions, steps), 1):
        parsed = _STEP.fullmatch(observed)
        if parsed is None or int(parsed[1]) != position:
            raise ValueError("malformed_observed_step")
        observed_state = (int(parsed[2]), int(parsed[3]))
        if any(value >= 97 for value in observed_state):
            raise ValueError("observed_state_outside_mod97")
        operand_value = registers[operand] if operand in registers else int(operand)
        if operand not in registers and operand_value >= 97:
            raise ValueError("literal_outside_mod97")
        before = registers[target]
        registers[target] = ({"ADD": before + operand_value, "SUB": before - operand_value,
                              "MUL": before * operand_value}[operation]) % 97
        if first_error == 0 and observed_state != (registers["A"], registers["B"]):
            first_error = position
        shape.append(f"{operation} {target} {'R' if operand in registers else 'C'}")
    return first_error, ";".join(shape)


def independently_label(question: str, steps: tuple[str, ...]) -> dict[str, Any]:
    """Return a public replay-derived label or a fail-closed abstention."""
    try:
        label, structure = _parse_and_replay(question, steps)
    except (TypeError, ValueError, OverflowError):
        return {"disposition": "ABSTAIN", "first_error_step": None,
                "reason": "malformed_or_unsupported_program"}
    return {"disposition": "ALL_CORRECT" if label == 0 else "ERROR_STEP",
            "first_error_step": None if label == 0 else label,
            "reason": "independently_replayed", "structure": structure}


def _wilson(successes: int, total: int) -> list[float] | None:
    if total < 1:
        return None
    z, p = 1.959963984540054, successes / total
    base = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / base
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / base
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _score(labels: Sequence[int], predictions: Sequence[int | None]) -> dict[str, Any]:
    total, exact, false_approvals, false_rejections, abstentions = len(labels), 0, 0, 0, 0
    confusion: dict[str, int] = {}
    for actual, predicted in zip(labels, predictions):
        if predicted is None:
            abstentions += 1
            key = f"{actual}->ABSTAIN"
        else:
            exact += int(actual == predicted)
            false_approvals += int(actual != 0 and predicted == 0)
            false_rejections += int(actual == 0 and predicted != 0)
            key = f"{actual}->{predicted}"
        confusion[key] = confusion.get(key, 0) + 1
    answered, errors = total - abstentions, sum(actual != 0 for actual in labels)
    clean = total - errors
    return {"total": total, "non_abstentions": answered, "abstentions": abstentions,
            "selective_coverage": answered / total, "exact_first_error_correct": exact,
            "exact_first_error_accuracy": exact / total,
            "selective_exact_first_error_accuracy": exact / answered if answered else None,
            "false_approvals": false_approvals, "false_rejections": false_rejections,
            "false_approval_rate": false_approvals / errors if errors else None,
            "false_rejection_rate": false_rejections / clean if clean else None,
            "confusion": dict(sorted(confusion.items()))}


def evaluate_predictions(cases: Sequence[TraceEvaluationCase], predictions: Mapping[str, Any], *,
                         training_structures: Sequence[str] = (),
                         iid_binomial_attested: bool = False) -> dict[str, Any]:
    """Audit labels, score predictions, and report negative-control baselines.

    Invalid predictions abstain. Any bad claimed label, duplicate ID, malformed
    case, or false supplied shape refuses the whole batch. Confidence intervals
    are omitted unless callers explicitly attest IID Bernoulli sampling.
    """
    if (not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(case) is not TraceEvaluationCase for case in cases)
            or not isinstance(predictions, Mapping) or type(iid_binomial_attested) is not bool
            or not isinstance(training_structures, (list, tuple))
            or any(type(item) is not str or len(item) > 256 for item in training_structures)):
        raise ValueError("invalid_evaluation_container")
    failures, seen, audited = [], set(), []
    for case in cases:
        if (type(case.case_id) is not str or not case.case_id or len(case.case_id) > 128
                or case.case_id in seen or type(case.slice_name) is not str or len(case.slice_name) > 64):
            failures.append({"case_id": str(case.case_id), "reason": "invalid_or_duplicate_case_id"})
            continue
        seen.add(case.case_id)
        try:
            actual, structure = _parse_and_replay(case.question, case.steps)
        except (TypeError, ValueError, OverflowError):
            failures.append({"case_id": case.case_id, "reason": "malformed_or_unsupported_program"})
            continue
        claimed = _label(case.claimed_first_error, len(case.steps), prediction=False)
        if claimed is _INVALID or claimed != actual:
            failures.append({"case_id": case.case_id, "reason": "claimed_label_disagrees_with_independent_replay"})
        elif case.structure is not None and (type(case.structure) is not str or case.structure != structure):
            failures.append({"case_id": case.case_id, "reason": "declared_structure_disagrees_with_program"})
        else:
            audited.append((case, actual, structure))
    overlap = sorted({shape for _, _, shape in audited} & set(training_structures))
    audit = {"total_cases": len(cases), "audited_cases": len(audited), "claimed_label_failures": failures,
             "derived_structures": sorted({shape for _, _, shape in audited}),
             "training_structure_overlap": overlap, "training_structure_overlap_count": len(overlap)}
    base = {"schema": SCHEMA, "label_source": "independent_program_replay", "audit": audit,
            "reasoning_qualification": "NOT_ESTABLISHED", "graduation_authority": "none"}
    if failures:
        return {**base, "disposition": "REFUSED", "reason": "label_or_input_audit_failed"}
    labels = [actual for _, actual, _ in audited]
    parsed = [_label(predictions.get(case.case_id), len(case.steps), prediction=True) for case, _, _ in audited]
    normalized = [None if value is _INVALID else value for value in parsed]
    report = _score(labels, normalized)
    controls = {"always_yes": _score(labels, [0] * len(labels)),
                "always_no_step_1": _score(labels, [1] * len(labels)),
                "position_cycle": _score(labels, [(i % len(case.steps)) + 1
                                                    for i, (case, _, _) in enumerate(audited)])}
    cis: Any = "not_computed_without_iid_binomial_attestation"
    if iid_binomial_attested:
        cis = {"accuracy_wilson_95": _wilson(report["exact_first_error_correct"], report["total"]),
               "coverage_wilson_95": _wilson(report["non_abstentions"], report["total"]),
               "condition": "conditional_on_caller_iid_binomial_attestation"}
    return {**base, "disposition": "SCORED", "metrics": report, "negative_controls": controls,
            "confidence_intervals": cis,
            "limitations": ["Public printed programs are not a private blind holdout.",
                            "Shape overlap is not proof of semantic dependence."]}
