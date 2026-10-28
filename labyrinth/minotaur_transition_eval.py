"""Fail-closed, dependency-free evaluation for Minotaur transition traces.

This is a public arithmetic checker, not a training oracle or graduation gate.
It deliberately imports neither generators nor model code (and therefore has no
Torch dependency).  A prediction may name its provenance, but provenance is a
diagnostic: symbolic gold states are never reported as evidence of learning.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

SCHEMA = "labyrinth.minotaur-transition-eval.v1"
MAX_CASES, MAX_STEPS, MAX_TEXT = 4096, 32, 8192
_INITIAL = re.compile(r"Initial registers: A=([0-9]{1,2}), B=([0-9]{1,2})")
_INSTRUCTION = re.compile(r"([1-9][0-9]*)\. (ADD|SUB|MUL) ([AB]) ([AB]|[0-9]{1,2})")
_STEP = re.compile(r"(?:Step |)([1-9][0-9]*): A=([0-9]{1,2}), B=([0-9]{1,2})")
_VERDICT = re.compile(r"ERROR_STEP:([1-9][0-9]*)", re.ASCII)
_PREAMBLES = {
    "Execute operations.",
    "Execute this register program in order. ADD, SUB, and MUL update the named target register using its current value and the operand. An operand is either a literal integer or the current value of the named register. After each instruction, replace the updated value with its nonnegative remainder modulo 97. Other registers do not change.",
    "Execute this register program in order. Each instruction updates its target using the current operand.",
}


@dataclass(frozen=True)
class TransitionEvaluationCase:
    case_id: str
    question: str
    observed_steps: tuple[str, ...]
    claimed_first_error: int | None
    group_id: str = "ungrouped"
    declared_structure: str | None = None


@dataclass(frozen=True)
class TransitionPrediction:
    """A candidate's states plus its trace-verdict; ``None`` verdict abstains."""
    predicted_states: tuple[tuple[int, int], ...] | None
    predicted_first_error: int | str | None
    source_provenance: str = "model_inference"


def _parse_verdict(value: Any, steps: int, *, claimed: bool) -> int | None | object:
    # Claimed None is the only canonical representation of all-correct.
    if claimed:
        if value is None:
            return 0
        return value if type(value) is int and not isinstance(value, bool) and 1 <= value <= steps else _BAD
    if value is None:
        return None
    if type(value) is int and not isinstance(value, bool):
        return value if 0 <= value <= steps else _BAD
    if type(value) is str:
        if value == "ALL_CORRECT":
            return 0
        match = _VERDICT.fullmatch(value)
        if match and int(match[1]) <= steps:
            return int(match[1])
    return _BAD


_BAD = object()


def _replay(question: str, observed_steps: tuple[str, ...]) -> tuple[tuple[tuple[int, int], ...], int, str]:
    if type(question) is not str or len(question) > MAX_TEXT or type(observed_steps) is not tuple:
        raise ValueError("malformed_program_or_trace")
    if not 1 <= len(observed_steps) <= MAX_STEPS or any(type(x) is not str or len(x) > 128 for x in observed_steps):
        raise ValueError("invalid_trace_shape")
    lines = question.splitlines()
    if len(lines) < 3 or any(not line for line in lines):
        raise ValueError("malformed_program_text")
    starts = [i for i, line in enumerate(lines) if _INITIAL.fullmatch(line)]
    valid_single = len(starts) == 1 and starts[0] == 1 and lines[0] in _PREAMBLES
    valid_double = (len(starts) == 1 and starts[0] == 2
                    and lines[0] == "Execute this register program in order. Each instruction updates its target using the current operand."
                    and lines[1] == "After every instruction take the nonnegative remainder modulo 97.")
    if not (valid_single or valid_double):
        raise ValueError("unsupported_program_grammar")
    start = _INITIAL.fullmatch(lines[starts[0]])
    assert start
    program = lines[starts[0] + 1:]
    instructions = []
    for position, line in enumerate(program, 1):
        match = _INSTRUCTION.fullmatch(line)
        if not match:
            if position == len(program) and line.startswith("What is the final value of A?"):
                break
            raise ValueError("unsupported_program_line")
        if int(match[1]) != len(instructions) + 1:
            raise ValueError("nonsequential_instruction")
        instructions.append((match[2], match[3], match[4]))
    if len(instructions) != len(observed_steps) or not instructions:
        raise ValueError("program_trace_length_mismatch")
    state = {"A": int(start[1]), "B": int(start[2])}
    if any(x >= 97 for x in state.values()):
        raise ValueError("initial_state_outside_mod97")
    truth, first_error, shape = [], 0, []
    for index, ((op, target, operand), observed) in enumerate(zip(instructions, observed_steps), 1):
        shown = _STEP.fullmatch(observed)
        if not shown or int(shown[1]) != index or int(shown[2]) >= 97 or int(shown[3]) >= 97:
            raise ValueError("malformed_observed_state")
        operand_value = state[operand] if operand in state else int(operand)
        if operand_value >= 97:
            raise ValueError("literal_outside_mod97")
        before = state[target]
        state[target] = {"ADD": before + operand_value, "SUB": before - operand_value,
                         "MUL": before * operand_value}[op] % 97
        expected = (state["A"], state["B"])
        truth.append(expected)
        if not first_error and (int(shown[2]), int(shown[3])) != expected:
            first_error = index
        shape.append(f"{op} {target} {'R' if operand in state else 'C'}")
    return tuple(truth), first_error, ";".join(shape)


def independently_replay(question: str, observed_steps: tuple[str, ...]) -> dict[str, Any]:
    """Replay only the strict printed language; malformed input becomes abstention."""
    try:
        states, first, structure = _replay(question, observed_steps)
    except (TypeError, ValueError, OverflowError):
        return {"disposition": "ABSTAIN", "reason": "malformed_or_unsupported_program"}
    return {"disposition": "REPLAYED", "true_states": states,
            "first_error_step": None if first == 0 else first, "structure": structure}


def _state_metrics(truth: Sequence[tuple[int, int]], predicted: Sequence[tuple[int, int]] | None) -> dict[str, Any]:
    if predicted is None:
        return {"state_steps": len(truth), "state_abstentions": len(truth), "state_correct": 0,
                "per_step_state_accuracy": None, "state_coverage": 0.0}
    correct = sum(a == b for a, b in zip(truth, predicted))
    return {"state_steps": len(truth), "state_abstentions": 0, "state_correct": correct,
            "per_step_state_accuracy": correct / len(truth), "state_coverage": 1.0}


def _verdict_metrics(actual: Sequence[int], predicted: Sequence[int | None]) -> dict[str, Any]:
    answered = [p is not None for p in predicted]
    exact = sum(a == p for a, p in zip(actual, predicted) if p is not None)
    errors = sum(a != 0 for a in actual)
    clean = len(actual) - errors
    false_approvals = sum(a != 0 and p == 0 for a, p in zip(actual, predicted))
    false_rejections = sum(a == 0 and p not in (None, 0) for a, p in zip(actual, predicted))
    localization = sum(a != 0 and p == a for a, p in zip(actual, predicted))
    return {"cases": len(actual), "abstentions": len(actual) - sum(answered),
            "coverage": sum(answered) / len(actual), "exact_first_error": exact,
            "exact_first_error_accuracy": exact / len(actual),
            "selective_exact_first_error_accuracy": exact / sum(answered) if sum(answered) else None,
            "false_approvals": false_approvals, "false_rejections": false_rejections,
            "false_approval_rate": false_approvals / errors if errors else None,
            "false_rejection_rate": false_rejections / clean if clean else None,
            "erroneous_step_localization_correct": localization,
            "erroneous_step_localization_accuracy": localization / errors if errors else None}


def evaluate_transition_predictions(cases: Sequence[TransitionEvaluationCase],
                                    predictions: Mapping[str, TransitionPrediction], *,
                                    training_structures: Sequence[str] = (),
                                    training_group_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Fail closed on bad inputs; score only independently replayed arithmetic."""
    if (not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= MAX_CASES
            or any(type(c) is not TransitionEvaluationCase for c in cases)
            or not isinstance(predictions, Mapping) or not isinstance(training_structures, (list, tuple))
            or not isinstance(training_group_ids, (list, tuple))
            or any(type(x) is not str for x in (*training_structures, *training_group_ids))):
        raise ValueError("invalid_evaluation_container")
    failures, audited, seen = [], [], set()
    for case in cases:
        if (type(case.case_id) is not str or not case.case_id or case.case_id in seen
                or type(case.group_id) is not str or not case.group_id):
            failures.append({"case_id": str(case.case_id), "reason": "invalid_or_duplicate_case_id"}); continue
        seen.add(case.case_id)
        try:
            truth, actual, structure = _replay(case.question, case.observed_steps)
        except (TypeError, ValueError, OverflowError):
            failures.append({"case_id": case.case_id, "reason": "malformed_or_unsupported_program"}); continue
        if _parse_verdict(case.claimed_first_error, len(truth), claimed=True) != actual:
            failures.append({"case_id": case.case_id, "reason": "unverifiable_claimed_label"}); continue
        if case.declared_structure is not None and (type(case.declared_structure) is not str or case.declared_structure != structure):
            failures.append({"case_id": case.case_id, "reason": "declared_structure_disagrees_with_program"}); continue
        candidate = predictions.get(case.case_id)
        if type(candidate) is not TransitionPrediction:
            failures.append({"case_id": case.case_id, "reason": "missing_or_invalid_prediction"}); continue
        if candidate.predicted_states is not None and (type(candidate.predicted_states) is not tuple or len(candidate.predicted_states) != len(truth)
                or any(type(s) is not tuple or len(s) != 2 or any(type(x) is not int or isinstance(x, bool) or not 0 <= x < 97 for x in s) for s in candidate.predicted_states)):
            failures.append({"case_id": case.case_id, "reason": "wrong_predicted_state_shape"}); continue
        verdict = _parse_verdict(candidate.predicted_first_error, len(truth), claimed=False)
        if verdict is _BAD or type(candidate.source_provenance) is not str or not candidate.source_provenance:
            failures.append({"case_id": case.case_id, "reason": "invalid_prediction_verdict_or_provenance"}); continue
        audited.append((case, truth, actual, structure, candidate, verdict))
    audit = {"total_cases": len(cases), "audited_cases": len(audited), "failures": failures}
    base = {"schema": SCHEMA, "oracle": "strict_public_printed_program_replay", "audit": audit,
            "qualification": "NOT_ESTABLISHED", "graduation_authority": "none",
            "confidence_claim": "No IID confidence claim is made."}
    if failures:
        return {**base, "disposition": "REFUSED"}
    actual = [x[2] for x in audited]
    verdicts = [x[5] for x in audited]
    state_rows = [_state_metrics(x[1], x[4].predicted_states) for x in audited]
    total_steps = sum(x["state_steps"] for x in state_rows)
    state_correct = sum(x["state_correct"] for x in state_rows)
    sources = sorted({x[4].source_provenance for x in audited})
    structures = {x[0].group_id: x[3] for x in audited}
    copied = [_state_metrics(x[1], tuple((int(m[2]), int(m[3])) for m in map(_STEP.fullmatch, x[0].observed_steps))) for x in audited]
    return {**base, "disposition": "SCORED", "metrics": {**_verdict_metrics(actual, verdicts),
            "state_steps": total_steps, "state_correct": state_correct,
            "per_step_state_accuracy": state_correct / total_steps,
            "state_coverage": sum(x["state_steps"] - x["state_abstentions"] for x in state_rows) / total_steps},
            "provenance": {"sources": sources, "diagnostic_only": any(s != "model_inference" for s in sources),
                           "learning_evidence": False if any("symbolic_gold" in s for s in sources) else "not_established"},
            "overlaps": {"structural": sorted(set(structures.values()) & set(training_structures)),
                         "group_ids": sorted(set(structures) & set(training_group_ids)),
                         "group_structures": dict(sorted(structures.items()))},
            "negative_controls": {"always_approve": _verdict_metrics(actual, [0] * len(actual)),
                                  "always_reject": _verdict_metrics(actual, [1] * len(actual)),
                                  "copied_observed_state": {"state_steps": total_steps, "state_correct": sum(x["state_correct"] for x in copied),
                                                            "per_step_state_accuracy": sum(x["state_correct"] for x in copied) / total_steps}},
            "limitations": ["Symbolic gold-state provenance is a diagnostic control, not learning.",
                            "Public arithmetic replay is not a blind or IID evaluation."]}
