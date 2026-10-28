"""Strict offline Qwen/ProcessBench adapter for Minotaur evaluation.

ProcessBench's published first-error labels are **zero based**, with -1 for
all steps correct (see QwenLM/ProcessBench code/run_eval.py).  Labyrinth's
reviewer response contract is **one based**: ALL_CORRECT or ERROR_STEP:N.

The data and predictions are supplied by the caller and are NOT authenticated
as an untouched ProcessBench split, a hidden exam, or an independent score.
No data is fetched and no arbitrary dataset code is executed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Sequence


_SPLITS = frozenset({"gsm8k", "math", "olympiadbench", "omnimath"})
_FIELDS = frozenset({"id", "generator", "problem", "steps",
                     "final_answer_correct", "label"})
_MAX_BYTES = 4 * 1024 * 1024
_MAX_CASES = 1024
_MAX_STEPS = 64
_MAX_TEXT = 40000
_MAX_ID = 200
_PREDICTION = re.compile(r"ERROR_STEP:([1-9][0-9]*)", re.ASCII)


@dataclass(frozen=True)
class ProcessBenchCase:
    """Caller-supplied example conforming to the six official JSON fields."""

    case_id: str
    generator: str
    problem: str
    steps: tuple[str, ...]
    final_answer_correct: bool
    label_zero_based: int
    split: str

    @property
    def first_error_step(self) -> int | None:
        return None if self.label_zero_based == -1 else self.label_zero_based + 1


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _text(value: object, *, field: str, limit: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {field}")
    return value


def load_processbench_jsonl(data: bytes, *, split: str) -> tuple[ProcessBenchCase, ...]:
    """Parse a bounded *locally supplied* JSONL export without network access.

    This is not a downloader or a claim that ``data`` came from ProcessBench.
    The fixture provenance must be managed separately by the experimenter.
    Reject malformed records, duplicate IDs/fields, and unbounded payloads.
    """
    if type(data) is not bytes:
        raise TypeError("ProcessBench JSONL input must be bytes")
    if type(split) is not str or split not in _SPLITS:
        raise ValueError("unknown ProcessBench split")
    if not 0 < len(data) <= _MAX_BYTES:
        raise ValueError("JSONL input must be nonempty and <= 4 MiB")
    try:
        lines = data.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise ValueError("JSONL input must be UTF-8") from exc
    if not 0 < len(lines) <= _MAX_CASES or any(not line.strip() for line in lines):
        raise ValueError("nonempty JSONL with 1..1024 records required")
    output: list[ProcessBenchCase] = []
    ids: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if len(line) > 128 * 1024:
            raise ValueError(f"record {line_number}: line too long")
        try:
            row = json.loads(line, object_pairs_hook=_unique_fields,
                             parse_constant=_bad_constant)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"record {line_number}: invalid JSON") from exc
        if type(row) is not dict or frozenset(row) != _FIELDS:
            raise ValueError(f"record {line_number}: official six-field schema required")
        case_id = _text(row["id"], field="id", limit=_MAX_ID)
        generator = _text(row["generator"], field="generator", limit=_MAX_ID)
        problem = _text(row["problem"], field="problem", limit=_MAX_TEXT)
        if type(row["steps"]) is not list or not 1 <= len(row["steps"]) <= _MAX_STEPS:
            raise ValueError(f"record {line_number}: invalid steps")
        steps = tuple(_text(step, field="step", limit=_MAX_TEXT) for step in row["steps"])
        correct = row["final_answer_correct"]
        label = row["label"]
        if type(correct) is not bool or type(label) is not int or not -1 <= label < len(steps):
            raise ValueError(f"record {line_number}: invalid answer correctness/label")
        if case_id in ids:
            raise ValueError(f"record {line_number}: duplicate case ID")
        ids.add(case_id)
        output.append(ProcessBenchCase(case_id, generator, problem, steps, correct,
                                       label, split))
    return tuple(output)


def evaluate_processbench(
    cases: Sequence[ProcessBenchCase],
    reviewer: Callable[[str, tuple[str, ...]], str],
) -> dict[str, str | int | float | None]:
    """Compute first-error agreement, matching ProcessBench's class accuracies.

    No answer label, generator, or final-answer correctness is shown to the
    reviewer, just problem text and complete numbered-step sequence. Return
    labels and predictions are omitted from the report to avoid accidental
    training contamination. This score is not a graduation credential.
    """
    if (not isinstance(cases, (list, tuple)) or not 0 < len(cases) <= _MAX_CASES
            or any(type(c) is not ProcessBenchCase for c in cases)
            or len({c.case_id for c in cases}) != len(cases)):
        raise ValueError("nonempty bounded unique ProcessBenchCase sequence required")
    if any(
        type(c.label_zero_based) is not int or not -1 <= c.label_zero_based < len(c.steps)
        or type(c.final_answer_correct) is not bool
        or not c.steps or type(c.problem) is not str or not c.problem.strip()
        for c in cases
    ):
        raise ValueError("malformed ProcessBenchCase; use validated loader")
    if not callable(reviewer):
        raise TypeError("reviewer must be callable")
    if len({case.split for case in cases}) != 1:
        raise ValueError("mixed source splits are not a reproducible evaluation")

    right = right_errors = right_correct = abstentions = false_approvals = false_rejections = 0
    error_cases = sum(c.first_error_step is not None for c in cases)
    correct_cases = len(cases) - error_cases
    for case in cases:
        response = reviewer(case.problem, case.steps)
        if type(response) is not str:
            response = ""
        matched = _PREDICTION.fullmatch(response)
        predicted_step = int(matched.group(1)) if matched else None
        accepted = response == "ALL_CORRECT" or (
            predicted_step is not None and predicted_step <= len(case.steps))
        if not accepted:
            abstentions += 1
            continue
        expected_step = case.first_error_step
        if expected_step is None:
            if response == "ALL_CORRECT":
                right += 1
                right_correct += 1
            else:
                false_rejections += 1
        elif response == "ALL_CORRECT":
            false_approvals += 1
        elif predicted_step == expected_step:
            right += 1
            right_errors += 1
    error_acc = right_errors / error_cases if error_cases else 0.0
    correct_acc = right_correct / correct_cases if correct_cases else 0.0
    harmonic = (2 * error_acc * correct_acc / (error_acc + correct_acc)
                if error_acc + correct_acc else 0.0)
    return {
        "benchmark_schema": "Qwen/ProcessBench",
        "split": cases[0].split,
        "evidence_status": "UNVERIFIED_CALLER_SUPPLIED_DATA",
        "reasoning_qualification": "NOT_ESTABLISHED",
        "total": len(cases),
        "error_cases": error_cases,
        "correct_cases": correct_cases,
        "accuracy": right / len(cases),
        "error_accuracy": error_acc,
        "correct_accuracy": correct_acc,
        "harmonic_f1": harmonic,
        "coverage": (len(cases) - abstentions) / len(cases),
        "abstentions": abstentions,
        "false_approval_rate": false_approvals / error_cases if error_cases else None,
        "false_rejection_rate": false_rejections / correct_cases if correct_cases else None,
    }


def evaluate_isomorphic_pairs(
    pairs: Sequence[tuple[ProcessBenchCase, ProcessBenchCase]],
    reviewer: Callable[[str, tuple[str, ...]], str],
) -> dict[str, str | int | float]:
    """Check *claimed* equivalent reformulations for verifier gaming.

    Inspired by Helff et al. (2026), Isomorphic Perturbation Testing. This
    function does not prove that the caller's variants are isomorphic, and
    pairwise consistency by itself does not imply correctness. It requires
    both variants to have the same earliest erroneous step and reports
    whether a critic produces the same strictly formatted verdict on each.
    """
    if not isinstance(pairs, (tuple, list)) or not 0 < len(pairs) <= 256:
        raise ValueError("1..256 variant pairs required")
    if not callable(reviewer):
        raise TypeError("reviewer must be callable")
    seen: set[str] = set()
    consistent = both_correct = either_correct = valid_pairs = 0
    for pair in pairs:
        if (type(pair) not in (tuple, list) or len(pair) != 2
                or any(type(item) is not ProcessBenchCase for item in pair)):
            raise ValueError("each pair must contain two ProcessBenchCase records")
        left, right = pair
        if (left.case_id == right.case_id or left.first_error_step != right.first_error_step
                or left.split != right.split or left.steps == right.steps
                and left.problem == right.problem
                or {left.case_id, right.case_id} & seen):
            raise ValueError("reformulations must be distinct, unique and share oracle step")
        seen.update((left.case_id, right.case_id))
        guesses = [reviewer(c.problem, c.steps) for c in pair]
        normalized: list[str | None] = []
        for guess, case in zip(guesses, pair):
            if type(guess) is not str:
                normalized.append(None)
                continue
            match = _PREDICTION.fullmatch(guess)
            normalized.append(guess if guess == "ALL_CORRECT" or
                              match is not None and int(match.group(1)) <= len(case.steps)
                              else None)
        valid_pairs += int(all(pred is not None for pred in normalized))
        consistent += int(normalized[0] is not None and normalized == [normalized[0]] * 2)
        actual = ("ALL_CORRECT" if left.first_error_step is None
                  else f"ERROR_STEP:{left.first_error_step}")
        hits = sum(pred == actual for pred in normalized)
        both_correct += int(hits == 2)
        either_correct += int(hits >= 1)
    return {
        "test_type": "CALLER_ASSERTED_ISOMORPHIC_VARIANTS",
        "equivalence_proof": "NOT_VERIFIED",
        "reasoning_qualification": "NOT_ESTABLISHED",
        "pairs": len(pairs),
        "pair_coverage": valid_pairs / len(pairs),
        "verdict_consistency": consistent / len(pairs),
        "both_correct_rate": both_correct / len(pairs),
        "one_sided_failure_rate": (either_correct - both_correct) / len(pairs),
    }
