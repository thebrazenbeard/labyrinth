"""Fail-closed, dependency-free scorer for public modular-arithmetic submissions.

This is an evaluator, not an oracle API: labels are derived internally and a
submission may contain only canonical operand-pair -> residue predictions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from types import MappingProxyType
from typing import Mapping


SCHEMA = "labyrinth.minotaur.v8-examiner.v1"
MODULUS = 97
UNSEEN_MODULUS = 101
WITHHELD = 53


@dataclass(frozen=True)
class ExamPlan:
    modulus: int
    unseen_modulus: int
    withheld_value: int
    training_members: tuple[tuple[int, int], ...]
    evaluation_groups: Mapping[str, tuple[tuple[int, int], ...]]
    custody_sha256: str


@dataclass(frozen=True)
class PredictionSubmission:
    """Plain predictions plus public custody and content hashes; no targets."""
    predictions: Mapping[tuple[int, int], int]
    plan_sha256: str
    prediction_sha256: str


def _canonical_pair(pair: object, modulus: int) -> tuple[int, int]:
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
            any(type(value) is not int for value in pair)):
        raise ValueError("operand pair must be two integer residues")
    a, b = pair
    if not (0 <= a < modulus and 0 <= b < modulus):
        raise ValueError("operand pair is outside its modulus")
    return (a, b) if a <= b else (b, a)


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_digest(value: object) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _target(pair: tuple[int, int], modulus: int) -> int:
    return (pair[0] * pair[1]) % modulus


def _plan_payload(plan: ExamPlan) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "modulus": plan.modulus,
        "unseen_modulus": plan.unseen_modulus,
        "withheld_value": plan.withheld_value,
        "training_members": [list(pair) for pair in plan.training_members],
        "evaluation_groups": {
            name: [list(pair) for pair in pairs]
            for name, pairs in plan.evaluation_groups.items()
        },
    }


def build_exam_plan(*, seed: int = 20261028, holdout_fraction: float = 0.10) -> ExamPlan:
    """Create a public split whose samples are canonical unordered pairs."""
    if type(seed) is not int or seed < 0 or type(holdout_fraction) is not float or not 0.02 <= holdout_fraction <= 0.30:
        raise ValueError("invalid bounded exam-plan configuration")
    all_97 = [(a, b) for a in range(MODULUS) for b in range(a, MODULUS)]
    output_53 = [pair for pair in all_97 if WITHHELD not in pair and _target(pair, MODULUS) == WITHHELD]
    input_53 = [pair for pair in all_97 if WITHHELD in pair]
    eligible = [pair for pair in all_97 if pair not in output_53 and pair not in input_53]
    random.Random(seed).shuffle(eligible)
    holdout_count = max(1, round(len(eligible) * holdout_fraction))
    # Values 97--100 make this a genuine unseen-residue evaluation while avoiding
    # ambiguous pair identities across the two moduli in a plain submission map.
    modulus_101 = [(a, b) for a in range(UNSEEN_MODULUS) for b in range(a, UNSEEN_MODULUS)
                   if a >= MODULUS or b >= MODULUS]
    groups = MappingProxyType({
        "independent_holdout": tuple(sorted(eligible[:holdout_count])),
        "output_53": tuple(output_53),
        "input_53": tuple(input_53),
        "modulus_101": tuple(modulus_101),
    })
    draft = ExamPlan(MODULUS, UNSEEN_MODULUS, WITHHELD,
                     tuple(sorted(eligible[holdout_count:])), groups, "")
    return ExamPlan(draft.modulus, draft.unseen_modulus, draft.withheld_value,
                    draft.training_members, draft.evaluation_groups,
                    _json_digest(_plan_payload(draft)))


def _prediction_rows(plan: ExamPlan, predictions: Mapping[tuple[int, int], int]) -> list[list[int]]:
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must be a mapping, not target-bearing rows or an oracle")
    expected: dict[tuple[int, int], int] = {}
    for name, pairs in plan.evaluation_groups.items():
        modulus = plan.unseen_modulus if name == "modulus_101" else plan.modulus
        for pair in pairs:
            expected[pair] = modulus
    if len(predictions) != len(expected) or set(predictions) != set(expected):
        raise ValueError("predictions must contain exactly one canonical evaluation pair")
    rows = []
    for pair, modulus in expected.items():
        if _canonical_pair(pair, modulus) != pair:
            raise ValueError("prediction pair is not canonical")
        label = predictions[pair]
        if type(label) is not int or not 0 <= label < modulus:
            raise ValueError("prediction label must be an integer residue")
        rows.append([pair[0], pair[1], label])
    return sorted(rows)


def make_prediction_submission(plan: ExamPlan, predictions: Mapping[tuple[int, int], int]) -> PredictionSubmission:
    """Validate and hash a prediction-only submission before it is transported."""
    rows = _prediction_rows(plan, predictions)
    return PredictionSubmission(MappingProxyType(dict(predictions)), plan.custody_sha256,
                                _json_digest(rows))


def constant_predictions(plan: ExamPlan, value: int = 0) -> dict[tuple[int, int], int]:
    """Convenience constructor for a prediction-only constant control/submission."""
    result: dict[tuple[int, int], int] = {}
    for name, pairs in plan.evaluation_groups.items():
        modulus = plan.unseen_modulus if name == "modulus_101" else plan.modulus
        if type(value) is not int or not 0 <= value < modulus:
            raise ValueError("constant must be an integer residue for every group")
        result.update({pair: value for pair in pairs})
    return result


def _validate_training(plan: ExamPlan, training_members: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(training_members, (tuple, list)):
        raise ValueError("training members must be a pair sequence without labels")
    members = tuple(_canonical_pair(pair, plan.modulus) for pair in training_members)
    if len(set(members)) != len(members):
        raise ValueError("training members must be unique unordered pairs")
    evaluation = {pair for pairs in plan.evaluation_groups.values() for pair in pairs}
    if set(members) & evaluation:
        raise ValueError("training member overlaps evaluation group (including reversal)")
    if set(members) != set(plan.training_members):
        raise ValueError("training members do not match plan custody")
    return members


def _score_values(pairs: tuple[tuple[int, int], ...], modulus: int,
                  values: Mapping[tuple[int, int], int]) -> dict[str, float | int]:
    correct = sum(values[pair] == _target(pair, modulus) for pair in pairs)
    count = len(pairs)
    return {"correct": correct, "count": count, "accuracy": correct / count if count else 0.0}


def _controls(plan: ExamPlan, train_majority: int) -> dict[str, dict[str, dict[str, float | int]]]:
    controls: dict[str, dict[str, dict[str, float | int]]] = {
        "zero": {}, "copy_x": {}, "train_majority": {}, "frozen_random": {},
    }
    for name, pairs in plan.evaluation_groups.items():
        modulus = plan.unseen_modulus if name == "modulus_101" else plan.modulus
        frozen = {pair: int(hashlib.sha256(
            f"{plan.custody_sha256}:{modulus}:{pair[0]}:{pair[1]}".encode("ascii")
        ).hexdigest(), 16) % modulus for pair in pairs}
        variants = {
            "zero": {pair: 0 for pair in pairs},
            "copy_x": {pair: pair[0] for pair in pairs},
            "train_majority": {pair: train_majority % modulus for pair in pairs},
            "frozen_random": frozen,
        }
        for control, predictions in variants.items():
            controls[control][name] = _score_values(pairs, modulus, predictions)
    return controls


def score_submission(plan: ExamPlan, training_members: object,
                     submission: PredictionSubmission) -> dict[str, object]:
    """Fail closed on custody, predictions, or split contamination before scoring."""
    if not isinstance(plan, ExamPlan) or not _is_digest(plan.custody_sha256) or _json_digest(_plan_payload(plan)) != plan.custody_sha256:
        raise ValueError("invalid plan custody")
    members = _validate_training(plan, training_members)
    if not isinstance(submission, PredictionSubmission) or submission.plan_sha256 != plan.custody_sha256:
        raise ValueError("submission custody does not match exam plan")
    if not _is_digest(submission.prediction_sha256):
        raise ValueError("invalid prediction digest")
    rows = _prediction_rows(plan, submission.predictions)
    if _json_digest(rows) != submission.prediction_sha256:
        raise ValueError("prediction digest does not match prediction contents")
    labels = [_target(pair, plan.modulus) for pair in members]
    train_majority = min(label for label, count in Counter(labels).items()
                        if count == max(Counter(labels).values()))
    groups = {}
    for name, pairs in plan.evaluation_groups.items():
        modulus = plan.unseen_modulus if name == "modulus_101" else plan.modulus
        groups[name] = _score_values(pairs, modulus, submission.predictions)
    return {
        "schema": SCHEMA,
        "verdict": "SCORED",
        "custody_sha256": plan.custody_sha256,
        "prediction_sha256": submission.prediction_sha256,
        "row_policy": "one canonical unordered pair is one sample",
        "groups": groups,
        "sample_group_accuracy": sum(group["accuracy"] for group in groups.values()) / len(groups),
        "controls": _controls(plan, train_majority),
    }
