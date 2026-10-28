"""Independent, pure-Python audit protocol for public V7 multiplication claims.

This is deliberately an evaluator, not a trainer and not a private examiner.
It regenerates every target from operands and validates the exact public split
before accepting predictions from a later model integration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import random
from typing import Mapping, Sequence

MOD97 = 97
MOD101 = 101
ABSENT_LABEL = 53
CaseRows = tuple[tuple[int, int, int], ...]


def recompute_label(left: int, right: int, modulus: int) -> int:
    """Independent arithmetic oracle, restricted to the two published domains."""
    if (type(left) is not int or type(right) is not int or type(modulus) is not int
            or modulus not in (MOD97, MOD101)
            or not 0 <= left < modulus or not 0 <= right < modulus):
        raise ValueError("invalid modular case")
    return (left * right) % modulus


def canonical_pair(left: int, right: int) -> tuple[int, int]:
    if type(left) is not int or type(right) is not int:
        raise ValueError("operands must be integers")
    return (left, right) if left <= right else (right, left)


def _digest(rows: Sequence[tuple[int, int, int]]) -> str:
    payload = "\n".join(f"{x}\t{y}\t{label}" for x, y, label in rows)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _directed(groups: Sequence[tuple[int, int]], modulus: int) -> CaseRows:
    rows = []
    for left, right in groups:
        rows.append((left, right, recompute_label(left, right, modulus)))
        if left != right:
            rows.append((right, left, recompute_label(right, left, modulus)))
    return tuple(sorted(rows))


def _groups(modulus: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(modulus) for right in range(left, modulus)]


@dataclass(frozen=True)
class BlindishProtocol:
    """Published challenge split.  ``blindish`` means hostile, never private."""
    seed: int
    modulus: int
    absent_label: int
    train: CaseRows
    challenges: dict[str, CaseRows]
    counts: dict[str, int]
    digests: dict[str, str]
    integrity: dict


def _integrity(train: CaseRows, challenges: Mapping[str, CaseRows]) -> dict:
    # Coordinates such as (1, 1) legitimately occur in both moduli.  Pair
    # leakage is a within-domain property, so mod101 must not be compared to
    # the mod97 public split by raw coordinate alone.
    partitions = {"train": train,
                  **{name: rows for name, rows in challenges.items() if name != "mod101"}}
    pair_locations: dict[tuple[int, int], str] = {}
    pair_overlap: dict[str, tuple[str, ...]] = {}
    labels = {name: {label for _x, _y, label in rows} for name, rows in partitions.items()}
    train_labels = labels["train"]
    for name, rows in partitions.items():
        for left, right, _label in rows:
            group = canonical_pair(left, right)
            old = pair_locations.setdefault(group, name)
            if old != name:
                pair_overlap["/".join(sorted((old, name)))] = (old, name)
    expected_zero = {"missing_output_53"}
    label_overlap = {name: tuple(sorted(train_labels & {label for _x, _y, label in rows}))
                     for name, rows in challenges.items()}
    modulus_101_rows = challenges.get("mod101", ())
    return {
        "canonical_pair_overlap": pair_overlap,
        "label_overlap": label_overlap,
        "zero_label_overlap_expected": tuple(sorted(expected_zero)),
        "mod101_training_rows": 0,
        "ok": (not pair_overlap and not label_overlap["missing_output_53"]
               and not any(x >= MOD101 or y >= MOD101 or label >= MOD101
                           for x, y, label in modulus_101_rows)),
    }


def build_protocol(seed: int = 20261028, holdout_fraction: float = 0.10) -> BlindishProtocol:
    """Build V7's public challenge layout without importing V7 data or model code."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a bounded integer")
    if type(holdout_fraction) is not float or not .02 <= holdout_fraction <= .30:
        raise ValueError("holdout_fraction outside protocol bounds")
    groups = _groups(MOD97)
    random.Random(seed).shuffle(groups)
    eligible: list[tuple[int, int]] = []
    missing_output: list[tuple[int, int]] = []
    missing_input: list[tuple[int, int]] = []
    for left, right in groups:
        label = recompute_label(left, right, MOD97)
        if left == ABSENT_LABEL or right == ABSENT_LABEL:
            missing_input.append((left, right))
        elif label == ABSENT_LABEL:
            missing_output.append((left, right))
        else:
            eligible.append((left, right))
    heldout_count = max(1, round(len(eligible) * holdout_fraction))
    train = _directed(eligible[heldout_count:], MOD97)
    challenges = {
        "unseen_group_pairs": _directed(eligible[:heldout_count], MOD97),
        "missing_output_53": _directed(missing_output, MOD97),
        "missing_input_53": _directed(missing_input, MOD97),
        "mod101": _directed(_groups(MOD101), MOD101),
    }
    counts = {"train": len(train), **{name: len(rows) for name, rows in challenges.items()}}
    digests = {"train": _digest(train), **{name: _digest(rows) for name, rows in challenges.items()}}
    integrity = _integrity(train, challenges)
    if not integrity["ok"]:
        raise RuntimeError("protocol construction violated a leakage invariant")
    return BlindishProtocol(seed, MOD97, ABSENT_LABEL, train, challenges, counts, digests, integrity)


def _normalise_predictions(rows: CaseRows, supplied: Mapping[tuple[int, int], int | None]) -> list[int | None]:
    expected = {(left, right) for left, right, _label in rows}
    extra = set(supplied) - expected
    if extra:
        raise ValueError("prediction contains cases outside the declared challenge")
    values = []
    for left, right, _label in rows:
        prediction = supplied.get((left, right))
        if prediction is not None and type(prediction) is not int:
            raise ValueError("predictions must be integers or None for abstention")
        values.append(prediction)
    return values


def _score(rows: CaseRows, predictions: Sequence[int | None], modulus: int) -> dict:
    correct = sum(prediction == label for prediction, (_x, _y, label) in zip(predictions, rows))
    abstained = sum(prediction is None for prediction in predictions)
    groups: dict[tuple[int, int], list[bool]] = {}
    for prediction, (left, right, label) in zip(predictions, rows):
        groups.setdefault(canonical_pair(left, right), []).append(prediction == label)
    complete_groups = sum(all(hits) for hits in groups.values())
    return {
        "count": len(rows), "correct": correct, "abstained": abstained,
        "coverage": (len(rows) - abstained) / len(rows) if rows else 0.0,
        "accuracy": correct / len(rows) if rows else 0.0,
        "canonical_groups": len(groups), "pair_complete_correct": complete_groups,
        "pair_complete_accuracy": complete_groups / len(groups) if groups else 0.0,
        "pair_complete": abstained == 0,
        "iid_confidence_interval": None,
        "iid_confidence_note": "withheld reversals share canonical pairs; IID confidence is not reported",
    }


def _validate_training_rows(training_rows: Sequence[Sequence[int]]) -> None:
    for row in training_rows:
        if len(row) not in (3, 4):
            raise ValueError("training rows must have x, y, label[, modulus]")
        left, right, label = row[:3]
        modulus = row[3] if len(row) == 4 else MOD97
        if modulus == MOD101:
            raise ValueError("mod101 is evaluation-only and may not appear in training")
        if modulus != MOD97:
            raise ValueError("unknown training modulus")
        if recompute_label(left, right, modulus) != label:
            raise ValueError("training label disagrees with independent oracle")


def _validate_protocol(protocol: BlindishProtocol) -> None:
    """Do not trust stored metadata: regenerate labels, counts, and digests."""
    if protocol.modulus != MOD97 or protocol.absent_label != ABSENT_LABEL:
        raise ValueError("unexpected protocol domain")
    expected_names = {"unseen_group_pairs", "missing_output_53", "missing_input_53", "mod101"}
    if set(protocol.challenges) != expected_names:
        raise ValueError("protocol challenge names do not match this evaluator")
    rows_by_name = {"train": protocol.train, **protocol.challenges}
    recomputed_counts = {name: len(rows) for name, rows in rows_by_name.items()}
    recomputed_digests = {name: _digest(rows) for name, rows in rows_by_name.items()}
    if protocol.counts != recomputed_counts or protocol.digests != recomputed_digests:
        raise ValueError("protocol counts or digests do not match its rows")
    for name, rows in rows_by_name.items():
        modulus = MOD101 if name == "mod101" else MOD97
        if any(recompute_label(left, right, modulus) != label for left, right, label in rows):
            raise ValueError("protocol labels disagree with independent oracle")
    if (any(ABSENT_LABEL in (left, right, label) for left, right, label in protocol.train)
            or any(ABSENT_LABEL in (left, right, label)
                   for left, right, label in protocol.challenges["unseen_group_pairs"])
            or any(label != ABSENT_LABEL or ABSENT_LABEL in (left, right)
                   for left, right, label in protocol.challenges["missing_output_53"])
            or any(ABSENT_LABEL not in (left, right)
                   for left, right, _label in protocol.challenges["missing_input_53"])):
        raise ValueError("protocol violates its stated residue-53 exclusions")
    integrity = _integrity(protocol.train, protocol.challenges)
    if protocol.integrity != integrity or not integrity["ok"]:
        raise ValueError("protocol integrity metadata or separation checks failed")


def audit_predictions(protocol: BlindishProtocol,
                      predictions: Mapping[str, Mapping[tuple[int, int], int | None]],
                      *, training_rows: Sequence[Sequence[int]] = ()) -> dict:
    """Audit model outputs; omitted values are explicit abstentions, never successes."""
    if not isinstance(protocol, BlindishProtocol):
        raise TypeError("protocol must come from build_protocol")
    _validate_protocol(protocol)
    _validate_training_rows(training_rows)
    unknown = set(predictions) - set(protocol.challenges)
    if unknown:
        raise ValueError("predictions contain unknown challenge names")
    cases = {}
    for name, rows in protocol.challenges.items():
        values = _normalise_predictions(rows, predictions.get(name, {}))
        cases[name] = _score(rows, values, MOD101 if name == "mod101" else MOD97)
    baselines = _baselines(protocol)
    return {
        "schema": "labyrinth.minotaur.v7-blindish-eval.v1",
        "protocol_counts": protocol.counts, "protocol_digests": protocol.digests,
        "integrity": protocol.integrity, "cases": cases, "baselines": baselines,
        "examiner_status": "public deterministic hostile protocol, not a private examiner",
        "qualification": "NOT_ESTABLISHED",
        "scientific_warning": ("No positive training example for residue 53 exists; a categorical "
                               "classification head cannot learn class 53 from this training set."),
    }


def _baselines(protocol: BlindishProtocol) -> dict:
    majority = Counter(label for _x, _y, label in protocol.train).most_common(1)[0][0]
    definitions = {
        "abstain": (lambda x, y, label, modulus: None, "explicit no-answer baseline"),
        "zero": (lambda x, y, label, modulus: 0, "fixed public baseline"),
        "train_majority": (lambda x, y, label, modulus: majority, "fit only to train labels"),
        "copy_left": (lambda x, y, label, modulus: x, "operand-copy baseline"),
        "anti_oracle_adversarial": (lambda x, y, label, modulus: (label + 1) % modulus,
                                     "uses labels to be wrong; not a valid predictive baseline"),
    }
    report = {}
    for name, (predict, note) in definitions.items():
        report[name] = {"note": note}
        for case, rows in protocol.challenges.items():
            modulus = MOD101 if case == "mod101" else MOD97
            report[name][case] = _score(rows, [predict(x, y, label, modulus) for x, y, label in rows], modulus)
    return report
