"""Leakage-resistant finite algebra data for the Minotaur V5 study.

The public atomic interface contains only opaque operation ids and operands;
the modular answer is a separate supervision field.  In particular, this is
not a renderer which places an answer-derived feature beside an example.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Literal

from .minotaur_transition_curriculum import CurriculumConfig, build_transition_splits


MODULUS = 97
OPERATIONS = ("ADD", "SUB", "MUL")
ATOM_COUNT = len(OPERATIONS) * MODULUS * MODULUS
SplitName = Literal["train", "val", "holdout"]
Challenge = Literal["none", "value", "diagonal"]
Atom = tuple[int, int, int, int]


@dataclass(frozen=True)
class AtomicSplits:
    """Ordered, disjoint atomic examples and auditable SHA-256 commitments."""
    train: tuple[Atom, ...]
    val: tuple[Atom, ...]
    holdout: tuple[Atom, ...]
    commitments: dict[str, str]
    op_ids: tuple[int, ...]
    challenge: Challenge
    withheld_value: int | None

    def __iter__(self):
        # Conveniently permits ``train, val, holdout = atomic_split()``.
        yield self.train
        yield self.val
        yield self.holdout


@dataclass(frozen=True)
class ProgramCompositionSplits:
    """Read-only projection of the causal curriculum under V5 split names."""
    train: tuple[object, ...]
    val: tuple[object, ...]
    holdout: tuple[object, ...]
    commitments: dict[str, str]


def _validate(seed: int, train_fraction: float, challenge: str) -> None:
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be a non-boolean 31-bit integer")
    if type(train_fraction) is not float or not .5 <= train_fraction <= .9:
        raise ValueError("train_fraction must be a float from .5 through .9")
    if challenge not in ("none", "value", "diagonal"):
        raise ValueError("challenge must be 'none', 'value', or 'diagonal'")


def _seeded(seed: int, tag: str) -> random.Random:
    raw = f"minotaur-algebraic-data-v5\0{seed}\0{tag}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def _truth(operation: str, x: int, y: int) -> int:
    """Private generator arithmetic; this is never used as an input feature."""
    if operation == "ADD":
        return (x + y) % MODULUS
    if operation == "SUB":
        return (x - y) % MODULUS
    if operation == "MUL":
        return (x * y) % MODULUS
    raise ValueError("unknown operation")


def verify_mod97(operation: str, x: int, y: int) -> int:
    """Independent, deliberately differently-shaped label oracle for audits."""
    if operation not in OPERATIONS or type(x) is not int or type(y) is not int:
        raise ValueError("invalid modular operation")
    if not 0 <= x < MODULUS or not 0 <= y < MODULUS:
        raise ValueError("operands outside modulo-97 domain")
    values = {"ADD": x + y, "SUB": x + (MODULUS - y), "MUL": x * y}
    return values[operation] % MODULUS


def atomic_features(examples: tuple[Atom, ...]) -> tuple[tuple[int, int, int], ...]:
    """Return model-visible inputs, explicitly excluding the label column."""
    return tuple((op_id, x, y) for op_id, x, y, _ in examples)


def _groups() -> tuple[tuple[tuple[int, int], ...], ...]:
    # Every reversal pair is indivisible, including SUB controls.  Thus no
    # commutative answer alias, nor its noncommutative control, crosses splits.
    return tuple(((x, x),) if x == y else ((x, y), (y, x))
                 for x in range(MODULUS) for y in range(x, MODULUS))


def _commitment(examples: tuple[Atom, ...]) -> str:
    payload = "\n".join(f"{op}\t{x}\t{y}\t{label}" for op, x, y, label in examples)
    return hashlib.sha256(payload.encode()).hexdigest()


def _allocate_one_operation(seed: int, operation_index: int, targets: dict[SplitName, int],
                            challenge: Challenge, withheld_value: int | None
                            ) -> dict[SplitName, list[tuple[int, int]]]:
    buckets: dict[SplitName, list[tuple[int, int]]] = {name: [] for name in targets}
    remaining = dict(targets)
    groups = list(_groups())
    forced = []
    if challenge == "diagonal":
        forced = [group for group in groups if group[0][0] == group[0][1]]
    elif challenge == "value":
        assert withheld_value is not None
        forced = [group for group in groups if withheld_value in group[0]]
    if sum(map(len, forced)) > remaining["holdout"]:
        raise RuntimeError("challenge cannot fit requested holdout")
    for group in forced:
        buckets["holdout"].extend(group)
        remaining["holdout"] -= len(group)
    unused = [group for group in groups if group not in forced]

    # Seed each split with a zero-containing group when it was not already
    # forced there.  This makes zero operands a deliberate coverage stratum.
    for name in ("train", "val", "holdout"):
        if any(0 in pair for pair in buckets[name]):
            continue
        candidate = next((group for group in unused if 0 in group[0] and len(group) <= remaining[name]), None)
        if candidate is not None:
            buckets[name].extend(candidate)
            remaining[name] -= len(candidate)
            unused.remove(candidate)

    rank = _seeded(seed, f"allocate/{operation_index}/{challenge}")
    rank.shuffle(unused)
    # Distribute all singleton diagonals first, using one in every odd bucket
    # and then consuming the rest in pairs.  The remaining items are reversal
    # pairs, so this is also the parity proof for exact allocation.
    singles = [group for group in unused if len(group) == 1]
    for name in ("train", "val", "holdout"):
        if remaining[name] % 2:
            group = singles.pop()
            buckets[name].extend(group)
            remaining[name] -= 1
            unused.remove(group)
    while singles:
        name = max((name for name in targets if remaining[name] >= 2),
                   key=lambda current: remaining[current])
        for _ in range(2):
            group = singles.pop()
            buckets[name].extend(group)
            remaining[name] -= 1
            unused.remove(group)
    for name in ("train", "val", "holdout"):
        while remaining[name]:
            group = next(group for group in unused if len(group) == 2)
            buckets[name].extend(group)
            remaining[name] -= 2
            unused.remove(group)
    if unused or any(remaining.values()):
        raise RuntimeError("atomic allocation did not exhaust operation domain")
    return buckets


def atomic_split(seed: int = 20261028, train_fraction: float = .8, *,
                 challenge: Challenge = "none") -> AtomicSplits:
    """Return train/val/holdout tuples covering all 3*97*97 atoms exactly once.

    ``value`` reserves every pair involving one deterministic nonzero value in
    holdout; ``diagonal`` reserves all x==y pairs.  Both retain reversal-group
    integrity and zero coverage in every operation/split.
    """
    _validate(seed, train_fraction, challenge)
    per_operation = MODULUS * MODULUS
    train_each = round(per_operation * train_fraction)
    remainder = per_operation - train_each
    targets: dict[SplitName, int] = {"train": train_each,
                                     "val": remainder // 2,
                                     "holdout": remainder - remainder // 2}
    if challenge == "diagonal":
        # All 97 diagonal singleton groups are held out.  The other two
        # splits must therefore have even cardinality under reversal grouping.
        diagonal_train = train_each if train_each % 2 == 0 else train_each - 1
        diagonal_remainder = per_operation - diagonal_train
        diagonal_val = diagonal_remainder // 2
        if diagonal_val % 2:
            diagonal_val -= 1
        targets = {"train": diagonal_train, "val": diagonal_val,
                   "holdout": per_operation - diagonal_train - diagonal_val}
    # A nonzero withheld value avoids accidentally making the zero stratum a
    # challenge artifact.  Operation ids are a seed-permuted anonymization.
    withheld_value = (_seeded(seed, "withheld-value").randrange(1, MODULUS)
                       if challenge == "value" else None)
    op_order = list(range(len(OPERATIONS)))
    _seeded(seed, "opaque-operation-ids").shuffle(op_order)
    output: dict[SplitName, list[Atom]] = {name: [] for name in targets}
    for semantic_index, operation in enumerate(OPERATIONS):
        allocated = _allocate_one_operation(seed, semantic_index, targets, challenge, withheld_value)
        opaque_id = op_order[semantic_index]
        for name, pairs in allocated.items():
            output[name].extend((opaque_id, x, y, _truth(operation, x, y)) for x, y in pairs)
    # Shuffle only within split: it cannot affect membership or pair grouping.
    frozen: dict[SplitName, tuple[Atom, ...]] = {}
    for name, entries in output.items():
        _seeded(seed, f"order/{name}/{challenge}").shuffle(entries)
        frozen[name] = tuple(entries)
    commitments = {name: _commitment(entries) for name, entries in frozen.items()}
    result = AtomicSplits(frozen["train"], frozen["val"], frozen["holdout"], commitments,
                          tuple(op_order), challenge, withheld_value)
    _validate_atomic_result(result)
    return result


def _validate_atomic_result(result: AtomicSplits) -> None:
    groups = (result.train, result.val, result.holdout)
    keys = [{entry[:3] for entry in group} for group in groups]
    if sum(map(len, groups)) != ATOM_COUNT or any(len(group) != len(key) for group, key in zip(groups, keys)):
        raise RuntimeError("atom identities are not a partition")
    if any(keys[left] & keys[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("atom split leakage")
    for group in groups:
        if {op for op, _, _, _ in group} != set(result.op_ids):
            raise RuntimeError("operation imbalance")
        if not all(any(x == 0 for op2, x, _, _ in group if op2 == op) and
                   any(y == 0 for op2, _, y, _ in group if op2 == op)
                   for op in result.op_ids):
            raise RuntimeError("zero operand stratum missing")
    location = {key: index for index, keyset in enumerate(keys) for key in keyset}
    if any(location[(op, x, y)] != location[(op, y, x)]
           for op, x, y in location):
        raise RuntimeError("reversal alias crossed split")
    inverse = {opaque: OPERATIONS[index] for index, opaque in enumerate(result.op_ids)}
    if any(label != verify_mod97(inverse[op], x, y)
           for group in groups for op, x, y, label in group):
        raise RuntimeError("independent label audit failed")


def program_composition_split(seed: int = 20261028, *, train_cases: int = 64,
                              val_cases: int = 32, holdout_cases: int = 32
                              ) -> ProgramCompositionSplits:
    """Build the separate causal program split with canonical structures disjoint."""
    corpus = build_transition_splits(CurriculumConfig(seed=seed, train_cases=train_cases,
        dev_cases=val_cases, transfer_cases=holdout_cases))
    result = ProgramCompositionSplits(corpus["train"].cases, corpus["dev"].cases,
        corpus["structure_disjoint_transfer"].cases,
        {"train": corpus["train"].commitment, "val": corpus["dev"].commitment,
         "holdout": corpus["structure_disjoint_transfer"].commitment})
    structures = [{case.canonical_sequence for case in group}
                  for group in (result.train, result.val, result.holdout)]
    if any(structures[i] & structures[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("composition canonical structures leaked")
    return result
