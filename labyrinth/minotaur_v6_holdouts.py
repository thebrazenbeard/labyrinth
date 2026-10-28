"""Reproducible, grouped multiplication holdouts for the Minotaur V6 study.

The triples in this module are ``(left, right, label)``.  They are data-owner
records, not model inputs: use :func:`model_features` to obtain the two
model-visible operands.  Labels are generated and audited by deliberately
different implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Literal


MOD97 = 97
MOD101 = 101
Example = tuple[int, int, int]
Pair = tuple[int, int]
ReverseGroup = tuple[Pair, ...]
SplitName = Literal["train", "dev", "test"]
Challenge = Literal["none", "withheld_value", "cartesian_block"]
_SPLIT_NAMES: tuple[SplitName, ...] = ("train", "dev", "test")


@dataclass(frozen=True)
class MultiplicationSplits:
    """A complete, reversal-grouped partition of multiplication modulo 97."""

    train: tuple[Example, ...]
    dev: tuple[Example, ...]
    test: tuple[Example, ...]
    digests: dict[str, str]
    modulus: int
    challenge: Challenge
    withheld_value: int | None
    block_values: tuple[int, ...]


@dataclass(frozen=True)
class OutOfDomainEvaluation:
    """A separately named and labeled full-domain multiplication evaluation."""

    examples: tuple[Example, ...]
    digest: str
    modulus: int
    domain: str


def _rng(seed: int, tag: str) -> random.Random:
    material = f"minotaur-v6-holdouts\0{seed}\0{tag}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be a non-boolean 31-bit integer")


def _label_from_generator(left: int, right: int, modulus: int) -> int:
    """Generator arithmetic; independent audits use repeated addition below."""
    return (left * right) % modulus


def verify_multiplication_label(left: int, right: int, modulus: int = MOD97) -> int:
    """Independently calculate a label without using multiplication.

    This deliberately slow audit implementation makes a shared arithmetic bug
    between data construction and validation less likely.
    """
    if type(left) is not int or type(right) is not int or type(modulus) is not int:
        raise ValueError("operands and modulus must be integers")
    if modulus < 2 or not 0 <= left < modulus or not 0 <= right < modulus:
        raise ValueError("operands must be inside the modular domain")
    total = 0
    for _ in range(right):
        total += left
        if total >= modulus:
            # ``total`` and ``left`` are each strictly below ``modulus``, so
            # one subtraction is sufficient.  Keeping this to addition and
            # subtraction makes the audit independent of the generator's
            # multiplication expression.
            total -= modulus
    return total


def model_features(examples: tuple[Example, ...]) -> tuple[tuple[int, int], ...]:
    """Return the only model-visible fields: operands, never answer labels."""
    return tuple((left, right) for left, right, _label in examples)


def _digest(examples: tuple[Example, ...]) -> str:
    payload = "\n".join(f"{left}\t{right}\t{label}" for left, right, label in examples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reverse_groups(modulus: int) -> list[ReverseGroup]:
    return [((left, left),) if left == right else ((left, right), (right, left))
            for left in range(modulus) for right in range(left, modulus)]


def _split_targets(modulus: int, train_fraction: float, dev_fraction: float) -> dict[SplitName, int]:
    if (type(train_fraction) is not float or type(dev_fraction) is not float or
            not 0.0 < train_fraction < 1.0 or not 0.0 < dev_fraction < 1.0 or
            train_fraction + dev_fraction >= 1.0):
        raise ValueError("train_fraction and dev_fraction must be positive floats summing to less than one")
    total = modulus * modulus
    train = round(total * train_fraction)
    dev = round(total * dev_fraction)
    targets: dict[SplitName, int] = {"train": train, "dev": dev,
                                     "test": total - train - dev}
    if any(size < 1 for size in targets.values()):
        raise ValueError("fractions must leave at least one example in every split")
    return targets


def _challenge_groups(groups: list[ReverseGroup], challenge: Challenge,
                      withheld_value: int | None, block_values: tuple[int, ...]
                      ) -> list[ReverseGroup]:
    if challenge == "none":
        return []
    if challenge == "withheld_value":
        assert withheld_value is not None
        return [group for group in groups if withheld_value in group[0]]
    block = set(block_values)
    return [group for group in groups if group[0][0] in block and group[0][1] in block]


def _singleton_allocation(capacity: dict[SplitName, int], singleton_count: int,
                          seed: int, challenge: Challenge) -> dict[SplitName, int]:
    """Pick an exact, parity-compatible distribution of diagonal groups.

    Every non-diagonal reversal group has two records.  The 97 diagonal
    singletons therefore supply the parity needed to hit exact split targets
    instead of silently drifting by one example.
    """
    candidates: list[dict[SplitName, int]] = []
    train_minimum = capacity["train"] % 2
    train_maximum = min(capacity["train"], singleton_count)
    for train_singletons in range(train_minimum, train_maximum + 1, 2):
        dev_minimum = capacity["dev"] % 2
        dev_maximum = min(capacity["dev"], singleton_count - train_singletons)
        for dev_singletons in range(dev_minimum, dev_maximum + 1, 2):
            test_singletons = singleton_count - train_singletons - dev_singletons
            if (0 <= test_singletons <= capacity["test"] and
                    test_singletons % 2 == capacity["test"] % 2):
                candidates.append({"train": train_singletons, "dev": dev_singletons,
                                   "test": test_singletons})
    if not candidates:
        raise ValueError("requested split sizes cannot preserve reverse-pair grouping")
    _rng(seed, f"singleton-allocation/{challenge}").shuffle(candidates)
    return candidates[0]


def _allocate_exact_groups(groups: list[ReverseGroup], forced: list[ReverseGroup],
                           targets: dict[SplitName, int], seed: int,
                           challenge: Challenge) -> dict[SplitName, list[Pair]]:
    """Allocate every group once while meeting exact record-count targets."""
    forced_count = sum(map(len, forced))
    capacity: dict[SplitName, int] = {"train": targets["train"], "dev": targets["dev"],
                                      "test": targets["test"] - forced_count}
    if any(size < 0 for size in capacity.values()):
        raise ValueError("challenge does not fit the requested test fraction")

    forced_set = set(forced)
    remaining = [group for group in groups if group not in forced_set]
    singletons = [group for group in remaining if len(group) == 1]
    pairs = [group for group in remaining if len(group) == 2]
    singleton_counts = _singleton_allocation(capacity, len(singletons), seed, challenge)

    _rng(seed, f"singleton-groups/{challenge}").shuffle(singletons)
    _rng(seed, f"pair-groups/{challenge}").shuffle(pairs)
    group_buckets: dict[SplitName, list[ReverseGroup]] = {
        "train": [], "dev": [], "test": list(forced),
    }
    singleton_offset = 0
    pair_offset = 0
    for name in _SPLIT_NAMES:
        singleton_count = singleton_counts[name]
        pair_count = (capacity[name] - singleton_count) // 2
        group_buckets[name].extend(singletons[singleton_offset:singleton_offset + singleton_count])
        group_buckets[name].extend(pairs[pair_offset:pair_offset + pair_count])
        singleton_offset += singleton_count
        pair_offset += pair_count
    if singleton_offset != len(singletons) or pair_offset != len(pairs):
        raise RuntimeError("group allocation did not exhaust the multiplication domain")
    return {name: [pair for group in group_buckets[name] for pair in group]
            for name in _SPLIT_NAMES}


def build_mod97_splits(seed: int = 20261028, *, train_fraction: float = 0.8,
                        dev_fraction: float = 0.1, challenge: Challenge = "none",
                        withheld_value: int | None = None, block_size: int = 20
                        ) -> MultiplicationSplits:
    """Build a deterministic grouped train/dev/test partition for multiplication mod 97.

    Reversal pairs are indivisible in every condition.  ``withheld_value``
    puts its complete row and column in test; ``cartesian_block`` puts every
    pair whose operands both belong to a deterministic block in test.
    """
    _validate_seed(seed)
    if challenge not in ("none", "withheld_value", "cartesian_block"):
        raise ValueError("challenge must be 'none', 'withheld_value', or 'cartesian_block'")
    if withheld_value is not None and (type(withheld_value) is not int or not 0 <= withheld_value < MOD97):
        raise ValueError("withheld_value must be in the mod-97 domain")
    if type(block_size) is not int or not 1 <= block_size < MOD97:
        raise ValueError("block_size must be an integer from 1 through 96")
    targets = _split_targets(MOD97, train_fraction, dev_fraction)
    if challenge == "withheld_value":
        value = withheld_value if withheld_value is not None else _rng(seed, "withheld-value").randrange(MOD97)
    else:
        value = None
    block = (tuple(sorted(_rng(seed, "cartesian-block").sample(range(MOD97), block_size)))
             if challenge == "cartesian_block" else ())
    groups = _reverse_groups(MOD97)
    forced = _challenge_groups(groups, challenge, value, block)
    buckets = _allocate_exact_groups(groups, forced, targets, seed, challenge)
    frozen: dict[str, tuple[Example, ...]] = {}
    for name, pairs in buckets.items():
        records = [(left, right, _label_from_generator(left, right, MOD97)) for left, right in pairs]
        _rng(seed, f"order/{name}/{challenge}").shuffle(records)
        frozen[name] = tuple(records)
    result = MultiplicationSplits(frozen["train"], frozen["dev"], frozen["test"],
                                  {name: _digest(rows) for name, rows in frozen.items()}, MOD97,
                                  challenge, value, block)
    _validate_splits(result)
    return result


def _validate_splits(data: MultiplicationSplits) -> None:
    rows = (data.train, data.dev, data.test)
    identities = [{(left, right) for left, right, _ in group} for group in rows]
    if sum(map(len, rows)) != data.modulus ** 2 or any(len(group) != len(keys) for group, keys in zip(rows, identities)):
        raise RuntimeError("examples are not an exhaustive identity partition")
    if any(identities[left] & identities[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("split leakage detected")
    location = {pair: index for index, keys in enumerate(identities) for pair in keys}
    if any(location[left, right] != location[right, left] for left, right in location):
        raise RuntimeError("reverse pair crossed a split")
    if any(label != verify_multiplication_label(left, right, data.modulus)
           for group in rows for left, right, label in group):
        raise RuntimeError("independent label audit failed")
    if any(data.digests[name] != _digest(group) for name, group in
           (("train", data.train), ("dev", data.dev), ("test", data.test))):
        raise RuntimeError("split digest audit failed")


def build_mod101_evaluation(seed: int = 20261028) -> OutOfDomainEvaluation:
    """Return a separately labeled, full out-of-domain multiplication-mod-101 set."""
    _validate_seed(seed)
    records = [(left, right, _label_from_generator(left, right, MOD101))
               for left in range(MOD101) for right in range(MOD101)]
    _rng(seed, "mod101-evaluation-order").shuffle(records)
    examples = tuple(records)
    if any(label != verify_multiplication_label(left, right, MOD101) for left, right, label in examples):
        raise RuntimeError("independent mod-101 label audit failed")
    return OutOfDomainEvaluation(examples, _digest(examples), MOD101, "out_of_domain_mod101")


# Descriptive aliases for experiment callers.
multiplication_mod97_splits = build_mod97_splits
mod101_evaluation = build_mod101_evaluation
