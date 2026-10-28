"""A small, deliberately constrained group-composition multiplication study.

``GroupPrototypeMultiplier`` is *not* given multiplication at inference.  It
looks up two trainable residue vectors, adds them, and compares that composed
vector to trainable output-class prototypes.  Thus its only algebraic
assumption is the supplied, commutative latent composition prior
``theta[x] + theta[y]``.  It does not learn or prove a group axiom.

Multiplication labels exist only in the offline example builders below.  In
particular, neither ``forward`` nor model parameters contain pair-indexed
tables, modular products, or a discrete-log representation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODULUS = 97
CHALLENGE_VALUE = 53
PARAMETER_LIMIT = 10_000


@dataclass(frozen=True)
class GroupConfig:
    """Bounded CPU configuration; two independently seeded runs are required."""

    seeds: tuple[int, int] = (20261028, 20261029)
    dimension: int = 24
    epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 0.04
    weight_decay: float = 0.0001
    holdout_fraction: float = 0.20
    max_optimizer_seconds: float = 8.0

    def validate(self) -> "GroupConfig":
        if (type(self.seeds) is not tuple or len(self.seeds) != 2
                or len(set(self.seeds)) != 2
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in self.seeds)):
            raise ValueError("seeds must be two distinct integer seeds")
        if type(self.dimension) is not int or not 1 <= self.dimension <= 50:
            raise ValueError("dimension must be an integer in [1, 50]")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 500:
            raise ValueError("epochs must be an integer in [1, 500]")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= MODULUS ** 2:
            raise ValueError("batch_size is invalid")
        if type(self.learning_rate) is not float or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be a float in (0, 1]")
        if type(self.weight_decay) is not float or not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be a float in [0, 1]")
        if type(self.holdout_fraction) is not float or not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be a float in (0, 1)")
        if type(self.max_optimizer_seconds) is not float or not 0.1 <= self.max_optimizer_seconds <= 60.0:
            raise ValueError("max_optimizer_seconds must be a float in [0.1, 60]")
        if parameter_count_for_dimension(self.dimension) >= PARAMETER_LIMIT:
            raise ValueError("configuration is not strictly below 10k parameters")
        return self


@dataclass(frozen=True)
class PairExamples:
    x: Tensor
    y: Tensor
    target: Tensor

    def __len__(self) -> int:
        return int(self.x.numel())


def parameter_count_for_dimension(dimension: int) -> int:
    """Exact model parameter count (residue vectors and class prototypes)."""
    return 2 * MODULUS * dimension


def multiplication_targets(x: Tensor, y: Tensor) -> Tensor:
    """Offline teacher only; never call from model inference or optimization."""
    return torch.remainder(torch.mul(x, y), MODULUS)


def _examples(pairs: Iterable[tuple[int, int]]) -> PairExamples:
    listed = list(pairs)
    x = torch.tensor([pair[0] for pair in listed], dtype=torch.long)
    y = torch.tensor([pair[1] for pair in listed], dtype=torch.long)
    return PairExamples(x=x, y=y, target=multiplication_targets(x, y))


def build_grouped_split(*, seed: int, holdout_fraction: float = 0.20) -> dict[str, PairExamples]:
    """80/20 split by unordered pairs, so a reversal can never leak."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if type(holdout_fraction) is not float or not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be a float in (0, 1)")
    groups = [(x, y) for x in range(MODULUS) for y in range(x, MODULUS)]
    random.Random(seed).shuffle(groups)
    held = set(groups[:round(len(groups) * holdout_fraction)])
    train, holdout = [], []
    for x in range(MODULUS):
        for y in range(MODULUS):
            (holdout if (min(x, y), max(x, y)) in held else train).append((x, y))
    return {"train": _examples(train), "holdout": _examples(holdout)}


def build_withheld_value_split(value: int = CHALLENGE_VALUE) -> dict[str, PairExamples]:
    """Training never observes ``value`` as an operand; test contains it."""
    if type(value) is not int or not 0 <= value < MODULUS:
        raise ValueError("value must be a residue")
    train = [(x, y) for x in range(MODULUS) for y in range(MODULUS)
             if x != value and y != value]
    challenge = [(x, y) for x in range(MODULUS) for y in range(MODULUS)
                 if x == value or y == value]
    return {"train": _examples(train), "withheld_value": _examples(challenge)}


def build_absent_output_split(value: int = CHALLENGE_VALUE) -> dict[str, PairExamples]:
    """Training is fully denied examples whose multiplication label is ``value``."""
    if type(value) is not int or not 0 <= value < MODULUS:
        raise ValueError("value must be a residue")
    all_examples = _examples((x, y) for x in range(MODULUS) for y in range(MODULUS))
    mask = all_examples.target == value
    return {
        "train": PairExamples(all_examples.x[~mask], all_examples.y[~mask], all_examples.target[~mask]),
        "absent_output": PairExamples(all_examples.x[mask], all_examples.y[mask], all_examples.target[mask]),
    }


class GroupPrototypeMultiplier(nn.Module):
    """Trainable residue vectors plus output prototypes under a supplied sum prior."""

    def __init__(self, config: GroupConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.residue_embedding = nn.Embedding(MODULUS, config.dimension)
        self.class_prototypes = nn.Parameter(torch.empty(MODULUS, config.dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.residue_embedding.weight, std=0.15)
        nn.init.normal_(self.class_prototypes, std=0.15)

    @staticmethod
    def _validate_inputs(x: Tensor, y: Tensor) -> None:
        for name, value in (("x", x), ("y", y)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.long or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional long tensor")
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= MODULUS):
                raise ValueError(f"{name} contains a non-residue")
        if x.shape != y.shape:
            raise ValueError("x and y must have identical shapes")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_inputs(x, y)
        # This is an explicit inductive bias supplied by the experimenter.
        composed = self.residue_embedding(x) + self.residue_embedding(y)
        return -torch.sum((composed.unsqueeze(1) - self.class_prototypes.unsqueeze(0)) ** 2, dim=-1)


def build_group_model(config: GroupConfig, seed: int) -> GroupPrototypeMultiplier:
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GroupPrototypeMultiplier(config)


def _digest(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _evaluate(model: nn.Module, examples: PairExamples) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(examples.x, examples.y)
        loss = F.cross_entropy(logits, examples.target).item()
        correct = int((logits.argmax(-1) == examples.target).sum().item())
    return {"count": len(examples), "accuracy": correct / len(examples), "cross_entropy": float(loss)}


def _constant(examples: PairExamples, value: int) -> dict[str, Any]:
    return {"count": len(examples), "accuracy": float((examples.target == value).float().mean().item()),
            "cross_entropy": None}


def _train_one(config: GroupConfig, seed: int, split: dict[str, PairExamples], evaluation_name: str) -> dict[str, Any]:
    model = build_group_model(config, seed).cpu()
    initial = _digest(model)
    frozen = {name: _evaluate(model, examples) for name, examples in split.items()}
    constant_value = int(torch.bincount(split["train"].target, minlength=MODULUS).argmax())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    order_generator = torch.Generator().manual_seed(seed + 1)
    losses: list[float] = []
    started = time.perf_counter()
    timed_out = False
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(split["train"]), generator=order_generator)
        for start in range(0, len(order), config.batch_size):
            if time.perf_counter() - started >= config.max_optimizer_seconds:
                timed_out = True
                break
            index = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(split["train"].x[index], split["train"].y[index]), split["train"].target[index])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite group-model loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if timed_out:
            break
    if not losses:
        raise RuntimeError("optimizer budget expired before first update")
    candidate = {name: _evaluate(model, examples) for name, examples in split.items()}
    return {"seed": seed, "evaluation": evaluation_name, "candidate": candidate,
            "controls": {"frozen_random": frozen,
                         "constant": {name: _constant(examples, constant_value) for name, examples in split.items()}},
            "training": {"optimizer": "AdamW", "optimizer_steps": len(losses),
                         "optimizer_seconds": time.perf_counter() - started,
                         "stopped_at_time_budget": timed_out, "initial_loss": losses[0],
                         "final_loss": losses[-1], "initial_weight_sha256": initial,
                         "trained_weight_sha256": _digest(model)}}


def run_experiment(config: GroupConfig = GroupConfig()) -> dict[str, Any]:
    """Run all three evaluations for exactly two seeds, with frozen controls."""
    config.validate()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    split_builders = {
        "grouped_pair_holdout": lambda seed: build_grouped_split(seed=seed, holdout_fraction=config.holdout_fraction),
        "withheld_value_53": lambda _seed: build_withheld_value_split(CHALLENGE_VALUE),
        "fully_absent_output_53": lambda _seed: build_absent_output_split(CHALLENGE_VALUE),
    }
    studies: dict[str, list[dict[str, Any]]] = {}
    for name, builder in split_builders.items():
        studies[name] = [_train_one(config, seed, builder(seed), name) for seed in config.seeds]
    negative = []
    for name, runs in studies.items():
        eval_key = "holdout" if name == "grouped_pair_holdout" else ("withheld_value" if name.startswith("withheld") else "absent_output")
        # An equal number of correct cases is not an improvement; tolerate
        # harmless float32/float64 division differences in the JSON metrics.
        if not all(
            run["candidate"][eval_key]["accuracy"]
            > max(run["controls"]["constant"][eval_key]["accuracy"],
                  run["controls"]["frozen_random"][eval_key]["accuracy"]) + 1e-12
            for run in runs
        ):
            negative.append(name)
    return {"schema": "labyrinth.minotaur.v6-group.v1", "config": asdict(config),
            "model": {"architecture": "learned_residue_vectors_plus_class_prototypes", "parameter_count": parameter_count_for_dimension(config.dimension),
                      "supplied_group_composition_prior": "theta[x] + theta[y]", "learned_group_axiom": False,
                      "pair_lookup": False, "pretrained_weights": False},
            "studies": studies, "negative_challenges": negative,
            "conclusion": "exploratory_only; supplied composition prior is not evidence of a learned group axiom"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compact V6 group-composition study")
    parser.add_argument("--epochs", type=int, default=GroupConfig.epochs)
    parser.add_argument("--dimension", type=int, default=GroupConfig.dimension)
    parser.add_argument("--max-seconds", type=float, default=GroupConfig.max_optimizer_seconds)
    args = parser.parse_args(argv)
    report = run_experiment(GroupConfig(epochs=args.epochs, dimension=args.dimension, max_optimizer_seconds=args.max_seconds))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CHALLENGE_VALUE", "GroupConfig", "GroupPrototypeMultiplier", "MODULUS", "PairExamples",
           "build_absent_output_split", "build_group_model", "build_grouped_split", "build_withheld_value_split",
           "multiplication_targets", "parameter_count_for_dimension", "run_experiment"]
