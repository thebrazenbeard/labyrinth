"""Bounded neural tensor completion for multiplication modulo 97.

The candidate is a symmetric CP-style spectral factorization.  The same
randomly initialized residue embedding encodes both operands, so there is no
parameter indexed by an operand pair.  Supervision is constructed separately:
``forward`` only embeds two observed residue IDs and composes latent features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODULUS = 97
PARAMETER_LIMIT = 150_000


@dataclass(frozen=True)
class SpectralConfig:
    """Controls for one CPU-only, random-initialization experiment."""

    seed: int = 20261028
    rank: int = 192
    epochs: int = 160
    batch_size: int = 1024
    learning_rate: float = 0.03
    weight_decay: float = 0.0001
    holdout_fraction: float = 0.20
    max_optimizer_seconds: float = 85.0

    def validate(self) -> "SpectralConfig":
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 10_000:
            raise ValueError("epochs must be an integer in [1, 10000]")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= MODULUS**2:
            raise ValueError("batch_size is invalid")
        if type(self.learning_rate) is not float or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be a float in (0, 1]")
        if type(self.weight_decay) is not float or not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be a float in [0, 1]")
        if type(self.holdout_fraction) is not float or not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be a float in (0, 1)")
        if (type(self.max_optimizer_seconds) is not float
                or not 0.1 <= self.max_optimizer_seconds <= 90.0):
            raise ValueError("max_optimizer_seconds must be a float in [0.1, 90]")
        if parameter_count_for_rank(self.rank) >= PARAMETER_LIMIT:
            raise ValueError("configuration is not strictly below 150k parameters")
        return self


@dataclass(frozen=True)
class PairExamples:
    x: Tensor
    y: Tensor
    target: Tensor

    def __len__(self) -> int:
        return int(self.x.numel())


def parameter_count_for_rank(rank: int) -> int:
    """Exact parameter count for :class:`SpectralMultiplier`."""

    # shared residue embedding, rank gains, decoder, and class bias
    return MODULUS * rank + rank + rank * MODULUS + MODULUS


def multiplication_targets(x: Tensor, y: Tensor) -> Tensor:
    """Teacher used only while constructing supervised examples."""

    return torch.remainder(torch.mul(x, y), MODULUS)


def build_grouped_split(*, seed: int, holdout_fraction: float = 0.20) -> dict[str, PairExamples]:
    """Split all pairs by unordered group, keeping reversals together."""

    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if type(holdout_fraction) is not float or not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be a float in (0, 1)")
    groups = [(x, y) for x in range(MODULUS) for y in range(x, MODULUS)]
    random.Random(seed).shuffle(groups)
    held_group_count = round(len(groups) * holdout_fraction)
    held_groups = set(groups[:held_group_count])
    values: dict[str, list[tuple[int, int]]] = {"train": [], "holdout": []}
    for x in range(MODULUS):
        for y in range(MODULUS):
            key = (x, y) if x <= y else (y, x)
            name = "holdout" if key in held_groups else "train"
            values[name].append((x, y))
    output = {}
    for name, pairs in values.items():
        x = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        y = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        output[name] = PairExamples(x=x, y=y, target=multiplication_targets(x, y))
    return output


class SpectralMultiplier(nn.Module):
    """Symmetric low-rank completion model with shared operand weights."""

    def __init__(self, config: SpectralConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.residue_embedding = nn.Embedding(MODULUS, config.rank)
        # Aliases document and expose the weight-sharing contract.  PyTorch
        # de-duplicates the shared module in state and parameter traversal.
        self.left_embedding = self.residue_embedding
        self.right_embedding = self.residue_embedding
        self.spectral_gain = nn.Parameter(torch.ones(config.rank))
        self.decoder = nn.Linear(config.rank, MODULUS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.residue_embedding.weight, mean=0.0, std=0.2)
        nn.init.ones_(self.spectral_gain)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate_inputs(x, y)
        left = self.residue_embedding(torch.remainder(x, MODULUS))
        right = self.residue_embedding(torch.remainder(y, MODULUS))
        # Hadamard composition is symmetric and learned entirely in latent
        # coordinates; it does not perform arithmetic on the operand IDs.
        interaction = torch.mul(left, right)
        return self.decoder(interaction * self.spectral_gain)

    @staticmethod
    def _validate_inputs(x: Tensor, y: Tensor) -> None:
        for name, value in (("x", x), ("y", y)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
            if value.ndim != 1:
                raise ValueError(f"{name} must have shape [batch]")
        if x.shape != y.shape:
            raise ValueError("x and y must have identical shapes")


def build_spectral_model(config: SpectralConfig) -> SpectralMultiplier:
    """Construct seeded random weights without changing caller RNG state."""

    if not isinstance(config, SpectralConfig):
        raise TypeError("config must be a SpectralConfig")
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return SpectralMultiplier(config)


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        raw = value.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
        digest.update(bytes(raw.tolist()))
    return digest.hexdigest()


def _evaluate(model: nn.Module, examples: PairExamples) -> dict[str, Any]:
    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), 2048):
            stop = start + 2048
            logits = model(examples.x[start:stop], examples.y[start:stop])
            target = examples.target[start:stop]
            correct += int((logits.argmax(dim=-1) == target).sum().item())
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
    return {"count": len(examples), "accuracy": correct / len(examples),
            "cross_entropy": loss_sum / len(examples)}


def audit_grouped_predictions(model: nn.Module, examples: PairExamples) -> dict[str, Any]:
    """Recompute actual labels independently and score canonical reverse pairs.

    Each (a,b)/(b,a) group is a SINGLE sampling unit, never two IID samples.
    This is a public arithmetic auditor, not a private examination or a
    source of training labels/gradients.
    """
    from .minotaur_v6_mul_eval import recompute_mul_label
    if len(examples) <= 0 or len(examples) > MODULUS**2:
        raise ValueError("invalid bounded examples for grouped audit")
    model.eval()
    predicted: list[int] = []
    with torch.no_grad():
        for start in range(0, len(examples), 1024):
            logits = model(examples.x[start:start + 1024], examples.y[start:start + 1024])
            if logits.shape != (min(1024, len(examples) - start), MODULUS):
                raise ValueError("malformed predictor logits")
            predicted.extend(int(v) for v in logits.argmax(-1).tolist())
    group: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]] = {}
    entries = []
    for x, y, claim, guess in zip(examples.x.tolist(), examples.y.tolist(),
                                  examples.target.tolist(), predicted):
        if claim != recompute_mul_label(x, y):
            raise ValueError("training-source claimed label differs from independent arithmetic")
        key = (min(x, y), max(x, y))
        directions = group.setdefault(key, {})
        if (x, y) in directions:
            raise ValueError("duplicate directed test input")
        directions[(x, y)] = (claim, guess)
        entries.append((x, y, guess))
    complete_correct = 0
    reversal_consistent = 0
    reversals = 0
    for (x, y), directions in group.items():
        expected_count = 1 if x == y else 2
        if len(directions) != expected_count:
            raise ValueError("incomplete reverse-pair group")
        if all(label == guess for label, guess in directions.values()):
            complete_correct += 1
        if expected_count == 2:
            reversals += 1
            if directions[(x, y)][1] == directions[(y, x)][1]:
                reversal_consistent += 1
    entries.sort()
    count = len(group)
    return {
        "canonical_groups": count, "directed_rows": len(entries),
        "correct_complete_groups": complete_correct,
        "pair_complete_accuracy": complete_correct / count,
        "directed_row_accuracy": sum(claim == guess for dirs in group.values()
                                     for claim, guess in dirs.values()) / len(entries),
        "reversal_consistency_rate": reversal_consistent / reversals if reversals else 1.0,
        "prediction_sha256": hashlib.sha256(
            json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "independent_label_replay": "minotaur_v6_mul_eval.recompute_mul_label",
        "confidence_intervals": "NOT_COMPUTED; paired public examples are not IID",
    }


def _constant_metrics(examples: PairExamples, prediction: int) -> dict[str, Any]:
    accuracy = float((examples.target == prediction).float().mean().item())
    return {"count": len(examples), "accuracy": accuracy,
            "cross_entropy": None}


def run_experiment(config: SpectralConfig = SpectralConfig()) -> dict[str, Any]:
    """Train and compare one candidate against frozen and constant controls."""

    config.validate()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    splits = build_grouped_split(seed=config.seed, holdout_fraction=config.holdout_fraction)
    model = build_spectral_model(config).cpu()
    frozen = {name: _evaluate(model, examples) for name, examples in splits.items()}
    initial_digest = _parameter_digest(model)
    constant = int(torch.bincount(splits["train"].target, minlength=MODULUS).argmax())
    constant_results = {
        name: _constant_metrics(examples, constant) for name, examples in splits.items()
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    started = time.perf_counter()
    losses: list[float] = []
    gradient_norms: list[float] = []
    timed_out = False
    model.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(len(splits["train"]), generator=generator)
        for start in range(0, len(order), config.batch_size):
            if time.perf_counter() - started >= config.max_optimizer_seconds:
                timed_out = True
                break
            indices = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(splits["train"].x[indices], splits["train"].y[indices])
            loss = F.cross_entropy(logits, splits["train"].target[indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite spectral training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite spectral gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(norm.detach().item()))
        if timed_out:
            break
    elapsed = time.perf_counter() - started
    if not losses:
        raise RuntimeError("optimizer budget expired before the first update")
    trained_digest = _parameter_digest(model)
    if trained_digest == initial_digest:
        raise RuntimeError("training did not update model parameters")

    candidate = {name: _evaluate(model, examples) for name, examples in splits.items()}
    grouped_holdout = audit_grouped_predictions(model, splits["holdout"])
    return {
        "schema": "labyrinth.minotaur.v6-spectral.v1",
        "config": asdict(config),
        "model": {
            "architecture": "shared_symmetric_cp_spectral_factorization",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "random_initialization": True,
            "pretrained_weights": False,
        },
        "split": {
            "method": "grouped_unordered_pair_80_20",
            "train_count": len(splits["train"]),
            "holdout_count": len(splits["holdout"]),
            "reversals_grouped": True,
            "holdout_used_for_training": False,
        },
        "training": {
            "optimizer": "AdamW",
            "optimizer_steps": len(losses),
            "optimizer_seconds": elapsed,
            "stopped_at_time_budget": timed_out,
            "initial_minibatch_loss": losses[0],
            "final_minibatch_loss": losses[-1],
            "minimum_minibatch_loss": min(losses),
            "gradient_norm_max": max(gradient_norms),
            "initial_weight_sha256": initial_digest,
            "trained_weight_sha256": trained_digest,
        },
        "candidate": candidate,
        "pair_group_audit": grouped_holdout,
        "controls": {
            "constant": {**constant_results},
            "frozen_random": frozen,
        },
        "conclusion": (
            "positive_holdout_signal" if candidate["holdout"]["accuracy"]
            > max(constant_results["holdout"]["accuracy"],
                  frozen["holdout"]["accuracy"])
            else "no_holdout_improvement_over_controls"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded modulo-97 spectral study")
    parser.add_argument("--output", help="optional new JSON report path")
    parser.add_argument("--epochs", type=int, default=SpectralConfig.epochs)
    parser.add_argument("--rank", type=int, default=SpectralConfig.rank)
    parser.add_argument("--max-seconds", type=float,
                        default=SpectralConfig.max_optimizer_seconds)
    args = parser.parse_args(argv)
    report = run_experiment(SpectralConfig(
        rank=args.rank, epochs=args.epochs, max_optimizer_seconds=args.max_seconds
    ))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        if path.exists():
            parser.error("refusing to overwrite existing report")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODULUS", "PairExamples", "SpectralConfig", "SpectralMultiplier",
    "build_grouped_split", "build_spectral_model", "multiplication_targets",
    "parameter_count_for_rank", "run_experiment",
]
