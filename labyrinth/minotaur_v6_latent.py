"""A compact latent neural experiment for multiplication modulo 97.

This is deliberately an experiment, rather than an arithmetic implementation.
``LatentMultiplier.forward`` receives only two operands and returns learned
class logits.  It has no pair-indexed table, no target argument, and does not
perform modular multiplication.  Labels live solely in the data helpers.

The held-out split is a four-by-four operand-group grid: one quarter of the
cells are withheld, while every individual residue remains represented in the
training rows.  Thus a held-out score measures unseen *pairs*, not unseen
tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


MODULUS = 97
_PAIR_COUNT = MODULUS * MODULUS


@dataclass(frozen=True)
class LatentConfig:
    """Architecture controls; defaults stay below 100k trainable parameters."""

    seed: int = 20261028
    harmonics: int = 16
    width: int = 96
    depth: int = 2

    def validate(self) -> "LatentConfig":
        for name in ("seed", "harmonics", "width", "depth"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0 or not 1 <= self.harmonics <= 48:
            raise ValueError("seed must be non-negative and harmonics in [1, 48]")
        if not 8 <= self.width <= 128 or not 1 <= self.depth <= 3:
            raise ValueError("width must be in [8, 128] and depth in [1, 3]")
        return self


def _features(harmonics: int) -> Tensor:
    """Fixed real Fourier features for every residue; they contain no labels."""
    residues = torch.arange(MODULUS, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(1, harmonics + 1, dtype=torch.float32).unsqueeze(0)
    angle = residues * frequencies * (2.0 * math.pi / MODULUS)
    return torch.cat((angle.sin(), angle.cos()), dim=1)


class LatentMultiplier(nn.Module):
    """Shared low-dimensional interaction network over two residue features."""

    def __init__(self, config: LatentConfig = LatentConfig()) -> None:
        super().__init__()
        self.config = config.validate()
        feature_width = 2 * config.harmonics
        self.register_buffer("fourier", _features(config.harmonics), persistent=True)
        self.left = nn.Linear(feature_width, config.width, bias=False)
        self.right = nn.Linear(feature_width, config.width, bias=False)
        # A Hadamard product is a compact latent bilinear interaction.  Its
        # weights are shared across all 9,409 input pairs.
        blocks: list[nn.Module] = []
        for _ in range(config.depth):
            blocks.extend((nn.Linear(config.width, config.width), nn.GELU()))
        self.latent = nn.Sequential(*blocks)
        self.classifier = nn.Linear(config.width, MODULUS)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self._validate(x, y)
        x_feature = self.fourier[torch.remainder(x, MODULUS)]
        y_feature = self.fourier[torch.remainder(y, MODULUS)]
        hidden = self.left(x_feature) * self.right(y_feature)
        return self.classifier(self.latent(hidden))

    @staticmethod
    def _validate(x: Tensor, y: Tensor) -> None:
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError("x and y must be torch tensors")
        if x.dtype != torch.long or y.dtype != torch.long:
            raise TypeError("x and y must have dtype torch.long")
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("x and y must have matching shape [batch]")


def build_latent_model(config: LatentConfig = LatentConfig()) -> LatentMultiplier:
    """Build reproducibly without perturbing the caller's Torch RNG state."""
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return LatentMultiplier(config)


def grouped_pair_split(groups: int = 4, heldout_cell: int = 0) -> dict[str, Tensor]:
    """Return train/test pair tensors with complete residue coverage in train.

    Pair ``(x, y)`` is held out when ``(x % groups + (y % groups)) %
    groups == heldout_cell``.  This creates a deterministic, balanced
    checkerboard-like group holdout and never puts the target in the features.
    """
    if type(groups) is not int or not 2 <= groups <= 16:
        raise ValueError("groups must be an integer in [2, 16]")
    if type(heldout_cell) is not int or not 0 <= heldout_cell < groups:
        raise ValueError("heldout_cell must be a valid group index")
    x = torch.arange(MODULUS, dtype=torch.long).repeat_interleave(MODULUS)
    y = torch.arange(MODULUS, dtype=torch.long).repeat(MODULUS)
    heldout = ((torch.remainder(x, groups) + torch.remainder(y, groups)) % groups
               == heldout_cell)
    # Labels are data-generation metadata and are intentionally not passed to
    # LatentMultiplier.forward.
    label = torch.remainder(x * y, MODULUS)
    return {"train_x": x[~heldout], "train_y": y[~heldout], "train_label": label[~heldout],
            "test_x": x[heldout], "test_y": y[heldout], "test_label": label[heldout]}


def train_and_evaluate(config: LatentConfig = LatentConfig(), *, steps: int = 2_000,
                       batch_size: int = 256, lr: float = 3e-3,
                       heldout_cell: int = 0) -> dict[str, Any]:
    """Train only on the grouped training cells and return both real scores."""
    if type(steps) is not int or not 1 <= steps <= 20_000:
        raise ValueError("steps must be an integer in [1, 20000]")
    if type(batch_size) is not int or not 1 <= batch_size <= _PAIR_COUNT:
        raise ValueError("batch_size must be in [1, 9409]")
    if not isinstance(lr, (float, int)) or isinstance(lr, bool) or not 0 < lr <= 1:
        raise ValueError("lr must be positive")
    config.validate()
    split = grouped_pair_split(heldout_cell=heldout_cell)
    model = build_latent_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    generator = torch.Generator().manual_seed(config.seed + 1)
    model.train()
    for _ in range(steps):
        index = torch.randint(len(split["train_x"]), (batch_size,), generator=generator)
        logits = model(split["train_x"][index], split["train_y"][index])
        loss = nn.functional.cross_entropy(logits, split["train_label"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        def score(prefix: str) -> dict[str, float | int]:
            logits = model(split[f"{prefix}_x"], split[f"{prefix}_y"])
            prediction = logits.argmax(dim=1)
            labels = split[f"{prefix}_label"]
            correct = int((prediction == labels).sum().item())
            return {"correct": correct, "total": len(labels), "accuracy": correct / len(labels)}
        return {"architecture": "fixed Fourier features + shared latent Hadamard interaction",
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "steps": steps, "heldout_cell": heldout_cell,
                "train": score("train"), "withheld_groups": score("test")}


__all__ = ["MODULUS", "LatentConfig", "LatentMultiplier", "build_latent_model",
           "grouped_pair_split", "train_and_evaluate"]
