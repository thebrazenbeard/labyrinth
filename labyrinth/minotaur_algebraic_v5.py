"""Compact trainable neural operator for arithmetic over the residues modulo 97.

The operator deliberately has no parameter indexed by an operand *pair*.  Each
residue gets an embedding and fixed periodic features; a low-rank bilinear
tensor shared across all pairs supplies the interaction.  The module is a
predictor, not an arithmetic implementation: no result is computed in
``forward`` and its randomly initialized logits are not expected to be right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


MODULUS = 97
OPERATION_COUNT = 3
_PARAMETER_LIMIT = 150_000


@dataclass(frozen=True)
class AlgebraicConfig:
    """Shape and initialization controls for :class:`AlgebraicNeuralOperator`."""

    seed: int = 0
    width: int = 48
    rank: int = 20

    def validate(self) -> "AlgebraicConfig":
        for name in ("seed", "width", "rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.width < 1:
            raise ValueError("width must be positive")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if _parameter_count(self.width, self.rank) >= _PARAMETER_LIMIT:
            raise ValueError(
                f"configuration has {_parameter_count(self.width, self.rank):,} "
                f"trainable parameters; limit is strictly below {_PARAMETER_LIMIT:,}"
            )
        return self


def _parameter_count(width: int, rank: int) -> int:
    # residue embedding, op bias, three CP factors, two op-specific linear maps,
    # two LayerNorms, classifier, periodic input gain, and periodic output gain.
    return 6 * width * width + 9 * width * rank + 202 * width + 98


def _fourier_features(width: int) -> Tensor:
    """Return fixed real Fourier features on the cyclic group Z/97Z."""

    residues = torch.arange(MODULUS, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(width, dtype=torch.long)
    harmonics = torch.div(columns, 2, rounding_mode="floor").add(1).float()
    angles = (2.0 * math.pi / MODULUS) * residues * harmonics.unsqueeze(0)
    features = torch.where((columns % 2).unsqueeze(0) == 0, angles.sin(), angles.cos())
    return features


class AlgebraicNeuralOperator(nn.Module):
    """Low-rank, operation-conditioned neural operator over residue pairs.

    Operation IDs are ``0`` (ADD), ``1`` (SUB), and ``2`` (MUL).  Inputs may
    be any integer representatives; only their residue classes are observed.
    """

    def __init__(self, config: AlgebraicConfig) -> None:
        super().__init__()
        self.config = config.validate()
        width, rank = config.width, config.rank

        self.residue_embedding = nn.Embedding(MODULUS, width)
        self.periodic_input_gain = nn.Parameter(torch.ones(width))
        self.operation_bias = nn.Embedding(OPERATION_COUNT, width)

        # CP-style factorization of one width x width x width interaction tensor
        # per operation.  Indexing is by operation only, never by an operand pair.
        self.left_factor = nn.Parameter(torch.empty(OPERATION_COUNT, width, rank))
        self.right_factor = nn.Parameter(torch.empty(OPERATION_COUNT, width, rank))
        self.output_factor = nn.Parameter(torch.empty(OPERATION_COUNT, rank, width))
        self.left_linear = nn.Parameter(torch.empty(OPERATION_COUNT, width, width))
        self.right_linear = nn.Parameter(torch.empty(OPERATION_COUNT, width, width))

        self.input_norm = nn.LayerNorm(width)
        self.hidden_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, MODULUS)
        self.periodic_output_gain = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("periodic_features", _fourier_features(width), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        width, rank = self.config.width, self.config.rank
        nn.init.normal_(self.residue_embedding.weight, std=width**-0.5)
        nn.init.normal_(self.operation_bias.weight, std=width**-0.5)
        nn.init.normal_(self.left_factor, std=width**-0.5)
        nn.init.normal_(self.right_factor, std=width**-0.5)
        nn.init.normal_(self.output_factor, std=rank**-0.5)
        nn.init.xavier_uniform_(self.left_linear)
        nn.init.xavier_uniform_(self.right_linear)
        self.input_norm.reset_parameters()
        self.hidden_norm.reset_parameters()
        self.classifier.reset_parameters()
        nn.init.ones_(self.periodic_input_gain)
        with torch.no_grad():
            self.periodic_output_gain.fill_(0.1)

    def forward(self, op: Tensor, x: Tensor, y: Tensor) -> Tensor:
        self._validate_inputs(op, x, y)
        x_index = torch.remainder(x, MODULUS)
        y_index = torch.remainder(y, MODULUS)

        left = self.input_norm(
            self.residue_embedding(x_index)
            + self.periodic_input_gain * self.periodic_features[x_index]
        )
        right = self.input_norm(
            self.residue_embedding(y_index)
            + self.periodic_input_gain * self.periodic_features[y_index]
        )

        selected_left = self.left_factor[op]
        selected_right = self.right_factor[op]
        left_low_rank = torch.einsum("bw,bwr->br", left, selected_left)
        right_low_rank = torch.einsum("bw,bwr->br", right, selected_right)
        interaction = left_low_rank * right_low_rank
        swapped_left = torch.einsum("bw,bwr->br", right, selected_left)
        swapped_right = torch.einsum("bw,bwr->br", left, selected_right)
        swapped_interaction = swapped_left * swapped_right
        commutative = (op != 1).unsqueeze(1)
        interaction = torch.where(
            commutative, (interaction + swapped_interaction) * 0.5, interaction
        )
        bilinear = torch.einsum(
            "br,brw->bw",
            interaction,
            self.output_factor[op],
        ) / math.sqrt(self.config.rank)
        selected_left_linear = self.left_linear[op]
        selected_right_linear = self.right_linear[op]
        linear = torch.einsum("bw,bwv->bv", left, selected_left_linear)
        linear = linear + torch.einsum("bw,bwv->bv", right, selected_right_linear)
        swapped_linear = torch.einsum("bw,bwv->bv", right, selected_left_linear)
        swapped_linear = swapped_linear + torch.einsum(
            "bw,bwv->bv", left, selected_right_linear
        )
        linear = torch.where(commutative, (linear + swapped_linear) * 0.5, linear)
        hidden = torch.tanh(self.hidden_norm(bilinear + linear + self.operation_bias(op)))

        learned_logits = self.classifier(hidden)
        periodic_logits = hidden @ self.periodic_features.transpose(0, 1)
        return learned_logits + self.periodic_output_gain * periodic_logits / math.sqrt(
            self.config.width
        )

    @staticmethod
    def _validate_inputs(op: Tensor, x: Tensor, y: Tensor) -> None:
        for name, value in (("op", op), ("x", x), ("y", y)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
            if value.ndim != 1:
                raise ValueError(f"{name} must have shape [batch]")
        if op.shape != x.shape or op.shape != y.shape:
            raise ValueError("op, x, and y must have identical shapes")
        if bool(((op < 0) | (op >= OPERATION_COUNT)).any()):
            raise ValueError("op values must be 0 (ADD), 1 (SUB), or 2 (MUL)")


def build_algebraic_model(config: AlgebraicConfig) -> nn.Module:
    """Build a reproducibly initialized model without changing caller RNG state."""

    if not isinstance(config, AlgebraicConfig):
        raise TypeError("config must be an AlgebraicConfig")
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return AlgebraicNeuralOperator(config)


__all__ = ["AlgebraicConfig", "AlgebraicNeuralOperator", "build_algebraic_model"]
