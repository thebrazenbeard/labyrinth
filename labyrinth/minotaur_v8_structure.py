"""V8: a learned recurrent binary-digit state machine for modular products.

The model sees the seven binary digits of each operand and modulus, in order,
and evolves a shared GRU state.  Its only supplied arithmetic prior is that
integers have a fixed-width binary representation and an ordered digit stream.
There are no residue embeddings, output label table, input-product feature, or
modular operation in ``forward``.  ``multiplication_targets`` is an offline
teacher used exclusively while constructing labelled datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
import time
from typing import Any

import torch
from torch import nn

MOD = 97
BITS = 7
CHALLENGE = 53


@dataclass(frozen=True)
class StructureConfig:
    seed: int = 20261028
    seeds: tuple[int, int] = (20261028, 20261029)
    hidden_size: int = 56
    layers: int = 1
    epochs: int = 28
    batch_size: int = 512
    learning_rate: float = .003
    max_optimizer_seconds: float = 40.0
    dev_fraction: float = .10

    def validate(self):
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.seeds) is not tuple or len(self.seeds) != 2
                or len(set(self.seeds)) != 2
                or any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds)
                or type(self.hidden_size) is not int or not 16 <= self.hidden_size <= 96
                or type(self.layers) is not int or not 1 <= self.layers <= 2
                or type(self.epochs) is not int or not 1 <= self.epochs <= 160
                or type(self.batch_size) is not int or not 16 <= self.batch_size <= 2048
                or type(self.learning_rate) is not float or not 0 < self.learning_rate <= .02
                or type(self.max_optimizer_seconds) is not float or not 1 <= self.max_optimizer_seconds <= 90
                or type(self.dev_fraction) is not float or not .02 <= self.dev_fraction <= .30):
            raise ValueError("invalid bounded V8 configuration")
        return self


@dataclass(frozen=True)
class Batch:
    x: Any
    y: Any
    target: Any

    def __len__(self):
        return int(self.x.numel())


@dataclass(frozen=True)
class StructureSplit:
    train: Batch
    dev: Batch
    missing_output: Batch
    unseen_operand: Batch
    mod101_test: Batch


def multiplication_targets(x, y, modulus=MOD):
    """Offline label oracle.  Never call this from a neural forward pass."""
    return torch.remainder(x * y, modulus)


def _batch(pairs, modulus=MOD):
    pairs = list(pairs)
    x = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    y = torch.tensor([b for _, b in pairs], dtype=torch.long)
    return Batch(x, y, multiplication_targets(x, y, modulus))


def _directions(groups):
    return [(a, b) for a, b in groups for a, b in (((a, b), (b, a)) if a != b else ((a, b),))]


def build_structure_split(*, seed=20261028, dev_fraction=.10):
    """Train/dev contain neither challenge label nor operand; challenges are test-only."""
    if type(seed) is not int or seed < 0 or not .02 <= dev_fraction <= .30:
        raise ValueError("invalid split")
    groups = [(a, b) for a in range(MOD) for b in range(a, MOD)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    eligible, missing, operand = [], [], []
    for a, b in groups:
        target = (a * b) % MOD
        # Keep this at V7's exact 94-row missing-output comparison: the two
        # rows that also use operand 53 belong only to the operand challenge.
        if target == CHALLENGE and a != CHALLENGE and b != CHALLENGE:
            missing.append((a, b))
        if a == CHALLENGE or b == CHALLENGE:
            operand.append((a, b))
        if target != CHALLENGE and a != CHALLENGE and b != CHALLENGE:
            eligible.append((a, b))
    n = max(1, round(len(eligible) * dev_fraction))
    return StructureSplit(
        _batch(_directions(eligible[n:])), _batch(_directions(eligible[:n])),
        _batch(_directions(missing)), _batch(_directions(operand)),
        _batch(((a, b) for a in range(101) for b in range(101)), 101),
    )


class DigitStateMultiplier(nn.Module):
    """A small recurrent state machine over binary digit triples, with scalar head."""
    def __init__(self, config: StructureConfig):
        super().__init__()
        self.digit_encoder = nn.Sequential(nn.Linear(3, config.hidden_size), nn.Tanh())
        self.state_machine = nn.GRU(config.hidden_size, config.hidden_size, config.layers,
                                    batch_first=True)
        self.readout = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2),
                                     nn.Tanh(), nn.Linear(config.hidden_size // 2, 1), nn.Sigmoid())

    @staticmethod
    def _digits(values):
        shifts = torch.arange(BITS - 1, -1, -1, device=values.device)
        return torch.bitwise_and(torch.bitwise_right_shift(values[:, None], shifts), 1).float()

    def forward(self, x, y, modulus):
        if (not all(isinstance(value, torch.Tensor) for value in (x, y, modulus))
                or x.dtype != torch.long or y.dtype != torch.long or modulus.dtype != torch.long
                or x.ndim != 1 or x.shape != y.shape or x.shape != modulus.shape
                or bool(((modulus < 3) | (modulus > 101)).any())
                or bool(((x < 0) | (y < 0) | (x >= modulus) | (y >= modulus)).any())):
            raise ValueError("invalid bounded residues")
        left, right, base = self._digits(x), self._digits(y), self._digits(modulus)
        # Sum and absolute difference make the learned state symmetric in operands.
        stream = torch.stack((left + right, torch.abs(left - right), base), dim=-1)
        encoded = self.digit_encoder(stream)
        states, _ = self.state_machine(encoded)
        normalized = self.readout(states[:, -1, :]).squeeze(-1)
        return torch.mul(normalized, modulus.float())


def build_structure_model(config=StructureConfig()):
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return DigitStateMultiplier(config)


def _digest(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _score(model, batch, modulus):
    model.eval()
    answers = []
    with torch.no_grad():
        for start in range(0, len(batch), 1024):
            x, y = batch.x[start:start + 1024], batch.y[start:start + 1024]
            predicted = model(x, y, torch.full_like(x, modulus))
            answers.extend(predicted.round().long().clamp(0, modulus - 1).tolist())
    correct = sum(answer == target for answer, target in zip(answers, batch.target.tolist()))
    return {"correct": correct, "count": len(batch), "accuracy": correct / len(batch) if len(batch) else 0.0}


def _fit(model, train, config, seed, shuffled_labels=False):
    targets = train.target.clone()
    if shuffled_labels:
        targets = targets[torch.randperm(len(targets), generator=torch.Generator().manual_seed(seed + 91))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    started, steps, timed_out = time.perf_counter(), 0, False
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(train), generator=generator)
        for begin in range(0, len(order), config.batch_size):
            if time.perf_counter() - started >= config.max_optimizer_seconds:
                timed_out = True
                break
            pick = order[begin:begin + config.batch_size]
            prediction = model(train.x[pick], train.y[pick], torch.full_like(train.x[pick], MOD))
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets[pick].float())
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite learned-state loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        if timed_out:
            break
    return {"optimizer_steps": steps, "optimizer_seconds": time.perf_counter() - started,
            "stopped_at_budget": timed_out}


def run_structure_study(config=StructureConfig()):
    """Two fixed seeds; test-only challenges are never inspected by the fitting loop."""
    config.validate()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    split = build_structure_split(seed=config.seed, dev_fraction=config.dev_fraction)
    named = {"dev": split.dev, "missing_output_53": split.missing_output,
             "unseen_operand_53": split.unseen_operand, "modulus_101": split.mod101_test}
    reports = []
    for seed in config.seeds:
        cfg = StructureConfig(seed=seed, seeds=config.seeds, hidden_size=config.hidden_size,
                              layers=config.layers, epochs=config.epochs, batch_size=config.batch_size,
                              learning_rate=config.learning_rate, max_optimizer_seconds=config.max_optimizer_seconds,
                              dev_fraction=config.dev_fraction)
        model = build_structure_model(cfg)
        initial = _digest(model)
        frozen = {name: _score(model, batch, 101 if name == "modulus_101" else MOD)
                  for name, batch in named.items()}
        training = _fit(model, split.train, cfg, seed)
        training["initial_sha256"] = initial
        training["trained_sha256"] = _digest(model)
        evaluation = {name: _score(model, batch, 101 if name == "modulus_101" else MOD)
                      for name, batch in named.items()}
        constant = int(torch.bincount(split.train.target, minlength=MOD).argmax())
        constants = {name: {"accuracy": (batch.target == constant).float().mean().item()}
                     for name, batch in named.items()}
        shuffled = build_structure_model(cfg)
        _fit(shuffled, split.train, cfg, seed, shuffled_labels=True)
        shuffled_scores = {name: _score(shuffled, batch, 101 if name == "modulus_101" else MOD)
                           for name, batch in named.items()}
        reports.append({"seed": seed,
            "split": {"train_count": len(split.train), "dev_count": len(split.dev),
                      "missing_output_train_rows": int((split.train.target == CHALLENGE).sum()),
                      "operand_53_train_rows": int(((split.train.x == CHALLENGE) | (split.train.y == CHALLENGE)).sum())},
            "controls": {"constant": constants, "frozen_random": frozen, "shuffled_labels": shuffled_scores},
            "evaluation": evaluation, "training": training,
            "conclusion": "reported_without_test_selection"})
    return {"schema": "labyrinth.minotaur.v8-structure.v1", "config": asdict(config),
            "model": {"architecture": "symmetric_binary_digit_gru_scalar_residue_head",
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "hardcoded_arithmetic_priors": "seven-bit integer representation, digit order, operand symmetry",
                      "forward_product_or_modulo": False}, "runs": reports,
            "promotion_recommendation": "DO_NOT_PROMOTE", "qualification": "NOT_ESTABLISHED",
            "limits": "Public synthetic experiment; 101 is evaluation-only and no test split feeds fitting."}
