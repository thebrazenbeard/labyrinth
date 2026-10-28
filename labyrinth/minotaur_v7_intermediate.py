"""Compact multi-head neural arithmetic probe, intentionally not an arithmetic program.

Supplied inductive priors: shared operand encoder, exchange symmetry through
sum/absolute-difference interaction, bounded normalized coordinates, and four
fixed sinusoidal coordinate features.  The model has no multiplication,
modular reduction, lookup table, or discrete-log routine in its forward pass.
Offline labels are used only by split construction and training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
import time
from typing import Any

import torch
from torch import nn

MOD = 97
ABSENT = 53


@dataclass(frozen=True)
class IntermediateConfig:
    seed: int = 20261028
    seeds: tuple[int, int] = (20261028, 20261029)
    width: int = 28
    depth: int = 2
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = .003
    max_optimizer_seconds: float = 90.0
    holdout_fraction: float = .10

    def validate(self):
        valid = (type(self.seed) is int and 0 <= self.seed < 2**31 and
                 type(self.seeds) is tuple and len(self.seeds) == 2 and len(set(self.seeds)) == 2 and
                 all(type(s) is int and 0 <= s < 2**31 for s in self.seeds) and
                 type(self.width) is int and 8 <= self.width <= 64 and
                 type(self.depth) is int and 1 <= self.depth <= 4 and
                 type(self.epochs) is int and 1 <= self.epochs <= 200 and
                 type(self.batch_size) is int and 16 <= self.batch_size <= 2048 and
                 type(self.learning_rate) is float and 0 < self.learning_rate <= .02 and
                 type(self.max_optimizer_seconds) is float and 1 <= self.max_optimizer_seconds <= 90 and
                 type(self.holdout_fraction) is float and .02 <= self.holdout_fraction <= .30)
        if not valid:
            raise ValueError("invalid bounded intermediate configuration")
        return self


@dataclass(frozen=True)
class Batch:
    x: Any
    y: Any
    remainder: Any
    quotient: Any
    product_scale: Any

    def __len__(self): return int(self.x.numel())


@dataclass(frozen=True)
class IntermediateSplit:
    train: Batch
    independent_holdout: Batch
    absent_output: Batch
    absent_operand: Batch


def offline_targets(x, y, modulus=MOD):
    """Teacher-only labels.  Never call this function from a model forward."""
    product = x * y
    return (torch.remainder(product, modulus), torch.div(product, modulus, rounding_mode="floor"),
            product.float() / float((modulus - 1) ** 2))


def _batch(pairs, modulus=MOD):
    pairs = list(pairs)
    x = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    y = torch.tensor([b for _, b in pairs], dtype=torch.long)
    remainder, quotient, product_scale = offline_targets(x, y, modulus)
    return Batch(x, y, remainder, quotient, product_scale)


def _directions(groups):
    return [(a, b) for a, b in groups for a, b in (((a, b), (b, a)) if a != b else ((a, b),))]


def build_intermediate_split(*, seed=20261028, holdout_fraction=.10):
    if type(seed) is not int or seed < 0 or not .02 <= holdout_fraction <= .30:
        raise ValueError("invalid split")
    groups = [(a, b) for a in range(MOD) for b in range(a, MOD)]
    random.Random(seed).shuffle(groups)
    available, output, operand = [], [], []
    for a, b in groups:
        label = (a * b) % MOD  # offline partition label only
        if a == ABSENT or b == ABSENT:
            operand.append((a, b))
        elif label == ABSENT:
            output.append((a, b))
        else:
            available.append((a, b))
    n = max(1, round(len(available) * holdout_fraction))
    return IntermediateSplit(_batch(_directions(available[n:])), _batch(_directions(available[:n])),
                             _batch(_directions(output)), _batch(_directions(operand)))


def audit_group_pairs(split):
    names = ("train", "independent_holdout", "absent_output", "absent_operand")
    groups = {name: {(min(a, b), max(a, b)) for a, b in zip(getattr(split, name).x.tolist(), getattr(split, name).y.tolist())}
              for name in names}
    return {"groups": {name: len(values) for name, values in groups.items()},
            "all_disjoint": all(not (groups[left] & groups[right]) for index, left in enumerate(names) for right in names[index + 1:]),
            "reversals_together": all((b, a) in set(zip(getattr(split, name).x.tolist(), getattr(split, name).y.tolist()))
                                      for name in names for a, b in zip(getattr(split, name).x.tolist(), getattr(split, name).y.tolist()))}


class IntermediateNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.operand = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        layers = [nn.Linear(2 * width, width), nn.GELU()]
        for _ in range(config.depth - 1): layers.extend((nn.Linear(width, width), nn.GELU()))
        self.interaction = nn.Sequential(*layers)
        self.remainder_head = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())
        self.quotient_head = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())
        self.product_head = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())

    @staticmethod
    def _features(values, modulus):
        coordinate = values.float() / (modulus.float() - 1.0)
        features = [coordinate]
        for frequency in (1, 2, 4, 8):
            angle = coordinate * (2.0 * math.pi * frequency)
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.stack(features, dim=-1)

    def forward(self, x, y, modulus):
        if (not all(isinstance(v, torch.Tensor) for v in (x, y, modulus)) or x.dtype != torch.long or
                y.dtype != torch.long or modulus.dtype != torch.long or x.ndim != 1 or x.shape != y.shape or
                x.shape != modulus.shape or bool(((modulus < 3) | (modulus > 101)).any()) or
                bool(((x < 0) | (y < 0) | (x >= modulus) | (y >= modulus)).any())):
            raise ValueError("invalid bounded residues")
        left, right = self.operand(self._features(x, modulus)), self.operand(self._features(y, modulus))
        state = self.interaction(torch.cat((left + right, torch.abs(left - right)), dim=-1))
        scale = modulus.float() - 1.0
        return {"remainder": self.remainder_head(state).squeeze(-1) * scale,
                "quotient": self.quotient_head(state).squeeze(-1) * scale,
                "product_scale": self.product_head(state).squeeze(-1)}


def build_intermediate_model(config=IntermediateConfig()):
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return IntermediateNetwork(config)


def _digest(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values(): digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _predict_values(model, batch, modulus):
    """Return neural predictions only, never a gold arithmetic result."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(batch), 1024):
            item = slice(start, start + 1024)
            out = model(batch.x[item], batch.y[item], torch.full_like(batch.x[item], modulus))["remainder"]
            predictions.extend(out.round().long().clamp(0, modulus - 1).tolist())
    return predictions


def _score(model, batch, modulus):
    prediction = _predict_values(model, batch, modulus)
    hits = sum(a == b for a, b in zip(prediction, batch.remainder.tolist()))
    return {"correct": hits, "count": len(batch), "accuracy": hits / len(batch) if len(batch) else 0.0}


def _loss(prediction, batch, modulus):
    scale = float(modulus - 1)
    rem = torch.nn.functional.smooth_l1_loss(prediction["remainder"] / scale, batch.remainder.float() / scale)
    quo = torch.nn.functional.smooth_l1_loss(prediction["quotient"] / scale, batch.quotient.float() / scale)
    prod = torch.nn.functional.mse_loss(prediction["product_scale"], batch.product_scale)
    return rem + .35 * quo + .15 * prod, {"remainder": float(rem.detach()), "quotient": float(quo.detach()), "product_scale": float(prod.detach())}


def run_intermediate_study(config=IntermediateConfig(), *, prediction_sink=None):
    config.validate()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    split = build_intermediate_split(seed=config.seed, holdout_fraction=config.holdout_fraction)
    external = _batch(((a, b) for a in range(101) for b in range(101)), 101)
    names = {"independent_holdout": split.independent_holdout, "absent_output_53": split.absent_output,
             "absent_operand_53": split.absent_operand, "modulus_101": external}
    reports = []
    for seed in config.seeds:
        cfg = IntermediateConfig(seed=seed, seeds=config.seeds, width=config.width, depth=config.depth, epochs=config.epochs,
                                 batch_size=config.batch_size, learning_rate=config.learning_rate,
                                 max_optimizer_seconds=config.max_optimizer_seconds, holdout_fraction=config.holdout_fraction)
        model = build_intermediate_model(cfg)
        initial = _digest(model)
        frozen = {name: _score(model, batch, 101 if name == "modulus_101" else MOD) for name, batch in names.items()}
        optimizer, generator = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate), torch.Generator().manual_seed(seed + 1)
        started, steps, timed_out, last_losses = time.perf_counter(), 0, False, {}
        model.train()
        for _ in range(cfg.epochs):
            order = torch.randperm(len(split.train), generator=generator)
            for begin in range(0, len(order), cfg.batch_size):
                if time.perf_counter() - started >= cfg.max_optimizer_seconds:
                    timed_out = True; break
                chosen = order[begin:begin + cfg.batch_size]
                batch = Batch(*(getattr(split.train, field)[chosen] for field in ("x", "y", "remainder", "quotient", "product_scale")))
                prediction = model(batch.x, batch.y, torch.full_like(batch.x, MOD))
                loss, last_losses = _loss(prediction, batch, MOD)
                if not torch.isfinite(loss): raise RuntimeError("nonfinite neural loss")
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); steps += 1
            if timed_out: break
        elapsed = time.perf_counter() - started
        evaluated = {name: _score(model, batch, 101 if name == "modulus_101" else MOD) for name, batch in names.items()}
        if prediction_sink is not None:
            for name, batch in names.items():
                modulus = 101 if name == "modulus_101" else MOD
                values = _predict_values(model, batch, modulus)
                mapping = {(int(a), int(b)): int(value) for a, b, value in
                           zip(batch.x.tolist(), batch.y.tolist(), values)}
                if len(mapping) != len(batch):
                    raise ValueError("Duplicate prediction coordinates")
                external_name = {"independent_holdout":"unseen_group_pairs",
                                 "absent_output_53":"missing_output_53",
                                 "absent_operand_53":"missing_input_53",
                                 "modulus_101":"mod101"}[name]
                prediction_sink(seed, external_name, mapping)
        constant = int(torch.bincount(split.train.remainder, minlength=MOD).argmax())
        controls = {"constant": {name: {"accuracy": (batch.remainder == constant).float().mean().item()} for name, batch in names.items()},
                    "frozen_random": frozen}
        reports.append({"seed": seed, "split": {"train_count": len(split.train), "target_53_train_rows": int((split.train.remainder == ABSENT).sum()),
                        "operand_53_train_rows": int(((split.train.x == ABSENT) | (split.train.y == ABSENT)).sum())},
                        "controls": controls, "evaluation": evaluated,
                        "training": {"optimizer_steps": steps, "optimizer_seconds": elapsed, "stopped_at_budget": timed_out,
                                     "last_auxiliary_losses": last_losses, "initial_sha256": initial, "trained_sha256": _digest(model)}})
    return {"schema": "labyrinth.minotaur.v7-intermediate.v1", "config": asdict(config),
            "model": {"architecture": "symmetric_shared_encoder_learned_sum_absdiff_multihead", "parameters": sum(p.numel() for p in model.parameters()),
                      "forward_product_or_modulo": False, "supplied_inductive_priors": ["shared operand encoder", "exchange symmetry", "normalized bounded coordinates", "fixed Fourier coordinate features"]},
            "auxiliary_supervision": {"remainder": "offline residue label", "quotient": "offline floor(product/modulus)", "product_scale": "offline normalized integer product"},
            "group_pair_audit": audit_group_pairs(split), "runs": reports,
            "promotion_recommendation": "DO_NOT_PROMOTE", "qualification": "NOT_ESTABLISHED",
            "limits": "Public synthetic data; mod101 is evaluation only; no private-exam claim."}
