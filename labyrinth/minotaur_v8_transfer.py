"""V8 cross-modulus neural multiplication study (research only).

Unlike V7's single-modulus-97 experiment, this preregistered study trains one
small neural network on the numeric operands and modulus for 83, 89, and 97.
Modulus 101 is constructed only after the training data are fixed and is never
used for fitting or development selection.  The teacher below creates labels;
the network's forward pass receives only operands and their modulus.
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

TRAIN_MODULI = (83, 89, 97)
FINAL_MODULUS = 101
STRICT_MODULUS = 97
STRICT_RESIDUE = 53


@dataclass(frozen=True)
class TransferConfig:
    seed: int = 20261028
    seeds: tuple[int, int] = (20261028, 20261029)
    width: int = 24
    depth: int = 2
    epochs: int = 18
    batch_size: int = 512
    learning_rate: float = 0.003
    max_optimizer_seconds: float = 35.0
    dev_fraction: float = 0.10

    def validate(self):
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31 or
                type(self.seeds) is not tuple or len(self.seeds) != 2 or
                len(set(self.seeds)) != 2 or
                any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds) or
                type(self.width) is not int or not 8 <= self.width <= 48 or
                type(self.depth) is not int or not 1 <= self.depth <= 3 or
                type(self.epochs) is not int or not 1 <= self.epochs <= 160 or
                type(self.batch_size) is not int or not 16 <= self.batch_size <= 4096 or
                type(self.learning_rate) is not float or not 0 < self.learning_rate <= .02 or
                type(self.max_optimizer_seconds) is not float or not 1 <= self.max_optimizer_seconds <= 90 or
                type(self.dev_fraction) is not float or not .02 <= self.dev_fraction <= .30):
            raise ValueError("invalid bounded transfer configuration")
        return self


@dataclass(frozen=True)
class Batch:
    x: Any
    y: Any
    target: Any

    def __len__(self):
        return int(self.x.numel())


@dataclass(frozen=True)
class TransferSplit:
    train: dict[int, Batch]
    dev: dict[int, Batch]
    strict_input_53: Batch
    strict_output_53: Batch
    train_moduli: tuple[int, ...] = TRAIN_MODULI
    final_modulus: int = FINAL_MODULUS


def multiplication_targets(x, y, modulus):
    """Offline label teacher only; never call this from ``forward``."""
    return torch.remainder(x * y, modulus)


def _batch(pairs: list[tuple[int, int]], modulus: int) -> Batch:
    x = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    y = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return Batch(x, y, multiplication_targets(x, y, modulus))


def _directed(groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(a, b) for a, b in groups for a, b in
            (((a, b), (b, a)) if a != b else ((a, b),))]


def build_transfer_split(*, seed: int = 20261028, dev_fraction: float = .10) -> TransferSplit:
    """Make grouped public dev data and V8's fixed p=97 residue-53 challenge."""
    if type(seed) is not int or seed < 0 or not .02 <= dev_fraction <= .30:
        raise ValueError("invalid transfer split")
    train, dev = {}, {}
    strict_input, strict_output = [], []
    for modulus in TRAIN_MODULI:
        groups = [(a, b) for a in range(modulus) for b in range(a, modulus)]
        permitted = []
        for a, b in groups:
            # Label generation is deliberately kept in the offline teacher.
            label = int(multiplication_targets(torch.tensor([a]), torch.tensor([b]), modulus).item())
            if modulus == STRICT_MODULUS and (a == STRICT_RESIDUE or b == STRICT_RESIDUE):
                strict_input.append((a, b))
            elif modulus == STRICT_MODULUS and label == STRICT_RESIDUE:
                strict_output.append((a, b))
            else:
                permitted.append((a, b))
        rng = random.Random(seed + modulus)
        rng.shuffle(permitted)
        count = max(1, round(len(permitted) * dev_fraction))
        dev[modulus] = _batch(_directed(permitted[:count]), modulus)
        train[modulus] = _batch(_directed(permitted[count:]), modulus)
    return TransferSplit(train, dev, _batch(_directed(strict_input), STRICT_MODULUS),
                         _batch(_directed(strict_output), STRICT_MODULUS))


class TransferNetwork(nn.Module):
    """Shared coordinate-token relation network, with no residue/pair tables."""
    def __init__(self, config: TransferConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.token = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, width), nn.Tanh())
        layers: list[nn.Module] = [nn.Linear(3 * width, width), nn.GELU()]
        for _ in range(config.depth - 1):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        layers.extend((nn.Linear(width, 1), nn.Sigmoid()))
        self.relation = nn.Sequential(*layers)

    @staticmethod
    def _coordinates(value, modulus):
        fraction = value.float() / modulus.float()
        features = [fraction, modulus.float() / FINAL_MODULUS,
                    torch.log(modulus.float()) / math.log(FINAL_MODULUS)]
        for harmonic in (1, 2, 4):
            angle = fraction * (2.0 * math.pi * harmonic)
            features.extend((torch.sin(angle), torch.cos(angle),
                             torch.sin(angle * .5)))
        return torch.stack(features, dim=-1)

    def forward(self, x, y, modulus):
        if (not all(isinstance(v, torch.Tensor) for v in (x, y, modulus)) or
                x.dtype != torch.long or y.dtype != torch.long or modulus.dtype != torch.long or
                x.ndim != 1 or x.shape != y.shape or x.shape != modulus.shape or
                bool(((modulus < 3) | (modulus > FINAL_MODULUS)).any()) or
                bool(((x < 0) | (y < 0) | (x >= modulus) | (y >= modulus)).any())):
            raise ValueError("invalid bounded transfer residues")
        left = self.token(self._coordinates(x, modulus))
        right = self.token(self._coordinates(y, modulus))
        # Additive, distance, and max relations leave all multiplication to the
        # learned affine/nonlinear decoder; no operand or latent product is a
        # forward feature.
        relation = torch.cat((left + right, torch.abs(left - right), torch.maximum(left, right)), dim=-1)
        return self.relation(relation).squeeze(-1) * modulus.float()


def build_transfer_model(config: TransferConfig = TransferConfig()):
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return TransferNetwork(config)


def _digest(model) -> str:
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _score(model, batch: Batch, modulus: int) -> dict[str, float | int]:
    model.eval()
    predicted: list[int] = []
    with torch.no_grad():
        for offset in range(0, len(batch), 1024):
            x, y = batch.x[offset:offset + 1024], batch.y[offset:offset + 1024]
            p = torch.full_like(x, modulus)
            predicted.extend(model(x, y, p).round().long().clamp(0, modulus - 1).tolist())
    hits = sum(a == b for a, b in zip(predicted, batch.target.tolist()))
    return {"correct": hits, "count": len(batch), "accuracy": hits / len(batch) if len(batch) else 0.0}


def _joined_train(split: TransferSplit) -> tuple[Batch, torch.Tensor]:
    x = torch.cat([split.train[p].x for p in TRAIN_MODULI])
    y = torch.cat([split.train[p].y for p in TRAIN_MODULI])
    target = torch.cat([split.train[p].target for p in TRAIN_MODULI])
    modulus = torch.cat([torch.full_like(split.train[p].x, p) for p in TRAIN_MODULI])
    return Batch(x, y, target), modulus


def _evaluation_modulus(name: str) -> int:
    if name == "final_modulus_101":
        return FINAL_MODULUS
    if name.startswith("modulus_"):
        return int(name.rsplit("_", 1)[-1])
    return STRICT_MODULUS


def run_study(config: TransferConfig = TransferConfig()):
    config.validate()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    split = build_transfer_split(seed=config.seed, dev_fraction=config.dev_fraction)
    joined, joined_modulus = _joined_train(split)
    final_101 = _batch([(a, b) for a in range(FINAL_MODULUS) for b in range(FINAL_MODULUS)], FINAL_MODULUS)
    reports = []
    for seed in config.seeds:
        cfg = TransferConfig(seed=seed, seeds=config.seeds, width=config.width, depth=config.depth,
                             epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.learning_rate,
                             max_optimizer_seconds=config.max_optimizer_seconds, dev_fraction=config.dev_fraction)
        model = build_transfer_model(cfg)
        initial = _digest(model)
        development = {f"modulus_{p}": split.dev[p] for p in TRAIN_MODULI}
        challenges = {"strict_input_53_mod97": split.strict_input_53,
                      "strict_output_53_mod97": split.strict_output_53,
                      "final_modulus_101": final_101}
        frozen = {name: _score(model, data, _evaluation_modulus(name))
                  for name, data in {**development, **challenges}.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        generator = torch.Generator().manual_seed(seed + 1)
        started, updates, timed_out = time.perf_counter(), 0, False
        model.train()
        for _ in range(cfg.epochs):
            order = torch.randperm(len(joined), generator=generator)
            for offset in range(0, len(order), cfg.batch_size):
                if time.perf_counter() - started >= cfg.max_optimizer_seconds:
                    timed_out = True
                    break
                pick = order[offset:offset + cfg.batch_size]
                prediction = model(joined.x[pick], joined.y[pick], joined_modulus[pick])
                loss = torch.nn.functional.smooth_l1_loss(prediction, joined.target[pick].float())
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite neural loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                updates += 1
            if timed_out:
                break
        elapsed = time.perf_counter() - started
        all_sets = {**development, **challenges}
        evaluated = {name: _score(model, data, _evaluation_modulus(name))
                     for name, data in all_sets.items()}
        constant = int(torch.bincount(joined.target, minlength=FINAL_MODULUS).argmax())
        controls = {"constant": {name: {"accuracy": (data.target == min(constant, modulus - 1)).float().mean().item()}
                                  for name, data, modulus in ((n, d, _evaluation_modulus(n))
                                     for n, d in all_sets.items())},
                    "frozen_random": frozen}
        reports.append({"seed": seed,
                        "split": {"train_rows": len(joined), "mod97_input_53_train_rows": int(((split.train[97].x == 53) | (split.train[97].y == 53)).sum()),
                                  "mod97_output_53_train_rows": int((split.train[97].target == 53).sum())},
                        "controls": controls,
                        "development": {name: evaluated[name] for name in development},
                        "evaluation": {name: evaluated[name] for name in challenges},
                        "training": {"optimizer_steps": updates, "optimizer_seconds": elapsed,
                                     "stopped_at_budget": timed_out, "initial_sha256": initial, "trained_sha256": _digest(model)}})
    return {"schema": "labyrinth.minotaur.v8-transfer.v1", "config": asdict(config),
            "protocol": {"train_moduli": list(TRAIN_MODULI), "final_evaluation_modulus": FINAL_MODULUS,
                         "development_selection": "public grouped dev only; final 101 not used for selection",
                         "strict_mod97": "input and output residue 53 excluded from training"},
            "model": {"architecture": "shared modulus-conditioned coordinate tokens plus symmetric relational decoder",
                      "parameters": sum(p.numel() for p in model.parameters()), "forward_product_or_modulo": False,
                      "no_residue_or_pair_tables": True}, "runs": reports,
            "promotion_recommendation": "DO_NOT_PROMOTE", "qualification": "NOT_ESTABLISHED",
            "limits": "Public synthetic study; passing software tests do not establish learned arithmetic."}
