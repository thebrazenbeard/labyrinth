"""Deterministic, split-respecting trainer for learned atomic operators.

This module deliberately knows no arithmetic rule and generates no labels.  A
caller supplies already-created ``(op, x, y, label)`` rows and a neural model
whose ``forward(op, x, y)`` returns class logits.  Development data is reported
only; it is never used for optimizer updates or for choosing a checkpoint.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Any


_MAX_ROWS = 100_000


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _rows(rows: Sequence[Any], name: str) -> tuple[tuple[int, int, int, int], ...]:
    if not isinstance(rows, (list, tuple)) or not rows or len(rows) > _MAX_ROWS:
        raise ValueError(f"{name} must be a nonempty list or tuple of at most {_MAX_ROWS} rows")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"{name}[{index}] must be (op, x, y, label)")
        result.append(tuple(_integer(value, f"{name}[{index}][{field}]")
                            for field, value in zip(("op", "x", "y", "label"), row)))
    return tuple(result)  # type: ignore[return-value]


def _weight_digest(model: Any) -> str:
    """Hash every parameter and buffer, not merely the final layer."""
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8") + b"\0")
        hasher.update(str(tensor.dtype).encode("ascii") + b"\0")
        hasher.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def _batch_tensors(rows: Sequence[tuple[int, int, int, int]], torch: Any) -> tuple[Any, Any, Any, Any]:
    # Labels are intentionally constructed separately and are never an input to
    # model.forward.  This is kept explicit because label leakage invalidates an
    # algebraic generalization experiment.
    op = torch.tensor([row[0] for row in rows], dtype=torch.long)
    x = torch.tensor([row[1] for row in rows], dtype=torch.long)
    y = torch.tensor([row[2] for row in rows], dtype=torch.long)
    label = torch.tensor([row[3] for row in rows], dtype=torch.long)
    return op, x, y, label


def _empty_score() -> dict[str, Any]:
    return {"correct": 0, "total": 0, "accuracy": None, "mean_confidence": None}


def _score(model: Any, rows: Sequence[tuple[int, int, int, int]], train_pairs: set[tuple[int, int, int]],
           batch_size: int, torch: Any) -> dict[str, Any]:
    """Score rows by operation and train-pair membership without updating model."""
    buckets: dict[int, dict[str, dict[str, float | int]]] = {}
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            op, x, y, label = _batch_tensors(batch, torch)
            logits = model(op, x, y)
            if logits.ndim != 2 or logits.shape[0] != len(batch):
                raise ValueError("model.forward(op, x, y) must return [batch, classes] logits")
            if logits.shape[1] <= int(label.max().item()):
                raise ValueError("a label is outside the model logits class range")
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            for i, row in enumerate(batch):
                op_bucket = buckets.setdefault(row[0], {
                    "seen_pairs": {"correct": 0, "total": 0, "confidence": 0.0},
                    "truly_unseen_pairs": {"correct": 0, "total": 0, "confidence": 0.0},
                    "all": {"correct": 0, "total": 0, "confidence": 0.0},
                })
                group = "seen_pairs" if row[:3] in train_pairs else "truly_unseen_pairs"
                correct = int(prediction[i].item() == row[3])
                value = float(confidence[i].item())
                for key in (group, "all"):
                    op_bucket[key]["correct"] += correct
                    op_bucket[key]["total"] += 1
                    op_bucket[key]["confidence"] += value
    if was_training:
        model.train()

    def finish(raw: dict[str, float | int]) -> dict[str, Any]:
        total = int(raw["total"])
        if not total:
            return _empty_score()
        return {"correct": int(raw["correct"]), "total": total,
                "accuracy": float(raw["correct"]) / total,
                "mean_confidence": float(raw["confidence"]) / total}

    per_op = {str(op): {name: finish(raw) for name, raw in groups.items()}
              for op, groups in sorted(buckets.items())}
    macro: dict[str, Any] = {}
    for group in ("seen_pairs", "truly_unseen_pairs", "all"):
        available = [values[group] for values in per_op.values() if values[group]["total"]]
        macro[group] = {
            "operations": len(available),
            "accuracy": (sum(item["accuracy"] for item in available) / len(available)
                         if available else None),
            "mean_confidence": (sum(item["mean_confidence"] for item in available) / len(available)
                                if available else None),
        }
    return {"per_op": per_op, "macro": macro}


def train_atomic_operator(model: Any, train_rows: Sequence[Any], dev_rows: Sequence[Any], *,
                          steps: int | None = None, epochs: int | None = None,
                          batch_size: int = 32, lr: float = 1e-3,
                          time_budget: float = 60.0, torch: Any) -> dict[str, Any]:
    """Train a neural ``forward(op, x, y)`` operator using only ``train_rows``.

    Exactly one of ``steps`` or ``epochs`` is required.  Batches are sequential
    and cyclic rather than shuffled, making the result reproducible for a fixed
    starting model and deterministic PyTorch configuration.  ``time_budget`` is
    a hard pre-update limit and is capped at 90 CPU seconds.
    """
    train = _rows(train_rows, "train_rows")
    dev = _rows(dev_rows, "dev_rows")
    if (steps is None) == (epochs is None):
        raise ValueError("specify exactly one of steps or epochs")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 1_000_000):
        raise ValueError("steps must be a positive bounded int")
    if epochs is not None and (type(epochs) is not int or not 1 <= epochs <= 100_000):
        raise ValueError("epochs must be a positive bounded int")
    if type(batch_size) is not int or not 1 <= batch_size <= len(train):
        raise ValueError("batch_size must be between one and the training-row count")
    if type(lr) not in (float, int) or isinstance(lr, bool) or not 0.0 < float(lr) <= 10.0:
        raise ValueError("lr must be a positive finite number")
    if type(time_budget) not in (float, int) or isinstance(time_budget, bool) or not 0.0 < float(time_budget) <= 90.0:
        raise ValueError("time_budget must be in (0, 90] CPU seconds")
    if not callable(getattr(model, "forward", None)) or not list(model.parameters()):
        raise ValueError("model must be a trainable torch.nn.Module")

    updates_target = (steps if steps is not None else
                      epochs * ((len(train) + batch_size - 1) // batch_size))
    train_pairs = {row[:3] for row in train}
    initial_digest = _weight_digest(model)
    frozen_baseline = _score(model, dev, train_pairs, batch_size, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    deadline = time.monotonic() + float(time_budget)
    model.train()
    losses: list[float] = []
    updates = 0
    timed_out = False
    for update in range(updates_target):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        start = (update * batch_size) % len(train)
        batch = tuple(train[(start + offset) % len(train)] for offset in range(batch_size))
        op, x, y, label = _batch_tensors(batch, torch)
        logits = model(op, x, y)
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise ValueError("model.forward(op, x, y) must return [batch, classes] logits")
        if logits.shape[1] <= int(label.max().item()):
            raise ValueError("a label is outside the model logits class range")
        loss = torch.nn.functional.cross_entropy(logits, label)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite cross-entropy loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        updates += 1
    final_digest = _weight_digest(model)
    final = _score(model, dev, train_pairs, batch_size, torch)
    return {
        "schema": "labyrinth.minotaur-algebraic-training-v5.v1",
        "split_policy": "dev is evaluation-only; no heldout/test rows are accepted or used for selection",
        "optimizer": "AdamW(weight_decay=0.0)",
        "optimizer_updates": updates,
        "requested_updates": updates_target,
        "timed_out": timed_out,
        "mean_training_loss": sum(losses) / len(losses) if losses else None,
        "frozen_baseline": frozen_baseline,
        "final": final,
        "weight_digest": {"initial_sha256": initial_digest, "final_sha256": final_digest,
                          "changed": initial_digest != final_digest},
        "reproducibility": {"batch_order": "sequential cyclic; no trainer RNG",
                            "time_budget_seconds": float(time_budget)},
    }
