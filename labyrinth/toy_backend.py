"""A tiny CPU-only trainable linear policy, NOT a language model.

The parser and the two-input linear hypothesis class are hand engineered. Four
floating-point coefficients learn the meanings of '+' and '-' from training
gold via normalized SGD. This demonstrates parameter persistence and controls,
not general reasoning, LLM fine-tuning, or structural generalization.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import re
import stat
from typing import Sequence

from .adapters import AdapterError, _validate_prompt_seed
from .evaluation import score_response
from .maze import Challenge, generate_maze
from .runner import _source_state


TOY_VERSION = "toy-linear-nlms-v1"
CHECKPOINT_SCHEMA = "labyrinth.toy-checkpoint.v1"
_PROMPT = re.compile(r"Calculate ([0-9]{1,3}) ([+-]) ([0-9]{1,3})\.\nRespond with exactly one line: ANSWER: <integer>\.", re.ASCII)
_MAX_CHECKPOINT = 8192


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")


def _parse(prompt: str) -> tuple[float, str, float]:
    _validate_prompt_seed(prompt, None)
    match = _PROMPT.fullmatch(prompt)
    if match is None:
        raise AdapterError("toy policy supports only canonical tier-1 arithmetic prompts")
    return float(match[1]), match[2], float(match[3])


def _deranged_labels(labels: list[float], seed: int) -> list[float]:
    """Permute the actual targets with no value-level collisions.

    Try seeded random permutations first. As a bounded fallback, sorted blocks
    rotated by the largest block size form a value derangement when no label
    occupies more than half the data. Refuse impossible shams.
    """
    largest = max(Counter(labels).values())
    if largest * 2 > len(labels):
        raise ValueError("cannot shuffle every label: use more diverse training targets")
    rng = random.Random(seed ^ 0x1A8E1)
    corrupted = list(labels)
    for _ in range(128):
        rng.shuffle(corrupted)
        if all(left != right for left, right in zip(labels, corrupted)):
            return corrupted
    # A deterministic fallback also handles frequent duplicate values.
    ordered = sorted(range(len(labels)), key=lambda index: labels[index])
    corrupted = list(labels)
    for index, recipient in enumerate(ordered):
        corrupted[recipient] = labels[ordered[(index + largest) % len(labels)]]
    if any(left == right for left, right in zip(labels, corrupted)):
        raise ValueError("label derangement failed")
    return corrupted


class ToyLinearAdapter:
    """A four-parameter linear policy implementing the ModelAdapter protocol."""

    def __init__(self, *, seed: int = 0) -> None:
        _integer(seed, "seed", 0, 2**31 - 1)
        self.seed = seed
        rng = random.Random(seed)
        self._weights = {operator: [rng.uniform(-0.01, 0.01) for _ in range(2)] for operator in ("+", "-")}

    @property
    def model_id(self) -> str:
        return f"{TOY_VERSION}:{self.model_hash}"

    def _payload(self) -> dict:
        return {"schema": CHECKPOINT_SCHEMA, "version": TOY_VERSION, "seed": self.seed,
                "weights": {op: list(values) for op, values in self._weights.items()}}

    @property
    def model_hash(self) -> str:
        """Canonical parameters/configuration digest, not an authenticity proof."""
        return _hash(self._payload())

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        _validate_prompt_seed(prompt, seed)
        a, operator, b = _parse(prompt)
        weights = self._weights[operator]
        return f"ANSWER: {round(weights[0] * a + weights[1] * b)}"

    def train(
        self, training_cases: Sequence[Challenge], *, epochs: int = 80,
        learning_rate: float = 0.5, shuffled_labels: bool = False,
        update_enabled: bool = True,
    ) -> dict:
        """Use private TRAIN gold only; validate the entire batch before mutation.

        Epochs are fixed in advance, never selected using evaluation scores.
        Zero-update controls traverse the same cases and calculate the same
        residuals; they intentionally do not mutate parameters.
        """
        _integer(epochs, "epochs", 1, 1000)
        if type(learning_rate) not in (int, float) or not 0 <= learning_rate <= 1 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and between 0 and 1")
        if type(shuffled_labels) is not bool or type(update_enabled) is not bool:
            raise ValueError("control flags must be boolean")
        if not isinstance(training_cases, (list, tuple)) or not 2 <= len(training_cases) <= 1000:
            # Check a singleton's split too, without reading private gold.
            if isinstance(training_cases, (list, tuple)) and any(getattr(c, "split", None) != "train" for c in training_cases):
                raise ValueError("only training split may supply supervision")
            raise ValueError("training requires 2 to 1000 cases")
        if any(getattr(c, "split", None) != "train" for c in training_cases):
            raise ValueError("only training split may supply supervision")
        if any(not isinstance(c, Challenge) or c.tier != 1 or c.kind != "arithmetic" for c in training_cases):
            raise ValueError("only tier-1 arithmetic training challenges are supported")
        if len({c.id for c in training_cases}) != len(training_cases) or len({c.prompt for c in training_cases}) != len(training_cases):
            raise ValueError("duplicate training case ID or prompt")
        features = [_parse(c.prompt) for c in training_cases]
        labels = []
        for case in training_cases:
            if not isinstance(case.answer, str) or re.fullmatch(r"-?[0-9]{1,4}", case.answer, re.ASCII) is None:
                raise ValueError("training targets must be bounded integers")
            labels.append(float(case.answer))
        targets = _deranged_labels(labels, self.seed) if shuffled_labels else labels
        weights = {op: list(values) for op, values in self._weights.items()}
        before = self.model_hash
        updates = 0
        for _ in range(epochs):
            for (a, operator, b), target in zip(features, targets):
                values = weights[operator]
                residual = target - (values[0] * a + values[1] * b)
                step = learning_rate * residual / max(a * a + b * b, 1.0)
                if update_enabled and learning_rate != 0:
                    changed = [values[0] + step * a, values[1] + step * b]
                    updates += int(changed != values)
                    weights[operator] = changed
        self._weights = weights
        return {"epochs": epochs, "learning_rate": learning_rate,
                "case_visits": epochs * len(training_cases), "parameter_updates": updates,
                "update_enabled": update_enabled, "changed_labels": sum(a != b for a, b in zip(labels, targets)),
                "label_control": "value-deranged permutation" if shuffled_labels else "true training gold",
                "before_model_hash": before, "after_model_hash": self.model_hash}

    def save_checkpoint(self, path: str | Path) -> str:
        """Exclusively create bounded JSON; never overwrite even a symlink.

        Returns the canonical payload integrity hash. The checksum detects
        accidental corruption, not maliciously re-signed replacement files.
        """
        payload = self._payload()
        integrity = _hash(payload)
        data = _canonical({**payload, "integrity_sha256": integrity}) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(os.fspath(path), flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return integrity

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "ToyLinearAdapter":
        """Reject unknown fields/versions, duplicate keys, bad numbers and hashes."""
        # O_NOFOLLOW is absent on Windows; reject a link there as well.
        if Path(path).is_symlink():
            raise ValueError("checkpoint must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(os.fspath(path), flags)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("checkpoint must be a regular file")
            raw = stream.read(_MAX_CHECKPOINT + 1)
        if len(raw) > _MAX_CHECKPOINT:
            raise ValueError("checkpoint exceeds size limit")

        def unique_object(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate checkpoint key")
                obj[key] = value
            return obj

        try:
            data = json.loads(raw, object_pairs_hook=unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid checkpoint JSON") from exc
        if not isinstance(data, dict) or set(data) != {"schema", "version", "seed", "weights", "integrity_sha256"}:
            raise ValueError("invalid checkpoint fields")
        if data["schema"] != CHECKPOINT_SCHEMA or data["version"] != TOY_VERSION:
            raise ValueError("unsupported checkpoint version")
        _integer(data["seed"], "seed", 0, 2**31 - 1)
        weights = data["weights"]
        if not isinstance(weights, dict) or set(weights) != {"+", "-"}:
            raise ValueError("invalid checkpoint operators")
        for values in weights.values():
            if not isinstance(values, list) or len(values) != 2 or any(
                type(v) not in (int, float) or abs(v) > 1_000_000 or not math.isfinite(v) for v in values
            ):
                raise ValueError("invalid checkpoint weights")
        integrity = data.pop("integrity_sha256")
        if not isinstance(integrity, str) or re.fullmatch(r"[0-9a-f]{64}", integrity) is None or not hmac.compare_digest(_hash(data), integrity):
            raise ValueError("checkpoint integrity mismatch")
        model = cls(seed=data["seed"])
        model._weights = {op: list(values) for op, values in weights.items()}
        return model


def _evaluate_cases(model: ToyLinearAdapter, cases: Sequence[Challenge], seed: int) -> dict:
    trials = []
    for case in cases:
        # Only public text and the fixed seed cross the generation boundary.
        response = model.generate(case.prompt, seed=seed)
        result = score_response(case, response)
        trials.append({"challenge_id": case.id, "correct": result.correct,
                       "response": response, "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest()})
    correct = sum(t["correct"] for t in trials)
    return {"accuracy": correct / len(trials), "correct": correct, "total": len(trials),
            "model_hash": model.model_hash, "trials": trials}


def evaluate_toy_model(model: ToyLinearAdapter, *, seed: int, eval_count: int) -> dict:
    """Read-only heldout scoring using the experiment's fixed eval-seed scheme.

    No training data or optimizer is consulted. Callers are responsible for
    retaining the checkpoint's training provenance and checking split overlap;
    a parameter-only checkpoint cannot establish which examples trained it.
    """
    _integer(seed, "seed", 0, 2**31 - 1)
    _integer(eval_count, "eval_count", 1, 1000)
    evaluated = generate_maze(seed=seed ^ 0x5C18A7, tiers=(1,), count=eval_count, split="eval")
    return {"schema": "labyrinth.toy-evaluation.v1", "backend_version": TOY_VERSION,
            "adaptation_kind": "toy linear parameter learning (not an LLM)",
            "parameters": {"seed": seed, "eval_count": eval_count},
            **_evaluate_cases(model, evaluated, seed),
            "limitations": ["Parameter-only checkpoints do not establish training data provenance or disjointness.",
                            "This is a four-parameter arithmetic toy, not evidence of general LLM reasoning."]}


def run_toy_experiment(
    *, seed: int = 0, train_count: int = 64, eval_count: int = 64,
    epochs: int = 80, learning_rate: float = 0.5,
) -> dict:
    """Paired fixed-budget heldout comparison; no eval feedback enters training.

    Baseline and zero_update intentionally preserve identical parameters via
    different switches. Equal budget means case visits and eval calls, NOT
    identical optimizer work or wall-clock time. No checkpoint files are written.
    """
    _integer(seed, "seed", 0, 2**31 - 1)
    _integer(train_count, "train_count", 2, 1000)
    _integer(eval_count, "eval_count", 1, 1000)
    train = generate_maze(seed=seed, tiers=(1,), count=train_count, split="train")
    evaluated = generate_maze(seed=seed ^ 0x5C18A7, tiers=(1,), count=eval_count, split="eval")
    train_ids, eval_ids = [c.id for c in train], [c.id for c in evaluated]
    train_prompts, eval_prompts = [c.prompt for c in train], [c.prompt for c in evaluated]
    if set(train_ids) & set(eval_ids) or set(train_prompts) & set(eval_prompts):
        raise ValueError("training/evaluation overlap")
    if {_parse(c.prompt)[1] for c in train} != {"+", "-"}:
        raise ValueError("training data must include both operators; increase train_count")
    arms = {}
    for name in ("baseline", "trained", "zero_update", "shuffled_labels"):
        model = ToyLinearAdapter(seed=seed)
        training = model.train(train, epochs=epochs,
                               learning_rate=0 if name == "zero_update" else learning_rate,
                               shuffled_labels=name == "shuffled_labels", update_enabled=name != "baseline")
        arms[name] = {**_evaluate_cases(model, evaluated, seed), "training": training}
    return {"schema": "labyrinth.toy-experiment.v1", "backend_version": TOY_VERSION,
            "source": _source_state(),
            "adaptation_kind": "toy linear parameter learning (not an LLM)",
            "parameters": {"seed": seed, "train_count": train_count, "eval_count": eval_count,
                           "epochs": epochs, "learning_rate": learning_rate},
            "train_challenge_ids": train_ids, "eval_challenge_ids": eval_ids,
            "train_prompt_hashes": [hashlib.sha256(p.encode()).hexdigest() for p in train_prompts],
            "eval_prompt_hashes": [hashlib.sha256(p.encode()).hexdigest() for p in eval_prompts],
            "arms": arms, "paired_accuracy_gain": arms["trained"]["accuracy"] - arms["baseline"]["accuracy"],
            "limitations": ["Four floating-point parameters in a hand-engineered linear hypothesis class; not LLM weights.",
                            "Heldout arithmetic operands are not structural or general reasoning transfer.",
                            "All arms have equal case visits and evaluation calls, not equal wall time; baseline and zero_update intentionally coincide.",
                            "Public deterministic generators are not a secret benchmark; this is a lifecycle demonstration."]}
