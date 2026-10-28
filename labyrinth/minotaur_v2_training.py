"""Tiny reproducible CPU training experiment for random Minotaur V2.

The data are public synthetic development data.  This module has no promotion,
signing, checkpoint-loading, or graduation authority and never writes weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from typing import Any, Sequence

from .minotaur_scratch_v2 import (
    V2Config,
    build_v2_model,
    pad_v2_tokens,
    parameter_digest_v2,
    tokenize_observed_trace_v2,
    v2_loss,
)
from .program_maze import generate_program_challenges


SCHEMA = "labyrinth.minotaur.v2-tiny-training.v1"
_SPLIT_SPECS = {
    "train": (7301, "v2train", "train"),
    "dev": (7302, "v2dev", "train"),
    "composition_transfer": (7303, "v2composition", "matched_length_ood"),
    "longer_transfer": (7304, "v2longer", "structural_holdout"),
}
_CORRUPTIONS = ("target_plus", "target_minus", "non_target_shift", "register_swap")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261027
    train_programs: int = 64
    eval_programs: int = 32
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 0.003
    width: int = 32
    heads: int = 4
    layers: int = 1
    max_optimizer_seconds: float = 90.0

    def validate(self) -> "ExperimentConfig":
        if (type(self.seed) is not int or not 0 <= self.seed < 2**31
                or type(self.train_programs) is not int
                or not 4 <= self.train_programs <= 128 or self.train_programs % 4
                or type(self.eval_programs) is not int
                or not 4 <= self.eval_programs <= 64 or self.eval_programs % 4
                or type(self.epochs) is not int or not 1 <= self.epochs <= 40
                or type(self.batch_size) is not int or not 4 <= self.batch_size <= 64
                or type(self.learning_rate) is not float
                or not 0.00001 <= self.learning_rate <= 0.05
                or type(self.width) is not int or not 16 <= self.width <= 64
                or type(self.heads) is not int or not 1 <= self.heads <= 8
                or self.width % self.heads
                or type(self.layers) is not int or not 1 <= self.layers <= 2
                or type(self.max_optimizer_seconds) is not float
                or not 1.0 <= self.max_optimizer_seconds <= 90.0):
            raise ValueError("invalid bounded V2 training configuration")
        return self


@dataclass(frozen=True)
class V2TrainingCase:
    case_id: str
    split: str
    question: str
    steps: tuple[str, ...]
    first_error_step: int | None
    structure_fingerprint: str
    corruption_family: str


def _program(question: str) -> list[tuple[str, str, str]]:
    return re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", question, re.MULTILINE)


def _clean_trace(question: str) -> tuple[str, ...]:
    initial = re.search(r"Initial registers: A=(\d+), B=(\d+)", question)
    operations = _program(question)
    if initial is None or not operations:
        raise ValueError("invalid generated register program")
    registers = {"A": int(initial[1]), "B": int(initial[2])}
    steps = []
    for index, (operation, target, operand) in enumerate(operations, 1):
        value = registers[operand] if operand in registers else int(operand)
        if operation == "ADD":
            result = registers[target] + value
        elif operation == "SUB":
            result = registers[target] - value
        else:
            result = registers[target] * value
        registers[target] = result % 97
        steps.append(f"Step {index}: A={registers['A']}, B={registers['B']}")
    return tuple(steps)


def _corrupt(question: str, clean: tuple[str, ...], index: int) -> tuple[tuple[str, ...], int, str]:
    operations = _program(question)
    step = 1 + (index * 3 + 1) % len(clean)
    family = _CORRUPTIONS[index % len(_CORRUPTIONS)]
    match = re.fullmatch(r"Step (\d+): A=(\d+), B=(\d+)", clean[step - 1])
    if match is None:
        raise ValueError("invalid clean trace")
    values = {"A": int(match[2]), "B": int(match[3])}
    target = operations[step - 1][1]
    other = "B" if target == "A" else "A"
    delta = 2 + (index * 5) % 17  # deliberately never the old fixed +1 fault
    if family == "target_plus":
        values[target] = (values[target] + delta) % 97
    elif family == "target_minus":
        values[target] = (values[target] - delta) % 97
    elif family == "non_target_shift":
        values[other] = (values[other] + delta) % 97
    else:
        values["A"], values["B"] = values["B"], values["A"]
        if values["A"] == int(match[2]) and values["B"] == int(match[3]):
            values[other] = (values[other] + delta) % 97
    changed = list(clean)
    changed[step - 1] = f"Step {step}: A={values['A']}, B={values['B']}"
    if changed[step - 1] == clean[step - 1]:
        raise RuntimeError("corruption failed to alter its designated step")
    return tuple(changed), step, family


def _selected_programs(seed: int, split: str, profile: str, count: int) -> list[Any]:
    pool = generate_program_challenges(seed, max(64, count * 8), split, profile=profile)
    buckets: dict[str, list[Any]] = {}
    for challenge in pool:
        structure = dict(challenge.metadata)["structure_fingerprint"]
        buckets.setdefault(structure, []).append(challenge)
    if len(buckets) != 4 or any(len(items) < count // 4 for items in buckets.values()):
        raise RuntimeError("could not form an exactly four-template balanced split")
    chosen = []
    for structure in sorted(buckets):
        chosen.extend(buckets[structure][:count // 4])
    return chosen


def _make_split(name: str, seed: int, split: str, profile: str, count: int) -> tuple[V2TrainingCase, ...]:
    output = []
    for index, challenge in enumerate(_selected_programs(seed, split, profile, count)):
        clean = _clean_trace(challenge.prompt)
        faulty, first_error, family = _corrupt(challenge.prompt, clean, index)
        structure = dict(challenge.metadata)["structure_fingerprint"]
        for label, steps, corruption in ((None, clean, "none"), (first_error, faulty, family)):
            payload = json.dumps({"split": name, "question": challenge.prompt, "steps": steps,
                                  "first_error_step": label, "structure": structure},
                                 sort_keys=True, separators=(",", ":"))
            case_id = "v2train-" + hashlib.sha256(payload.encode()).hexdigest()[:32]
            output.append(V2TrainingCase(case_id, name, challenge.prompt, steps, label,
                                         structure, corruption))
    return tuple(output)


def build_splits(config: ExperimentConfig) -> dict[str, tuple[V2TrainingCase, ...]]:
    config.validate()
    groups = {}
    for name, (seed, split, profile) in _SPLIT_SPECS.items():
        count = config.train_programs if name == "train" else config.eval_programs
        groups[name] = _make_split(name, seed, split, profile, count)
    ids = [case.case_id for cases in groups.values() for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("case identity overlaps across splits")
    train_structures = {case.structure_fingerprint for case in groups["train"]}
    if train_structures & {case.structure_fingerprint for case in groups["composition_transfer"]}:
        raise RuntimeError("composition transfer overlaps training structures")
    if train_structures & {case.structure_fingerprint for case in groups["longer_transfer"]}:
        raise RuntimeError("longer transfer overlaps training structures")
    return groups


def dataset_fingerprint(cases: Sequence[V2TrainingCase]) -> str:
    if not cases or any(not isinstance(case, V2TrainingCase) for case in cases):
        raise ValueError("nonempty V2TrainingCase sequence required")
    payload = json.dumps([asdict(case) for case in cases], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def score_predictions(cases: Sequence[V2TrainingCase], predictions: Sequence[int]) -> dict[str, Any]:
    if (not cases or len(cases) != len(predictions)
            or any(type(prediction) is not int or prediction < 0
                   or prediction > len(case.steps)
                   for case, prediction in zip(cases, predictions))):
        raise ValueError("complete in-range predictions required")
    actual = [case.first_error_step or 0 for case in cases]
    ac_pc = sum(a == 0 and p == 0 for a, p in zip(actual, predictions))
    ac_pi = sum(a == 0 and p != 0 for a, p in zip(actual, predictions))
    ai_pc = sum(a != 0 and p == 0 for a, p in zip(actual, predictions))
    ai_pi = sum(a != 0 and p != 0 for a, p in zip(actual, predictions))
    correct_count, incorrect_count = ac_pc + ac_pi, ai_pc + ai_pi
    binary_right = ac_pc + ai_pi
    exact = sum(a == p for a, p in zip(actual, predictions))
    balanced = ((ac_pc / correct_count) + (ai_pi / incorrect_count)) / 2
    return {
        "total": len(cases),
        "accuracy": exact / len(cases),
        "binary_accuracy": binary_right / len(cases),
        "correct_trace_recall": ac_pc / correct_count,
        "incorrect_trace_recall": ai_pi / incorrect_count,
        "balanced_accuracy": balanced,
        "balanced_binary_accuracy": balanced,
        "false_approval_rate": ai_pc / incorrect_count,
        "false_rejection_rate": ac_pi / correct_count,
        "confusion": {
            "actual_correct_predicted_correct": ac_pc,
            "actual_correct_predicted_incorrect": ac_pi,
            "actual_incorrect_predicted_correct": ai_pc,
            "actual_incorrect_predicted_incorrect": ai_pi,
        },
        "predictions_sha256": hashlib.sha256(
            json.dumps(list(predictions), separators=(",", ":")).encode()).hexdigest(),
    }


def _predict(model: Any, cases: Sequence[V2TrainingCase], config: V2Config, torch: Any) -> list[int]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(cases), 64):
            batch = cases[start:start + 64]
            ids = pad_v2_tokens([tokenize_observed_trace_v2(c.question, c.steps)
                                 for c in batch], config, torch)
            output = model(ids)
            correct = output["correctness_logits"].argmax(dim=-1)
            locations = output["earliest_error_logits"].argmax(dim=-1) + 1
            predictions.extend([0 if int(ok) == 1 else int(location)
                                for ok, location in zip(correct, locations)])
    return predictions


def _source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> dict[str, Any]:
    """Train and evaluate one deterministic CPU run; optimizer work is <=90 s."""
    config.validate()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the explicit experiment interpreter must provide PyTorch") from exc
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model_config = V2Config(seed=config.seed, width=config.width, heads=config.heads,
                            layers=config.layers)
    groups = build_splits(config)
    model = build_v2_model(model_config)
    initial_digest = parameter_digest_v2(model)
    evaluated = {name: cases for name, cases in groups.items() if name != "train"}
    evaluations: dict[str, dict[str, Any]] = {
        "frozen_random": {name: score_predictions(cases, _predict(model, cases, model_config, torch))
                          for name, cases in evaluated.items()},
        "always_approve": {name: score_predictions(cases, [0] * len(cases))
                           for name, cases in evaluated.items()},
        "always_reject_step_1": {name: score_predictions(cases, [1] * len(cases))
                                 for name, cases in evaluated.items()},
    }
    train_cases = list(groups["train"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    rng = random.Random(config.seed)
    started = time.perf_counter()
    losses = []
    optimizer_steps = 0
    model.train()
    for _epoch in range(config.epochs):
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            if time.perf_counter() - started > config.max_optimizer_seconds:
                raise RuntimeError("optimizer work exceeded the configured <=90 second budget")
            batch = [train_cases[index] for index in order[start:start + config.batch_size]]
            ids = pad_v2_tokens([tokenize_observed_trace_v2(c.question, c.steps)
                                 for c in batch], model_config, torch)
            labels = torch.tensor([c.first_error_step or 0 for c in batch], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = v2_loss(model(ids), labels, torch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().item()))
    optimizer_seconds = time.perf_counter() - started
    if optimizer_seconds > config.max_optimizer_seconds:
        raise RuntimeError("optimizer work exceeded the configured <=90 second budget")
    trained_digest = parameter_digest_v2(model)
    if trained_digest == initial_digest:
        raise RuntimeError("training did not update random V2 parameters")
    evaluations["trained_candidate"] = {
        name: score_predictions(cases, _predict(model, cases, model_config, torch))
        for name, cases in evaluated.items()
    }
    root = Path(__file__).resolve().parents[1]
    split_evidence = {}
    for name, cases in groups.items():
        split_evidence[name] = {
            "case_count": len(cases),
            "correct_count": sum(c.first_error_step is None for c in cases),
            "incorrect_count": sum(c.first_error_step is not None for c in cases),
            "dataset_sha256": dataset_fingerprint(cases),
            "case_ids_sha256": hashlib.sha256(
                json.dumps([c.case_id for c in cases], separators=(",", ":")).encode()).hexdigest(),
            "structure_fingerprints": sorted({c.structure_fingerprint for c in cases}),
            "corruption_families": dict(sorted(Counter(
                c.corruption_family for c in cases if c.first_error_step is not None).items())),
            "program_lengths": sorted({len(c.steps) for c in cases}),
        }
    return {
        "schema": SCHEMA,
        "qualification": "NOT_ESTABLISHED",
        "promotion_recommendation": "DO_NOT_PROMOTE",
        "source": {
            "requested_base_commit": "f2fe07bd1ee78610c5f2f2c415984f1fb98f49e5",
            "git_head_at_run": _source_commit(),
            "training_module_sha256": _file_sha256(Path(__file__)),
            "architecture_module_sha256": _file_sha256(root / "labyrinth" / "minotaur_scratch_v2.py"),
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "config": asdict(config),
        "model_config": asdict(model_config),
        "splits": split_evidence,
        "training": {"lineage": "random_from_seed", "optimizer": "AdamW",
                     "optimizer_steps": optimizer_steps,
                     "optimizer_seconds": optimizer_seconds,
                     "initial_loss": losses[0], "final_loss": losses[-1],
                     "losses_sha256": hashlib.sha256(
                         json.dumps(losses, separators=(",", ":")).encode()).hexdigest(),
                     "initial_weight_sha256": initial_digest,
                     "trained_weight_sha256": trained_digest,
                     "weights_saved": False},
        "evaluations": evaluations,
        "limitations": [
            "Each split draws from only four public structural templates.",
            "There is no independent private benchmark or evaluator custody.",
            "Synthetic generator, seeds, labels, and corruption rules are all public.",
            "The longer-program result is reserved structural transfer within this tiny generator, not generalization.",
            "This bounded CPU experiment cannot establish graduation or production readiness.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded random Minotaur V2 CPU training")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite existing evidence")
    report = run_experiment()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"qualification": report["qualification"],
                      "promotion_recommendation": report["promotion_recommendation"],
                      "optimizer_seconds": report["training"]["optimizer_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
