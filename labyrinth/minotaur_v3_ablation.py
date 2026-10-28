"""Bounded scratch-only ablations for the Minotaur V2 learning failure.

This module has no checkpoint loading, saving, promotion, or deployment path.
Every neural arm starts from seeded random weights and is evaluated on fixed,
identity-disjoint public synthetic holdouts.  Development and transfer examples
are never augmented or used by the train-only decision calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from typing import Any, Sequence

from .minotaur_scratch_v2 import (
    MAX_STEPS,
    V2Config,
    build_v2_model,
    pad_v2_tokens,
    parameter_digest_v2,
    tokenize_observed_trace_v2,
)
from .minotaur_v2_training import (
    ExperimentConfig,
    V2TrainingCase,
    _corrupt,
    build_splits,
    dataset_fingerprint,
    score_predictions,
)


SCHEMA = "labyrinth.minotaur.v3-scratch-ablation.v1"
REQUESTED_BASE = "d0263dbf6e431f0b8f3140b29ee72c140318265e"
VARIANTS = ("v2_joint", "weighted_correctness", "joint_label",
            "augmented_joint_label")


@dataclass(frozen=True)
class AblationConfig:
    seeds: tuple[int, ...] = (20261027, 20261028)
    train_programs: int = 64
    eval_programs: int = 32
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 0.003
    width: int = 32
    heads: int = 4
    layers: int = 1
    correctness_weight: float = 3.0
    max_optimizer_seconds: float = 90.0

    def validate(self) -> "AblationConfig":
        if (not isinstance(self.seeds, tuple) or len(self.seeds) < 2
                or len(self.seeds) > 4 or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in self.seeds)
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
                or type(self.correctness_weight) is not float
                or not 1.0 <= self.correctness_weight <= 10.0
                or type(self.max_optimizer_seconds) is not float
                or not 1.0 <= self.max_optimizer_seconds <= 90.0):
            raise ValueError("invalid bounded V3 ablation configuration")
        return self


def _v2_config(config: AblationConfig, seed: int) -> V2Config:
    return V2Config(seed=seed, width=config.width, heads=config.heads,
                    layers=config.layers)


def _augmented_train(cases: Sequence[V2TrainingCase]) -> tuple[V2TrainingCase, ...]:
    """Add deterministic train-only corruption diversity without touching holdouts."""
    output = list(cases)
    clean_cases = [case for case in cases if case.first_error_step is None]
    for clean_case in clean_cases:
        seen = {(case.first_error_step, case.corruption_family)
                for case in cases if case.question == clean_case.question
                and case.first_error_step is not None}
        # The V2 corruption function maps index deterministically to both family
        # and position.  Twelve probes span all families and positions 1..4.
        for probe in range(12):
            steps, error, family = _corrupt(clean_case.question, clean_case.steps, probe)
            key = (error, family)
            if key in seen:
                continue
            seen.add(key)
            payload = json.dumps({"parent": clean_case.case_id, "steps": steps,
                                  "error": error, "family": family},
                                 sort_keys=True, separators=(",", ":"))
            output.append(V2TrainingCase(
                "v3aug-" + hashlib.sha256(payload.encode()).hexdigest()[:32],
                "train_augmented", clean_case.question, steps, error,
                clean_case.structure_fingerprint, family,
            ))
    return tuple(output)


def build_ablation_splits(config: AblationConfig) -> dict[str, tuple[V2TrainingCase, ...]]:
    config.validate()
    base = build_splits(ExperimentConfig(
        seed=config.seeds[0], train_programs=config.train_programs,
        eval_programs=config.eval_programs, epochs=1,
        batch_size=config.batch_size, learning_rate=config.learning_rate,
        width=config.width, heads=config.heads, layers=config.layers,
        max_optimizer_seconds=config.max_optimizer_seconds,
    ))
    result = dict(base)
    result["train_augmented"] = _augmented_train(base["train"])
    result["dev_evaluation_copy"] = base["dev"]
    original_ids = {case.case_id for group in base.values() for case in group}
    augmented_ids = [case.case_id for case in result["train_augmented"]
                     if case.case_id.startswith("v3aug-")]
    if original_ids & set(augmented_ids) or len(augmented_ids) != len(set(augmented_ids)):
        raise RuntimeError("augmentation identity collision")
    return result


def _batch(model: Any, cases: Sequence[V2TrainingCase], model_config: V2Config,
           torch: Any) -> tuple[dict[str, Any], Any]:
    ids = pad_v2_tokens([tokenize_observed_trace_v2(case.question, case.steps)
                         for case in cases], model_config, torch)
    labels = torch.tensor([case.first_error_step or 0 for case in cases], dtype=torch.long)
    return model(ids), labels


def _loss(outputs: dict[str, Any], labels: Any, variant: str,
          correctness_weight: float, torch: Any) -> Any:
    correct = (labels == 0).to(dtype=torch.long)
    binary = torch.nn.functional.cross_entropy(outputs["correctness_logits"], correct)
    wrong = labels != 0
    location = (torch.nn.functional.cross_entropy(
        outputs["earliest_error_logits"][wrong], labels[wrong] - 1)
        if bool(wrong.any()) else binary * 0.0)
    if variant == "v2_joint":
        return binary + location
    if variant == "weighted_correctness":
        return correctness_weight * binary + location
    if variant in ("joint_label", "augmented_joint_label"):
        # Class zero means fully correct.  Classes 1..6 jointly encode
        # incorrectness and the earliest location, preventing contradictory heads.
        logits = torch.cat((outputs["correctness_logits"][:, 1:2],
                            outputs["correctness_logits"][:, 0:1]
                            + outputs["earliest_error_logits"]), dim=-1)
        class_weights = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                     dtype=logits.dtype, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
    raise ValueError("unknown V3 ablation variant")


def _raw_outputs(model: Any, cases: Sequence[V2TrainingCase], model_config: V2Config,
                 torch: Any) -> tuple[list[float], list[int]]:
    margins: list[float] = []
    locations: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), 64):
            batch_cases = cases[start:start + 64]
            output, _ = _batch(model, batch_cases, model_config, torch)
            margin = output["correctness_logits"][:, 1] - output["correctness_logits"][:, 0]
            margins.extend(float(value) for value in margin.tolist())
            location_logits = output["earliest_error_logits"].clone()
            for row, case in enumerate(batch_cases):
                location_logits[row, len(case.steps):] = -torch.inf
            locations.extend(int(value) + 1 for value in location_logits.argmax(dim=-1).tolist())
    return margins, locations


def _train_calibration(model: Any, cases: Sequence[V2TrainingCase], model_config: V2Config,
                       torch: Any) -> tuple[float, float]:
    """Select a binary threshold on training predictions only, never a holdout."""
    margins, _ = _raw_outputs(model, cases, model_config, torch)
    labels = [case.first_error_step is None for case in cases]
    candidates = sorted(set([-math.inf, 0.0, math.inf] + margins))
    best_threshold, best_balanced = 0.0, -1.0
    for threshold in candidates:
        tp = sum(label and margin >= threshold for label, margin in zip(labels, margins))
        tn = sum(not label and margin < threshold for label, margin in zip(labels, margins))
        positives = sum(labels)
        negatives = len(labels) - positives
        balanced = ((tp / positives) + (tn / negatives)) / 2
        # Deterministic conservative tie break: closest to the natural zero boundary.
        key = (balanced, -abs(threshold) if math.isfinite(threshold) else -math.inf)
        incumbent = (best_balanced, -abs(best_threshold)
                     if math.isfinite(best_threshold) else -math.inf)
        if key > incumbent:
            best_threshold, best_balanced = threshold, balanced
    return best_threshold, best_balanced


def _evaluate(model: Any, cases: Sequence[V2TrainingCase], model_config: V2Config,
              threshold: float, torch: Any) -> dict[str, Any]:
    margins, locations = _raw_outputs(model, cases, model_config, torch)
    predictions = [0 if margin >= threshold else location
                   for margin, location in zip(margins, locations)]
    return score_predictions(cases, predictions)


def _train_seed(config: AblationConfig, variant: str, seed: int,
                splits: dict[str, tuple[V2TrainingCase, ...]], torch: Any) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model_config = _v2_config(config, seed)
    model = build_v2_model(model_config)
    initial_digest = parameter_digest_v2(model)
    train_name = "train_augmented" if variant == "augmented_joint_label" else "train"
    train_cases = list(splits[train_name])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    rng = random.Random(seed)
    started = time.perf_counter()
    losses: list[float] = []
    gradients: list[float] = []
    epoch_losses: list[float] = []
    steps = 0
    model.train()
    for _epoch in range(config.epochs):
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        current: list[float] = []
        for start in range(0, len(order), config.batch_size):
            if time.perf_counter() - started > config.max_optimizer_seconds:
                raise RuntimeError("V3 optimizer work exceeded per-experiment budget")
            cases = [train_cases[index] for index in order[start:start + config.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            outputs, labels = _batch(model, cases, model_config, torch)
            loss = _loss(outputs, labels, variant, config.correctness_weight, torch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite V3 training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite V3 gradient norm")
            optimizer.step()
            value = float(loss.detach().item())
            losses.append(value)
            current.append(value)
            gradients.append(float(norm.detach().item()))
            steps += 1
        epoch_losses.append(sum(current) / len(current))
    elapsed = time.perf_counter() - started
    if elapsed > config.max_optimizer_seconds:
        raise RuntimeError("V3 optimizer work exceeded per-experiment budget")
    trained_digest = parameter_digest_v2(model)
    if trained_digest == initial_digest:
        raise RuntimeError("V3 training did not update parameters")
    calibration_threshold, calibration_score = _train_calibration(
        model, train_cases, model_config, torch)
    evaluation_groups = {
        "train": splits["train"], "dev": splits["dev"],
        "composition_transfer": splits["composition_transfer"],
        "longer_transfer": splits["longer_transfer"],
    }
    evaluations = {name: _evaluate(model, cases, model_config,
                                   calibration_threshold, torch)
                   for name, cases in evaluation_groups.items()}
    return {
        "seed": seed,
        "training": {
            "lineage": "random_from_seed", "pretrained_weights": False,
            "training_split": train_name, "optimizer": "AdamW",
            "optimizer_steps": steps, "optimizer_seconds": elapsed,
            "initial_loss": losses[0], "final_loss": losses[-1],
            "minimum_loss": min(losses), "epoch_mean_losses": epoch_losses,
            "losses_sha256": hashlib.sha256(json.dumps(
                losses, separators=(",", ":")).encode()).hexdigest(),
            "gradient_norm_first": gradients[0],
            "gradient_norm_last": gradients[-1],
            "gradient_norm_max": max(gradients),
            "initial_weight_sha256": initial_digest,
            "trained_weight_sha256": trained_digest, "weights_saved": False,
        },
        "calibration": {
            "source_split": train_name, "holdout_used": False,
            "correctness_margin_threshold": calibration_threshold,
            "train_balanced_binary_accuracy": calibration_score,
        },
        "evaluations": evaluations,
    }


def _aggregate(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "dev", "composition_transfer", "longer_transfer"):
        result[split] = {}
        for metric in ("accuracy", "binary_accuracy", "balanced_accuracy",
                       "false_approval_rate", "false_rejection_rate"):
            values = [item["evaluations"][split][metric] for item in seed_results]
            result[split][metric + "_mean"] = sum(values) / len(values)
            result[split][metric + "_min"] = min(values)
            result[split][metric + "_max"] = max(values)
    return result


def choose_promotion(variants: dict[str, dict[str, Any]]) -> str:
    """A cautious signal gate; it never grants deployment or qualification."""
    for result in variants.values():
        scores = [seed["evaluations"]["longer_transfer"]["balanced_accuracy"]
                  for seed in result["seed_results"]]
        if len(scores) >= 2 and min(scores) >= 0.60:
            return "STRUCTURAL_SIGNAL_ONLY"
    return "DO_NOT_PROMOTE"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_evidence(splits: dict[str, tuple[V2TrainingCase, ...]]) -> dict[str, Any]:
    evidence = {}
    for name in ("train", "train_augmented", "dev", "composition_transfer",
                 "longer_transfer"):
        cases = splits[name]
        evidence[name] = {
            "case_count": len(cases), "dataset_sha256": dataset_fingerprint(cases),
            "correct_count": sum(case.first_error_step is None for case in cases),
            "incorrect_count": sum(case.first_error_step is not None for case in cases),
            "structure_fingerprints": sorted({case.structure_fingerprint for case in cases}),
            "program_lengths": sorted({len(case.steps) for case in cases}),
            "corruption_families": dict(sorted(Counter(
                case.corruption_family for case in cases
                if case.first_error_step is not None).items())),
        }
    return evidence


def run_ablation(config: AblationConfig = AblationConfig(),
                 *, variants: Sequence[str] = VARIANTS) -> dict[str, Any]:
    config.validate()
    if (not variants or len(set(variants)) != len(variants)
            or any(variant not in VARIANTS for variant in variants)):
        raise ValueError("variants must be a unique nonempty subset of VARIANTS")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("explicit V3 experiment interpreter must provide PyTorch") from exc
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    splits = build_ablation_splits(config)
    evaluation_groups = {name: splits[name] for name in
                         ("train", "dev", "composition_transfer", "longer_transfer")}
    trivial = {}
    for control, prediction in (("always_approve", 0), ("always_reject_step_1", 1)):
        trivial[control] = {name: score_predictions(cases, [prediction] * len(cases))
                            for name, cases in evaluation_groups.items()}
    frozen_results = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        model_config = _v2_config(config, seed)
        model = build_v2_model(model_config)
        frozen_results.append({
            "seed": seed,
            "evaluations": {name: _evaluate(model, cases, model_config, 0.0, torch)
                            for name, cases in evaluation_groups.items()},
        })
    controls = {**trivial, "frozen_random": {"seed_results": frozen_results}}
    variant_results = {}
    for variant in variants:
        results = [_train_seed(config, variant, seed, splits, torch)
                   for seed in config.seeds]
        variant_results[variant] = {"seed_results": results,
                                    "aggregate": _aggregate(results)}
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": SCHEMA, "qualification": "NOT_ESTABLISHED",
        "source": {
            "requested_base_commit": REQUESTED_BASE, "git_head_at_run": _git_head(),
            "ablation_module_sha256": _sha(Path(__file__)),
            "v2_training_module_sha256": _sha(root / "labyrinth" / "minotaur_v2_training.py"),
            "architecture_module_sha256": _sha(root / "labyrinth" / "minotaur_scratch_v2.py"),
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "config": asdict(config), "splits": _split_evidence(splits),
        "controls": controls, "variants": variant_results,
        "diagnosis": [
            "V2 binary balanced accuracy was 0.5 on every holdout despite lower loss.",
            "Paired correct/faulty traces received nearly paired binary decisions, consistent with a shortcut rather than learned fault detection.",
            "These ablations test loss coupling, correctness weighting, unified labels, train-only augmentation, and train-only margin calibration.",
        ],
        "limitations": [
            "All cases are public synthetic data from four templates per split.",
            "Train-only threshold selection can overfit training and is not independent calibration evidence.",
            "The same generator and corruption mechanism underlie training and transfer.",
            "Two seeds are replication probes, not a statistical generalization guarantee.",
            "No pretrained weights, private evaluator, checkpoint saving, deployment, or production authority is involved.",
        ],
    }
    report["promotion_recommendation"] = choose_promotion(variant_results)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Minotaur V3 scratch ablations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite existing evidence")
    report = run_ablation()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"qualification": report["qualification"],
                      "promotion_recommendation": report["promotion_recommendation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
