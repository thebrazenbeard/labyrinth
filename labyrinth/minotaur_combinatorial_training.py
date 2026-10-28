"""Bounded two-seed Minotaur V2 training on the combinatorial curriculum.

This is a CPU-only, random-initialization development experiment.  It has no
checkpoint writer, promotion authority, production loader, or legacy
``program_maze`` data path.  Holdout labels are scored by independent replay.
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
import subprocess
import time
from typing import Any, Mapping, Sequence

from .minotaur_combinatorial_curriculum import (
    CombinatorialCase,
    CombinatorialSplit,
    CurriculumConfig,
    build_combinatorial_splits,
)
from .minotaur_independent_eval import TraceEvaluationCase, evaluate_predictions
from .minotaur_scratch_v2 import (
    MAX_STEPS,
    V2Config,
    build_v2_model,
    pad_v2_tokens,
    parameter_digest_v2,
    tokenize_observed_trace_v2,
)


SCHEMA = "labyrinth.minotaur-combinatorial-training.v1"
REQUESTED_BASE = "d6fe2ee01a3d1e0e046ab62718dbaa943cd46779"


@dataclass(frozen=True)
class CombinatorialTrainingConfig:
    seeds: tuple[int, int] = (20261027, 20261028)
    dataset_seed: int = 20261027
    train_cases: int = 128
    dev_cases: int = 64
    transfer_cases: int = 64
    epochs: int = 24
    batch_size: int = 16
    learning_rate: float = 0.003
    width: int = 32
    heads: int = 4
    layers: int = 1
    max_optimizer_seconds: float = 89.0

    def validate(self) -> "CombinatorialTrainingConfig":
        counts = (self.train_cases, self.dev_cases, self.transfer_cases)
        if (type(self.seeds) is not tuple or len(self.seeds) != 2
                or len(set(self.seeds)) != 2
                or any(type(seed) is not int or not 0 <= seed < 2**31
                       for seed in self.seeds)
                or type(self.dataset_seed) is not int
                or not 0 <= self.dataset_seed < 2**31
                or any(type(count) is not int or count % 2 for count in counts)
                or not 128 <= self.train_cases <= 256
                or not 64 <= self.dev_cases <= 128
                or not 64 <= self.transfer_cases <= 128
                or type(self.epochs) is not int or not 1 <= self.epochs <= 30
                or type(self.batch_size) is not int or not 4 <= self.batch_size <= 64
                or type(self.learning_rate) is not float
                or not 0.00001 <= self.learning_rate <= 0.05
                or type(self.width) is not int or not 16 <= self.width <= 128
                or type(self.heads) is not int or not 1 <= self.heads <= 8
                or self.width % self.heads
                or type(self.layers) is not int or not 1 <= self.layers <= 3
                or type(self.max_optimizer_seconds) is not float
                or not 1.0 <= self.max_optimizer_seconds < 90.0):
            raise ValueError("invalid bounded combinatorial training configuration")
        return self


def build_training_splits(
        config: CombinatorialTrainingConfig) -> dict[str, CombinatorialSplit]:
    """Use only the new combinatorial curriculum with its canonical disjointness."""
    config.validate()
    return build_combinatorial_splits(CurriculumConfig(
        seed=config.dataset_seed, train_cases=config.train_cases,
        dev_cases=config.dev_cases, transfer_cases=config.transfer_cases))


def visible_tokens(case: CombinatorialCase) -> list[int]:
    """Tokenize only question and displayed states; labels are inaccessible."""
    return tokenize_observed_trace_v2(case.question, case.steps)


def mask_invalid_step_logits(logits: Any, lengths: Any, torch: Any) -> Any:
    """Return cloned logits with every nonexistent step set to negative infinity."""
    if (logits.ndim != 2 or logits.shape[1] != MAX_STEPS or lengths.ndim != 1
            or lengths.shape[0] != logits.shape[0]
            or not bool(((lengths >= 1) & (lengths <= MAX_STEPS)).all())):
        raise ValueError("step logits require one valid trace length per row")
    positions = torch.arange(MAX_STEPS, device=logits.device).unsqueeze(0)
    return logits.clone().masked_fill(positions >= lengths.unsqueeze(1), -torch.inf)


def _batch(model: Any, cases: Sequence[CombinatorialCase], model_config: V2Config,
           torch: Any) -> tuple[dict[str, Any], Any, Any]:
    ids = pad_v2_tokens([visible_tokens(case) for case in cases], model_config, torch)
    labels = torch.tensor([case.first_error_step or 0 for case in cases], dtype=torch.long)
    lengths = torch.tensor([len(case.steps) for case in cases], dtype=torch.long)
    outputs = model(ids)
    outputs["earliest_error_logits"] = mask_invalid_step_logits(
        outputs["earliest_error_logits"], lengths, torch)
    return outputs, labels, lengths


def _loss(outputs: Mapping[str, Any], labels: Any, lengths: Any, torch: Any) -> Any:
    if not bool((labels <= lengths).all()):
        raise ValueError("first-error label points beyond the visible trace")
    correct_targets = (labels == 0).to(dtype=torch.long)
    binary = torch.nn.functional.cross_entropy(outputs["correctness_logits"], correct_targets)
    wrong = labels != 0
    if not bool(wrong.any()):
        return binary
    location = torch.nn.functional.cross_entropy(
        outputs["earliest_error_logits"][wrong], labels[wrong] - 1)
    return binary + location


def _predict(model: Any, cases: Sequence[CombinatorialCase], model_config: V2Config,
             torch: Any) -> dict[str, int]:
    predictions: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), 64):
            group = cases[start:start + 64]
            outputs, _, _ = _batch(model, group, model_config, torch)
            correct = outputs["correctness_logits"].argmax(dim=-1) == 1
            locations = outputs["earliest_error_logits"].argmax(dim=-1) + 1
            for case, is_correct, location in zip(group, correct.tolist(), locations.tolist()):
                predictions[case.case_id] = 0 if is_correct else int(location)
    return predictions


def _evaluation_cases(cases: Sequence[CombinatorialCase]) -> list[TraceEvaluationCase]:
    return [TraceEvaluationCase(
        case_id=case.case_id, question=case.question, steps=case.steps,
        claimed_first_error=case.first_error_step, slice_name=case.split,
        structure=case.canonical_sequence,
    ) for case in cases]


def _independent_score(cases: Sequence[CombinatorialCase], predictions: Mapping[str, int],
                       training_structures: Sequence[str]) -> dict[str, Any]:
    report = evaluate_predictions(
        _evaluation_cases(cases), predictions,
        training_structures=training_structures, iid_binomial_attested=False)
    if report["disposition"] != "SCORED":
        raise RuntimeError("independent evaluator refused generated curriculum")
    return report


def _controls(cases: Sequence[CombinatorialCase],
              training_structures: Sequence[str]) -> dict[str, Any]:
    return {
        "always_approve": _independent_score(
            cases, {case.case_id: 0 for case in cases}, training_structures),
        "always_reject_step_1": _independent_score(
            cases, {case.case_id: 1 for case in cases}, training_structures),
    }


def _train_seed(config: CombinatorialTrainingConfig, seed: int,
                splits: Mapping[str, CombinatorialSplit], torch: Any) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model_config = V2Config(seed=seed, width=config.width, heads=config.heads,
                            layers=config.layers)
    model = build_v2_model(model_config)
    initial_digest = parameter_digest_v2(model)
    training_structures = sorted({case.canonical_sequence
                                  for case in splits["train"].cases})
    evaluated = ("dev", "structure_disjoint_transfer")
    frozen = {name: _independent_score(
        splits[name].cases, _predict(model, splits[name].cases, model_config, torch),
        training_structures) for name in evaluated}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=0.0)
    train_cases = list(splits["train"].cases)
    rng = random.Random(seed)
    losses: list[float] = []
    started = time.perf_counter()
    optimizer_steps = 0
    model.train()
    for _epoch in range(config.epochs):
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            if time.perf_counter() - started >= config.max_optimizer_seconds:
                raise RuntimeError("optimizer exceeded its per-seed under-90-second budget")
            batch = [train_cases[index] for index in order[start:start + config.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            outputs, labels, lengths = _batch(model, batch, model_config, torch)
            loss = _loss(outputs, labels, lengths, torch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            optimizer_steps += 1
    elapsed = time.perf_counter() - started
    if elapsed >= 90.0 or elapsed > config.max_optimizer_seconds:
        raise RuntimeError("optimizer exceeded its per-seed under-90-second budget")
    trained_digest = parameter_digest_v2(model)
    if trained_digest == initial_digest:
        raise RuntimeError("training did not update random parameters")
    trained = {name: _independent_score(
        splits[name].cases, _predict(model, splits[name].cases, model_config, torch),
        training_structures) for name in evaluated}
    return {
        "seed": seed,
        "training": {
            "lineage": "random_from_seed", "pretrained_weights": False,
            "optimizer": "AdamW", "optimizer_steps": optimizer_steps,
            "optimizer_seconds": elapsed, "initial_loss": losses[0],
            "final_loss": losses[-1], "minimum_loss": min(losses),
            "losses_sha256": hashlib.sha256(json.dumps(
                losses, separators=(",", ":")).encode()).hexdigest(),
            "initial_weight_sha256": initial_digest,
            "trained_weight_sha256": trained_digest, "weights_saved": False,
        },
        "frozen_random": frozen,
        "trained_candidate": trained,
    }


def _dataset_sha(cases: Sequence[CombinatorialCase]) -> str:
    payload = [{"case_id": case.case_id, "question": case.question,
                "steps": case.steps, "first_error_step": case.first_error_step,
                "canonical_sequence": case.canonical_sequence,
                "fault_family": case.fault_family} for case in cases]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_evidence(splits: Mapping[str, CombinatorialSplit]) -> dict[str, Any]:
    return {name: {
        "source_group": name,
        "case_count": len(split.cases),
        "correct_count": sum(case.first_error_step is None for case in split.cases),
        "incorrect_count": sum(case.first_error_step is not None for case in split.cases),
        "curriculum_commitment_sha256": split.commitment,
        "dataset_sha256": _dataset_sha(split.cases),
        "canonical_structure_count": len({case.canonical_sequence for case in split.cases}),
        "program_lengths": sorted({len(case.steps) for case in split.cases}),
        "fault_families": dict(sorted(Counter(
            case.fault_family for case in split.cases).items())),
    } for name, split in splits.items()}


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.run(["git", *command], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(
        config: CombinatorialTrainingConfig = CombinatorialTrainingConfig()) -> dict[str, Any]:
    """Execute the frozen two-seed CPU protocol and return development evidence."""
    config.validate()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the explicit experiment interpreter must provide PyTorch") from exc
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    splits = build_training_splits(config)
    train_structures = sorted({case.canonical_sequence for case in splits["train"].cases})
    evaluated = ("dev", "structure_disjoint_transfer")
    controls = {name: _controls(splits[name].cases, train_structures) for name in evaluated}
    seed_results = [_train_seed(config, seed, splits, torch) for seed in config.seeds]
    root = Path(__file__).resolve().parents[1]
    head = _git(("rev-parse", "HEAD"))
    dirty = _git(("status", "--porcelain"))
    return {
        "schema": SCHEMA,
        "qualification": "NOT_ESTABLISHED",
        "promotion_recommendation": "DO_NOT_PROMOTE",
        "source": {
            "requested_base_commit": REQUESTED_BASE,
            "git_head_at_run": head,
            "source_tree_clean_at_run": dirty == "",
            "training_module_sha256": _sha(Path(__file__)),
            "curriculum_module_sha256": _sha(root / "labyrinth" / "minotaur_combinatorial_curriculum.py"),
            "independent_eval_module_sha256": _sha(root / "labyrinth" / "minotaur_independent_eval.py"),
            "architecture_module_sha256": _sha(root / "labyrinth" / "minotaur_scratch_v2.py"),
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "config": asdict(config),
        "controls": controls,
        "splits": _split_evidence(splits),
        "seed_results": seed_results,
        "protocol_assertions": {
            "data_source": "labyrinth.minotaur_combinatorial_curriculum",
            "legacy_four_template_generator_used": False,
            "invalid_step_logits_masked_to_negative_infinity": True,
            "holdouts_used_for_training_or_threshold_selection": False,
            "labels_present_in_model_tokens": False,
            "independent_replay_required_before_scoring": True,
            "weights_saved": False,
        },
        "limitations": [
            "Public synthetic data and public replay are not a private blind evaluation.",
            "Two deterministic seeds are replication probes, not a statistical guarantee.",
            "No checkpoint is saved and this experiment has no production authority.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train bounded combinatorial Minotaur V2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite existing evidence")
    report = run_experiment()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"git_head_at_run": report["source"]["git_head_at_run"],
                      "promotion_recommendation": report["promotion_recommendation"],
                      "seed_optimizer_seconds": [item["training"]["optimizer_seconds"]
                                                 for item in report["seed_results"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
