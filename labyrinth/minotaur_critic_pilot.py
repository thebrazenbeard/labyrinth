"""Opt-in pretrained Minotaur correctness critic, not an exit authority.

Runs genuinely trainable PEFT LoRA against a *local* safetensors causal LM,
with paired Yes/No judgments and frozen/zero-update/corrupted-label controls.
All split generators are public, so pilot metrics are development evidence,
not private blind verification or established reasoning generalization.
No model download, package installation, or production route mutation occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Sequence

from .hf_pilot import (
    PilotConfig, _adapter_manifest, _assert_adapter_unchanged,
    _assert_checkpoint_unchanged, _checkpoint_manifest, _deps,
    _device_headroom, _load_base, _lora_value_digest, _parameter_digest,
    _save_new, _sha_file, validate_config,
)
from .maze import generate_maze
from .guardian_benchmark import generate_process_cases
from .runner import _source_state, load_json


SCHEMA = "labyrinth.minotaur-critic-pilot.v2"
RELOAD_SCHEMA = "labyrinth.minotaur-critic-reload.v2"
_ARMS = ("frozen", "candidate", "zero_update", "corrupted")


@dataclass(frozen=True)
class CriticExample:
    example_id: str
    case_id: str
    case_prompt_sha256: str
    kind: str
    prompt: str
    trainee_answer: int
    is_correct: bool


@dataclass(frozen=True)
class CriticConfig:
    model_path: str
    output_dir: str
    seed: int = 777
    train_seed: int = 1357
    dev_seed: int = 2468
    transfer_seed: int = 3579
    train_count_per_tier: int = 24
    eval_count_per_tier: int = 8
    steps: int = 24
    max_tokens: int = 384
    learning_rate: float = 0.0003
    lora_rank: int = 4
    lora_alpha: int = 8
    device: str = "cpu"
    objective: str = "answer_correctness"  # or process_first_error

    def validate_fields(self) -> None:
        for name in ("seed", "train_seed", "dev_seed", "transfer_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**31:
                raise ValueError(f"{name} must be a bounded integer")
        for name, maximum in (("train_count_per_tier", 32), ("eval_count_per_tier", 32),
                              ("steps", 64), ("max_tokens", 1024),
                              ("lora_rank", 16), ("lora_alpha", 64)):
            value = getattr(self, name)
            minimum = 32 if name == "max_tokens" else 1
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if (type(self.learning_rate) is not float
                or not math.isfinite(self.learning_rate)
                or not 1e-7 <= self.learning_rate <= 0.001):
            raise ValueError("learning_rate must be a finite supported float")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.objective not in ("answer_correctness", "process_first_error"):
            raise ValueError("unsupported Minotaur critic objective")
        if len({self.train_seed, self.dev_seed, self.transfer_seed}) != 3:
            raise ValueError("training and evaluation seeds must differ")
        if self.steps > self.train_count_per_tier * 2:
            raise ValueError("one paired optimizer update per distinct training case maximum")

    def validate(self, *, output_must_be_new: bool = True) -> tuple[Path, Path, list[Path]]:
        self.validate_fields()
        # Reuse Labyrinth's established local-safetensors, path, weight-size,
        # tokenizer-metadata and output-isolation preflight.
        return validate_config(PilotConfig(
            model_path=self.model_path, output_dir=self.output_dir,
            seed=self.seed, tiers=(1,), steps=min(self.steps, 32),
            max_tokens=self.max_tokens, device=self.device,
        ), output_must_be_new=output_must_be_new)


def critic_examples(seed: int, split: str, count_per_tier: int,
                    tiers: tuple[int, ...]) -> list[CriticExample]:
    challenges = generate_maze(seed=seed, tiers=tiers, count=count_per_tier, split=split)
    examples: list[CriticExample] = []
    for challenge in challenges:
        answer = int(challenge.answer)
        # Always an incorrect integer, including wraparound for register tasks.
        wrong = ((answer + 1 + int(hashlib.sha256(challenge.id.encode()).hexdigest()[:4], 16) % 95) % 97
                 if challenge.tier == 4 else answer + 1 + int(hashlib.sha256(challenge.id.encode()).hexdigest()[:4], 16) % 19)
        for value, truth in ((answer, True), (wrong, False)):
            identifier = hashlib.sha256(f"{challenge.id}:{truth}".encode()).hexdigest()
            text = ("You are checking a trainee's proposed answer, not solving as the trainee.\n"
                    "Read the task and proposed answer. Decide whether the proposed integer\n"
                    "is substantively correct. Respond Yes or No.\n"
                    f"Task:\n{challenge.prompt}\n"
                    f"Trainee answer: ANSWER: {value}\nVerdict:")
            examples.append(CriticExample(
                example_id=identifier, case_id=challenge.id,
                case_prompt_sha256=hashlib.sha256(challenge.prompt.encode()).hexdigest(),
                kind=challenge.kind, prompt=text, trainee_answer=value,
                is_correct=truth,
            ))
    return examples


def process_critic_examples(
    seed: int, split: str, count_per_tier: int, tiers: tuple[int, ...],
) -> list[CriticExample]:
    """Matched first-error-claim verification, NOT direct step localization.

    The oracle label remains outside the model prompt. Each process trace yields
    a correct and a deliberately false proposed first-error claim.
    """
    names = {1: "fresh_numbers", 2: "new_composition", 4: "longer_program"}
    if (type(split) is not str or not split or type(tiers) is not tuple
            or not tiers or len(set(tiers)) != len(tiers)
            or any(tier not in names for tier in tiers)):
        raise ValueError("unsupported process curriculum profile")
    examples: list[CriticExample] = []
    for case in generate_process_cases(seed, count_per_slice=count_per_tier):
        if case.slice_name not in {names[tier] for tier in tiers}:
            continue
        truth = case.first_error_step or 0
        # For a bad trace alternate missed-error and wrong-location hypotheses.
        wrong = (1 if truth == 0 else
                 (0 if seed % 2 == 0 or len(case.steps) == 1
                  else 1 + truth % len(case.steps)))
        if wrong == truth:
            raise ValueError("process claim negative control equals truth")
        case_text = case.question + "\n" + "\n".join(case.steps)
        binding = hashlib.sha256(case_text.encode("utf-8")).hexdigest()
        for claimed, correct in ((truth, True), (wrong, False)):
            guess = "ALL_CORRECT" if claimed == 0 else f"ERROR_STEP:{claimed}"
            prompt = (
                "You are checking a trainee's proposed first error location, "
                "not accepting a persuasive explanation.\n"
                "Check every step against the printed program. Identify whether "
                "the proposed first error is exactly correct. Respond Yes or No.\n"
                f"Task:\n{case.question}\nTrace:\n"
                + "\n".join(case.steps)
                + f"\nTrainee proposed first error: {guess}\nVerdict:"
            )
            examples.append(CriticExample(
                example_id=hashlib.sha256((case.case_id + ":" + str(claimed)).encode()).hexdigest(),
                case_id=case.case_id, case_prompt_sha256=binding,
                kind="first_error." + case.slice_name,
                prompt=prompt, trainee_answer=claimed, is_correct=correct,
            ))
    return examples


def critic_datasets(config: CriticConfig) -> tuple[list[CriticExample], list[CriticExample], list[CriticExample]]:
    """The same objective-selected, split-bound cases for training and reload."""
    config.validate_fields()
    factory = (critic_examples if config.objective == "answer_correctness"
               else process_critic_examples)
    return (
        factory(config.train_seed, "train", config.train_count_per_tier, (1, 2)),
        factory(config.dev_seed, "critic_dev", config.eval_count_per_tier, (1, 2)),
        factory(config.transfer_seed, "critic_transfer", config.eval_count_per_tier, (4,)),
    )


def validate_dataset_separation(*groups: Sequence[CriticExample]) -> None:
    ids: set[str] = set()
    cases: set[str] = set()
    task_prompts: set[str] = set()
    for group in groups:
        own_ids = {ex.example_id for ex in group}
        own_cases = {ex.case_id for ex in group}
        own_prompts = {ex.case_prompt_sha256 for ex in group}
        if (len(own_ids) != len(group) or len(own_cases) * 2 != len(group)
                or len(own_prompts) != len(own_cases)):
            raise ValueError("malformed or duplicated paired examples")
        if ids & own_ids or cases & own_cases or task_prompts & own_prompts:
            raise ValueError("training and evaluation sets overlap")
        ids |= own_ids
        cases |= own_cases
        task_prompts |= own_prompts


def training_schedule(examples: Sequence[CriticExample], *, steps: int,
                      seed: int) -> list[tuple[CriticExample, CriticExample]]:
    """One balanced correct/wrong *pair per task* per optimizer update.

    Distinct case IDs prevent an update schedule from consuming the same
    problem twice, while pairing prevents class-prior learning from an
    entire optimizer step seeing only a positive or negative example.
    """
    if type(steps) is not int or steps < 1 or type(seed) is not int:
        raise ValueError("training steps must be positive with a valid seed")
    grouped: dict[str, list[CriticExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.case_id, []).append(ex)
    if steps > len(grouped) or any(
        len(pair) != 2 or {x.is_correct for x in pair} != {True, False}
        or len({x.example_id for x in pair}) != 2
        or len({x.case_prompt_sha256 for x in pair}) != 1
        or len({x.trainee_answer for x in pair}) != 2
        for pair in grouped.values()
    ):
        raise ValueError("need one unique balanced pair per distinct training task")
    rng = random.Random(seed)
    case_ids = sorted(grouped)
    rng.shuffle(case_ids)
    result: list[tuple[CriticExample, CriticExample]] = []
    for case_id in case_ids[:steps]:
        pair = list(grouped[case_id])
        rng.shuffle(pair)
        result.append((pair[0], pair[1]))
    return result


def encode_supervised_verdict(tokenizer: Any, example: CriticExample,
                              *, max_tokens: int) -> tuple[list[int], list[int]]:
    prefix = tokenizer.encode(example.prompt, add_special_tokens=True)
    label = tokenizer.encode(" Yes" if example.is_correct else " No", add_special_tokens=False)
    if len(label) != 1:
        raise ValueError("verdict must map to a single next-token ID")
    if not prefix or len(prefix) + 1 > max_tokens:
        raise ValueError("example exceeds max token length")
    return prefix + label, [-100] * len(prefix) + label


def prepare_verdict_pair(
    tokenizer: Any, pair: tuple[CriticExample, CriticExample],
    *, max_tokens: int, corrupt: bool,
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """Supervise both opposite labels on an identical task each step."""
    if (type(pair) is not tuple or len(pair) != 2
            or any(type(ex) is not CriticExample for ex in pair)
            or pair[0].case_id != pair[1].case_id
            or pair[0].case_prompt_sha256 != pair[1].case_prompt_sha256
            or pair[0].example_id == pair[1].example_id
            or pair[0].trainee_answer == pair[1].trainee_answer
            or {ex.is_correct for ex in pair} != {True, False}
            or type(corrupt) is not bool):
        raise ValueError("each update must have correct/wrong examples of one distinct task")
    prepared: list[tuple[list[int], list[int]]] = []
    for ex in pair:
        target = (ex if not corrupt else
                  CriticExample(**{**asdict(ex), "is_correct": not ex.is_correct}))
        prepared.append(encode_supervised_verdict(tokenizer, target,
                                                  max_tokens=max_tokens))
    return prepared[0], prepared[1]


def score_binary(examples: Sequence[CriticExample], predictions: Sequence[bool]) -> dict[str, Any]:
    if len(examples) != len(predictions) or not examples or any(type(v) is not bool for v in predictions):
        raise ValueError("complete boolean predictions required")
    tp = sum(ex.is_correct and p for ex, p in zip(examples, predictions))
    tn = sum(not ex.is_correct and not p for ex, p in zip(examples, predictions))
    fp = sum(not ex.is_correct and p for ex, p in zip(examples, predictions))
    fn = sum(ex.is_correct and not p for ex, p in zip(examples, predictions))
    positives, negatives = tp + fn, tn + fp
    return {
        "total": len(examples), "accuracy": (tp + tn) / len(examples),
        "balanced_accuracy": ((tp / positives if positives else 0)
                              + (tn / negatives if negatives else 0)) / 2,
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "false_approval_rate": fp / negatives if negatives else 0.0,
        "false_rejection_rate": fn / positives if positives else 0.0,
    }


def score_selective_margins(
    examples: Sequence[CriticExample], margins: Sequence[float], *, threshold: float,
) -> dict[str, Any]:
    """Report reject/approve/abstain at a frozen absolute Yes-minus-No logit margin.

    This is a development diagnostic, NEVER graduation evidence. Thresholds
    selected by looking at evaluation scores have no independent calibration.
    """
    if (not examples or len(examples) != len(margins)
            or any(type(ex) is not CriticExample for ex in examples)
            or type(threshold) not in (int, float)
            or not math.isfinite(threshold) or threshold < 0
            or any(type(m) not in (int, float) or not math.isfinite(m) for m in margins)):
        raise ValueError("finite aligned margins and a nonnegative threshold required")
    approved = rejected = abstained = right = false_approved = false_rejected = 0
    for ex, margin in zip(examples, margins):
        if abs(margin) <= threshold:
            abstained += 1
        elif margin > threshold:
            approved += 1
            right += int(ex.is_correct)
            false_approved += int(not ex.is_correct)
        else:
            rejected += 1
            right += int(not ex.is_correct)
            false_rejected += int(ex.is_correct)
    positives = sum(ex.is_correct for ex in examples)
    negatives = len(examples) - positives
    covered = approved + rejected
    return {
        "total": len(examples), "threshold": threshold,
        "coverage": covered / len(examples), "abstentions": abstained,
        "approvals": approved, "rejections": rejected,
        "approval_rate": approved / len(examples),
        "selective_accuracy": right / covered if covered else None,
        "false_approvals": false_approved, "false_rejections": false_rejected,
        "false_approval_rate": false_approved / negatives if negatives else None,
        "false_rejection_rate": false_rejected / positives if positives else None,
        "qualification": "NOT_ESTABLISHED",
    }


def verify_reload_contract(*, training_pid: int, reload_pid: int,
                           measured: dict[str, Any],
                           expected: dict[str, Any]) -> None:
    """A same-process load or changed predictions cannot certify persistence."""
    if type(training_pid) is not int or type(reload_pid) is not int:
        raise RuntimeError("Minotaur reload process identity is invalid")
    if training_pid == reload_pid:
        raise RuntimeError("Minotaur reload must run in a fresh process")
    # CPU inference may vary in the last few float32 units. Permit only
    # descriptive Yes-minus-No margin summaries to differ by at most 1e-4;
    # classification hashes, scores and all abstention decisions remain exact.
    margin_fields = frozenset({
        "margin_min", "margin_max", "margin_mean",
        "mean_margin_on_correct", "mean_margin_on_incorrect",
    })

    def normalize(value: Any, reference: Any, path: tuple[str, ...]) -> Any:
        if type(value) is dict and type(reference) is dict:
            if value.keys() != reference.keys():
                raise RuntimeError("Minotaur reloaded predictions or metrics differ")
            return {key: normalize(value[key], reference[key], path + (key,))
                    for key in value}
        if (len(path) >= 2 and path[-2] == "verdict_margin_diagnostic"
                and path[-1] in margin_fields):
            if value is None and reference is None:
                return None
            if (type(value) not in (int, float)
                    or type(reference) not in (int, float)
                    or not math.isfinite(value) or not math.isfinite(reference)
                    or abs(value - reference) > 0.0001):
                raise RuntimeError("Minotaur reloaded margin summary exceeds tolerance")
            return reference
        return value

    if normalize(measured, expected, ()) != expected:
        raise RuntimeError("Minotaur reloaded predictions or metrics differ")


def _runtime_manifest(torch: Any, device: str) -> dict[str, Any]:
    """Reproducibility metadata, not evidence of software authenticity."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": metadata.version("transformers"),
        "peft": metadata.version("peft"),
        "device": device,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _score(model: Any, tokenizer: Any, torch: Any, examples: list[CriticExample],
           *, device: str, max_tokens: int) -> dict[str, Any]:
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError("verdict tokens unavailable or ambiguous")
    model.eval()
    preds: list[bool] = []
    margins: list[float] = []
    with torch.no_grad():
        for example in examples:
            tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
            if not tokens or len(tokens) + 1 > max_tokens:
                raise ValueError("held-out prompt exceeds token budget")
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
            if not bool(torch.isfinite(logits[yes[0]])) or not bool(torch.isfinite(logits[no[0]])):
                raise RuntimeError("nonfinite Minotaur verdict logits")
            margin = float((logits[yes[0]] - logits[no[0]]).detach().cpu().item())
            if not math.isfinite(margin):
                raise RuntimeError("nonfinite Minotaur verdict margin")
            margins.append(margin)
            preds.append(margin > 0)
    report = score_binary(examples, preds)
    report["predictions_sha256"] = hashlib.sha256(
        bytes(int(p) for p in preds)).hexdigest()
    report["verdict_margin_diagnostic"] = {
        "qualification": "NOT_ESTABLISHED",
        "margin_min": min(margins),
        "margin_max": max(margins),
        "margin_mean": sum(margins) / len(margins),
        "mean_margin_on_correct": (sum(m for m, ex in zip(margins, examples)
                                      if ex.is_correct) /
                                   sum(ex.is_correct for ex in examples)
                                   if any(ex.is_correct for ex in examples) else None),
        "mean_margin_on_incorrect": (sum(m for m, ex in zip(margins, examples)
                                        if not ex.is_correct) /
                                     sum(not ex.is_correct for ex in examples)
                                     if any(not ex.is_correct for ex in examples) else None),
        "predeclared_thresholds": {
            str(threshold): score_selective_margins(examples, margins, threshold=threshold)
            for threshold in (0.0, 0.5, 1.0, 2.0)
        },
    }
    report["by_kind"] = {
        kind: score_binary([ex for ex in examples if ex.kind == kind],
                           [p for ex, p in zip(examples, preds) if ex.kind == kind])
        for kind in sorted({ex.kind for ex in examples})
    }
    return report


def _updates(model: Any, tokenizer: Any, torch: Any,
             pairs: list[tuple[CriticExample, CriticExample]],
             config: CriticConfig, *, mode: str) -> dict[str, Any]:
    if mode not in ("candidate", "zero_update", "corrupted"):
        raise ValueError("unsupported arm")
    prepared = [prepare_verdict_pair(tokenizer, pair,
                                     max_tokens=config.max_tokens,
                                     corrupt=mode == "corrupted") for pair in pairs]
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate, weight_decay=0.0,
    )
    losses: list[float] = []
    for prepared_pair in prepared:
        optimizer.zero_grad(set_to_none=True)
        paired_loss = 0.0
        # Accumulate equal positive/negative gradients before one step.
        # Sequential backward saves memory versus retaining both graphs.
        for tokens, labels in prepared_pair:
            ids = torch.tensor([tokens], dtype=torch.long, device=config.device)
            targets = torch.tensor([labels], dtype=torch.long, device=config.device)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=targets).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite critic loss")
            (loss / 2).backward()
            paired_loss += float(loss.detach().cpu().item()) / 2
        if mode == "zero_update":
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
        else:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()
                                            if p.requires_grad), 1.0)
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
               if parameter.requires_grad):
            raise RuntimeError("nonfinite critic adapter value after optimizer step")
        losses.append(paired_loss)
    return {"optimizer_steps": len(losses), "forward_backward_steps": 2 * len(losses),
            "examples_per_optimizer_step": 2,
            "visited_example_ids": [ex.example_id for pair in pairs for ex in pair],
            "zeroed_gradients": mode == "zero_update", "losses": losses}


def run_critic(config: CriticConfig) -> dict[str, Any]:
    base_path, output, weights = config.validate()
    base_manifest = _checkpoint_manifest(base_path, weights)
    train, dev, transfer = critic_datasets(config)
    validate_dataset_separation(train, dev, transfer)
    selected = training_schedule(train, steps=config.steps, seed=config.seed)
    torch, auto_model, auto_tokenizer, peft, device = _deps(config.device)
    _device_headroom(torch, device, weights)
    lora_cls, _, get_peft_model = peft
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    base, tokenizer = _load_base(base_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(base_path, weights, base_manifest)
    for ex in train + dev + transfer:
        encode_supervised_verdict(tokenizer, ex, max_tokens=config.max_tokens)
    frozen = {name: _score(base, tokenizer, torch, group, device=device,
                           max_tokens=config.max_tokens)
              for name, group in (("dev", dev), ("transfer", transfer))}
    model = get_peft_model(base, lora_cls(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
    ))
    initial = _parameter_digest(model, torch)
    init_values = {name: p.detach().clone()
                   for name, p in model.named_parameters() if p.requires_grad}

    def reset() -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in init_values:
                    parameter.copy_(init_values[name])
        if _parameter_digest(model, torch) != initial:
            raise RuntimeError("control failed identical LoRA initialization")

    results: dict[str, dict[str, Any]] = {"frozen": frozen}
    training: dict[str, Any] = {}
    digests = {"initial": initial}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for mode in ("candidate", "zero_update", "corrupted"):
        reset()
        training[mode] = _updates(model, tokenizer, torch, selected, config, mode=mode)
        digests[mode] = _parameter_digest(model, torch)
        if (mode == "candidate" and digests[mode] == initial) or (
                mode == "zero_update" and digests[mode] != initial):
            raise RuntimeError("candidate did not adapt or sham changed parameters")
        if mode == "candidate":
            tensor_digest = _lora_value_digest(model, torch)
            if tensor_digest != digests[mode]:
                raise RuntimeError("adapter tensor manifest mismatch")
            model.save_pretrained(str(output / "candidate_adapter"), safe_serialization=True)
        results[mode] = {name: _score(model, tokenizer, torch, group, device=device,
                                      max_tokens=config.max_tokens)
                         for name, group in (("dev", dev), ("transfer", transfer))}
    _assert_checkpoint_unchanged(base_path, weights, base_manifest)
    adapter_manifest = _adapter_manifest(output / "candidate_adapter")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "model_role": "pretrained_lm_correctness_critic_advisory_only",
        "qualification": "NOT_ESTABLISHED",
        "source": _source_state(),
        "process_id": os.getpid(),
        "runtime": _runtime_manifest(torch, device),
        "config": asdict(config),
        "base": base_manifest,
        "adapter_sha256": adapter_manifest,
        "candidate_lora_value_sha256": tensor_digest,
        "parameter_digests": digests,
        "training": training,
        "train_examples": [ex.example_id for ex in train],
        "dev_examples": [ex.example_id for ex in dev],
        "transfer_examples": [ex.example_id for ex in transfer],
        "evaluations": results,
        "limitations": [
            "Development cases and labels are generated from public finite templates.",
            "One training seed and a small holdout cannot prove generalization.",
            "No external evaluator, private oracle, or qualification authority.",
            "A separate process must load adapter and re-run the same cases.",
        ],
    }
    _save_new(output / "critic_report.json", report)
    return report


def reload_critic(run_dir: str, report_path: str) -> dict[str, Any]:
    root = Path(run_dir).resolve(strict=True)
    prior = load_json(root / "critic_report.json", strict=True)
    if prior.get("schema") != SCHEMA:
        raise ValueError("unsupported Minotaur critic training report")
    source = prior.get("source")
    if (not isinstance(source, dict) or
            source.get("source_tree_sha256") != _source_state()["source_tree_sha256"]):
        raise ValueError("source tree changed between training and reload")
    config = CriticConfig(**prior["config"])
    base_path, output, weights = config.validate(output_must_be_new=False)
    if output != root:
        raise ValueError("output path changed after training")
    if _checkpoint_manifest(base_path, weights) != prior["base"]:
        raise ValueError("pretrained base changed after training")
    adapter = root / "candidate_adapter"
    expected_adapter = prior["adapter_sha256"]
    if _adapter_manifest(adapter) != expected_adapter:
        raise ValueError("saved adapter changed")
    _, dev, transfer = critic_datasets(config)
    torch, auto_model, auto_tokenizer, peft, device = _deps(config.device)
    torch.use_deterministic_algorithms(True)
    current_runtime = _runtime_manifest(torch, device)
    if current_runtime != prior.get("runtime"):
        raise RuntimeError("training/reload runtime versions or settings differ")
    _, peft_model, _ = peft
    base, tokenizer = _load_base(base_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    model = peft_model.from_pretrained(base, str(adapter), is_trainable=False,
                                      local_files_only=True)
    digest = _lora_value_digest(model, torch)
    if digest != prior["candidate_lora_value_sha256"]:
        raise ValueError("reloaded LoRA tensor values differ")
    measurements = {name: _score(model, tokenizer, torch, group, device=device,
                                 max_tokens=config.max_tokens)
                    for name, group in (("dev", dev), ("transfer", transfer))}
    _assert_adapter_unchanged(adapter, expected_adapter)
    _assert_checkpoint_unchanged(base_path, weights, prior["base"])
    verify_reload_contract(training_pid=prior["process_id"], reload_pid=os.getpid(),
                           measured=measurements,
                           expected=prior["evaluations"]["candidate"])
    outcome = {
        "schema": RELOAD_SCHEMA,
        "training_report_sha256": _sha_file(root / "critic_report.json"),
        "process_id": os.getpid(),
        "different_process_id": os.getpid() != prior["process_id"],
        "reloaded_lora_value_sha256": digest,
        "same_predictions_and_metrics": measurements == prior["evaluations"]["candidate"],
        "evaluations": measurements,
        "qualification": "NOT_ESTABLISHED",
    }
    _save_new(Path(report_path), outcome)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in Minotaur correctness-critic LoRA pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--model-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=24)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate = sub.add_parser("reload")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = run_critic(CriticConfig(model_path=args.model_path,
                                             output_dir=args.output_dir,
                                             steps=args.steps, device=args.device))
            print(json.dumps({"schema": result["schema"],
                              "parameter_digests": result["parameter_digests"],
                              "evaluations": result["evaluations"]}))
        else:
            result = reload_critic(args.run_dir, args.report)
            print(json.dumps({"schema": result["schema"],
                              "different_process_id": result["different_process_id"],
                              "same_predictions_and_metrics": result["same_predictions_and_metrics"]}))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
