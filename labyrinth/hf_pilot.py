"""Opt-in local Hugging Face/PEFT LoRA pilot: *actual* LM parameter updates.

This intentionally uses an already-present, small local safetensors causal-LM
checkpoint. It does not download a model, install dependencies, execute model
output, or attempt to fit multi-billion-parameter weights onto a 4 GiB GPU.

The generated mazes are public finite templates, so even a successful update
does not establish broad reasoning transfer. Running this module is not part of
the dependency-free Labyrinth test suite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

from .evaluation import score_response
from .maze import Challenge, generate_maze
from .runner import _source_state, load_json


LEGACY_SCHEMA = "labyrinth.hf-lora-pilot.v2"
SCHEMA = "labyrinth.hf-lora-pilot.v3"
MEZO_SCHEMA = "labyrinth.hf-lora-pilot.v4"
EVAL_SCHEMA = "labyrinth.hf-lora-reload-eval.v2"
EVAL_SEED_XOR = 0x5C18A7
# No quantization/CPU offload in v1: bound pre-existing weights to a small size.
MAX_LOCAL_WEIGHTS = 1_073_741_824


@dataclass(frozen=True)
class PilotConfig:
    model_path: str
    output_dir: str
    seed: int = 0
    train_count: int = 8
    eval_count: int = 8
    tiers: tuple[int, ...] = (1,)
    steps: int = 2
    max_tokens: int = 256
    max_new_tokens: int = 16
    learning_rate: float = 0.0002
    lora_rank: int = 4
    lora_alpha: int = 8
    device: str = "cpu"
    answer_only: bool = False
    optimizer: str = "adamw"
    zo_epsilon: float = 0.001


def _check_int(value: object, label: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")


def _paths(config: PilotConfig, *, output_must_be_new: bool) -> tuple[Path, Path, list[Path]]:
    if not isinstance(config.model_path, str) or not isinstance(config.output_dir, str):
        raise ValueError("model_path and output_dir must be strings")
    model = Path(config.model_path).expanduser().resolve(strict=True)
    if not model.is_dir() or not (model / "config.json").is_file() or (model / "config.json").is_symlink():
        raise ValueError("model_path must be an existing local HF model directory")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(not path.is_file() or path.is_symlink() for path in weights):
        raise ValueError("local safetensors weights required; pickle/.bin-only models refused")
    total = sum(path.stat().st_size for path in weights)
    if not 0 < total <= MAX_LOCAL_WEIGHTS:
        raise ValueError("checkpoint exceeds the 1 GiB no-quantization pilot limit")
    _validate_sharded_index(model, weights)
    raw_output = Path(config.output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError("output path must not be a symlink")
    output = raw_output.resolve(strict=False)
    if output == model or model in output.parents or output in model.parents:
        raise ValueError("output directory must be separate from local model directory")
    if output_must_be_new and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    if output_must_be_new and not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    return model, output, weights


def _validate_sharded_index(model: Path, weights: list[Path]) -> None:
    """Fail closed if a sharded checkpoint index escapes the pinned local files."""
    index = model / "model.safetensors.index.json"
    if not index.exists() and not index.is_symlink():
        return
    if not index.is_file() or index.is_symlink():
        raise ValueError("sharded index must be a regular non-symlink JSON file")
    try:
        parsed = load_json(index, strict=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid sharded index: {exc}") from exc
    mapping = parsed.get("weight_map") if isinstance(parsed, dict) else None
    filenames = {p.name for p in weights}
    if (not isinstance(mapping, dict) or not mapping
            or any(type(name) is not str or not name or type(target) is not str
                   or target not in filenames for name, target in mapping.items())
            or set(mapping.values()) != filenames):
        raise ValueError("sharded index refers to unknown, unused, or nonlocal weight files")


def validate_config(config: PilotConfig, *, output_must_be_new: bool = True) -> tuple[Path, Path, list[Path]]:
    """Reject unsupported resource scales before importing heavyweight libraries."""
    if not isinstance(config, PilotConfig):
        raise TypeError("expected PilotConfig")
    _check_int(config.seed, "seed", 0, 2**31 - 1)
    _check_int(config.train_count, "train_count", 2, 32)
    _check_int(config.eval_count, "eval_count", 2, 32)
    _check_int(config.steps, "steps", 1, 128)
    _check_int(config.max_tokens, "max_tokens", 32, 1024)
    _check_int(config.max_new_tokens, "max_new_tokens", 1, 32)
    _check_int(config.lora_rank, "lora_rank", 1, 16)
    _check_int(config.lora_alpha, "lora_alpha", 1, 64)
    if type(config.learning_rate) is not float or not math.isfinite(config.learning_rate) or not 1e-7 <= config.learning_rate <= 0.001:
        raise ValueError("learning_rate must be a finite float in [1e-7, 0.001]")
    if type(config.tiers) is not tuple or not config.tiers or any(type(t) is not int for t in config.tiers):
        raise ValueError("tiers must be a non-empty tuple of integers")
    # Invoke the generator's source-of-truth validation; no eval/gold read here.
    generate_maze(seed=config.seed, count=1, tiers=config.tiers, split="train")
    if len(config.tiers) * config.train_count > 64:
        raise ValueError("training set limited to 64 examples")
    if config.steps < len(config.tiers):
        raise ValueError("steps insufficient for each training tier")
    if config.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(config.answer_only) is not bool:
        raise ValueError("answer_only must be a boolean")
    if config.optimizer not in ("adamw", "mezo") or type(config.optimizer) is not str:
        raise ValueError("optimizer must be 'adamw' or 'mezo'")
    if (type(config.zo_epsilon) is not float or not math.isfinite(config.zo_epsilon)
            or not 1e-6 <= config.zo_epsilon <= 0.1):
        raise ValueError("zo_epsilon must be a finite float in [1e-6, 0.1]")
    resolved = _paths(config, output_must_be_new=output_must_be_new)
    _metadata_digest(resolved[0])
    return resolved


def training_cases(config: PilotConfig) -> tuple[list[Challenge], list[Challenge]]:
    """Keep supervision entirely on the TRAIN split; eval gold never enters SGD."""
    train = generate_maze(seed=config.seed, count=config.train_count, tiers=config.tiers, split="train")
    heldout = generate_maze(seed=config.seed ^ EVAL_SEED_XOR, count=config.eval_count,
                            tiers=config.tiers, split="eval")
    if {c.id for c in train} & {c.id for c in heldout}:
        raise ValueError("training/held-out challenge IDs overlap")
    if {c.prompt for c in train} & {c.prompt for c in heldout}:
        raise ValueError("training/held-out prompt text overlaps")
    return train, heldout


def derange_answers(cases: list[Challenge]) -> list[str]:
    """Permutation control preserving the answer multiset, with zero collisions."""
    labels = [case.answer for case in cases]
    if len(labels) < 2:
        raise ValueError("at least two training cases required")
    frequency = max(Counter(labels).values())
    if frequency * 2 > len(labels):
        raise ValueError("cannot fully derange training answers")
    indices = sorted(range(len(labels)), key=lambda index: (labels[index], index))
    changed = list(labels)
    for position, destination in enumerate(indices):
        changed[destination] = labels[indices[(position + frequency) % len(indices)]]
    if any(a == b for a, b in zip(labels, changed)):
        raise ValueError("answer derangement left unchanged supervision")
    return changed


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights_digest(weights: list[Path]) -> dict[str, str]:
    return {path.name: _sha_file(path) for path in weights}


def _metadata_digest(model_path: Path) -> dict[str, str]:
    """Bind tokenizer/config bytes alongside model weights for replay."""
    names = {"config.json", "generation_config.json", "special_tokens_map.json",
             "model.safetensors.index.json",
             "tokenizer_config.json", "tokenizer.json", "added_tokens.json",
             "vocab.json", "vocab.txt", "merges.txt", "sentencepiece.bpe.model",
             "tokenizer.model"}
    matches = sorted(path for path in model_path.iterdir()
                     if path.name in names or path.name.startswith("tokenizer"))
    if any(not p.is_file() or p.is_symlink() or p.stat().st_size > 268_435_456
           for p in matches):
        raise ValueError("local model metadata must be regular non-symlink files")
    return {p.name: _sha_file(p) for p in matches}


def _checkpoint_manifest(model_path: Path, weights: list[Path]) -> dict[str, dict[str, str]]:
    """Digest declared local model bytes before a load or training operation."""
    return {"weights": _weights_digest(weights), "metadata": _metadata_digest(model_path)}


def _assert_checkpoint_unchanged(model_path: Path, weights: list[Path], expected: dict[str, Any]) -> None:
    """Detect changes across load/train windows (not an atomic filesystem snapshot)."""
    current_weights = sorted(model_path.glob("*.safetensors"))
    if (current_weights != weights
            or any(not p.is_file() or p.is_symlink() for p in current_weights)
            or _checkpoint_manifest(model_path, weights) != expected):
        raise RuntimeError("base model checkpoint files changed during load or evaluation")


def _adapter_manifest(adapter_dir: Path) -> dict[str, str]:
    """Pin all local adapter file bytes, including PEFT metadata and scaling."""
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise RuntimeError("adapter checkpoint root must be a non-symlink directory")
    entries = sorted(adapter_dir.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise RuntimeError("adapter checkpoint contains a symlink or nonregular entry")
    files = [path for path in entries if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError("PEFT adapter safetensors checkpoint is missing")
    return {path.relative_to(adapter_dir).as_posix(): _sha_file(path) for path in files}


def _assert_adapter_unchanged(adapter_dir: Path, expected: dict[str, str]) -> None:
    """Detect persistent adapter swaps across a load/eval window, not atomic reads."""
    try:
        current = _adapter_manifest(adapter_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("adapter checkpoint files changed during load or evaluation") from exc
    if current != expected:
        raise RuntimeError("adapter checkpoint files changed during load or evaluation")


def _prompt_commitment(cases: list[Challenge]) -> str:
    payload = [{"id": case.id, "prompt": case.prompt} for case in cases]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _selected_parameter_digest(model: Any, torch: Any, *, include: Any) -> str:
    """Hash selected tensor *values*, names, shapes and dtypes, not just keys."""
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not include(name, parameter):
            continue
        count += 1
        raw = parameter.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        metadata = json.dumps([name, list(parameter.shape), str(parameter.dtype)], separators=(",", ":"))
        digest.update(metadata.encode("utf-8") + b"\x00" + raw)
    if count == 0:
        raise RuntimeError("PEFT model exposed no selected adapter parameters")
    return digest.hexdigest()


def _parameter_digest(model: Any, torch: Any) -> str:
    """Hash trainable tensors before/after optimization."""
    return _selected_parameter_digest(model, torch, include=lambda name, p: p.requires_grad)


def _frozen_parameter_digest(model: Any, torch: Any) -> str:
    """Hash all frozen neural parameters rather than only trainable LoRA tensors."""
    return _selected_parameter_digest(
        model, torch, include=lambda _name, parameter: not parameter.requires_grad,
    )


def _clip_trainable_gradients(torch: Any, trainable: Any) -> None:
    """Reject finite-loss/nonfinite-gradient cases before any optimizer step."""
    torch.nn.utils.clip_grad_norm_(
        trainable, 1.0, error_if_nonfinite=True,
    )


def _assert_frozen_state(
    model: Any, torch: Any, frozen_digest: str, buffer_digest: str, phase: str,
) -> None:
    if _frozen_parameter_digest(model, torch) != frozen_digest:
        raise RuntimeError(f"{phase}: frozen neural parameters changed")
    if _buffer_digest(model, torch) != buffer_digest:
        raise RuntimeError(f"{phase}: persistent model buffers changed")


def _buffer_digest(model: Any, torch: Any) -> str:
    """Value-hash persistent model buffers, including non-trainable state.

    Optimizer LR=0 does not guarantee training forwards preserve mutable
    running-statistics buffers. Reject changed state rather than treating
    a parameter-only digest as proof of identical starting conditions.
    """
    digest = hashlib.sha256()
    for name, buffer in sorted(model.named_buffers()):
        metadata = json.dumps([name, list(buffer.shape), str(buffer.dtype)],
                              separators=(",", ":")).encode("utf-8")
        raw = buffer.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        digest.update(metadata + b"\x00" + raw)
    return digest.hexdigest()


def _lora_value_digest(model: Any, torch: Any) -> str:
    """Hash LoRA tensor values regardless of frozen `requires_grad` on reload."""
    return _selected_parameter_digest(model, torch, include=lambda name, p: ".lora_" in name)


def _encode_pair(tokenizer: Any, prompt: str, answer: str, max_tokens: int, *,
                 answer_only: bool = False) -> tuple[list[int], list[int]]:
    # Pre-fill the constant ANSWER: scaffold, supervise only numerical output.
    task = f"Task:\n{prompt}\nAssistant:\n" + ("ANSWER:" if answer_only else "")
    completion = (f" {answer}" if answer_only else f"ANSWER: {answer}") + (tokenizer.eos_token or "")
    prefix = tokenizer(task, add_special_tokens=True)["input_ids"]
    suffix = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not suffix or not prefix or len(prefix) + len(suffix) > max_tokens:
        raise ValueError("example exceeds token budget or tokenizer produced empty tokens")
    return prefix + suffix, [-100] * len(prefix) + suffix


def _score_model(model: Any, tokenizer: Any, torch: Any, cases: list[Challenge],
                 *, device: str, max_tokens: int, max_new_tokens: int,
                 answer_only: bool = False) -> dict[str, Any]:
    model.eval()
    trials: list[dict[str, Any]] = []
    with torch.no_grad():
        for challenge in cases:
            prompt = f"Task:\n{challenge.prompt}\nAssistant:\n" + ("ANSWER:" if answer_only else "")
            encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
            ids = encoded["input_ids"]
            mask = encoded.get("attention_mask")
            if mask is None or mask.shape != ids.shape:
                raise ValueError("tokenizer must supply an aligned attention mask")
            if ids.shape[1] + max_new_tokens > max_tokens:
                raise ValueError("evaluation prompt exceeds context budget")
            ids = ids.to(device)
            output = model.generate(input_ids=ids, attention_mask=mask.to(device), do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            if answer_only:
                # Evaluation does not reward generating the supplied scaffold.
                response = f"ANSWER: {response}"
            result = score_response(challenge, response)
            trials.append({
                "id": challenge.id, "correct": result.correct,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "response_preview": response[:128], "score_reason": result.reason,
            })
    return _summarize_trials(trials)


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate strict-format adherence from answer correctness."""
    if not trials:
        raise ValueError("cannot summarize an empty evaluation")
    correct = sum(bool(t["correct"]) for t in trials)
    format_valid = sum(t["score_reason"] in ("correct", "incorrect") for t in trials)
    return {"correct": correct, "total": len(trials), "accuracy": correct / len(trials),
            "format_valid": format_valid, "format_rate": format_valid / len(trials),
            "trials": trials}


def _training_visit_indices(train: list[Challenge], *, seed: int, steps: int) -> list[int]:
    """Deterministic per-epoch, per-tier interleaving without silent tier omission.

    A short run can skip some examples but never an entire requested tier.
    Candidate, zero-LR sham and corrupted-label controls use identical visits.
    """
    if not train or type(seed) is not int or type(steps) is not int or steps < 1:
        raise ValueError("training visitation requires cases, seed and positive steps")
    tiers = tuple(dict.fromkeys(case.tier for case in train))
    if steps < len(tiers):
        raise ValueError("training steps insufficient for every tier")
    indices: list[int] = []
    epoch = 0
    while len(indices) < steps:
        pools: dict[int, list[int]] = {
            tier: [i for i, case in enumerate(train) if case.tier == tier]
            for tier in tiers
        }
        for tier in tiers:
            random.Random(f"{seed}:epoch:{epoch}:tier:{tier}").shuffle(pools[tier])
        for position in range(max(len(group) for group in pools.values())):
            for tier in tiers:
                if position < len(pools[tier]):
                    indices.append(pools[tier][position])
                    if len(indices) == steps:
                        return indices
        epoch += 1
    return indices


def _train_steps(model: Any, tokenizer: Any, torch: Any, train: list[Challenge],
                 labels: list[str], config: PilotConfig, *,
                 learning_rate: float | None = None) -> dict[str, Any]:
    if len(train) != len(labels) or not train:
        raise ValueError("training supervision must match every training case")
    indices = _training_visit_indices(train, seed=config.seed, steps=config.steps)
    prepared = [_encode_pair(tokenizer, case.prompt, answer, config.max_tokens,
                             answer_only=config.answer_only)
                for case, answer in zip(train, labels)]
    if learning_rate is not None and learning_rate != 0.0:
        raise ValueError("control override only admits exact zero learning rate")
    # The candidate and corrupted-label arms must start their update loops with
    # the same RNG state, even when a local model contains stochastic layers.
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    if config.optimizer == "mezo":
        # Paired ZO loss evaluations require deterministic, dropout-free forwards.
        # The frozen pretrained base remains frozen; only LoRA tensors are tuned.
        model.eval()
        from .mezo import mezo_step
        optimizer = None
    else:
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config.learning_rate if learning_rate is None else learning_rate)
    token_total = 0
    losses: list[float] = []
    positive_losses: list[float] = []
    negative_losses: list[float] = []
    direction_seeds: list[int] = []
    forward_evaluations = 0
    for step in range(config.steps):
        inputs, target = prepared[indices[step]]
        ids = torch.tensor([inputs], dtype=torch.long, device=config.device)
        targets = torch.tensor([target], dtype=torch.long, device=config.device)
        if config.optimizer == "mezo":
            direction_seed = config.seed + step
            outcome = mezo_step(
                model, torch,
                lambda: model(input_ids=ids, labels=targets).loss,
                epsilon=config.zo_epsilon,
                learning_rate=config.learning_rate if learning_rate is None else learning_rate,
                seed=direction_seed,
            )
            if type(outcome.get("forward_evaluations")) is not int or outcome["forward_evaluations"] != 2:
                raise RuntimeError("MeZO optimizer did not confirm exactly two paired loss forwards")
            forward_evaluations += outcome["forward_evaluations"]
            direction_seeds.append(direction_seed)
            value = float(outcome["loss"])
            positive_losses.append(float(outcome["loss_plus"]))
            negative_losses.append(float(outcome["loss_minus"]))
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=targets).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite SFT loss")
            loss.backward()
            _clip_trainable_gradients(
                torch, (p for p in model.parameters() if p.requires_grad)
            )
            optimizer.step()
            value = float(loss.detach().cpu().item())
        for parameter in model.parameters():
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                raise RuntimeError("non-finite trained LoRA parameter")
        losses.append(value)
        token_total += len(inputs)
    receipt = {"steps": config.steps, "example_visits": config.steps,
               "visited_case_ids": [train[index].id for index in indices],
               "input_tokens_seen": token_total, "losses": losses}
    if config.optimizer == "mezo":
        receipt.update({"forward_evaluations": forward_evaluations,
                        "direction_seeds": direction_seeds,
                        "positive_losses": positive_losses,
                        "negative_losses": negative_losses})
    return receipt


def _deps(device: str) -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError("optional torch, transformers and peft are required; nothing was installed") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training unavailable in this Python environment; no CPU fallback")
    return torch, AutoModelForCausalLM, AutoTokenizer, (LoraConfig, PeftModel, get_peft_model), device


def _device_headroom(torch: Any, device: str, weights: list[Path]) -> dict[str, int] | None:
    """Conservative CUDA preflight; not a memory-use guarantee for any model."""
    if device != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # File size is not in-memory size. Float32 expansion, transient copies,
    # activations and CUDA allocator overhead can exceed this guard estimate.
    estimated_minimum = 4 * sum(path.stat().st_size for path in weights) + 1_073_741_824
    if estimated_minimum >= free_bytes:
        raise RuntimeError("insufficient free CUDA memory for conservative fp32 LoRA pilot preflight")
    return {"free_before_bytes": int(free_bytes), "total_bytes": int(total_bytes),
            "estimated_minimum_bytes": int(estimated_minimum)}


def _load_base(model_path: Path, device: str, *, torch: Any,
               auto_model: Any, auto_tokenizer: Any) -> tuple[Any, Any]:
    tokenizer = auto_tokenizer.from_pretrained(str(model_path), local_files_only=True,
                                               trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("checkpoint tokenizer must have an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = auto_model.from_pretrained(str(model_path), local_files_only=True,
                                       trust_remote_code=False, use_safetensors=True,
                                       dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    return model, tokenizer


def _report_config(config: PilotConfig, model_path: Path, output_dir: Path) -> dict[str, Any]:
    """Preserve the original exact v3 AdamW receipt keys for historical audits."""
    payload = asdict(config)
    if config.optimizer == "adamw":
        payload.pop("optimizer")
        payload.pop("zo_epsilon")
    return {**payload, "model_path": str(model_path), "output_dir": str(output_dir)}


def run_pilot(config: PilotConfig) -> dict[str, Any]:
    """Run fixed-budget LoRA-SFT and controls, save a new adapter and report.

    The baseline and zero-update sham share the same base checkpoint. A
    corrupted-label arm starts from a restored identical LoRA initialization.
    A separate process must run `reload_evaluate` to qualify persistence.
    """
    model_path, output_dir, weights = validate_config(config)
    initial_manifest = _checkpoint_manifest(model_path, weights)
    train, heldout = training_cases(config)
    corrupted_answers = derange_answers(train)
    torch, auto_model, auto_tokenizer, peft_parts, device = _deps(config.device)
    cuda_headroom = _device_headroom(torch, device, weights)
    lora_config, _, get_peft_model = peft_parts
    random.seed(config.seed)
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # Deterministic algorithms fail closed rather than silently switching to
    # nondeterministic kernels. Full cross-hardware determinism is not claimed.
    torch.use_deterministic_algorithms(True)
    base, tokenizer = _load_base(model_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(model_path, weights, initial_manifest)
    # Validate all token budgets before creating an output directory.
    for case in train:
        _encode_pair(tokenizer, case.prompt, case.answer, config.max_tokens,
                     answer_only=config.answer_only)
    for case, answer in zip(train, corrupted_answers):
        _encode_pair(tokenizer, case.prompt, answer, config.max_tokens,
                     answer_only=config.answer_only)
    baseline = _score_model(base, tokenizer, torch, heldout, device=device,
                            max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    model = get_peft_model(base, lora_config(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                             lora_dropout=0, bias="none", task_type="CAUSAL_LM",
                                             target_modules="all-linear"))
    initial_digest = _parameter_digest(model, torch)
    initial_frozen_digest = _frozen_parameter_digest(model, torch)
    initial_buffer_digest = _buffer_digest(model, torch)
    initial_state = {name: parameter.detach().clone()
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
    # Compute/exposure-matched *zero-effective-update* control: same supervised
    # examples and optimiser steps, but LR=0. Verify all trainable tensors are
    # unchanged, then start the candidate from the same initialization.
    sham_training = _train_steps(
        model, tokenizer, torch, train, [c.answer for c in train], config,
        learning_rate=0.0,
    )
    if _parameter_digest(model, torch) != initial_digest:
        raise RuntimeError("zero-LR placebo changed trainable adapter parameters")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "zero-LR placebo",
    )
    sham = _score_model(model, tokenizer, torch, heldout, device=device,
                        max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "zero-LR placebo evaluation",
    )
    correct_training = _train_steps(model, tokenizer, torch, train, [c.answer for c in train], config)
    trained_digest = _parameter_digest(model, torch)
    if trained_digest == initial_digest:
        raise RuntimeError("no trainable LoRA parameter changed; refusing learning claim")
    trained_saved_digest = _lora_value_digest(model, torch)
    if trained_saved_digest != trained_digest:
        raise RuntimeError("trainable parameter selection differs from LoRA tensor selection")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "correct-label training",
    )
    candidate = _score_model(model, tokenizer, torch, heldout, device=device,
                             max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "correct-label evaluation",
    )
    # A clean, newly created output directory is used; no existing run changes.
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    candidate_dir = output_dir / "candidate_adapter"
    if candidate_dir.is_symlink():
        raise RuntimeError("candidate adapter root must not be a symlink")
    model.save_pretrained(str(candidate_dir), safe_serialization=True)
    if candidate_dir.is_symlink():
        raise RuntimeError("candidate adapter root was replaced by a symlink")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in initial_state:
                parameter.copy_(initial_state[name])
    if _parameter_digest(model, torch) != initial_digest:
        raise RuntimeError("failed to restore LoRA initialization for corrupted-label control")
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "restoration before corrupted-label training",
    )
    corrupted_training = _train_steps(model, tokenizer, torch, train, corrupted_answers, config)
    corrupted_digest = _parameter_digest(model, torch)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "corrupted-label training",
    )
    corrupted = _score_model(model, tokenizer, torch, heldout, device=device,
                             max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_frozen_state(
        model, torch, initial_frozen_digest, initial_buffer_digest,
        "corrupted-label evaluation",
    )
    _assert_checkpoint_unchanged(model_path, weights, initial_manifest)
    candidate_adapter_manifest = _adapter_manifest(candidate_dir)
    result: dict[str, Any] = {
        "schema": MEZO_SCHEMA if config.optimizer == "mezo" else SCHEMA,
        "adaptation_kind": "local HF causal-LM PEFT LoRA supervised parameter update",
        "source": _source_state(),
        "training_process_id": os.getpid(),
        "config": _report_config(config, model_path, output_dir),
        "base_weights_sha256": initial_manifest["weights"],
        "model_metadata_sha256": initial_manifest["metadata"],
        "runtime_versions": {"python": sys.version.split()[0], "torch": package_version("torch"),
                             "transformers": package_version("transformers"),
                             "peft": package_version("peft")},
        "candidate_adapter_sha256": candidate_adapter_manifest,
        "training_case_ids": [c.id for c in train],
        "heldout_case_ids": [c.id for c in heldout],
        "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        "initial_lora_parameter_sha256": initial_digest,
        "initial_persistent_buffers_sha256": initial_buffer_digest,
        "initial_frozen_parameter_sha256": initial_frozen_digest,
        "candidate_lora_parameter_sha256": trained_digest,
        "candidate_lora_tensor_value_sha256": trained_saved_digest,
        "corrupted_lora_parameter_sha256": corrupted_digest,
        "control_data": {"corrupted_label_collisions": sum(a.answer == b for a, b in zip(train, corrupted_answers)),
                         "sham_updates": 0, "sham_optimizer_steps": config.steps,
                         "sham_train_case_visits": config.steps,
                         "sham_evaluation_before_updates": False,
                         "sham_is_compute_matched": True,
                         "answer_scaffold_supplied": config.answer_only},
        "cuda_preflight": cuda_headroom,
        "training": {"sham": sham_training, "candidate": correct_training,
                     "corrupted": corrupted_training},
        "evaluations": {"baseline": baseline, "sham": sham, "candidate": candidate, "corrupted": corrupted},
        "fresh_process_reload": "NOT RUN: execute `python -m labyrinth.hf_pilot eval` separately",
        "limitations": ["Finite public templates are not evidence of general reasoning transfer.",
                        "One-seed and small-N pilot; do not generalize accuracy deltas.",
                        "The zero-LR sham matches visits and optimizer steps but not guaranteed hardware timing.",
                        "Only the adapter was persisted; base weights remain unchanged.",
                        "Same-process scores do not qualify independent checkpoint reload."],
    }
    if config.optimizer == "mezo":
        result["control_data"].update({
            "mezo_loss_forward_evaluations_per_arm": 2 * config.steps,
            "mezo_snapshot_restoration": True,
        })
        result["limitations"].extend([
            "MeZO tunes LoRA parameters only; this does not measure full-model MeZO memory savings.",
            "Exact restoration uses a snapshot of trainable adapter tensors; peak memory includes this copy.",
            "Two loss forwards per ZO step are not compute-matched to one AdamW forward/backward step.",
        ])
    _save_new(output_dir / "pilot_report.json", result)
    return result


def _save_new(path: Path, result: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def reload_evaluate(run_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Separate-process command: load adapter from disk and score fixed holdouts.

    This function cannot by itself attest process separation. The module CLI
    writes its own PID and a different-run CLI invocation is required.
    """
    root = Path(run_dir).expanduser().resolve(strict=True)
    source = root / "pilot_report.json"
    report = load_json(source)
    if report.get("schema") not in (SCHEMA, LEGACY_SCHEMA, MEZO_SCHEMA):
        raise ValueError("unknown training pilot report schema")
    source_state = report.get("source")
    if not isinstance(source_state, dict) or source_state.get("source_tree_sha256") != _source_state()["source_tree_sha256"]:
        raise ValueError("source revision changed since training")
    source_config = report.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("invalid training pilot config")
    config = PilotConfig(**{**source_config, "tiers": tuple(source_config["tiers"])})
    model_path, output_dir, weights = validate_config(config, output_must_be_new=False)
    expected_manifest = {"weights": report.get("base_weights_sha256"),
                         "metadata": report.get("model_metadata_sha256")}
    if output_dir.resolve(strict=True) != root:
        raise ValueError("report output path does not match run directory")
    if _weights_digest(weights) != report.get("base_weights_sha256"):
        raise ValueError("base model files changed after training")
    if _metadata_digest(model_path) != report.get("model_metadata_sha256"):
        raise ValueError("local model tokenizer/config changed after training")
    adapter_dir = root / "candidate_adapter"
    if adapter_dir.is_symlink():
        raise ValueError("candidate adapter root must not be a symlink")
    actual_adapters = _adapter_manifest(adapter_dir)
    if actual_adapters != report.get("candidate_adapter_sha256"):
        raise ValueError("candidate adapter changed after training")
    _, heldout = training_cases(config)
    if [c.id for c in heldout] != report.get("heldout_case_ids"):
        raise ValueError("heldout generation differs from training report")
    if _prompt_commitment(heldout) != report.get("heldout_public_prompt_sha256"):
        raise ValueError("heldout prompt text changed since training")
    torch, auto_model, auto_tokenizer, peft_parts, device = _deps(config.device)
    _, peft_model, _ = peft_parts
    base, tokenizer = _load_base(model_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(model_path, weights, expected_manifest)
    model = peft_model.from_pretrained(base, str(adapter_dir), is_trainable=False,
                                      local_files_only=True)
    _assert_adapter_unchanged(adapter_dir, actual_adapters)
    observed_adapter_digest = _lora_value_digest(model, torch)
    if observed_adapter_digest != report.get("candidate_lora_tensor_value_sha256"):
        raise ValueError("reloaded LoRA tensor values differ from trained candidate")
    observed = _score_model(model, tokenizer, torch, heldout, device=device,
                            max_tokens=config.max_tokens, max_new_tokens=config.max_new_tokens,
                            answer_only=config.answer_only)
    _assert_adapter_unchanged(adapter_dir, actual_adapters)
    _assert_checkpoint_unchanged(model_path, weights, expected_manifest)
    value = {"schema": EVAL_SCHEMA, "training_report_sha256": _sha_file(source),
             "adapter_sha256": actual_adapters, "process_id": os.getpid(),
             "reloaded_lora_tensor_value_sha256": observed_adapter_digest,
             "evaluation": observed, "claimed_fresh_process": None,
             "process_id_differs": os.getpid() != report.get("training_process_id"),
             "matches_original_candidate_trials": observed["trials"] == report["evaluations"]["candidate"]["trials"],
             "limitations": ["Different process must be independently established; PID alone is not proof.",
                             "Determinism depends on compatible model kernels/hardware."]}
    _save_new(Path(report_path).expanduser(), value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in local HF LoRA pilot (no downloads)")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train tiny local safetensors checkpoint")
    train.add_argument("--model-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--train-count", type=int, default=8)
    train.add_argument("--eval-count", type=int, default=8)
    train.add_argument("--tiers", default="1")
    train.add_argument("--steps", type=int, default=2)
    train.add_argument("--max-tokens", type=int, default=256)
    train.add_argument("--max-new-tokens", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.0002)
    train.add_argument("--lora-rank", type=int, default=4)
    train.add_argument("--lora-alpha", type=int, default=8)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--optimizer", choices=("adamw", "mezo"), default="adamw",
                       help="MeZO uses two forward-only loss queries per step on trainable LoRA tensors")
    train.add_argument("--zo-epsilon", type=float, default=0.001,
                       help="Two-sided MeZO perturbation scale (ignored for AdamW)")
    train.add_argument("--answer-only", action="store_true",
                       help="Pre-fill ANSWER: and supervise only numerical completion + EOS")
    evaluate = commands.add_parser("eval", help="Reload saved adapter in a new Python process")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config_dict = {key: value for key, value in vars(args).items() if key != "command"}
            config_dict["tiers"] = tuple(int(t) for t in args.tiers.split(",")) if args.tiers else ()
            result = run_pilot(PilotConfig(**config_dict))
            print(json.dumps({"schema": result["schema"], "output_dir": result["config"]["output_dir"],
                              "changed": result["initial_lora_parameter_sha256"] != result["candidate_lora_parameter_sha256"]}))
        else:
            result = reload_evaluate(args.run_dir, args.report)
            # The diagnostic report was persisted by reload_evaluate even on
            # mismatch; the CLI must not represent mismatch as successful
            # qualification by returning zero status.
            if result["matches_original_candidate_trials"] is not True:
                raise RuntimeError("reloaded candidate did not reproduce training trials")
            if result["process_id_differs"] is not True:
                raise RuntimeError("reload was not observed in a different process")
            print(json.dumps({"schema": result["schema"],
                              "matches_original_candidate_trials": True,
                              "process_id_differs": True,
                              "independent_executor_attestation": False}))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
