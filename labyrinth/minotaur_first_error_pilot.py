"""Opt-in, offline pretrained LoRA first-error pilot for register traces.

Public synthetic ProcessBench-*style* data, never official ProcessBench scores.
No imports of heavyweight neural dependencies until explicit ``train``/``reload``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

from .program_maze import generate_program_challenges
from .minotaur_critic_pilot import (
    CriticExample, _runtime_manifest, encode_supervised_verdict,
    verify_reload_contract,
)
from .hf_pilot import (
    PilotConfig, validate_config, _checkpoint_manifest, _assert_checkpoint_unchanged,
    _adapter_manifest, _assert_adapter_unchanged, _deps, _device_headroom,
    _load_base, _parameter_digest, _lora_value_digest, _save_new, _sha_file,
)
from .runner import _source_state, load_json


SCHEMA = "labyrinth.minotaur-first-error-pilot.v1"
RELOAD_SCHEMA = "labyrinth.minotaur-first-error-reload.v1"
_INITIAL = re.compile(r"Initial registers: A=(\d+), B=(\d+)")
_INSTRUCTION = re.compile(r"^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", re.M)


@dataclass(frozen=True)
class TraceCase:
    case_id: str
    family_id: str
    structure: str
    profile: str
    variant: str
    problem: str
    steps: tuple[str, ...]
    first_error: int | None  # one-based; None = all correct
    injected_step: int | None


@dataclass(frozen=True)
class FirstErrorConfig:
    model_path: str
    output_dir: str
    seed: int = 777
    train_seed: int = 1357
    dev_seed: int = 2468
    transfer_seed: int = 3579
    length_seed: int = 4680
    train_count: int = 24
    eval_count: int = 8
    steps: int = 16
    max_tokens: int = 512
    learning_rate: float = 0.0003
    lora_rank: int = 4
    lora_alpha: int = 8
    device: str = "cpu"

    def validate_fields(self) -> None:
        for name in ("seed", "train_seed", "dev_seed", "transfer_seed", "length_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**31:
                raise ValueError(f"invalid {name}")
        if len({self.train_seed, self.dev_seed, self.transfer_seed, self.length_seed}) != 4:
            raise ValueError("split seeds must differ")
        for name, lower, upper in (("train_count", 2, 32), ("eval_count", 2, 32),
                                   ("steps", 1, 32), ("max_tokens", 64, 1024),
                                   ("lora_rank", 1, 16), ("lora_alpha", 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"invalid {name}")
        if self.steps > self.train_count:
            raise ValueError("steps exceed unique training programs")
        if type(self.learning_rate) is not float or not 1e-7 <= self.learning_rate <= .001:
            raise ValueError("invalid learning rate")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("unsupported device")

    def validate(self, *, output_must_be_new: bool = True):
        self.validate_fields()
        return validate_config(PilotConfig(
            self.model_path, self.output_dir, seed=self.seed, steps=self.steps,
            max_tokens=self.max_tokens, device=self.device,
        ), output_must_be_new=output_must_be_new)


def _execute(operation: str, target: str, operand: str, state: dict[str, int]) -> None:
    value = state[operand] if operand in state else int(operand)
    before = state[target]
    state[target] = ({"ADD": lambda: before + value,
                      "SUB": lambda: before - value,
                      "MUL": lambda: before * value}[operation]()) % 97


def _oracle(problem: str, steps: Sequence[str]) -> int | None:
    """Independently replay supplied trace; first mismatching state is the label."""
    start = _INITIAL.search(problem)
    instructions = _INSTRUCTION.findall(problem)
    if start is None or not instructions or len(steps) != len(instructions):
        raise ValueError("malformed register trace")
    state = {"A": int(start[1]), "B": int(start[2])}
    first: int | None = None
    for index, (number, op, target, operand) in enumerate(instructions, 1):
        if int(number) != index:
            raise ValueError("nonsequential program")
        _execute(op, target, operand, state)
        expected = f"{index}. A={state['A']}, B={state['B']}"
        if first is None and steps[index - 1] != expected:
            first = index
    return first


def make_cases(seed: int, split: str, count: int, profile: str) -> list[TraceCase]:
    challenges = generate_program_challenges(seed, count, split, profile=profile)
    result: list[TraceCase] = []
    for challenge in challenges:
        # The displayed program is stripped of the final-answer request/answer.
        problem = challenge.prompt.split("\nWhat is the final value of A?")[0]
        start = _INITIAL.search(problem)
        if start is None:
            raise ValueError("missing initial registers")
        instructions = _INSTRUCTION.findall(problem)
        states = {"A": int(start[1]), "B": int(start[2])}
        correct: list[str] = []
        for index, (number, op, target, operand) in enumerate(instructions, 1):
            if int(number) != index:
                raise ValueError("bad instruction sequence")
            _execute(op, target, operand, states)
            correct.append(f"{index}. A={states['A']}, B={states['B']}")
        if states["A"] != int(challenge.answer):
            raise RuntimeError("trace oracle disagrees with source answer")
        digest = hashlib.sha256(challenge.id.encode()).digest()
        bad_at = int.from_bytes(digest[:4], "big") % len(correct) + 1
        corrupted = list(correct)
        # Change exactly the selected state line. Subsequent lines remain
        # oracle-correct, so earliest error is unambiguous and checkable.
        match = re.fullmatch(r"(\d+)\. A=(\d+), B=(\d+)", corrupted[bad_at - 1])
        assert match is not None
        corrupted[bad_at - 1] = (f"{bad_at}. A={(int(match[2]) + 1) % 97}, "
                                  f"B={match[3]}")
        structure = dict(challenge.metadata)["structure_fingerprint"]
        for variant, trace, expected in (("clean", correct, None),
                                         ("error", corrupted, bad_at)):
            measured = _oracle(problem, trace)
            if measured != expected:
                raise RuntimeError("first-error oracle verification failed")
            result.append(TraceCase(f"{challenge.id}:{variant}", challenge.id,
                                    structure, profile, variant, problem,
                                    tuple(trace), measured,
                                    bad_at if variant == "error" else None))
    return result


def validate_splits(*groups: Sequence[TraceCase]) -> None:
    seen: set[str] = set()
    problems: set[str] = set()
    for group in groups:
        if not group or len(group) % 2:
            raise ValueError("paired, nonempty split required")
        families: dict[str, list[TraceCase]] = {}
        for case in group:
            families.setdefault(case.family_id, []).append(case)
        if any(len(pair) != 2 or {c.variant for c in pair} != {"clean", "error"}
               or len({c.problem for c in pair}) != 1 for pair in families.values()):
            raise ValueError("malformed clean/error pair")
        ids = {c.case_id for c in group}
        prompts = {c.problem for c in group}
        if len(ids) != len(group) or ids & seen or prompts & problems:
            raise ValueError("duplicate or overlapping splits")
        seen |= ids
        problems |= prompts


def _prefix_prompt(problem: str, steps: Sequence[str]) -> str:
    return ("Check the proposed execution of the register program. Each line gives "
            "the registers after that numbered instruction. Are ALL displayed "
            "lines correct? Respond Yes or No.\nProgram:\n" + problem +
            "\nProposed execution:\n" + "\n".join(steps) + "\nVerdict:")


def prefix_examples(cases: Sequence[TraceCase]) -> list[CriticExample]:
    examples: list[CriticExample] = []
    for case in cases:
        for step in range(1, len(case.steps) + 1):
            examples.append(CriticExample(
                example_id=f"{case.case_id}:{step}", case_id=case.case_id,
                case_prompt_sha256=hashlib.sha256(case.problem.encode()).hexdigest(),
                kind=case.profile, prompt=_prefix_prompt(case.problem, case.steps[:step]),
                trainee_answer=step,
                is_correct=case.first_error is None or step < case.first_error,
            ))
    return examples


def training_pairs(cases: Sequence[TraceCase], steps: int, seed: int):
    families: dict[str, dict[str, TraceCase]] = {}
    for case in cases:
        families.setdefault(case.family_id, {})[case.variant] = case
    if type(steps) is not int or steps < 1 or steps > len(families) or any(
        set(pair) != {"clean", "error"} for pair in families.values()
    ):
        raise ValueError("distinct clean/error training families required")
    rng = random.Random(seed)
    keys = sorted(families)
    rng.shuffle(keys)
    result = []
    for key in keys[:steps]:
        clean, bad = families[key]["clean"], families[key]["error"]
        assert bad.first_error is not None
        index = bad.first_error
        pair = []
        for case in (clean, bad):
            pair.append(CriticExample(
                f"{case.case_id}:{index}", key,
                hashlib.sha256(case.problem.encode()).hexdigest(), case.profile,
                _prefix_prompt(case.problem, case.steps[:index]), index,
                case.variant == "clean",
            ))
        rng.shuffle(pair)
        result.append(tuple(pair))
    return result


def encode_pair(tokenizer: Any, pair, max_tokens: int, corrupt: bool):
    if len(pair) != 2 or {x.is_correct for x in pair} != {True, False}:
        raise ValueError("balanced pair required")
    return tuple(encode_supervised_verdict(tokenizer, CriticExample(
        **{**asdict(ex), "is_correct": not ex.is_correct if corrupt else ex.is_correct}),
        max_tokens=max_tokens) for ex in pair)


def score_first_errors(cases: Sequence[TraceCase], predictions: Sequence[int | None]):
    if len(cases) != len(predictions) or not cases or any(
        p is not None and (type(p) is not int or not 1 <= p <= len(c.steps))
        for c, p in zip(cases, predictions)
    ):
        raise ValueError("complete valid first-error predictions required")
    errors = sum(c.first_error is not None for c in cases)
    correct = len(cases) - errors
    hit_errors = sum(c.first_error == p for c, p in zip(cases, predictions)
                     if c.first_error is not None)
    hit_clean = sum(p is None for c, p in zip(cases, predictions)
                    if c.first_error is None)
    approvals = sum(p is None for c, p in zip(cases, predictions)
                    if c.first_error is not None)
    rejections = sum(p is not None for c, p in zip(cases, predictions)
                     if c.first_error is None)
    ea, ca = hit_errors / errors if errors else 0., hit_clean / correct if correct else 0.
    return {"total": len(cases), "error_cases": errors, "correct_cases": correct,
            "accuracy": (hit_errors + hit_clean) / len(cases),
            "error_accuracy": ea, "correct_accuracy": ca,
            "harmonic_f1": 2 * ea * ca / (ea + ca) if ea + ca else 0.,
            "false_approval_rate": approvals / errors if errors else None,
            "false_rejection_rate": rejections / correct if correct else None,
            "predictions_sha256": hashlib.sha256(json.dumps(predictions).encode()).hexdigest()}


def _score(model, tokenizer, torch, cases, *, device, max_tokens):
    yes, no = (tokenizer.encode(s, add_special_tokens=False) for s in (" Yes", " No"))
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError("ambiguous verdict tokens")
    model.eval()
    predictions = []
    with torch.no_grad():
        for case in cases:
            first = None
            for ex in prefix_examples([case]):
                ids = tokenizer.encode(ex.prompt, add_special_tokens=True)
                if not ids or len(ids) + 1 > max_tokens:
                    raise ValueError("trace exceeds token budget")
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits[0, -1]
                if not bool(torch.isfinite(logits[yes[0]])) or not bool(torch.isfinite(logits[no[0]])):
                    raise RuntimeError("nonfinite logits")
                if bool(logits[no[0]] >= logits[yes[0]]):
                    first = ex.trainee_answer
                    break
            predictions.append(first)
    return score_first_errors(cases, predictions)


def _updates(model, tokenizer, torch, pairs, config, mode):
    if mode not in ("candidate", "zero_update", "corrupted"):
        raise ValueError("unknown control")
    prepared = [encode_pair(tokenizer, pair, config.max_tokens,
                            mode == "corrupted") for pair in pairs]
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate, weight_decay=0.)
    losses = []
    for pair in prepared:
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for tokens, labels in pair:
            ids = torch.tensor([tokens], dtype=torch.long, device=config.device)
            targets = torch.tensor([labels], dtype=torch.long, device=config.device)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=targets).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            (loss / 2).backward()
            total += float(loss.detach().cpu().item()) / 2
        if mode == "zero_update":
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        else:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()
                                            if p.requires_grad), 1.)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()
               if p.requires_grad):
            raise RuntimeError("nonfinite adapter parameters")
        losses.append(total)
    return {"optimizer_steps": len(losses), "forward_backward_steps": 2 * len(losses),
            "examples_per_step": 2, "losses": losses,
            "visited_example_ids": [ex.example_id for pair in pairs for ex in pair],
            "labels_swapped": mode == "corrupted", "zero_gradients": mode == "zero_update"}


def _splits(config):
    groups = {
        "train": make_cases(config.train_seed, "train", config.train_count, "train"),
        "dev": make_cases(config.dev_seed, "firsterror_dev", config.eval_count, "train"),
        "transfer": make_cases(config.transfer_seed, "firsterror_transfer",
                               config.eval_count, "matched_length_ood"),
        "length_transfer": make_cases(config.length_seed, "firsterror_length",
                                      config.eval_count, "structural_holdout"),
    }
    validate_splits(*groups.values())
    train_shapes = {x.structure for x in groups["train"]}
    if any(train_shapes & {x.structure for x in groups[name]}
           for name in ("transfer", "length_transfer")):
        raise RuntimeError("transfer structure contamination")
    return groups


def run_first_error(config: FirstErrorConfig):
    base_path, output, weights = config.validate()
    manifest = _checkpoint_manifest(base_path, weights)
    groups = _splits(config)
    pairs = training_pairs(groups["train"], config.steps, config.seed)
    torch, auto_model, auto_tokenizer, peft, device = _deps(config.device)
    _device_headroom(torch, device, weights)
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    base, tokenizer = _load_base(base_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(base_path, weights, manifest)
    for group in groups.values():
        for ex in prefix_examples(group):
            encode_supervised_verdict(tokenizer, ex, max_tokens=config.max_tokens)
    frozen = {name: _score(base, tokenizer, torch, cases, device=device,
                           max_tokens=config.max_tokens)
              for name, cases in groups.items() if name != "train"}
    lora_cls, _, get_peft_model = peft
    model = get_peft_model(base, lora_cls(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    initial = _parameter_digest(model, torch)
    originals = {name: p.detach().clone() for name, p in model.named_parameters()
                 if p.requires_grad}

    def reset():
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in originals:
                    p.copy_(originals[name])
        if _parameter_digest(model, torch) != initial:
            raise RuntimeError("control initialization changed")

    output.mkdir(mode=0o700, exist_ok=False)
    evaluations = {"frozen": frozen}
    training = {}
    digests = {"initial": initial}
    for mode in ("candidate", "zero_update", "corrupted"):
        reset()
        training[mode] = _updates(model, tokenizer, torch, pairs, config, mode)
        digests[mode] = _parameter_digest(model, torch)
        if mode == "candidate" and digests[mode] == initial:
            raise RuntimeError("candidate made no parameter update")
        if mode == "zero_update" and digests[mode] != initial:
            raise RuntimeError("zero-update sham changed adapter")
        if mode == "candidate":
            value_digest = _lora_value_digest(model, torch)
            if value_digest != digests[mode]:
                raise RuntimeError("adapter digest mismatch")
            model.save_pretrained(str(output / "candidate_adapter"), safe_serialization=True)
        evaluations[mode] = {name: _score(model, tokenizer, torch, cases,
                                          device=device, max_tokens=config.max_tokens)
                             for name, cases in groups.items() if name != "train"}
    _assert_checkpoint_unchanged(base_path, weights, manifest)
    report = {
        "schema": SCHEMA, "qualification": "NOT_ESTABLISHED",
        "role": "advisory_synthetic_first_error_step_critic",
        "source": _source_state(), "process_id": os.getpid(),
        "runtime": _runtime_manifest(torch, device), "config": asdict(config),
        "base": manifest, "adapter_sha256": _adapter_manifest(output / "candidate_adapter"),
        "candidate_lora_value_sha256": value_digest,
        "parameter_digests": digests, "training": training,
        "split_cases": {name: [c.case_id for c in cases] for name, cases in groups.items()},
        "evaluations": evaluations,
        "limitations": ["Public deterministic synthetic templates, not official ProcessBench.",
                        "Labels and generator are known; not independent blinded evidence.",
                        "Small local CPU run is exploratory, not production authorization."],
    }
    _save_new(output / "first_error_report.json", report)
    return report


def reload_first_error(run_dir: str, report_path: str):
    root = Path(run_dir).resolve(strict=True)
    prior = load_json(root / "first_error_report.json", strict=True)
    if prior.get("schema") != SCHEMA:
        raise ValueError("invalid first-error report schema")
    if prior.get("source", {}).get("source_tree_sha256") != _source_state()["source_tree_sha256"]:
        raise ValueError("source tree changed")
    config = FirstErrorConfig(**prior["config"])
    base_path, output, weights = config.validate(output_must_be_new=False)
    if output != root or _checkpoint_manifest(base_path, weights) != prior["base"]:
        raise ValueError("base/output identity changed")
    adapter = root / "candidate_adapter"
    if _adapter_manifest(adapter) != prior["adapter_sha256"]:
        raise ValueError("adapter bytes changed")
    torch, auto_model, auto_tokenizer, peft, device = _deps(config.device)
    torch.use_deterministic_algorithms(True)
    if _runtime_manifest(torch, device) != prior["runtime"]:
        raise RuntimeError("runtime identity changed")
    base, tokenizer = _load_base(base_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    model = peft[1].from_pretrained(base, str(adapter), is_trainable=False,
                                   local_files_only=True)
    digest = _lora_value_digest(model, torch)
    if digest != prior["candidate_lora_value_sha256"]:
        raise ValueError("reloaded adapter tensors differ")
    measured = {name: _score(model, tokenizer, torch, cases,
                             device=device, max_tokens=config.max_tokens)
                for name, cases in _splits(config).items() if name != "train"}
    _assert_checkpoint_unchanged(base_path, weights, prior["base"])
    _assert_adapter_unchanged(adapter, prior["adapter_sha256"])
    verify_reload_contract(training_pid=prior["process_id"], reload_pid=os.getpid(),
                           measured=measured, expected=prior["evaluations"]["candidate"])
    result = {"schema": RELOAD_SCHEMA, "qualification": "NOT_ESTABLISHED",
              "training_report_sha256": _sha_file(root / "first_error_report.json"),
              "process_id": os.getpid(), "different_process_id": True,
              "reloaded_lora_value_sha256": digest,
              "same_predictions_and_metrics": True, "evaluations": measured}
    _save_new(Path(report_path), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline opt-in Minotaur first-error LoRA pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--model-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=16)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    reload_cmd = commands.add_parser("reload")
    reload_cmd.add_argument("--run-dir", required=True)
    reload_cmd.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        result = (run_first_error(FirstErrorConfig(args.model_path, args.output_dir,
                                                   steps=args.steps, device=args.device))
                  if args.command == "train" else
                  reload_first_error(args.run_dir, args.report))
        print(json.dumps({k: result[k] for k in ("schema", "qualification", "evaluations")}))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
