"""Opt-in offline MiniLM/LoRA interrupted-vs-uninterrupted CPU development probe.

The orchestrator launches *three fresh Python processes*: uninterrupted
reference, interrupted checkpoint producer, and resumed continuation.
The same public training cases, initialized parameters, AdamW steps and
random-state stream must produce identical final trainable tensors.

This is NOT a signed development-event replay, final examination, graduation
authority, or proof of general reasoning transfer. Only an independently
custodied digest and host-verified bytes could authenticate a real run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from scripts.qualify_minilm_cpu import HELDOUT, MODEL_ID, TRAIN, TRANSFER, validate_fixtures


SCHEMA = "labyrinth.minilm-interrupted-resume-development.v1"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
OBJECTIVE_ID = "sentiment-polarity-development-v1"
FENCE = 1


def plan_output(parent: str | Path, run_name: str) -> Path:
    if type(run_name) is not str or not RUN_NAME.fullmatch(run_name):
        raise ValueError("invalid bounded run name")
    parent = Path(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")
    root = parent.resolve(strict=True) / run_name
    if root.exists() or root.is_symlink():
        raise FileExistsError("the output run directory already exists")
    return root


def training_visits(steps: int) -> tuple[tuple[str, str, int], ...]:
    """Public deterministic case exposure, with different cases after regeneration."""
    if type(steps) is not int or not 2 <= steps <= 32 or steps % 2:
        raise ValueError("steps must be an even count from 2 to 32")
    half = steps // 2
    result = []
    for step in range(steps):
        cycle = 0 if step < half else 1
        rows = TRAIN[cycle]
        batch = [rows[(step * 4 + offset) % len(rows)] for offset in range(4)]
        for offset, (prompt, label) in enumerate(batch):
            case_id = f"cycle{cycle}-step{step}-item{offset}"
            result.append((case_id, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), label))
    return tuple(result)


def checkpoint_exposures(steps: int, split_at: int) -> tuple[tuple[str, str], ...]:
    if type(split_at) is not int or not 1 <= split_at < steps:
        raise ValueError("invalid interruption cursor")
    return tuple((case_id, digest)
                 for case_id, digest, _ in training_visits(steps)[:split_at * 4])


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("missing or oversized run receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("invalid run receipt")
    return value


def _sha_bytes(parts: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parts:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _neural_stage(root: Path, stage: str, seed: int, steps: int, split_at: int) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    from labyrinth.neural_resume_checkpoint import (
        FullCheckpointContext, load_full_checkpoint, save_full_checkpoint,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    adapter = get_peft_model(base, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["query", "value"],
    ))

    class Learner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = adapter
            self.head = torch.nn.Linear(base.config.hidden_size, 2)

        def forward(self, prompts: list[str]):
            tokens = tokenizer(prompts, padding=True, truncation=True,
                               max_length=64, return_tensors="pt")
            result = self.adapter(**tokens)
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (result.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            return self.head(pooled)

    learner = Learner()
    named = sorted((name, p) for name, p in learner.named_parameters()
                   if p.requires_grad)
    lora_named = [(name, p) for name, p in named if "lora_" in name]
    frozen = sorted((name, p) for name, p in learner.named_parameters()
                    if not p.requires_grad)
    if not named or not frozen or not lora_named:
        raise RuntimeError("pretrained and trainable parameter inventories required")
    base_digest = _sha_bytes(frozen)
    initial_digest = _sha_bytes(named)
    initial_lora_digest = _sha_bytes(lora_named)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=0.002)

    visits = training_visits(steps)
    exposures = checkpoint_exposures(steps, split_at)
    context = FullCheckpointContext(
        run_id=root.name, objective_id=OBJECTIVE_ID,
        base_checkpoint_sha256=base_digest,
        prior_checkpoint_sha256=initial_digest,
        fence=FENCE, cursor=len(exposures), exposures=exposures,
    )
    if stage == "resume":
        receipt = _read_json(root / "checkpoint_receipt.json")
        if (set(receipt) != {"checkpoint_sha256", "base_digest", "initial_digest",
                            "initial_lora_digest", "checkpoint_lora_digest",
                            "split_at", "seed", "steps", "process_pid"}
                or receipt["base_digest"] != base_digest
                or receipt["initial_digest"] != initial_digest
                or receipt["initial_lora_digest"] != initial_lora_digest
                or receipt["seed"] != seed
                or receipt["steps"] != steps
                or receipt["split_at"] != split_at):
            raise RuntimeError("resumed process context differs from checkpoint producer")
        load_full_checkpoint(
            root / "snapshot", model=learner, optimizer=optimizer,
            expected_sha256=receipt["checkpoint_sha256"],
            expected_context=context,
        )
        if _sha_bytes(lora_named) != receipt["checkpoint_lora_digest"]:
            raise RuntimeError("resumed LoRA tensors differ from checkpoint")
    start = split_at if stage == "resume" else 0
    stop = split_at if stage == "checkpoint" else steps
    learner.train()
    for step in range(start, stop):
        cycle = 0 if step < steps // 2 else 1
        rows = TRAIN[cycle]
        batch = [rows[(step * 4 + offset) % len(rows)] for offset in range(4)]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = learner([prompt for prompt, _ in batch])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    if stage == "checkpoint":
        saved = save_full_checkpoint(root / "snapshot", model=learner,
                                     optimizer=optimizer, context=context)
        _write_json(root / "checkpoint_receipt.json", {
            "checkpoint_sha256": saved.sha256, "base_digest": base_digest,
            "initial_digest": initial_digest,
            "initial_lora_digest": initial_lora_digest,
            "checkpoint_lora_digest": _sha_bytes(lora_named), "split_at": split_at,
            "seed": seed, "steps": steps, "process_pid": os.getpid(),
        })
        return

    learner.eval()
    with torch.no_grad():
        def predict(rows: tuple[tuple[str, int], ...]) -> list[int]:
            return learner([prompt for prompt, _ in rows]).argmax(dim=-1).tolist()

        heldout = predict(HELDOUT)
        transfer = predict(TRANSFER)
    _write_json(root / ("reference.json" if stage == "reference" else "resumed.json"), {
        "schema": SCHEMA, "stage": stage, "seed": seed, "steps": steps,
        "split_at": split_at, "process_pid": os.getpid(),
        "base_digest": base_digest, "initial_digest": initial_digest,
        "initial_lora_digest": initial_lora_digest,
        "final_lora_digest": _sha_bytes(lora_named),
        "final_trainable_digest": _sha_bytes(named),
        "training_exposures": [list(pair) for pair in
                              tuple((case_id, digest) for case_id, digest, _
                                    in visits)],
        "heldout_predictions": heldout, "transfer_predictions": transfer,
        "heldout_correct": sum(int(x == label)
                               for x, (_, label) in zip(heldout, HELDOUT)),
        "transfer_correct": sum(int(x == label)
                                for x, (_, label) in zip(transfer, TRANSFER)),
        "graduation_authorized": False,
    })


def verify_reference_resume(reference: dict[str, Any], resumed: dict[str, Any],
                            checkpoint: dict[str, Any]) -> None:
    """Reject mismatched lineage or outputs; no accuracy-only substitution."""
    if not all(type(item) is dict for item in (reference, resumed, checkpoint)):
        raise ValueError("invalid run receipt type")
    if (reference.get("schema") != SCHEMA or resumed.get("schema") != SCHEMA
            or reference.get("stage") != "reference"
            or resumed.get("stage") != "resume"
            or reference.get("graduation_authorized") is not False
            or resumed.get("graduation_authorized") is not False):
        raise ValueError("unqualified or mismatched run stage receipt")
    for name in ("base_digest", "initial_digest", "initial_lora_digest",
                 "seed", "steps", "split_at"):
        if (name not in reference or name not in resumed or name not in checkpoint
                or reference[name] != resumed[name]
                or reference[name] != checkpoint[name]):
            raise ValueError(f"checkpoint/reference lineage mismatch: {name}")
    for name in ("process_pid", "final_trainable_digest", "final_lora_digest",
                 "training_exposures",
                 "heldout_predictions", "transfer_predictions",
                 "heldout_correct", "transfer_correct"):
        if name not in reference or name not in resumed:
            raise ValueError(f"missing exact resume evidence: {name}")
    if (type(reference["process_pid"]) is not int
            or type(resumed["process_pid"]) is not int
            or type(checkpoint.get("process_pid")) is not int
            or len({reference["process_pid"], resumed["process_pid"],
                    checkpoint["process_pid"]}) != 3):
        raise ValueError("fresh-process isolation was not established")
    for name in ("final_trainable_digest", "final_lora_digest", "training_exposures",
                 "heldout_predictions", "transfer_predictions",
                 "heldout_correct", "transfer_correct"):
        if reference[name] != resumed[name]:
            raise ValueError(f"interrupted continuation differs from reference: {name}")
    if (reference["initial_digest"] == reference["final_trainable_digest"]
            or reference["initial_lora_digest"] == reference["final_lora_digest"]):
        raise ValueError("pretrained LoRA run did not update learned parameters")


def _orchestrate(root: Path, seed: int, steps: int, split_at: int) -> dict[str, Any]:
    root.mkdir(mode=0o700)
    command_base = [
        sys.executable, "-m", "scripts.qualify_minilm_resume_cpu",
        "--run-root", str(root), "--seed", str(seed), "--steps", str(steps),
        "--split-at", str(split_at), "--internal",
    ]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    for stage in ("reference", "checkpoint", "resume"):
        with (root / f"{stage}.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command_base + ["--stage", stage],
                cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=600, check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{stage} subprocess failed; see isolated log")

    reference = _read_json(root / "reference.json")
    resumed = _read_json(root / "resumed.json")
    checkpoint = _read_json(root / "checkpoint_receipt.json")
    verify_reference_resume(reference, resumed, checkpoint)
    report = {
        "schema": SCHEMA, "model_id": MODEL_ID, "seed": seed,
        "steps": steps, "interruption_step": split_at,
        "fresh_processes": 3,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "base_model_tensor_sha256": reference["base_digest"],
        "initial_trainable_sha256": reference["initial_digest"],
        "final_trainable_sha256": reference["final_trainable_digest"],
        "initial_lora_sha256": reference["initial_lora_digest"],
        "final_lora_sha256": reference["final_lora_digest"],
        "heldout_correct": reference["heldout_correct"],
        "heldout_total": len(HELDOUT),
        "transfer_correct": reference["transfer_correct"],
        "transfer_total": len(TRANSFER),
        "exact_resume_match": True,
        "graduation_authorized": False,
        "qualification": "PUBLIC_DEVELOPMENT_PRETRAINED_RESUME_ONLY",
        "signed_event_journal_integrated": False,
        "independent_exam_custody": False,
    }
    _write_json(root / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent")
    parser.add_argument("--run-name")
    parser.add_argument("--run-root")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--split-at", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--stage", choices=("reference", "checkpoint", "resume"))
    args = parser.parse_args(argv)
    try:
        validate_fixtures()
        checkpoint_exposures(args.steps, args.split_at)
        if args.internal:
            if not args.stage or not args.run_root:
                raise ValueError("internal stage requires stage and run root")
            root = Path(args.run_root).resolve(strict=True)
            if not root.is_dir() or root.is_symlink():
                raise ValueError("internal root missing or symlink")
            _neural_stage(root, args.stage, args.seed, args.steps, args.split_at)
            return 0
        if args.stage or args.run_root or not args.output_parent or not args.run_name:
            raise ValueError("external invocation requires output parent and run name")
        root = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "run_root": str(root), "model_id": MODEL_ID,
                              "steps": args.steps, "split_at": args.split_at,
                              "graduation_authorized": False}, sort_keys=True))
            return 0
        report = _orchestrate(root, args.seed, args.steps, args.split_at)
        print(json.dumps({"run_root": str(root), **report}, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
