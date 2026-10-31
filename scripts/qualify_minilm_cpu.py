"""Offline, bounded pretrained MiniLM/LoRA development experiment.

This is an opt-in *physical neural* probe, not a private final examination.
It records a genuinely untrained baseline and trains identical initial
adapters in correct-label, corrupted-label and zero-update arms; reloads the
adapter and classification head across two cycles; and scores disjoint
held-out and alternative-template examples. The untrained baseline has no
optimizer steps. The other three arms have matched forward/backward/optimizer
calls, with the zero-update arm using a zero learning rate.

Only cached model files are used. The supplied output run directory must not
exist, and all artifacts remain there. The default is a read-only plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "labyrinth.minilm-cpu-development.v2"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Frozen, disjoint public DEVELOPMENT fixtures. These are not final exams.
# The transfer set uses outcome narratives, not the train review construction.
TRAIN = (
    (("The service was excellent.", 1), ("The support was terrible.", 0),
     ("The explanation was clear.", 1), ("The answer was confusing.", 0),
     ("The result was helpful.", 1), ("The process was frustrating.", 0),
     ("The instructions were useful.", 1), ("The instructions were useless.", 0),
     ("The response was friendly.", 1), ("The response was hostile.", 0),
     ("The guidance was accurate.", 1), ("The guidance was misleading.", 0)),
    (("This was a wonderful experience.", 1), ("This was an awful experience.", 0),
     ("The work was impressive.", 1), ("The work was disappointing.", 0),
     ("The information was reliable.", 1), ("The information was unreliable.", 0),
     ("The outcome was satisfying.", 1), ("The outcome was upsetting.", 0),
     ("The help was valuable.", 1), ("The help was worthless.", 0),
     ("The solution was elegant.", 1), ("The solution was broken.", 0)),
)
HELDOUT = (
    ("I appreciated the thoughtful assistance.", 1),
    ("I regretted asking for help.", 0),
    ("I found the documentation genuinely helpful.", 1),
    ("I found the documentation deeply confusing.", 0),
    ("Everything worked smoothly and reliably.", 1),
    ("Everything failed and wasted my time.", 0),
    ("The explanation made the problem easier.", 1),
    ("The explanation made the problem harder.", 0),
    ("I would recommend this excellent service.", 1),
    ("I would avoid this dreadful service.", 0),
)
TRANSFER = (
    ("After the repair, the customer smiled and thanked the team.", 1),
    ("After the repair, the customer complained about more damage.", 0),
    ("The team finished early and everyone celebrated.", 1),
    ("The team missed the deadline and morale collapsed.", 0),
    ("The audit found no problems and praised the documentation.", 1),
    ("The audit found severe problems and rejected the documentation.", 0),
    ("The patient left reassured by the clear instructions.", 1),
    ("The patient left frightened by contradictory instructions.", 0),
    ("The new process prevented errors and earned trust.", 1),
    ("The new process caused errors and destroyed trust.", 0),
)
# Historical v1 helper API retained for source/tests expecting its exact names.
# Only PHYSICAL_ARMS is used for new v2 execution and output.
ARMS = ("correct", "zero_update", "inverted")
PHYSICAL_ARMS = ("baseline", "zero_update", "correct_labels", "corrupted_labels")
TRAINING_ARMS = PHYSICAL_ARMS[1:]
PUBLIC_TRAIN_BATCH_SIZE = 4


def validate_fixtures() -> dict[str, int]:
    seen: set[str] = set()
    sizes: dict[str, int] = {}
    for name, rows in (("cycle0", TRAIN[0]), ("cycle1", TRAIN[1]),
                       ("heldout", HELDOUT), ("transfer", TRANSFER)):
        if len(rows) < 8 or {label for _, label in rows} != {0, 1}:
            raise ValueError(f"invalid balanced {name} fixture")
        sizes[name] = len(rows)
        for prompt, label in rows:
            if type(prompt) is not str or type(label) is not int or label not in (0, 1):
                raise ValueError("invalid example")
            normalized = " ".join(prompt.casefold().split())
            if normalized in seen:
                raise ValueError(f"training/development prompt overlap: {name}")
            seen.add(normalized)
    return sizes


def plan_output(parent: str | Path, run_name: str) -> Path:
    if type(run_name) is not str or not RUN_NAME.fullmatch(run_name):
        raise ValueError("invalid bounded run-name")
    root = Path(parent).expanduser()
    if (root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction())
            or not root.is_dir()):
        raise ValueError("output parent must be an existing non-reparse directory")
    result = root.resolve(strict=True) / run_name
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"run directory already exists: {result}")
    return result


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(accuracy: dict[str, float]) -> dict[str, Any]:
    if set(accuracy) != set(ARMS) or any(type(v) not in (int, float)
                                           or not 0 <= v <= 1 for v in accuracy.values()):
        raise ValueError("all three finite accuracy arms are required")
    return {
        "correct_minus_zero": accuracy["correct"] - accuracy["zero_update"],
        "correct_minus_inverted": accuracy["correct"] - accuracy["inverted"],
        "correct_outperforms_both": (accuracy["correct"] > accuracy["zero_update"]
                                      and accuracy["correct"] > accuracy["inverted"]),
    }


def physical_comparison(accuracy: dict[str, float]) -> dict[str, Any]:
    """Four-arm development comparison; not a private exam or causal proof."""
    if (set(accuracy) != set(PHYSICAL_ARMS)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not 0 <= v <= 1 for v in accuracy.values())):
        raise ValueError("four finite physical control accuracies are required")
    correct = accuracy["correct_labels"]
    return {
        "correct_minus_baseline": correct - accuracy["baseline"],
        "correct_minus_zero": correct - accuracy["zero_update"],
        "correct_minus_corrupted": correct - accuracy["corrupted_labels"],
        "correct_outperforms_all_controls": all(correct > accuracy[arm]
                                                  for arm in PHYSICAL_ARMS
                                                  if arm != "correct_labels"),
    }


def arm_contract(arm: str, learning_rate: float) -> dict[str, Any]:
    """Describe executed work; never present the baseline as a matched sham."""
    if arm not in PHYSICAL_ARMS:
        raise ValueError(f"unknown physical arm: {arm}")
    if type(learning_rate) not in (float, int) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid physical optimizer learning rate")
    trained = arm != "baseline"
    return {
        "trained": trained,
        "optimizer_step": trained,
        "learning_rate": 0.0 if arm in ("baseline", "zero_update") else float(learning_rate),
        "invert_labels": arm == "corrupted_labels",
        "actual_weight_updates_expected": arm in ("correct_labels", "corrupted_labels"),
    }


def training_batches(cycle: int, *, steps: int,
                     batch_size: int = PUBLIC_TRAIN_BATCH_SIZE,
                     invert_labels: bool = False) -> list[dict[str, Any]]:
    """Public example-exposure receipts; one optimizer step has four visits."""
    if (type(cycle) is not int or not 0 <= cycle < len(TRAIN)
            or type(steps) is not int or not 1 <= steps <= 32
            or type(batch_size) is not int or not 1 <= batch_size <= len(TRAIN[cycle])):
        raise ValueError("invalid public training schedule")
    schedule = []
    for step in range(steps):
        examples = []
        for offset in range(batch_size):
            index = (step * batch_size + offset) % len(TRAIN[cycle])
            prompt, label = TRAIN[cycle][index]
            examples.append({
                "case_id": f"train-c{cycle}-r{index}",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "structure_key": f"public-sentiment-cycle-{cycle}",
                "target_label": label ^ int(invert_labels),
            })
        schedule.append({"step": step, "examples": examples})
    return schedule


def prediction_records(split: str, rows: tuple[tuple[str, int], ...],
                       predicted: list[int]) -> list[dict[str, Any]]:
    """Identify actual public predictions without asserting private provenance."""
    if (split not in ("heldout", "transfer") or len(rows) != len(predicted)
            or not rows):
        raise ValueError("invalid public prediction coverage")
    records = []
    for index, ((prompt, label), response) in enumerate(zip(rows, predicted)):
        if type(response) is not int or response not in (0, 1):
            raise ValueError("public predictions must be binary labels")
        records.append({
            "case_id": f"{split}-r{index}",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response": str(response),
            "predicted_label": response,
            "expected_label": label,
        })
    return records


def optimizer_learning_rate(arm: str, learning_rate: float) -> float:
    """Keep the AdamW computation in the sham without changing any weights."""
    if arm not in ARMS:
        raise ValueError(f"unknown training arm: {arm}")
    return 0.0 if arm == "zero_update" else learning_rate


def reload_logit_delta(expected: list[list[float]], observed: list[list[float]],
                       *, tolerance: float = 1e-5) -> float:
    """Verify a restored model produces equivalent outputs, not merely hashes."""
    if (not expected or len(expected) != len(observed)
            or not math.isfinite(tolerance) or tolerance < 0):
        raise RuntimeError("heldout reload logit shape/tolerance mismatch")
    delta = 0.0
    for before, after in zip(expected, observed):
        if not before or len(before) != len(after):
            raise RuntimeError("heldout reload logit shape mismatch")
        for a, b in zip(before, after):
            if (type(a) not in (int, float) or type(b) not in (int, float)
                    or not math.isfinite(a) or not math.isfinite(b)):
                raise RuntimeError("heldout reload logits must be finite numbers")
            delta = max(delta, abs(a - b))
    if delta > tolerance:
        raise RuntimeError(f"heldout logits changed on checkpoint reload: {delta}")
    return delta


def execute(root: Path, *, model_id: str, seed: int, steps_per_cycle: int,
            learning_rate: float) -> dict[str, Any]:
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    def digest_adapter(model: Any) -> str:
        digest = hashlib.sha256()
        trainable = 0
        for name, parameter in sorted(model.named_parameters()):
            if parameter.requires_grad:
                trainable += 1
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        if not trainable:
            raise RuntimeError("no trainable LoRA adapter tensors")
        return digest.hexdigest()

    def digest_state(model: Any, head: Any) -> str:
        digest = hashlib.sha256()
        digest.update(digest_adapter(model).encode("ascii") + b"\0")
        for name, parameter in sorted(head.named_parameters()):
            digest.update(("head." + name).encode("utf-8") + b"\0")
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def create_model(previous: Path | None) -> tuple[Any, Any]:
        torch.manual_seed(seed)
        base = AutoModel.from_pretrained(model_id, local_files_only=True)
        if previous is None:
            adapter = get_peft_model(base, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8,
                lora_dropout=0.0, bias="none", target_modules=["query", "value"],
            ))
        else:
            adapter = PeftModel.from_pretrained(base, previous / "adapter", is_trainable=True)
        head = torch.nn.Linear(base.config.hidden_size, 2)
        if previous is not None:
            head.load_state_dict(load_file(str(previous / "head.safetensors")))
        return adapter, head

    def logits(model: Any, head: Any, prompts: list[str]) -> Any:
        tokens = tokenizer(prompts, padding=True, truncation=True, max_length=64,
                           return_tensors="pt")
        result = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (result.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        return head(pooled)

    def score(model: Any, head: Any, rows: tuple[tuple[str, int], ...],
              split: str) -> tuple[float, list[dict[str, Any]]]:
        model.eval()
        head.eval()
        with torch.no_grad():
            predictions = logits(model, head, [prompt for prompt, _ in rows]).argmax(dim=-1)
        outputs = [int(predictions[i].item()) for i in range(len(rows))]
        return (sum(int(outputs[i] == label)
                    for i, (_, label) in enumerate(rows)) / len(rows),
                prediction_records(split, rows, outputs))

    def heldout_logits(model: Any, head: Any) -> list[list[float]]:
        model.eval()
        head.eval()
        with torch.no_grad():
            return logits(model, head, [prompt for prompt, _ in HELDOUT]).tolist()

    report: dict[str, Any] = {
        "schema": SCHEMA, "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "pretrained_origin": "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
        "model_id": model_id, "seed": seed, "run_id": root.name,
        "steps_per_cycle": steps_per_cycle, "learning_rate": learning_rate,
        "physical_training_batch_size": PUBLIC_TRAIN_BATCH_SIZE,
        "fixture_sizes": validate_fixtures(), "arms": {},
        "harness_sha256": _sha_file(Path(__file__)),
        "device": "cpu", "torch_version": str(torch.__version__),
        "cpu_torch_threads": 2,
        "reload_method": "NEW_MODEL_OBJECT_SAME_PROCESS",
        "zero_update_control": "SAME_FORWARD_BACKWARD_ADAMW_STEP_WITH_ZERO_LR",
        "private_final_exam": "NOT_RUN", "graduation_authorized": False,
        "physical_model_byte_custody": "SELF_REPORTED",
        "arm_contracts": {arm: arm_contract(arm, learning_rate)
                          for arm in PHYSICAL_ARMS},
        "baseline_compute_matched_to_training": False,
    }
    root.mkdir(mode=0o700)
    for arm in PHYSICAL_ARMS:
        print(f"arm={arm} starting", flush=True)
        contract = arm_contract(arm, learning_rate)
        arm_root = root / arm
        arm_root.mkdir()
        previous: Path | None = None
        cycle_results = []
        initial_digest = None
        initial_adapter_digest = None
        preceding_after_digest = None
        preceding_adapter_digest = None
        preceding_heldout_logits = None
        for cycle, rows in enumerate(TRAIN):
            model, head = create_model(previous)
            before = digest_state(model, head)
            adapter_before = digest_adapter(model)
            if initial_digest is None:
                initial_digest = before
                initial_adapter_digest = adapter_before
            if preceding_after_digest is not None and before != preceding_after_digest:
                raise RuntimeError("adapter/head checkpoint failed full trainable-value reload")
            if (preceding_adapter_digest is not None
                    and adapter_before != preceding_adapter_digest):
                raise RuntimeError("LoRA adapter value digest changed during reload")
            reload_delta = (reload_logit_delta(preceding_heldout_logits,
                                              heldout_logits(model, head))
                            if preceding_heldout_logits is not None else None)
            baseline_heldout, _ = score(model, head, HELDOUT, "heldout")
            batches = (training_batches(cycle, steps=steps_per_cycle,
                                        invert_labels=contract["invert_labels"])
                       if contract["trained"] else [])
            visits = [example["prompt_sha256"] for batch in batches
                      for example in batch["examples"]]
            changed_optimizer_steps = 0
            if contract["optimizer_step"]:
                params = [p for p in model.parameters() if p.requires_grad]
                if not params:
                    raise RuntimeError("no trainable pretrained-model adapter parameters")
                optimizer = torch.optim.AdamW(params + list(head.parameters()),
                                               lr=contract["learning_rate"])
                model.train()
                head.train()
                indexed_rows = {f"train-c{cycle}-r{index}": prompt
                                for index, (prompt, _) in enumerate(rows)}
                for batch_receipt in batches:
                    batch = [(indexed_rows[example["case_id"]], example["target_label"])
                             for example in batch_receipt["examples"]]
                    target = torch.tensor([label for _, label in batch], dtype=torch.long)
                    predicted = logits(model, head, [text for text, _ in batch])
                    loss = torch.nn.functional.cross_entropy(predicted, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    step_before = digest_state(model, head)
                    optimizer.step()
                    changed_optimizer_steps += int(digest_state(model, head) != step_before)
            after = digest_state(model, head)
            adapter_after = digest_adapter(model)
            if arm in ("baseline", "zero_update") and after != before:
                raise RuntimeError("frozen control arm unexpectedly mutated")
            if arm in ("baseline", "zero_update") and adapter_after != adapter_before:
                raise RuntimeError("frozen control LoRA adapter unexpectedly mutated")
            if contract["actual_weight_updates_expected"] and after == before:
                raise RuntimeError("trained adapter/head did not change")
            if contract["actual_weight_updates_expected"] and adapter_after == adapter_before:
                raise RuntimeError("pretrained-model LoRA adapter did not learn")
            checkpoint = arm_root / f"cycle-{cycle}"
            checkpoint.mkdir()
            (checkpoint / "adapter").mkdir()
            model.save_pretrained(str(checkpoint / "adapter"), safe_serialization=True)
            save_file({key: tensor.detach().cpu().contiguous()
                       for key, tensor in head.state_dict().items()},
                      str(checkpoint / "head.safetensors"))
            heldout_accuracy, heldout_predictions = score(model, head, HELDOUT, "heldout")
            transfer_accuracy, transfer_predictions = score(model, head, TRANSFER, "transfer")
            cycle_results.append({
                "cycle": cycle, "trainable_sha256_before": before,
                "trainable_sha256_after": after,
                "lora_sha256_before": adapter_before,
                "lora_sha256_after": adapter_after,
                "adapter_file_sha256": _sha_file(checkpoint / "adapter" / "adapter_model.safetensors"),
                "head_file_sha256": _sha_file(checkpoint / "head.safetensors"),
                "optimizer_updates": changed_optimizer_steps,
                "optimizer_step_calls": len(batches),
                "compute_steps": len(batches),
                "forward_steps": len(batches),
                "backward_steps": len(batches),
                "training_forward_steps": len(batches),
                "training_backward_steps": len(batches),
                "train_example_visits": len(visits),
                "training_batches": batches,
                "training_prompt_sha256": visits,
                "heldout_reload_max_abs_logit_delta": reload_delta,
                "heldout_accuracy_before": baseline_heldout,
                "heldout_accuracy": heldout_accuracy,
                "transfer_accuracy": transfer_accuracy,
                "heldout_predictions": heldout_predictions,
                "transfer_predictions": transfer_predictions,
            })
            preceding_after_digest = after
            preceding_adapter_digest = adapter_after
            preceding_heldout_logits = heldout_logits(model, head)
            previous = checkpoint
            print(f"arm={arm} cycle={cycle} saved", flush=True)
        report["arms"][arm] = {"initial_trainable_sha256": initial_digest,
                               "initial_lora_sha256": initial_adapter_digest,
                               "cycles": cycle_results}
    initial = {report["arms"][arm]["initial_trainable_sha256"] for arm in PHYSICAL_ARMS}
    if len(initial) != 1:
        raise RuntimeError("matched arms did not have identical initial trainable values")
    if len({report["arms"][arm]["initial_lora_sha256"] for arm in PHYSICAL_ARMS}) != 1:
        raise RuntimeError("matched arms did not start from the same LoRA adapter")
    for cycle in range(len(TRAIN)):
        exposures = {tuple(report["arms"][arm]["cycles"][cycle]["training_prompt_sha256"])
                     for arm in TRAINING_ARMS}
        if len(exposures) != 1:
            raise RuntimeError("matched arms did not receive identical training visits")
        if report["arms"]["baseline"]["cycles"][cycle]["train_example_visits"] != 0:
            raise RuntimeError("untrained baseline has training exposures")
    report["heldout"] = physical_comparison({
        arm: report["arms"][arm]["cycles"][-1]["heldout_accuracy"]
        for arm in PHYSICAL_ARMS})
    report["transfer"] = physical_comparison({
        arm: report["arms"][arm]["cycles"][-1]["transfer_accuracy"]
        for arm in PHYSICAL_ARMS})
    report["retention"] = {arm: (report["arms"][arm]["cycles"][1]["heldout_accuracy"]
                                  - report["arms"][arm]["cycles"][0]["heldout_accuracy"])
                           for arm in PHYSICAL_ARMS}
    report["learning_qualification"] = "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION"
    (root / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                       encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps_per_cycle <= 32 or not 0 < args.learning_rate <= 0.01:
        parser.error("steps must be 1..32 and learning rate in (0, .01]")
    try:
        sizes = validate_fixtures()
        root = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "run_root": str(root), "model_id": args.model_id,
                              "fixture_sizes": sizes, "graduation_authorized": False},
                             sort_keys=True))
            return 0
        report = execute(root, model_id=args.model_id, seed=args.seed,
                         steps_per_cycle=args.steps_per_cycle,
                         learning_rate=args.learning_rate)
        print(json.dumps({"run_root": str(root), "heldout": report["heldout"],
                          "transfer": report["transfer"], "graduation_authorized": False},
                         sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
