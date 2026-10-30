"""Offline MiniLM LoRA-versus-head causal ablation (public development only).

This experiment freezes either the PEFT LoRA adapter or the classifier head,
from identical initial values, while retaining the source pilot's exact
training/development fixtures. It never claims final examination or graduation.
The default mode only prints a plan; ``--execute`` is explicit and writes to a
new, isolated run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__:  # ``python -m scripts.qualify_minilm_ablation_cpu``
    from . import qualify_minilm_cpu as pilot
else:  # direct file execution; never resolve an unrelated installed scripts package
    from importlib.util import module_from_spec, spec_from_file_location

    _pilot_path = Path(__file__).with_name("qualify_minilm_cpu.py")
    _pilot_spec = spec_from_file_location("bt2_minilm_pilot_sibling", _pilot_path)
    if _pilot_spec is None or _pilot_spec.loader is None:
        raise ImportError("missing sibling MiniLM pilot source")
    pilot = module_from_spec(_pilot_spec)
    _pilot_spec.loader.exec_module(pilot)


SCHEMA = "labyrinth.minilm-lora-ablation.v1"
ARMS = ("full", "head_only", "lora_only", "zero_update")


def compare_arms(scores: dict[str, float]) -> dict[str, Any]:
    """Describe observed differences without turning them into causal proof."""
    if (type(scores) is not dict or set(scores) != set(ARMS)
            or any(type(score) not in (int, float) or not math.isfinite(score)
                   or not 0 <= score <= 1 for score in scores.values())):
        raise ValueError("all four finite ablation accuracy arms are required")
    return {
        "full_minus_head_only": scores["full"] - scores["head_only"],
        "lora_only_minus_zero": scores["lora_only"] - scores["zero_update"],
        "head_only_minus_zero": scores["head_only"] - scores["zero_update"],
        "full_outperforms_head_only": scores["full"] > scores["head_only"],
    }


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

    def tensor_digest(parameters: list[tuple[str, Any]]) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(parameters):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def lora_parameters(model: Any) -> list[tuple[str, Any]]:
        result = [(name, parameter) for name, parameter in model.named_parameters()
                  if "lora_A" in name or "lora_B" in name]
        if not result:
            raise RuntimeError("pretrained encoder has no LoRA trainable tensors")
        return result

    def create_model(previous: Path | None, arm: str) -> tuple[Any, Any]:
        torch.manual_seed(seed)
        base = AutoModel.from_pretrained(model_id, local_files_only=True)
        if previous is None:
            model = get_peft_model(base, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8,
                lora_dropout=0.0, bias="none", target_modules=["query", "value"],
            ))
        else:
            model = PeftModel.from_pretrained(
                base, previous / "adapter", is_trainable=True,
            )
        head = torch.nn.Linear(base.config.hidden_size, 2)
        if previous is not None:
            head.load_state_dict(load_file(str(previous / "head.safetensors")))
        frozen_lora = arm == "head_only"
        frozen_head = arm == "lora_only"
        for _, parameter in lora_parameters(model):
            parameter.requires_grad_(not frozen_lora)
        for parameter in head.parameters():
            parameter.requires_grad_(not frozen_head)
        return model, head

    def logits(model: Any, head: Any, prompts: list[str]) -> Any:
        tokens = tokenizer(prompts, padding=True, truncation=True, max_length=64,
                           return_tensors="pt")
        output = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = ((output.last_hidden_state * mask).sum(dim=1)
                  / mask.sum(dim=1))
        return head(pooled)

    def score(model: Any, head: Any, rows: tuple[tuple[str, int], ...]) -> float:
        model.eval()
        head.eval()
        with torch.no_grad():
            prediction = logits(model, head, [text for text, _ in rows]).argmax(dim=-1)
        return sum(int(prediction[index].item() == label)
                   for index, (_, label) in enumerate(rows)) / len(rows)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "pretrained_origin": "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
        "model_id": model_id, "seed": seed,
        "steps_per_cycle": steps_per_cycle, "learning_rate": learning_rate,
        "fixture_sizes": pilot.validate_fixtures(), "arms": {},
        "private_final_exam": "NOT_RUN", "graduation_authorized": False,
        "physical_model_byte_custody": "SELF_REPORTED",
        "causal_attribution": "NOT_ESTABLISHED_BY_SOURCE_ALONE",
    }
    root.mkdir(mode=0o700)
    for arm in ARMS:
        print(f"arm={arm} starting", flush=True)
        arm_root = root / arm
        arm_root.mkdir()
        previous: Path | None = None
        cycles: list[dict[str, Any]] = []
        initial: tuple[str, str] | None = None
        predecessor: tuple[str, str] | None = None
        for cycle, rows in enumerate(pilot.TRAIN):
            model, head = create_model(previous, arm)
            lora = lora_parameters(model)
            head_parameters = [("head." + name, p) for name, p in head.named_parameters()]
            before = (tensor_digest(lora), tensor_digest(head_parameters))
            if initial is None:
                initial = before
            if predecessor is not None and before != predecessor:
                raise RuntimeError("LoRA/head checkpoint failed exact cycle reload")
            optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer_parameters += [p for p in head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate)
            model.train()
            head.train()
            visited: list[str] = []
            for step in range(steps_per_cycle):
                batch = [rows[(step * 4 + offset) % len(rows)] for offset in range(4)]
                visited.extend(hashlib.sha256(text.encode("utf-8")).hexdigest()
                               for text, _ in batch)
                target = torch.tensor([label for _, label in batch], dtype=torch.long)
                prediction = logits(model, head, [text for text, _ in batch])
                loss = torch.nn.functional.cross_entropy(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if arm != "zero_update":
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
            after = (tensor_digest(lora), tensor_digest(head_parameters))
            expected_lora_change = arm in ("full", "lora_only")
            expected_head_change = arm in ("full", "head_only")
            if (after[0] != before[0]) != expected_lora_change:
                raise RuntimeError(f"{arm} unexpected LoRA change/absence")
            if (after[1] != before[1]) != expected_head_change:
                raise RuntimeError(f"{arm} unexpected head change/absence")
            checkpoint = arm_root / f"cycle-{cycle}"
            checkpoint.mkdir()
            (checkpoint / "adapter").mkdir()
            model.save_pretrained(str(checkpoint / "adapter"), safe_serialization=True)
            save_file({name: tensor.detach().cpu().contiguous()
                       for name, tensor in head.state_dict().items()},
                      str(checkpoint / "head.safetensors"))
            cycles.append({
                "cycle": cycle,
                "lora_sha256_before": before[0], "lora_sha256_after": after[0],
                "head_sha256_before": before[1], "head_sha256_after": after[1],
                "adapter_file_sha256": pilot._sha_file(
                    checkpoint / "adapter" / "adapter_model.safetensors"),
                "head_file_sha256": pilot._sha_file(checkpoint / "head.safetensors"),
                "optimizer_updates": 0 if arm == "zero_update" else steps_per_cycle,
                "forward_steps": steps_per_cycle, "backward_steps": steps_per_cycle,
                "training_prompt_sha256": visited,
                "heldout_accuracy": score(model, head, pilot.HELDOUT),
                "transfer_accuracy": score(model, head, pilot.TRANSFER),
            })
            predecessor = after
            previous = checkpoint
            print(f"arm={arm} cycle={cycle} saved", flush=True)
        report["arms"][arm] = {
            "initial_lora_sha256": initial[0],
            "initial_head_sha256": initial[1], "cycles": cycles,
        }

    if len({report["arms"][arm]["initial_lora_sha256"] for arm in ARMS}) != 1:
        raise RuntimeError("ablation arms did not share initial LoRA values")
    if len({report["arms"][arm]["initial_head_sha256"] for arm in ARMS}) != 1:
        raise RuntimeError("ablation arms did not share initial head values")
    for cycle in range(len(pilot.TRAIN)):
        visits = {tuple(report["arms"][arm]["cycles"][cycle]["training_prompt_sha256"])
                  for arm in ARMS}
        if len(visits) != 1:
            raise RuntimeError("ablation arms did not share training exposures")

    report["heldout"] = compare_arms({
        arm: report["arms"][arm]["cycles"][-1]["heldout_accuracy"]
        for arm in ARMS
    })
    report["transfer"] = compare_arms({
        arm: report["arms"][arm]["cycles"][-1]["transfer_accuracy"]
        for arm in ARMS
    })
    report["causal_attribution"] = (
        "SINGLE_PUBLIC_FIXTURE_ABLATION_NO_INDEPENDENT_EXAM"
    )
    (root / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                                       encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-id", default=pilot.MODEL_ID)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps_per_cycle <= 32 or not 0 < args.learning_rate <= 0.01:
        parser.error("steps must be 1..32 and learning rate in (0, .01]")
    try:
        sizes = pilot.validate_fixtures()
        root = pilot.plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "run_root": str(root), "model_id": args.model_id,
                              "fixture_sizes": sizes,
                              "arms": ARMS, "graduation_authorized": False},
                             sort_keys=True))
            return 0
        report = execute(root, model_id=args.model_id, seed=args.seed,
                         steps_per_cycle=args.steps_per_cycle,
                         learning_rate=args.learning_rate)
        print(json.dumps({"run_root": str(root), "heldout": report["heldout"],
                          "transfer": report["transfer"],
                          "graduation_authorized": False}, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
