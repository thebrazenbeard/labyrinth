"""Package and run Minotaur's experimental, *advisory-only* neural adapter.

Minotaur's neural component cannot grant graduation. An independent,
deterministic Labyrinth gate must own all release decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from .hf_pilot import (
    EVAL_SCHEMA,
    SCHEMA,
    PilotConfig,
    _adapter_manifest,
    _metadata_digest,
    _sha_file,
    _weights_digest,
    validate_config,
)
from .runner import load_json


MODEL_SCHEMA = "labyrinth.minotaur.advisory-model.v1"
MODEL_NAME = "Minotaur"
MODEL_ROLE = "advisory_guardian_neural_component"


def _assert_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular, non-symlink file: {path.name}")


def _manifest_payload(run_dir: Path, reload_report: Path, base_repository: str,
                      base_revision: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", base_repository):
        raise ValueError("base repository must be a Hugging Face owner/name")
    if not re.fullmatch(r"[a-f0-9]{40}", base_revision):
        raise ValueError("base revision must be a 40-character lowercase commit SHA")
    training_file = run_dir / "pilot_report.json"
    _assert_file(training_file)
    _assert_file(reload_report)
    train = load_json(training_file, strict=True)
    fresh = load_json(reload_report, strict=True)
    if train.get("schema") != SCHEMA or fresh.get("schema") != EVAL_SCHEMA:
        raise ValueError("training and reload reports must use supported schemas")
    if fresh.get("training_report_sha256") != _sha_file(training_file):
        raise ValueError("reload receipt does not bind the training report")
    if not fresh.get("matches_original_candidate_trials"):
        raise ValueError("reloaded adapter does not reproduce the trained candidate")
    if (fresh.get("reloaded_lora_tensor_value_sha256")
            != train.get("candidate_lora_tensor_value_sha256")):
        raise ValueError("reloaded adapter parameter digest disagrees with training")
    if (train.get("initial_lora_parameter_sha256")
            == train.get("candidate_lora_parameter_sha256")):
        raise ValueError("training report does not establish changed LoRA values")
    steps = train.get("training", {}).get("candidate", {}).get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("training report lacks positive optimizer steps")
    source_adapter = run_dir / "candidate_adapter"
    if _adapter_manifest(source_adapter) != train.get("candidate_adapter_sha256"):
        raise ValueError("source adapter files do not match training report")
    if fresh.get("adapter_sha256") != train.get("candidate_adapter_sha256"):
        raise ValueError("reload receipt does not bind the source adapter")
    config = train.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("invalid training configuration")
    scores = {}
    for name in ("baseline", "sham", "candidate", "corrupted"):
        value = train.get("evaluations", {}).get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing evaluation arm: {name}")
        scores[name] = {key: value[key] for key in ("correct", "total", "format_valid")}
    return {
        "schema": MODEL_SCHEMA,
        "model_name": MODEL_NAME,
        "model_version": "0.1.0-experimental",
        "role": MODEL_ROLE,
        "graduation_authority": "none",
        "qualification": "NOT_ESTABLISHED",
        "trained_behavior": "Labyrinth maze-answer completion, not trainee verification",
        "next_training_target": "Evidence-grounded correctness critique; measure false passes and false rejects",
        "base_model": {
            "repository": base_repository,
            "revision": base_revision,
            "license": "apache-2.0",
            "weight_sha256": train["base_weights_sha256"],
            "metadata_sha256": train["model_metadata_sha256"],
        },
        "adaptation": {
            "kind": "pretrained_causal_lm_peft_lora",
            "optimizer": "AdamW",
            "completed_steps": steps,
            "training_source_commit": train.get("source", {}).get("commit"),
            "training_source_tree_sha256": train.get("source", {}).get("source_tree_sha256"),
            "training_report_sha256": _sha_file(training_file),
            "reload_report_sha256": _sha_file(reload_report),
            "initial_lora_parameter_sha256": train["initial_lora_parameter_sha256"],
            "trained_lora_parameter_sha256": train["candidate_lora_parameter_sha256"],
            "reloaded_lora_parameter_sha256": fresh["reloaded_lora_tensor_value_sha256"],
            "configuration": {key: config[key] for key in (
                "seed", "train_count", "eval_count", "tiers", "steps", "max_tokens",
                "max_new_tokens", "learning_rate", "lora_rank", "lora_alpha", "device")
                if key in config},
        },
        "development_scores": scores,
        "limitations": [
            "This adapter is an advisory guardian research component, not a graduation judge.",
            "Guardian/verifier capability has not been trained or established.",
            "No reasoning improvement or structural generalization is established.",
            "The sham arm performed no training visits and was not compute matched.",
            "Training evaluation omitted an explicit generation attention mask.",
            "Reports are local receipts, not independent or signed attestation.",
        ],
    }


def package_model(run_dir: str | Path, reload_report: str | Path, output_dir: str | Path,
                  *, base_repository: str, base_revision: str) -> dict[str, Any]:
    """Create a distinctly named adapter package with safe, local lineage."""
    source = Path(run_dir).resolve(strict=True)
    reload_path = Path(reload_report).resolve(strict=True)
    destination_arg = Path(output_dir)
    if destination_arg.is_symlink() or destination_arg.exists():
        raise FileExistsError("Minotaur destination must not already exist")
    destination = destination_arg.resolve(strict=False)
    if not destination.parent.is_dir() or destination == source or source in destination.parents:
        raise ValueError("Minotaur package must use a new directory outside the training run")
    manifest = _manifest_payload(source, reload_path, base_repository, base_revision)
    source_adapter = source / "candidate_adapter"
    config = load_json(source_adapter / "adapter_config.json", strict=True)
    if not isinstance(config, dict):
        raise ValueError("invalid PEFT adapter configuration")
    destination.mkdir(mode=0o700)
    adapter_dir = destination / "adapter"
    adapter_dir.mkdir(mode=0o700)
    for file in sorted(source_adapter.iterdir()):
        if file.is_symlink() or not file.is_file():
            raise ValueError("Minotaur source adapter must contain only regular files")
        if file.name in ("README.md", "adapter_config.json"):
            continue
        shutil.copyfile(file, adapter_dir / file.name)
    # Never publish a workstation's absolute checkpoint directory inside PEFT.
    config["base_model_name_or_path"] = base_repository
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest["adapter_files_sha256"] = _adapter_manifest(adapter_dir)
    (destination / "minotaur_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        "# Minotaur — experimental neural guardian component\n\n"
        "This is a distinct PEFT/LoRA adapter trained from the pinned pretrained "
        f"[{base_repository}](https://huggingface.co/{base_repository}) base at `{base_revision}`. "
        "It is **not** a standalone merged GGUF or new base model.\n\n"
        "Role: advisory guardian research. **No authority to graduate or release trainees.** "
        "Only an independently qualified, deterministic gate may make such decisions.\n\n"
        "Qualification: **NOT ESTABLISHED**. Training targeted maze-answer completion, "
        "not judging trainee correctness. Future work must evaluate evidence critique, "
        "false passes and false rejects before any neural advisory use.\n\n"
        "Exact development results and lineage are in `minotaur_manifest.json`. "
        "No reasoning improvement was established; do not use this adapter as "
        "a production evaluator. The zero-update sham is not compute matched.\n",
        encoding="utf-8")
    return manifest


def inference(artifact_dir: str | Path, base_model_path: str | Path, prompt: str,
              *, max_new_tokens: int = 32) -> dict[str, str]:
    """Load Minotaur's pinned adapter and run advisory-only raw-format inference."""
    artifact = Path(artifact_dir).resolve(strict=True)
    manifest_file = artifact / "minotaur_manifest.json"
    _assert_file(manifest_file)
    manifest = load_json(manifest_file, strict=True)
    if (manifest.get("schema") != MODEL_SCHEMA or manifest.get("model_name") != MODEL_NAME
            or manifest.get("role") != MODEL_ROLE
            or manifest.get("graduation_authority") != "none"
            or manifest.get("qualification") != "NOT_ESTABLISHED"):
        raise ValueError("not a supported advisory-only Minotaur package")
    adapter_dir = artifact / "adapter"
    if _adapter_manifest(adapter_dir) != manifest.get("adapter_files_sha256"):
        raise ValueError("Minotaur adapter has changed")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
        raise ValueError("prompt must contain 1 to 8192 characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be in [1, 128]")
    model_path = Path(base_model_path).resolve(strict=True)
    config = PilotConfig(model_path=str(model_path), output_dir=str(artifact))
    _, _, weights = validate_config(config, output_must_be_new=False)
    if (_weights_digest(weights) != manifest["base_model"]["weight_sha256"]
            or _metadata_digest(model_path) != manifest["base_model"]["metadata_sha256"]):
        raise ValueError("Minotaur base checkpoint does not match pinned lineage")
    from .hf_pilot import _deps, _load_base
    torch, auto_model, auto_tokenizer, (_, peft_model, _), device = _deps("cpu")
    base, tokenizer = _load_base(model_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    model = peft_model.from_pretrained(base, str(adapter_dir), is_trainable=False,
                                      local_files_only=True)
    model.eval()
    formatted = f"Task:\n{prompt}\nAssistant:\n"
    encoded = tokenizer(formatted, add_special_tokens=True, return_tensors="pt")
    if encoded["input_ids"].shape[1] + max_new_tokens > 1024:
        raise ValueError("prompt exceeds model input budget")
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
    if _adapter_manifest(adapter_dir) != manifest["adapter_files_sha256"]:
        raise ValueError("Minotaur adapter changed during inference")
    return {"model_name": MODEL_NAME, "role": MODEL_ROLE,
            "graduation_authority": "none", "qualification": "NOT_ESTABLISHED",
            "response": response,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minotaur advisory neural model artifact")
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--run-dir", required=True)
    pack.add_argument("--reload-report", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--base-repository", required=True)
    pack.add_argument("--base-revision", required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--artifact-dir", required=True)
    infer.add_argument("--base-model-path", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.action == "package":
            result = package_model(args.run_dir, args.reload_report, args.output_dir,
                                   base_repository=args.base_repository,
                                   base_revision=args.base_revision)
            print(json.dumps({"model_name": result["model_name"],
                              "role": result["role"],
                              "graduation_authority": result["graduation_authority"],
                              "qualification": result["qualification"]}))
        else:
            print(json.dumps(inference(args.artifact_dir, args.base_model_path, args.prompt,
                                       max_new_tokens=args.max_new_tokens), sort_keys=True))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
