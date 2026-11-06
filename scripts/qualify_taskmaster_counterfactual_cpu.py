"""Read-only, post-hoc PUBLIC development counterfactual test of a saved taskmaster.

This deliberately adversarial fixture probes reliance on surface keywords. It is
NOT an independent final examination, a preregistered test, proof of semantic
generalization, or authorization to graduate. The public taskmaster's saved
pretrained-encoder/linear-policy artifacts are evaluated without optimization.

Run with ``python -m scripts.qualify_taskmaster_counterfactual_cpu``. The CLI
defaults to a no-write preflight; ``--execute`` enables local CPU inference and
creation of a separate, exclusive result directory. Network is disabled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any

from scripts import qualify_minotaur_neural_taskmaster_cpu as trainer


SCHEMA = "labyrinth.taskmaster-counterfactual-public-development.v1"
QUALIFICATION = "PUBLIC_POSTHOC_COUNTERFACTUAL_DEVELOPMENT_ONLY"
EXPECTED_REPORT_SCHEMA = trainer.SCHEMA
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUES = "forgotten half almost every fundamental failure"
_MEASURES = (
    # prerequisite, competing-method, currently retained, prior retention
    (2, 9, 9, 9),  # prerequisite weakness
    (9, 5, 9, 9),  # closely related methods confused
    (9, 9, 9, 9),  # ready for unfamiliar transfer
    (9, 9, 4, 9),  # previously mastered skill lost
)
_FAMILIES = (
    ("manufacturing", "Machine-shop training audit, milling tolerances: prerequisite "
     "{basic}/10 successful calibrations; neighboring-method selection {paired}/10; "
     "former skill last shift {before}/10 but current shift {now}/10. "
     "Which next training tactic follows the audited attempts?"),
    ("robotics", "Robotics motion-control student, obstacle-course lab dashboard. "
     "HISTORICAL command mastery: {before}/10. CURRENT command retention: {now}/10. "
     "Two similar navigation controllers differentiated: {paired}/10. "
     "Foundational motor steps: {basic}/10. Pick the next exercise difficulty."),
    ("archives", "Archives cataloging case memo: the clerk previously indexed "
     "{before}/10 records correctly and now retains {now}/10. For the next "
     "unfamiliar collection, method comparisons pass {paired}/10 while "
     "necessary lookup basics pass {basic}/10. Assign a learning corridor."),
    ("field_diagnostics", "Electrical field-diagnostics packet for unfamiliar boards. "
     "Question A, prerequisite fault isolation, correct {basic}/10. "
     "Question B, distinguish close circuit faults, correct {paired}/10. "
     "Question C, retained prior procedure: {now}/10 now versus "
     "{before}/10 before. What is the taskmaster's next tactic?"),
)


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    prompt: str
    gold: int
    prerequisite: int
    paired_method: int
    retention_now: int
    retention_before: int


def inferred_tactic(prerequisite: int, paired_method: int,
                   retention_now: int, retention_before: int) -> int:
    """Public, explicit decision rubric; historical regression takes priority."""
    vals = (prerequisite, paired_method, retention_now, retention_before)
    if any(type(x) is not int or not 0 <= x <= 10 for x in vals):
        raise ValueError("counterfactual evidence must be 0..10 integer counts")
    if retention_before >= 8 and retention_now < 6:
        return 3
    if prerequisite < 6:
        return 0
    if paired_method < 7:
        return 1
    return 2


def cases() -> tuple[Case, ...]:
    result = []
    for family, structure in _FAMILIES:
        for index, (basic, paired, now, before) in enumerate(_MEASURES):
            # Same contradictory lexical cues in EVERY class. Only the audited
            # counts change within a family; case IDs/labels never enter text.
            prompt = (
                structure.format(basic=basic, paired=paired, now=now, before=before)
                + " "
                f"An unrelated search index has the noisy words {_CUES}; those "
                "words describe no actual learner result."
            )
            result.append(Case(f"{family}_{index}", family, prompt, index,
                               basic, paired, now, before))
    return tuple(result)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_cases(records: tuple[Case, ...] | None = None) -> dict[str, Any]:
    records = cases() if records is None else records
    if len(records) != 16 or len({x.case_id for x in records}) != 16:
        raise ValueError("counterfactual development case count/IDs invalid")
    if {x.family for x in records} != {name for name, _ in _FAMILIES}:
        raise ValueError("counterfactual family coverage invalid")
    if tuple(sum(x.gold == i for x in records) for i in range(4)) != (4, 4, 4, 4):
        raise ValueError("counterfactual tactic labels unbalanced")
    existing = {normalize(prompt) for split in trainer.fixtures().values()
                for prompt, _ in split}
    observed: set[str] = set()
    per_family: dict[str, list[Case]] = {}
    for case in records:
        if (not isinstance(case, Case) or not isinstance(case.prompt, str)
                or not 100 < len(case.prompt) <= 800
                or any(cue not in normalize(case.prompt) for cue in _CUES.split())
                or inferred_tactic(case.prerequisite, case.paired_method,
                                  case.retention_now, case.retention_before) != case.gold):
            raise ValueError("invalid counterfactual prompt/evidence/gold")
        norm = normalize(case.prompt)
        if norm in observed or norm in existing:
            raise ValueError("duplicate public development/training prompt")
        observed.add(norm)
        per_family.setdefault(case.family, []).append(case)
    for family, family_cases in per_family.items():
        if (len(family_cases) != 4 or {x.gold for x in family_cases} != set(range(4))
                or {tuple((x.prerequisite, x.paired_method, x.retention_now,
                           x.retention_before)) for x in family_cases} != set(_MEASURES)):
            raise ValueError(f"invalid four-way minimal-pair family {family}")
        # Only metric strings and labels may vary across a family's prompts.
        stripped = set()
        for case in family_cases:
            text = re.sub(r"\b(?:10|[0-9])/10\b", "?/10", case.prompt)
            stripped.add(text)
        if len(stripped) != 1:
            raise ValueError(f"non-minimal counterfactual prompts in {family}")
    commitment = hashlib.sha256("\n".join(
        f"{case.case_id}|{case.gold}|{normalize(case.prompt)}" for case in records
    ).encode("utf-8")).hexdigest()
    return {"count": 16, "family_count": 4, "tactic_counts": [4, 4, 4, 4],
            "fixture_commitment_sha256": commitment}


def keyword_baseline(prompt: str) -> int:
    """Post-hoc naive classifier; intentionally not a matched frozen-model arm."""
    p = normalize(prompt)
    if "forgotten" in p or "deteriorated" in p or "declined" in p:
        return 3
    if "half" in p or "confus" in p:
        return 1
    if "almost every" in p or "rising" in p:
        return 2
    return 0


def _validated_file(path: Path) -> None:
    if (not path.is_file() or path.is_symlink()
            or (hasattr(path, "is_junction") and path.is_junction())
            or path.stat().st_size == 0):
        raise ValueError(f"missing, linked, or empty input artifact: {path}")


def validate_training_receipt(training_run_dir: str | Path,
                              expected_report_sha256: str,
                              model_id: str = trainer.MODEL_ID) -> dict[str, Any]:
    """Require externally pinned bytes; do not accept a self-pinning report."""
    if not isinstance(expected_report_sha256, str) or not _SHA256.fullmatch(expected_report_sha256):
        raise ValueError("explicit lowercase SHA-256 training report pin is required")
    root = Path(training_run_dir)
    if (not root.is_dir() or root.is_symlink()
            or (hasattr(root, "is_junction") and root.is_junction())):
        raise ValueError("training run root must be an existing nonlinked directory")
    report_path = root / "report.json"
    _validated_file(report_path)
    if report_path.stat().st_size > (1 << 20):
        raise ValueError("unexpectedly large training report")
    if trainer._hash_file(report_path) != expected_report_sha256:
        raise ValueError("training report digest mismatches explicit pin")
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if (not isinstance(report, dict) or report.get("schema") != EXPECTED_REPORT_SCHEMA
            or report.get("model_id") != model_id or report.get("device") != "cpu"
            or report.get("tactics") != list(trainer.TACTICS)
            or report.get("graduation_authorized") is not False
            or report.get("independent_exam") != "NOT_RUN"
            or report.get("model_execution_authenticated") is not False
            or not isinstance(report.get("frozen_encoder_sha256"), str)
            or not _SHA256.fullmatch(report["frozen_encoder_sha256"])
            or not isinstance(report.get("arms"), dict)
            or set(report["arms"]) != set(trainer.ARMS)):
        raise ValueError("unexpected or escalated training receipt")
    checkpoints: dict[str, str] = {}
    for arm in trainer.ARMS:
        directory = root / arm
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"invalid arm directory: {arm}")
        arm_data = report["arms"][arm]
        if not isinstance(arm_data, dict) or not isinstance(arm_data.get("cycles"), list):
            raise ValueError(f"missing cycle receipts: {arm}")
        cycles = arm_data["cycles"]
        if len(cycles) != 2 or [entry.get("cycle") for entry in cycles] != [0, 1]:
            raise ValueError(f"unexpected training cycles: {arm}")
        last = cycles[-1]
        if (not isinstance(last, dict)
                or not isinstance(last.get("after_sha256"), str)
                or not _SHA256.fullmatch(last["after_sha256"])
                or not isinstance(last.get("checkpoint_sha256"), str)
                or not _SHA256.fullmatch(last["checkpoint_sha256"])):
            raise ValueError(f"unverifiable final head: {arm}")
        checkpoint = directory / "cycle1.safetensors"
        _validated_file(checkpoint)
        if trainer._hash_file(checkpoint) != last["checkpoint_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {arm}")
        checkpoints[arm] = last["checkpoint_sha256"]
    return {"root": root.resolve(strict=True), "report": report,
            "report_sha256": expected_report_sha256, "checkpoint_sha256": checkpoints}


def plan(training_run_dir: str | Path, output_parent: str | Path,
         run_name: str, expected_report_sha256: str,
         model_id: str = trainer.MODEL_ID) -> dict[str, Any]:
    observed = validate_training_receipt(training_run_dir, expected_report_sha256, model_id)
    target = trainer.plan_output(output_parent, run_name)
    training_root = observed["root"]
    if target.is_relative_to(training_root) or training_root.is_relative_to(target):
        raise ValueError("counterfactual output must be isolated from training")
    observed["result_root"] = target
    observed["cases"] = validate_cases()
    return observed


def execute(checked: dict[str, Any], *, model_id: str = trainer.MODEL_ID) -> dict[str, Any]:
    """Perform only frozen CPU forward inference, with before/after state checks."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder_before = trainer._tensor_digest(sorted(encoder.named_parameters()))
    if encoder_before != checked["report"]["frozen_encoder_sha256"]:
        raise ValueError("frozen pretrained encoder digest mismatch")
    rows = cases()
    with torch.no_grad():
        tokenized = tokenizer([x.prompt for x in rows], padding=True,
                              truncation=True, max_length=96, return_tensors="pt")
        output = encoder(**tokenized)
        mask = tokenized["attention_mask"].unsqueeze(-1)
        embeddings = ((output.last_hidden_state * mask).sum(dim=1)
                      / mask.sum(dim=1)).detach().contiguous()
    gold = torch.tensor([x.gold for x in rows], dtype=torch.int64)
    counts: dict[str, int] = {}
    predictions: dict[str, list[int]] = {}
    head_digests: dict[str, str] = {}
    for arm in trainer.ARMS:
        checkpoint = checked["root"] / arm / "cycle1.safetensors"
        state = load_file(str(checkpoint), device="cpu")
        if set(state) != {"weight", "bias"}:
            raise ValueError(f"unexpected policy state keys: {arm}")
        head = torch.nn.Linear(encoder.config.hidden_size, len(trainer.TACTICS))
        head.load_state_dict(state, strict=True)
        head.eval()
        head_before = trainer._tensor_digest(sorted(head.named_parameters()))
        expected = checked["report"]["arms"][arm]["cycles"][-1]["after_sha256"]
        if head_before != expected:
            raise ValueError(f"policy parameter digest mismatch: {arm}")
        with torch.no_grad():
            predicted = head(embeddings).argmax(dim=-1)
        counts[arm] = int((predicted == gold).sum().item())
        predictions[arm] = [int(x) for x in predicted.tolist()]
        head_digests[arm] = head_before
        if trainer._tensor_digest(sorted(head.named_parameters())) != head_before:
            raise RuntimeError(f"evaluation mutated saved policy: {arm}")
    if trainer._tensor_digest(sorted(encoder.named_parameters())) != encoder_before:
        raise RuntimeError("evaluation mutated pretrained encoder")
    checked_again = validate_training_receipt(checked["root"], checked["report_sha256"], model_id)
    if checked_again["checkpoint_sha256"] != checked["checkpoint_sha256"]:
        raise RuntimeError("checkpoints changed during counterfactual evaluation")
    baseline = sum(keyword_baseline(x.prompt) == x.gold for x in rows)
    report = {
        "schema": SCHEMA, "qualification": QUALIFICATION,
        "model_id": model_id, "model_origin": checked["report"].get("model_origin"),
        "training_report_sha256": checked["report_sha256"],
        "training_checkpoint_sha256": checked["checkpoint_sha256"],
        "encoder_sha256": encoder_before, "policy_head_sha256": head_digests,
        "cases": checked["cases"], "gold_labels": [x.gold for x in rows],
        "case_ids": [x.case_id for x in rows], "predictions": predictions,
        "correct_counts": counts, "keyword_baseline_correct": baseline,
        "total": len(rows), "graduation_authorized": False,
        "independent_exam": "NOT_RUN", "model_execution_authenticated": False,
        "evaluation_only": True, "optimizer_updates": 0,
        "interpretation": "Post-hoc public adversarial development probe; no independent generalization claim",
    }
    root = checked["result_root"]
    root.mkdir(mode=0o700)
    trainer._write_json(root / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-dir", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--model-id", default=trainer.MODEL_ID)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        checked = plan(args.training_run_dir, args.output_parent, args.run_name,
                       args.expected_report_sha256, args.model_id)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "result_root": str(checked["result_root"]),
                              "training_report_sha256": checked["report_sha256"],
                              "cases": checked["cases"],
                              "graduation_authorized": False}, sort_keys=True))
            return 0
        report = execute(checked, model_id=args.model_id)
        print(json.dumps({"schema": SCHEMA, "execute": True,
                          "result_root": str(checked["result_root"]),
                          "result_sha256": trainer._hash_file(checked["result_root"] / "report.json"),
                          "correct_counts": report["correct_counts"],
                          "keyword_baseline_correct": report["keyword_baseline_correct"],
                          "graduation_authorized": False}, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"counterfactual preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
