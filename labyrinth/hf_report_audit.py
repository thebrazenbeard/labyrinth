"""Dependency-free *internal* audit for HF pilot receipts, not model attestation.

Short saved responses may be rescored. The 128-character preview format cannot
fully verify longer generated answers, so those records are INCOMPLETE.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
import math
import re
from typing import Any

from .evaluation import score_response
from .hf_pilot import EVAL_SEED_XOR, SCHEMA, _prompt_commitment, derange_answers
from .maze import generate_maze
from .runner import _source_state

AUDIT_SCHEMA = "labyrinth.hf-pilot-audit.v1"
ARMS = ("baseline", "sham", "candidate", "corrupted")
CONFIG_FIELDS = frozenset({
    "model_path", "output_dir", "seed", "train_count", "eval_count", "tiers", "steps",
    "max_tokens", "max_new_tokens", "learning_rate", "lora_rank", "lora_alpha", "device",
})
PILOT_KIND = "local HF causal-LM PEFT LoRA supervised parameter update"
RELOAD_NOTE = "NOT RUN: execute `python -m labyrinth.hf_pilot eval` separately"


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _proportion(value: Any, expected: float) -> bool:
    return (type(value) in (int, float) and 0 <= value <= 1
            and math.isfinite(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-12))


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest_map(value: Any) -> bool:
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(name, str) and bool(name) and _digest(digest)
                    for name, digest in value.items()))


def audit_hf_report(report: Any) -> dict[str, Any]:
    failures: list[str] = []
    unknown: list[str] = []
    checked_trials = 0
    unverifiable_trials = 0

    def fail(message: str) -> None:
        if len(failures) < 40:
            failures.append(message)

    def incomplete(message: str) -> None:
        if message not in unknown and len(unknown) < 40:
            unknown.append(message)

    def result() -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEMA,
            "status": "FAIL" if failures else "INCOMPLETE" if unknown else "PASS",
            "meaning": "internal_consistency_only", "authenticated_model_execution": False,
            "checked_trials": checked_trials, "unverifiable_trials": unverifiable_trials,
            "failures": failures, "unchecked": unknown,
            "limitations": [
                "Self-reported training, optimizer steps, hashes and model execution are unauthenticated.",
                "A self-consistent fabricated pilot report can pass this audit without neural inference.",
                "PASS is neither checkpoint verification nor evidence of learned reasoning.",
            ],
        }

    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        fail("invalid HF pilot report schema")
        return result()
    source = report.get("source")
    if not isinstance(source, dict) or not _digest(source.get("source_tree_sha256")):
        fail("missing or malformed HF pilot source fingerprint")
    elif source["source_tree_sha256"] != _source_state()["source_tree_sha256"]:
        incomplete("source fingerprint mismatch: current generator/scorer is not pinned to historical report")
    for field in ("base_weights_sha256", "model_metadata_sha256", "candidate_adapter_sha256"):
        digests = report.get(field)
        has_required_file = (
            "config.json" in digests if field == "model_metadata_sha256"
            else any(name.endswith(".safetensors") for name in digests)
        ) if isinstance(digests, dict) else False
        if not _digest_map(digests) or not has_required_file:
            fail(f"{field}: missing or malformed file digest receipt")
    for field in ("initial_lora_parameter_sha256", "candidate_lora_parameter_sha256",
                  "candidate_lora_tensor_value_sha256", "corrupted_lora_parameter_sha256"):
        if not _digest(report.get(field)):
            fail(f"{field}: missing or malformed tensor digest receipt")
    initial = report.get("initial_lora_parameter_sha256")
    candidate = report.get("candidate_lora_parameter_sha256")
    if _digest(initial) and _digest(candidate) and initial == candidate:
        fail("candidate LoRA value digest indicates no update")
    if (_digest(candidate) and _digest(report.get("candidate_lora_tensor_value_sha256"))
            and candidate != report["candidate_lora_tensor_value_sha256"]):
        fail("trained and persisted LoRA value digest mismatch")
    if not _integer(report.get("training_process_id"), 1, 2**63 - 1):
        fail("missing or invalid training process id")
    config = report.get("config")
    if not isinstance(config, dict):
        fail("invalid pilot config")
        return result()
    if (set(config) != CONFIG_FIELDS
            or type(config.get("model_path")) is not str or not config["model_path"]
            or type(config.get("output_dir")) is not str or not config["output_dir"]
            or config.get("device") not in ("cpu", "cuda")
            or not _integer(config.get("max_tokens"), 32, 1024)
            or not _integer(config.get("max_new_tokens"), 1, 32)
            or not _integer(config.get("lora_rank"), 1, 16)
            or not _integer(config.get("lora_alpha"), 1, 64)
            or type(config.get("learning_rate")) is not float
            or not math.isfinite(config["learning_rate"])
            or not 1e-7 <= config["learning_rate"] <= 0.001):
        fail("missing or invalid local HF model, decoding, or optimizer config")
        return result()
    versions = report.get("runtime_versions")
    if (report.get("adaptation_kind") != PILOT_KIND
            or report.get("fresh_process_reload") != RELOAD_NOTE
            or not isinstance(versions, dict)
            or set(versions) != {"python", "torch", "transformers", "peft"}
            or any(type(v) is not str or not v or len(v) > 200 for v in versions.values())):
        fail("missing or invalid reported adaptation kind, runtime versions, or reload status")
    seed, ntrain, neval, tiers, steps = (
        config.get("seed"), config.get("train_count"), config.get("eval_count"),
        config.get("tiers"), config.get("steps"),
    )
    if (not _integer(seed, 0, 2**31 - 1) or not _integer(ntrain, 2, 32)
            or not _integer(neval, 2, 32) or not _integer(steps, 1, 32)
            or not isinstance(tiers, list) or not tiers or len(tiers) > 4
            or len(set(map(str, tiers))) != len(tiers)
            or any(type(t) is not int or t not in (1, 2, 3, 4) for t in tiers)
            or len(tiers) * ntrain > 64):
        fail("invalid seed/count/steps/tiers in pilot config")
        return result()
    try:
        train = generate_maze(seed=seed, count=ntrain, tiers=tuple(tiers), split="train")
        heldout = generate_maze(seed=seed ^ EVAL_SEED_XOR, count=neval, tiers=tuple(tiers), split="eval")
        corrupted_labels = derange_answers(train)
    except (ValueError, TypeError) as exc:
        fail(f"cannot reproduce training/evaluation controls: {exc}")
        return result()
    if report.get("training_case_ids") != [c.id for c in train]:
        fail("training case ids differ from regenerated train split")
    if report.get("heldout_case_ids") != [c.id for c in heldout]:
        fail("heldout case ids differ from regenerated evaluation split")
    if report.get("heldout_public_prompt_sha256") != _prompt_commitment(heldout):
        fail("heldout prompt commitment mismatch")
    controls = report.get("control_data")
    if not isinstance(controls, dict) or (
        controls.get("corrupted_label_collisions") != sum(c.answer == label for c, label in zip(train, corrupted_labels))
        or type(controls.get("corrupted_label_collisions")) is not int
        or controls.get("sham_updates") != 0 or type(controls.get("sham_updates")) is not int
        or controls.get("sham_train_case_visits") != 0 or type(controls.get("sham_train_case_visits")) is not int
        or controls.get("sham_evaluation_before_updates") is not True
        or controls.get("sham_is_compute_matched") is not False
    ):
        fail("control metadata inconsistent with declared unmatched-sham/corrupted protocol")
    training = report.get("training")
    if not isinstance(training, dict) or set(training) != {"candidate", "corrupted"}:
        fail("training arms missing or mislabeled")
    else:
        for arm in ("candidate", "corrupted"):
            record = training[arm]
            expected_visits = [train[i % len(train)].id for i in range(steps)]
            if not isinstance(record, dict) or record.get("visited_case_ids") != expected_visits:
                fail(f"{arm}: training visit order mismatch")
                continue
            if (record.get("steps") != steps or type(record.get("steps")) is not int
                    or record.get("example_visits") != steps or type(record.get("example_visits")) is not int
                    or not _integer(record.get("input_tokens_seen"), 1, 100000)
                    or not isinstance(record.get("losses"), list)
                    or len(record["losses"]) != steps
                    or any(type(v) not in (float, int) or v < 0 or v > 1e12 or not math.isfinite(v)
                           for v in record["losses"])):
                fail(f"{arm}: invalid declared training steps/tokens/losses")
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != set(ARMS):
        fail("expected baseline/sham/candidate/corrupted evaluations")
        return result()
    for arm in ARMS:
        summary = evaluations[arm]
        if not isinstance(summary, dict) or not isinstance(summary.get("trials"), list) or len(summary["trials"]) != len(heldout):
            fail(f"{arm}: invalid trial count")
            continue
        reported_correct = 0
        reported_format = 0
        for index, (trial, challenge) in enumerate(zip(summary["trials"], heldout)):
            tag = f"{arm}[{index}]"
            if not isinstance(trial, dict):
                fail(f"{tag}: invalid trial")
                continue
            if trial.get("id") != challenge.id:
                fail(f"{tag}: challenge id mismatch")
            response = trial.get("response_preview")
            digest = trial.get("response_sha256")
            if not isinstance(response, str) or len(response) > 128 or not _digest(digest):
                fail(f"{tag}: invalid response preview or digest")
                continue
            if type(trial.get("correct")) is not bool or trial.get("score_reason") not in (
                "correct", "incorrect", "invalid_format", "invalid_response"
            ):
                fail(f"{tag}: invalid score/format claim")
                continue
            reported_correct += int(trial["correct"])
            reported_format += int(trial["score_reason"] in ("correct", "incorrect"))
            try:
                preview_bytes = response.encode("utf-8")
            except UnicodeEncodeError:
                fail(f"{tag}: response preview contains invalid Unicode")
                continue
            preview_digest = hashlib.sha256(preview_bytes).hexdigest()
            if preview_digest != digest and len(response) == 128:
                unverifiable_trials += 1
                incomplete(f"{arm}: truncated model response cannot be independently rescored")
                continue
            if preview_digest != digest:
                fail(f"{tag}: response digest mismatch")
                continue
            rescored = score_response(challenge, response)
            if (trial["correct"] != rescored.correct
                    or trial["score_reason"] != rescored.reason):
                fail(f"{tag}: score differs from regenerated evaluator")
            checked_trials += 1
        if summary.get("total") != len(heldout) or type(summary.get("total")) is not int:
            fail(f"{arm}: declared total mismatch")
        if summary.get("correct") != reported_correct or type(summary.get("correct")) is not int:
            fail(f"{arm}: declared correct count mismatch")
        if summary.get("format_valid") != reported_format or type(summary.get("format_valid")) is not int:
            fail(f"{arm}: declared format count mismatch")
        if not _proportion(summary.get("accuracy"), reported_correct / len(heldout)):
            fail(f"{arm}: accuracy mismatch")
        if not _proportion(summary.get("format_rate"), reported_format / len(heldout)):
            fail(f"{arm}: format rate mismatch")

    # The optional diagnostic probe is explicitly *training-set* evidence;
    # never fold these counts into held-out wins or graduation thresholds.
    if "training_probe" in report:
        probes = report["training_probe"]
        if not isinstance(probes, dict) or set(probes) != {"candidate", "corrupted"}:
            fail("training probe must contain candidate and corrupted arms")
            return result()
        for arm in ("candidate", "corrupted"):
            summary = probes[arm]
            if (not isinstance(summary, dict) or not isinstance(summary.get("trials"), list)
                    or len(summary["trials"]) != len(train)):
                fail(f"training probe {arm}: invalid trial count")
                continue
            correct = 0
            formatted = 0
            target_hits = 0
            for index, (trial, challenge) in enumerate(zip(summary["trials"], train)):
                tag = f"training probe {arm}[{index}]"
                if not isinstance(trial, dict) or trial.get("id") != challenge.id:
                    fail(f"{tag}: train challenge id mismatch")
                    continue
                response = trial.get("response_preview")
                digest = trial.get("response_sha256")
                if not isinstance(response, str) or len(response) > 128 or not _digest(digest):
                    fail(f"{tag}: invalid response preview or digest")
                    continue
                if type(trial.get("correct")) is not bool or trial.get("score_reason") not in (
                    "correct", "incorrect", "invalid_format", "invalid_response"
                ):
                    fail(f"{tag}: invalid score claim")
                    continue
                correct += int(trial["correct"])
                formatted += int(trial["score_reason"] in ("correct", "incorrect"))
                if arm == "corrupted":
                    if type(trial.get("training_target_correct")) is not bool:
                        fail(f"{tag}: missing or invalid corrupted training target claim")
                        continue
                    target_hits += int(trial["training_target_correct"])
                elif "training_target_correct" in trial:
                    fail(f"{tag}: unexpected candidate alternate training target")
                try:
                    checksum = hashlib.sha256(response.encode("utf-8")).hexdigest()
                except UnicodeEncodeError:
                    fail(f"{tag}: invalid Unicode in response")
                    continue
                if checksum != digest and len(response) == 128:
                    unverifiable_trials += 1
                    incomplete(f"training probe {arm}: truncated response cannot be rescored")
                    continue
                if checksum != digest:
                    fail(f"{tag}: response digest mismatch")
                    continue
                scored = score_response(challenge, response)
                if trial["correct"] != scored.correct or trial["score_reason"] != scored.reason:
                    fail(f"{tag}: oracle score differs from regenerated evaluator")
                if arm == "corrupted":
                    target = replace(challenge, answer=corrupted_labels[index])
                    if trial["training_target_correct"] != score_response(target, response).correct:
                        fail(f"{tag}: training target claim differs from deranged supervision")
                checked_trials += 1
            if summary.get("total") != len(train) or type(summary.get("total")) is not int:
                fail(f"training probe {arm}: total mismatch")
            if summary.get("correct") != correct or type(summary.get("correct")) is not int:
                fail(f"training probe {arm}: oracle correct count mismatch")
            if summary.get("format_valid") != formatted or type(summary.get("format_valid")) is not int:
                fail(f"training probe {arm}: format valid count mismatch")
            if not _proportion(summary.get("accuracy"), correct / len(train)):
                fail(f"training probe {arm}: oracle accuracy mismatch")
            if not _proportion(summary.get("format_rate"), formatted / len(train)):
                fail(f"training probe {arm}: format rate mismatch")
            if arm == "corrupted":
                if summary.get("training_target_hits") != target_hits or type(summary.get("training_target_hits")) is not int:
                    fail("corrupted training target hits mismatch")
            elif "training_target_hits" in summary:
                fail("candidate training probe must not claim alternate target hits")
    return result()
