"""Dependency-free *internal* audit for HF pilot receipts, not model attestation.

Short saved responses may be rescored. The 128-character preview format cannot
fully verify longer generated answers, so those records are INCOMPLETE.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .evaluation import score_response
from .hf_pilot import (
    LEGACY_SCHEMA, SCHEMA, PilotConfig,
    RESEARCH_ONLY_RESUME, SIGNED_DEVELOPMENT_RESUME, _prompt_commitment,
    _arithmetic_semantic_commitments, _historical_arithmetic_case_count,
    _arithmetic_history_witnesses, _verified_arithmetic_history_witnesses,
    _scoped_training_cases, _training_visit_indices, derange_answers, training_cases,
)
from .runner import _source_state
from .training_scope import TrainingScopeSpec

AUDIT_SCHEMA = "labyrinth.hf-pilot-audit.v1"
ARMS = ("baseline", "sham", "candidate", "corrupted")
CONFIG_FIELDS = frozenset({
    "model_path", "output_dir", "seed", "train_count", "eval_count", "tiers", "steps",
    "max_tokens", "max_new_tokens", "learning_rate", "lora_rank", "lora_alpha", "device",
})
MATCHED_CONFIG_FIELDS = CONFIG_FIELDS | {"answer_only"}
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

    if not isinstance(report, dict) or report.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        fail("invalid HF pilot report schema")
        return result()
    matched = report["schema"] == SCHEMA
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
    if matched and not _digest(report.get("initial_persistent_buffers_sha256")):
        fail("v3 persistent-buffer digest missing or malformed")
    if "initial_frozen_parameter_sha256" in report and not _digest(
            report["initial_frozen_parameter_sha256"]):
        fail("frozen-parameter identity digest malformed")
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
    if (set(config) != (MATCHED_CONFIG_FIELDS if matched else CONFIG_FIELDS)
            or (matched and type(config.get("answer_only")) is not bool)
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
            or not _integer(neval, 2, 32) or not _integer(steps, 1, 128 if matched else 32)
            or not isinstance(tiers, list) or not tiers or len(tiers) > 4
            or len(set(map(str, tiers))) != len(tiers)
            or any(type(t) is not int or t not in (1, 2, 3, 4) for t in tiers)
            or len(tiers) * ntrain > 64):
        fail("invalid seed/count/steps/tiers in pilot config")
        return result()
    journey = report.get("resumed_journey")
    try:
        if journey is None:
            # Match the pilot's bounded semantic split regeneration. A fixed
            # eval seed would falsely reject a legitimate rerolled heldout
            # report whenever commutative TRAIN/EVAL expressions collide.
            train, heldout = training_cases(PilotConfig(
                **{**config, "tiers": tuple(tiers)},
            ))
        else:
            if not matched or type(journey) is not dict:
                raise ValueError("scoped receipt requires an explicit v3 journey object")
            scope = TrainingScopeSpec.from_dict(journey.get("scope"))
            cycle = journey.get("cycle_index")
            weak = journey.get("weak_skills")
            if (journey.get("scope_fingerprint") != scope.fingerprint
                    or type(cycle) is not int or not 0 <= cycle < scope.max_cycles
                    or type(weak) is not list
                    or any(type(skill) is not str or skill not in scope.skills for skill in weak)
                    or len(set(weak)) != len(weak)):
                raise ValueError("scoped objective/cycle/weakness receipt is inconsistent")
            pilot = PilotConfig(**{**config, "tiers": tuple(tiers)})
            train, heldout, plan = _scoped_training_cases(pilot, scope, cycle, tuple(weak))
            if journey.get("training_room_ids") != [room.room_id for room in plan.rooms]:
                raise ValueError("scoped training room identities disagree with regenerated maze")
            historical_ids = journey.get("historical_case_ids")
            historical_prompts = journey.get("historical_prompt_sha256")
            historical_arithmetic = journey.get("historical_arithmetic_sha256")
            historical_witnesses = journey.get("historical_arithmetic_witnesses_v1")
            historical_bindings = journey.get("historical_arithmetic_case_bindings_v1")
            historical_binding_root = journey.get("historical_arithmetic_binding_root_v1")
            current_case_ids = {case.id for case in train + heldout}
            current_prompt_hashes = {
                hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
                for case in train + heldout
            }
            current_arithmetic = _arithmetic_semantic_commitments(train + heldout)
            if (type(historical_ids) is not list or type(historical_prompts) is not list
                    or len(historical_ids) > 4096 or len(historical_prompts) > 4096
                    or len(historical_ids) != len(historical_prompts)
                    or any(type(case_id) is not str or not 1 <= len(case_id) <= 128
                           for case_id in historical_ids)
                    or any(not _digest(prompt) for prompt in historical_prompts)
                    or len(set(historical_ids)) != len(historical_ids)
                    or len(set(historical_prompts)) != len(historical_prompts)
                    or not current_case_ids.issubset(historical_ids)
                    or not current_prompt_hashes.issubset(historical_prompts)
                    or journey.get("same_initial_lora_for_all_controls") is not True):
                raise ValueError("scoped history lacks current TRAIN/public diagnostic commitments")
            if (type(historical_arithmetic) is not list
                    or len(historical_arithmetic) > 4096
                    or any(not _digest(digest) for digest in historical_arithmetic)
                    or len(set(historical_arithmetic)) != len(historical_arithmetic)
                    or not current_arithmetic.issubset(historical_arithmetic)
                    or len(historical_arithmetic) !=
                    _historical_arithmetic_case_count(historical_ids)):
                raise ValueError("scoped semantic arithmetic history missing, malformed or incomplete")
            _verified_arithmetic_history_witnesses(
                historical_witnesses, historical_ids, historical_prompts,
                historical_arithmetic, historical_bindings, historical_binding_root,
            )
            if historical_bindings is None or historical_binding_root is None:
                raise ValueError("scoped arithmetic case-to-prompt binding is missing")
            if cycle == 0:
                if journey.get("development_resume_authority") is not None:
                    raise ValueError("initial cycle cannot claim successor authorization")
                if (set(historical_ids) != current_case_ids
                        or set(historical_prompts) != current_prompt_hashes
                        or set(historical_arithmetic) != current_arithmetic
                        or historical_witnesses != _arithmetic_history_witnesses(train + heldout)):
                    raise ValueError("initial scoped history contains unexplained prior evidence")
            else:
                authority = journey.get("development_resume_authority")
                if authority == RESEARCH_ONLY_RESUME:
                    incomplete("successor explicitly used unauthenticated research-only authorization")
                elif authority == SIGNED_DEVELOPMENT_RESUME:
                    incomplete("signed development authorization is self-reported, not independently attested")
                else:
                    incomplete("legacy successor lacks recognized development-transition authority")
                incomplete("successor history cannot be authenticated without independent predecessor custody")
        corrupted_labels = derange_answers(train)
    except (ValueError, TypeError, KeyError) as exc:
        fail(f"cannot reproduce training/evaluation controls: {exc}")
        return result()
    if report.get("training_case_ids") != [c.id for c in train]:
        fail("training case ids differ from regenerated train split")
    if report.get("heldout_case_ids") != [c.id for c in heldout]:
        fail("heldout case ids differ from regenerated evaluation split")
    if report.get("heldout_public_prompt_sha256") != _prompt_commitment(heldout):
        fail("heldout prompt commitment mismatch")
    controls = report.get("control_data")
    common_control = (
        isinstance(controls, dict)
        and controls.get("corrupted_label_collisions")
            == sum(c.answer == label for c, label in zip(train, corrupted_labels))
        and type(controls.get("corrupted_label_collisions")) is int
        and controls.get("sham_updates") == 0
        and type(controls.get("sham_updates")) is int
    )
    if matched:
        control_valid = (
            common_control
            and controls.get("sham_optimizer_steps") == steps
            and type(controls.get("sham_optimizer_steps")) is int
            and controls.get("sham_train_case_visits") == steps
            and type(controls.get("sham_train_case_visits")) is int
            and controls.get("sham_evaluation_before_updates") is False
            and controls.get("sham_is_compute_matched") is True
            and controls.get("answer_scaffold_supplied") is config.get("answer_only")
        )
    else:
        control_valid = (
            common_control
            and controls.get("sham_train_case_visits") == 0
            and type(controls.get("sham_train_case_visits")) is int
            and controls.get("sham_evaluation_before_updates") is True
            and controls.get("sham_is_compute_matched") is False
        )
    if not control_valid:
        fail("control metadata inconsistent with declared sham/corrupted protocol")
    training = report.get("training")
    arms = ("sham", "candidate", "corrupted") if matched else ("candidate", "corrupted")
    if not isinstance(training, dict) or set(training) != set(arms):
        fail("training arms missing or mislabeled")
    else:
        if matched:
            try:
                expected_indices = _training_visit_indices(train, seed=seed, steps=steps)
            except ValueError as exc:
                fail(f"invalid tier visit schedule: {exc}")
                expected_indices = []
        else:
            expected_indices = [i % len(train) for i in range(steps)]
        expected_visits = [train[i].id for i in expected_indices]
        for arm in arms:
            record = training[arm]
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
    return result()
