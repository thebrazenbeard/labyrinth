"""Fail-closed custody audit for the public MiniLM-to-Journey integration gap.

This command does not train a model or run an examination. It validates the
existing opt-in CPU experiment's exact prompt exposure, checkpoint continuity,
matched controls, and saved bytes; then replays *one* training-only Journey
transition. Its inferred room assignments are explicitly NOT observed room
dispatch. In particular it cannot manufacture an independently scored exam,
a signed development FAIL, maze regeneration, or graduation authority.

Run with ``python -m scripts.qualify_neural_journey_e2e_cpu --run-dir PATH``.
Exit 3 means the report is internally coherent but the complete journey is
unqualified. Exit 2 means the report or saved-byte custody failed validation.
Reported aggregate control superiority is distinguished from independently
verified learning: the latter cannot PASS in this source-only auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from labyrinth.journey import AWAITING_EXAM, TrainingCompletionReceipt, begin_journey, complete_maze
from labyrinth.training_scope import TrainingScopeSpec
from scripts import qualify_minilm_cpu as minilm


SCHEMA = "labyrinth.neural-journey-custody-audit.v1"
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _actual_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint file missing or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_sha(root: Path, arm: str, cycle: int, *parts: str) -> str:
    # An arm/cycle directory symlink must not redirect byte checks outside the
    # operator-selected run. Checking just the final file is insufficient.
    parent = root
    for piece in (arm, f"cycle-{cycle}", *parts):
        parent = parent / piece
        if parent.is_symlink():
            raise ValueError("checkpoint path contains a symbolic link")
    return _actual_sha(parent)


def _expected_visits(rows: tuple[tuple[str, int], ...], steps: int) -> list[str]:
    return [hashlib.sha256(rows[(step * 4 + offset) % len(rows)][0].encode()).hexdigest()
            for step in range(steps) for offset in range(4)]


def _scope(seed: int) -> TrainingScopeSpec:
    # Two training skills, six rooms per skill, exactly twelve unique cases.
    return TrainingScopeSpec("public_sentiment_development", "sentiment",
                             ("negative", "positive"), 2, 3, 2, seed)


def audit_run(report: dict[str, Any], *, run_dir: Path | None = None) -> dict[str, Any]:
    """Verify public development lineage; never claim completed neural journey.

    ``run_dir=None`` only performs source/report checks. A real run's local
    checkpoint-byte digests must also be verified before asserting byte custody.
    No report field is treated as independent evidence of pretrained origin.
    """
    _require(type(report) is dict and report.get("schema") == minilm.SCHEMA,
             "unsupported MiniLM report schema")
    _require(report.get("qualification") == "PUBLIC_DEVELOPMENT_ONLY"
             and report.get("graduation_authorized") is False
             and report.get("private_final_exam") == "NOT_RUN",
             "report is not a non-graduating public development experiment")
    _require(report.get("physical_model_byte_custody") == "SELF_REPORTED",
             "unrecognized model-byte custody claim")
    _require(report.get("device") == "cpu" and
             report.get("pretrained_origin") == "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
             "unsupported device or pretrained model provenance claim")
    _require(type(report.get("seed")) is int and 0 <= report["seed"] < 2**31,
             "invalid training seed")
    _require(type(report.get("steps_per_cycle")) is int
             and 3 <= report["steps_per_cycle"] <= 32,
             "at least three steps are required to visit twelve cases per cycle")
    _require(report.get("fixture_sizes") == minilm.validate_fixtures(),
             "reported fixture sizes differ from declared public fixtures")
    _require(report.get("harness_sha256") == _actual_sha(Path(minilm.__file__)),
             "the MiniLM harness changed since the report was produced")
    _require(type(report.get("arms")) is dict and set(report["arms"]) == set(minilm.ARMS),
             "the three matched experimental arms are required")

    scope = _scope(report["seed"])
    initial = None
    initial_lora = None
    for arm in minilm.ARMS:
        arm_report = report["arms"][arm]
        _require(type(arm_report) is dict and type(arm_report.get("cycles")) is list
                 and len(arm_report["cycles"]) == 2, "missing arm cycles")
        _digest(arm_report.get("initial_trainable_sha256"), "initial trainable state")
        _digest(arm_report.get("initial_lora_sha256"), "initial LoRA state")
        if initial is None:
            initial = arm_report["initial_trainable_sha256"]
            initial_lora = arm_report["initial_lora_sha256"]
        _require(arm_report["initial_trainable_sha256"] == initial
                 and arm_report["initial_lora_sha256"] == initial_lora,
                 "matched arms have different initial trainable weights")
        prior = None
        prior_lora = None
        for cycle, rows in enumerate(minilm.TRAIN):
            record = arm_report["cycles"][cycle]
            _require(type(record) is dict and record.get("cycle") == cycle,
                     "cycle index mismatch")
            for field in ("trainable_sha256_before", "trainable_sha256_after",
                          "lora_sha256_before", "lora_sha256_after",
                          "adapter_file_sha256", "head_file_sha256"):
                _digest(record.get(field), field)
            before, after = record["trainable_sha256_before"], record["trainable_sha256_after"]
            lora_before, lora_after = record["lora_sha256_before"], record["lora_sha256_after"]
            _require(before == (initial if cycle == 0 else prior)
                     and lora_before == (initial_lora if cycle == 0 else prior_lora),
                     "learner checkpoint or LoRA adapter did not persist into next cycle")
            _require((before != after and lora_before != lora_after) if arm != "zero_update"
                     else (before == after and lora_before == lora_after),
                     "parameter mutation contradicts declared training/control arm")
            expected_updates = 0 if arm == "zero_update" else report["steps_per_cycle"]
            _require(record.get("optimizer_updates") == expected_updates
                     and record.get("forward_steps") == report["steps_per_cycle"]
                     and record.get("backward_steps") == report["steps_per_cycle"],
                     "optimizer/forward/backward counts disagree")
            _require(record.get("training_prompt_sha256") ==
                     _expected_visits(rows, report["steps_per_cycle"]),
                     "training exposure diverges from the frozen public fixture schedule")
            for score_name in ("heldout_accuracy_before", "heldout_accuracy", "transfer_accuracy"):
                score = record.get(score_name)
                _require(type(score) in (int, float) and 0 <= score <= 1,
                         f"invalid {score_name}")
            if run_dir is not None:
                _require(_checkpoint_sha(run_dir, arm, cycle, "adapter", "adapter_model.safetensors") ==
                         record["adapter_file_sha256"], "saved LoRA file digest mismatch")
                _require(_checkpoint_sha(run_dir, arm, cycle, "head.safetensors") ==
                         record["head_file_sha256"], "saved head file digest mismatch")
            prior, prior_lora = after, lora_after

    for split in ("heldout", "transfer"):
        comparison = minilm.comparison({arm: report["arms"][arm]["cycles"][1][
            "heldout_accuracy" if split == "heldout" else "transfer_accuracy"]
            for arm in minilm.ARMS})
        _require(report.get(split) == comparison, f"{split} control comparison mismatch")
    _require(type(report.get("retention")) is dict
             and report["retention"] == {
                 arm: report["arms"][arm]["cycles"][1]["heldout_accuracy"]
                 - report["arms"][arm]["cycles"][0]["heldout_accuracy"]
                 for arm in minilm.ARMS}, "retention calculation mismatch")

    # A deliberately bounded source replay, NOT evidence of true room dispatch.
    state = begin_journey(scope, checkpoint_sha256=initial)
    records = report["arms"]["correct"]["cycles"]
    room_case_ids: list[tuple[str, str, str]] = []
    for skill, label in (("negative", 0), ("positive", 1)):
        rooms = [room for room in state.maze.rooms if room.skill == skill]
        candidates = [(idx, prompt) for idx, (prompt, actual) in
                      enumerate(minilm.TRAIN[0]) if actual == label]
        _require(len(rooms) == len(candidates), "maze cannot cover frozen training cases")
        for room, (index, prompt) in zip(rooms, candidates):
            room_case_ids.append((room.room_id, f"public.train.c0.case{index}",
                                  hashlib.sha256(prompt.encode()).hexdigest()))
    state = complete_maze(state, TrainingCompletionReceipt(
        scope.objective_id, 0, records[0]["trainable_sha256_before"],
        records[0]["trainable_sha256_after"], tuple(room_case_ids)))
    _require(state.phase == AWAITING_EXAM, "source replay failed to reach exam boundary")
    # The original report's arithmetic has been checked, but its aggregate
    # correctness and physical causal attribution have NOT been independently
    # examined. Thus failure to beat either control on either reported split
    # can fail the specified comparison, whereas a reported win remains UNKNOWN
    # as independently verified learning evidence.
    reported_advantage = {
        split: "PASS" if report[split]["correct_outperforms_both"] else "FAIL"
        for split in ("heldout", "transfer")
    }
    reported_advantage["joint"] = (
        "PASS" if all(reported_advantage[split] == "PASS"
                      for split in ("heldout", "transfer")) else "FAIL"
    )
    return {
        "schema": SCHEMA,
        "result": "HOLD_INCOMPLETE_NEURAL_JOURNEY",
        "public_report_checks": "PASS",
        "public_report_checks_meaning":
            "SCHEMA_EXPOSURE_LINEAGE_ARITHMETIC_ONLY_NOT_LEARNING",
        "reported_control_advantage": reported_advantage,
        "control_benefit": (
            "FAIL" if reported_advantage["joint"] == "FAIL" else "UNKNOWN"
        ),
        "control_benefit_basis":
            "PUBLIC_SELF_REPORTED_AGGREGATES_WITHOUT_INDEPENDENT_SCORING",
        "local_checkpoint_bytes": "VERIFIED_HASHES" if run_dir is not None else "NOT_CHECKED",
        "scope_fingerprint": scope.fingerprint,
        "source_replay_phase": state.phase,
        "mapped_training_rooms": len(room_case_ids),
        "physical_neural_updates": "SOURCE_REPORT_ONLY_NOT_INDEPENDENTLY_ATTESTED",
        "journey_room_dispatch": "UNKNOWN_POSTHOC_SOURCE_MAPPING_ONLY",
        "development_case_scores": "MISSING_REPORT_ONLY_CONTAINS_AGGREGATES",
        "signed_development_failure": "MISSING",
        "maze_regeneration": "NOT_RUN_NO_AUTHENTICATED_DEVELOPMENT_FAILURE",
        "private_final_exam": "NOT_RUN",
        "graduation_authorized": False,
        "gates": {"declaration": "PASS_SOURCE", "generation": "PASS_SOURCE",
                  "experience": "UNKNOWN", "learning": "SOURCE_REPORT_ONLY",
                  "confront": "NOT_RUN", "evaluate": "NOT_RUN",
                  "regenerate": "NOT_RUN", "graduate": "NOT_AUTHORIZED"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Existing MiniLM report.json and three-arm checkpoint directory")
    args = parser.parse_args(argv)
    try:
        root = args.run_dir
        if root.is_symlink() or not root.is_dir():
            raise ValueError("run directory is absent or linked")
        report_path = root / "report.json"
        if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size > 1_048_576:
            raise ValueError("report missing, linked or unreasonably large")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = audit_run(report, run_dir=root)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 3  # A coherent public probe is deliberately not a qualified full loop.
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"schema": SCHEMA, "result": "FAIL_SOURCE_OR_BYTE_CUSTODY",
                          "reason": str(exc), "graduation_authorized": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
