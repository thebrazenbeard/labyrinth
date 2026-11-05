"""Fail-closed, source-only lineage for regenerated Minotaur HF development.

This module deliberately does NOT launch neural training. The existing HF
pilot's adaptive source and resume auditor cannot replay multi-cycle policy,
optimizer state, or sealed arithmetic history together. This v1 records an
exact local TRAIN proposal so that a future version can bridge that gap
without quietly erasing custody or overstating evidence.

All examination fields are identity commitments only. This is neither an
independent evaluator nor an authorized private qualification path.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .hf_adaptive_source_v1 import ADAPTIVE_SCHEMA, _budget, arithmetic_semantic_atoms
from .hf_pilot import PilotConfig
from .hf_resume_bridge import ResumeContext
from .journey import JourneyState, TRAINING, _validate_state
from .journey_policy_runner import TrainingCorridorRunner
from .maze import Challenge
from .minotaur_hf_roster_bridge import (
    MinotaurHfTrainRoster, _source_arithmetic, _source_arithmetic_atoms,
)
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.hf-development-only-successor-source.v1"
_DOMAIN = b"labyrinth.hf-development-only-successor-source.v1\x00"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class DevelopmentOnlyNeuralExecutionBlocked(ValueError):
    """Source preflight is not a resumable neural artifact or authority."""


def _canonical(value: Any) -> bytes:
    result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(result) > 262144:
        raise ValueError("development source exceeds bounded manifest size")
    return result


def _fingerprint(manifest: dict[str, Any]) -> str:
    """Operator-local consistency digest, never a signature or witness."""
    return hashlib.sha256(
        _DOMAIN + _canonical({k: v for k, v in manifest.items()
                              if k != "source_sha256"}),
    ).hexdigest()


def _distinct_atoms(source: Any) -> list[list[Any]]:
    if type(source) is not frozenset or not 1 <= len(source) <= 3:
        raise ValueError("missing TRAIN arithmetic-semantic atom history")
    return [list(atom) for atom in sorted(source)]


def _predecessor(
    scope: TrainingScopeSpec, journey: JourneyState,
    context: ResumeContext, predecessor_report_bytes: bytes,
) -> dict[str, Any]:
    if type(predecessor_report_bytes) is not bytes or not 0 < len(predecessor_report_bytes) <= 8 * 1024 * 1024:
        raise ValueError("predecessor report must be bounded exact local bytes")
    if hashlib.sha256(predecessor_report_bytes).hexdigest() != context.predecessor_report_sha256:
        raise ValueError("predecessor report digest differs from local pin")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("predecessor report contains duplicate JSON fields")
            result[key] = value
        return result

    try:
        report = json.loads(predecessor_report_bytes, object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("predecessor report is not strict JSON") from exc
    if (type(report) is not dict or report.get("schema") != ADAPTIVE_SCHEMA
            or report.get("candidate_lora_parameter_sha256") != context.predecessor_lora_value_sha256
            or report.get("candidate_lora_tensor_value_sha256") != context.predecessor_lora_value_sha256
            or report.get("training_case_ids") != list(journey.attempts[-1].training_case_ids)
            or report.get("training_prompt_sha256") != list(journey.attempts[-1].training_prompt_sha256)
            or report.get("scope_fingerprint") != scope.fingerprint
            or report.get("cycle_index") != journey.cycle_index - 1):
        raise ValueError("predecessor adaptive report is not bound to TRAIN lineage and retained neural checkpoint")
    return report


def _rooms(roster: MinotaurHfTrainRoster) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    known_prompts: set[str] = set()
    known_atoms: set[tuple[str, int, int]] = set()
    if len(roster.adaptive_feedback_rows) != len(roster.train_cases):
        raise ValueError("successor TRAIN feedback must cover every staged room")
    for case, feedback in zip(roster.train_cases, roster.adaptive_feedback_rows, strict=True):
        if (type(case.id) is not str or not case.id.startswith("train-taskmaster-")
                or case.split != "train" or type(feedback) is not tuple or len(feedback) != 6
                or any(type(value) is not str for index, value in enumerate(feedback) if index != 3)
                or type(feedback[3]) is not int or feedback[3] not in (0, 1)):
            raise ValueError("only exact scored Minotaur TRAIN rooms may enter development")
        room_id, case_id, prompt_sha, score, before, after = feedback
        actual_hash = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (case_id != case.id or prompt_sha != actual_hash
                or not all(_SHA.fullmatch(value) for value in (prompt_sha, before, after))
                or case.id in known_ids or prompt_sha in known_prompts):
            raise ValueError("successor TRAIN identity, commitment or feedback changed")
        atoms = arithmetic_semantic_atoms(case)
        if atoms & known_atoms:
            raise ValueError("successor TRAIN repeats an arithmetic semantic atom")
        known_ids.add(case.id)
        known_prompts.add(prompt_sha)
        known_atoms.update(atoms)
        rows.append({
            "room_id": room_id, "case_id": case.id, "prompt": case.prompt,
            "prompt_sha256": prompt_sha, "skill": "arithmetic", "score": score,
            "checkpoint_before_sha256": before, "checkpoint_after_sha256": after,
            "semantic_atoms": _distinct_atoms(atoms),
        })
    for current, following in zip(rows, rows[1:]):
        if current["checkpoint_after_sha256"] != following["checkpoint_before_sha256"]:
            raise ValueError("successor TRAIN room checkpoint chain is broken")
    return rows


def _historical_training_prompts(
    journey: JourneyState, witnesses: Any,
) -> tuple[list[dict[str, str]], list[list[list[Any]]]]:
    """Replay prior *public TRAIN* prompts; never use an exam prompt or gold.

    A case-ID/hash-only history cannot detect semantic arithmetic reuse. This
    explicit prompt witness closes that development-source gap without
    exposing any development or private exam content.
    """
    ids = journey.used_training_case_ids
    digests = journey.used_training_prompt_sha256
    if (type(witnesses) is not tuple or len(witnesses) != len(ids)
            or not 1 <= len(witnesses) <= 2048):
        raise ValueError("complete prior public TRAIN prompt witnesses required")
    payload: list[dict[str, str]] = []
    atoms: list[list[list[Any]]] = []
    history_seen: set[tuple[str, int, int]] = set()
    for witness, case_id, digest in zip(witnesses, ids, digests, strict=True):
        if (type(witness) is not tuple or len(witness) != 2
                or any(type(part) is not str for part in witness)
                or witness[0] != case_id
                or not case_id.startswith("train-taskmaster-")
                or not 0 < len(witness[1].encode("utf-8")) <= 512
                or hashlib.sha256(witness[1].encode("utf-8")).hexdigest() != digest):
            raise ValueError("historical TRAIN public prompt differs from sealed journey")
        challenge = Challenge(case_id, "arithmetic", 1, "train", witness[1], "")
        # Derive the correct gold locally and discard it; only public TRAIN
        # prompt text and semantic atoms enter this manifest.
        from .hf_adaptive_source_v1 import _CONTRAST
        kind = "arithmetic_contrast" if _CONTRAST.fullmatch(witness[1]) else "arithmetic"
        challenge = Challenge(case_id, kind, 1, "train", witness[1], "")
        gold, _ = _source_arithmetic(challenge)
        challenge = Challenge(case_id, kind, 1, "train", witness[1], gold)
        computed = _source_arithmetic_atoms(challenge)
        if computed & history_seen:
            raise ValueError("historical TRAIN reuses a previously exposed arithmetic atom")
        history_seen.update(computed)
        payload.append({"case_id": case_id, "prompt": witness[1], "prompt_sha256": digest})
        atoms.append(_distinct_atoms(computed))
    return payload, atoms


def _common(
    config: PilotConfig, scope: TrainingScopeSpec, journey: JourneyState,
    roster: MinotaurHfTrainRoster, context: ResumeContext,
    predecessor_report_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _budget(config)
    if (type(scope) is not TrainingScopeSpec or type(journey) is not JourneyState
            or type(roster) is not MinotaurHfTrainRoster
            or type(context) is not ResumeContext):
        raise ValueError("exact local development lineage types required")
    TrainingScopeSpec.__post_init__(scope)
    _validate_state(journey)
    ResumeContext.__post_init__(context)
    if (scope != journey.scope or journey.phase != TRAINING
            or journey.cycle_index < 1 or not journey.attempts
            or journey.attempts[-1].exam_status != "FAIL"
            or journey.cycle_index != context.cycle_index
            or scope.fingerprint != context.scope_fingerprint
            or roster.cycle_index != context.cycle_index
            or roster.scope_sha256 != scope.fingerprint
            or len(roster.train_cases) != config.train_count
            or context.allow_legacy_unscoped is not False
            or journey.checkpoint_sha256 != context.predecessor_lora_value_sha256
            or roster.initial_checkpoint_sha256 != context.predecessor_lora_value_sha256):
        raise ValueError("retained neural checkpoint, scope or failed development predecessor is unbound")
    predecessor = _predecessor(scope, journey, context, predecessor_report_bytes)
    rooms = _rooms(roster)
    if (any(room["case_id"] in journey.used_training_case_ids + journey.sealed_exam_case_ids
            or room["prompt_sha256"] in (journey.used_training_prompt_sha256
                                           + journey.sealed_exam_prompt_sha256)
            for room in rooms)
            or rooms[0]["checkpoint_before_sha256"] != context.predecessor_lora_value_sha256):
        raise ValueError("successor TRAIN overlaps sealed development cases or predecessor checkpoint")
    from .hf_pilot import _training_visit_exposures
    expected_exposures = _training_visit_exposures(
        list(roster.train_cases), seed=config.seed, steps=config.steps,
    )
    if roster.ordered_optimizer_exposures != expected_exposures:
        raise ValueError("successor TRAIN optimizer exposures changed")
    return rooms, predecessor


def prepare_local_successor_source(
    config: PilotConfig, scope: TrainingScopeSpec, journey: JourneyState,
    runner: TrainingCorridorRunner, roster: MinotaurHfTrainRoster,
    context: ResumeContext, predecessor_report_bytes: bytes,
    previous_public_training_prompts: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Snapshot a bounded, explicitly non-executable multi-cycle TRAIN source.

    The caller must hold the *real* regenerated operator runner, not construct
    a new one by fabricating future neural-weight digests. This prevents
    accidental promotion of an unlinked source prototype into an HF run.
    """
    rooms, _ = _common(config, scope, journey, roster, context, predecessor_report_bytes)
    previous_prompts, previous_atoms = _historical_training_prompts(
        journey, previous_public_training_prompts,
    )
    if (type(runner) is not TrainingCorridorRunner or not runner.is_complete
            or runner.state.scope != scope or runner.state.plan.cycle_index != context.cycle_index
            or runner.state.initial_checkpoint_sha256 != context.predecessor_lora_value_sha256
            or runner.state.pending is not None or runner.policy is None
            or len(runner.state.history) != len(rooms)
            or len(runner.observations) != len(rooms)):
        raise ValueError("exact complete regenerated TRAIN policy runner required")
    historical = runner._semantic_history
    approvals = runner.training_registry
    if (len(historical) != len(approvals)
            or len(historical) < len(rooms)
            or [entry.case_id for entry in approvals[-len(rooms):]]
               != [room["case_id"] for room in rooms]):
        raise ValueError("successor taskmaster admission and arithmetic history are incomplete")
    prior_atoms = set().union(*historical[:-len(rooms)])
    if [_distinct_atoms(atoms) for atoms in historical[:-len(rooms)]] != previous_atoms:
        raise ValueError("operator policy history differs from public TRAIN prompt witnesses")
    for index, (source_row, event, observation, atoms) in enumerate(zip(
        rooms, runner.state.history, runner.observations, historical[-len(rooms):], strict=True,
    )):
        if (event.feedback.case_id != source_row["case_id"]
                or event.feedback.prompt_sha256 != source_row["prompt_sha256"]
                or event.feedback.score != source_row["score"]
                or event.feedback.checkpoint_before_sha256 != source_row["checkpoint_before_sha256"]
                or event.feedback.checkpoint_after_sha256 != source_row["checkpoint_after_sha256"]
                or observation.case_id != source_row["case_id"]
                or observation.room_id != source_row["room_id"]
                or source_row["semantic_atoms"] != _distinct_atoms(atoms)
                or atoms & prior_atoms):
            raise ValueError(f"successor TRAIN policy or semantic history diverged at room {index}")
    manifest = {
        "schema": SCHEMA, "mode": "DEVELOPMENT_ONLY", "cycle_index": context.cycle_index,
        "scope_sha256": scope.fingerprint, "predecessor_report_sha256": context.predecessor_report_sha256,
        "predecessor_checkpoint_sha256": context.predecessor_lora_value_sha256,
        "sealed_development_exams": list(journey.sealed_exam_case_ids),
        "sealed_development_prompt_sha256": list(journey.sealed_exam_prompt_sha256),
        "historical_training_prompts": previous_prompts,
        "historical_train_semantic_atoms": previous_atoms,
        "training_rooms": rooms, "operator_policy_sha256": runner.policy.model_hash,
        "ordered_optimizer_exposures": [list(pair) for pair in roster.ordered_optimizer_exposures],
        "neural_execution": "BLOCKED_PENDING_ARTIFACT_REPLAY_BRIDGE",
        "qualification": "NOT_ESTABLISHED", "graduation_authorized": False,
    }
    manifest["source_sha256"] = _fingerprint(manifest)
    verify_local_successor_source(
        config, scope, journey, runner, roster, context, predecessor_report_bytes, manifest,
    )
    return manifest


def verify_local_successor_source(
    config: PilotConfig, scope: TrainingScopeSpec, journey: JourneyState,
    runner: TrainingCorridorRunner, roster: MinotaurHfTrainRoster, context: ResumeContext,
    predecessor_report_bytes: bytes, manifest: Any,
) -> str:
    """Replay a local TRAIN proposal, not the actual neural model state."""
    rooms, _ = _common(config, scope, journey, roster, context, predecessor_report_bytes)
    if (type(runner) is not TrainingCorridorRunner or not runner.is_complete
            or runner.policy is None or runner.state.scope != scope
            or runner.state.plan.cycle_index != context.cycle_index
            or runner.state.initial_checkpoint_sha256 != context.predecessor_lora_value_sha256
            or len(runner.observations) != len(rooms)
            or [approval.case_id for approval in runner.training_registry[-len(rooms):]]
               != [room["case_id"] for room in rooms]):
        raise ValueError("exact operator taskmaster runner is required for policy replay")
    if (type(manifest) is not dict
            or set(manifest) != {
                "schema", "mode", "cycle_index", "scope_sha256", "predecessor_report_sha256",
                "predecessor_checkpoint_sha256", "sealed_development_exams",
                "sealed_development_prompt_sha256", "historical_train_semantic_atoms",
                "historical_training_prompts",
                "training_rooms", "operator_policy_sha256", "ordered_optimizer_exposures",
                "neural_execution", "qualification", "graduation_authorized", "source_sha256",
            }
            or manifest["schema"] != SCHEMA or manifest["mode"] != "DEVELOPMENT_ONLY"
            or manifest["neural_execution"] != "BLOCKED_PENDING_ARTIFACT_REPLAY_BRIDGE"
            or manifest["qualification"] != "NOT_ESTABLISHED"
            or manifest["graduation_authorized"] is not False
            or manifest["cycle_index"] != context.cycle_index
            or manifest["scope_sha256"] != scope.fingerprint
            or manifest["predecessor_report_sha256"] != context.predecessor_report_sha256
            or manifest["predecessor_checkpoint_sha256"] != context.predecessor_lora_value_sha256
            or manifest["sealed_development_exams"] != list(journey.sealed_exam_case_ids)
            or manifest["sealed_development_prompt_sha256"] != list(journey.sealed_exam_prompt_sha256)
            or manifest["training_rooms"] != rooms
            or manifest["ordered_optimizer_exposures"] != [list(p) for p in roster.ordered_optimizer_exposures]
            or manifest["operator_policy_sha256"] != runner.policy.model_hash
            or type(manifest["source_sha256"]) is not str
            or manifest["source_sha256"] != _fingerprint(manifest)):
        raise ValueError("development-only successor source fields, custody or route do not replay")
    prior = manifest["historical_training_prompts"]
    if (type(prior) is not list or any(type(item) is not dict
            or set(item) != {"case_id", "prompt", "prompt_sha256"} for item in prior)):
        raise ValueError("development-only historical TRAIN prompt witness is malformed")
    replayed_prompts, prior_atoms = _historical_training_prompts(
        journey, tuple((row["case_id"], row["prompt"]) for row in prior),
    )
    if (replayed_prompts != prior
            or manifest["historical_train_semantic_atoms"] != prior_atoms):
        raise ValueError("development-only historical arithmetic provenance differs from public TRAIN")
    flattened = {tuple(atom) for row in prior_atoms for atom in row}
    if (len(flattened) != sum(len(row) for row in prior_atoms)
            or flattened & {tuple(atom) for row in rooms for atom in row["semantic_atoms"]}):
        raise ValueError("development-only successor repeats a historical TRAIN semantic atom")
    return "SOURCE_PRECHECK_ONLY"


def run_local_neural_successor(_source: Any) -> None:
    """Deliberate hard stop until pinned *actual* adapter replay exists."""
    raise DevelopmentOnlyNeuralExecutionBlocked(
        "multi-cycle adaptive TRAIN requires audited predecessor adapter bytes, "
        "replayable prior policy/history, and qualified LOCAL_WEIGHTS_ONLY HF controls; "
        "a source digest or arbitrary future checkpoint hash cannot authorize neural training"
    )
