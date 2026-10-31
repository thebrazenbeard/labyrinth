"""Opt-in, source-verified measured-development TRAIN schedule binding.

The scheduling evidence is reverified from signed development receipts before
the resulting cycle-one TRAIN batches can be committed for all five arms. This
is deliberately *not* host key custody, actual neural execution, a persistent
one-shot reservation, or a private graduation credential. A host must pin the
schedule digest and the checkpoint/fence independently before model loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

from scripts.qualify_maze_conditioned_minilm_cpu import (
    PREREG_ARMS, TrainingCycle, TrainingExample, validate_cycles,
    plan_signed_measured_sentiment_batches,
)
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.bound-measured-neural-batches.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _sha(value: Any, name: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {name} digest")


def _run(value: Any) -> None:
    if type(value) is not str or _RUN.fullmatch(value) is None:
        raise ValueError("invalid candidate run")


def _validate_batches(
    batches: Any, cycle: TrainingCycle, steps_per_cycle: int,
) -> tuple[tuple[str, ...], ...]:
    if (type(steps_per_cycle) is not int
            or not 4 <= steps_per_cycle <= 16
            or type(batches) is not tuple or len(batches) != steps_per_cycle):
        raise ValueError("measured schedule missing a complete step inventory")
    roster = {row.exposure_id: row for row in cycle.examples}
    if len(roster) != len(cycle.examples):
        raise ValueError("nonunique regenerated TRAIN roster")
    visited: set[str] = set()
    order = []
    for batch in batches:
        if (type(batch) is not tuple or len(batch) != 4
                or any(type(row) is not TrainingExample
                       or roster.get(row.exposure_id) != row for row in batch)):
            raise ValueError("measured batch contains non-TRAIN or altered rows")
        ids = tuple(row.exposure_id for row in batch)
        if len(set(ids)) != 4:
            raise ValueError("measured batch repeats one TRAIN exposure")
        visited.update(ids)
        order.append(ids)
    if visited != set(roster):
        raise ValueError("measured batches omit authentic regenerated TRAIN examples")
    return tuple(order)


def _canonical_digest(
    *, run_id: str, candidate_artifact_sha256: str,
    predecessor_checkpoint_sha256: str, scope_fingerprint: str,
    fence: int, source_event_id: str, source_exam_id: str,
    weak_skill: str, tactic: str, steps_per_cycle: int,
    cycle: TrainingCycle, ordered_visits: tuple[tuple[str, ...], ...],
) -> str:
    data = {
        "schema": SCHEMA, "run_id": run_id,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "predecessor_checkpoint_sha256": predecessor_checkpoint_sha256,
        "scope_fingerprint": scope_fingerprint, "fence": fence,
        "source_event_id": source_event_id, "source_exam_id": source_exam_id,
        "weak_skill": weak_skill, "tactic": tactic,
        "steps_per_cycle": steps_per_cycle,
        "cycle_index": cycle.plan.cycle_index,
        "regenerated_training_roster": [row.exposure_id for row in cycle.examples],
        "arms": list(PREREG_ARMS), "ordered_visits": ordered_visits,
    }
    return sha256((json.dumps(data, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True)
class BoundMeasuredNeuralBatches:
    """One exact immutable cycle-one schedule, shared across all controls."""

    run_id: str
    candidate_artifact_sha256: str
    predecessor_checkpoint_sha256: str
    scope_fingerprint: str
    fence: int
    source_event_id: str
    source_exam_id: str
    weak_skill: str
    tactic: str
    steps_per_cycle: int
    cycle: TrainingCycle
    batches: tuple[tuple[TrainingExample, ...], ...]
    schedule_sha256: str
    arms: tuple[str, ...] = PREREG_ARMS
    host_signature_attested: bool = False
    model_execution_authenticated: bool = False
    graduation_authorized: bool = False

    def batch_for_arm(self, arm: str, step: int) -> tuple[TrainingExample, ...]:
        if type(arm) is not str or arm not in PREREG_ARMS:
            raise ValueError("only the five precommitted control arms are admitted")
        if type(step) is not int or not 0 <= step < self.steps_per_cycle:
            raise ValueError("invalid step for committed measured TRAIN schedule")
        return self.batches[step]


def propose_signed_neural_batch_schedule(
    *, declared_scope: TrainingScopeSpec,
    training_cycles: tuple[TrainingCycle, ...], steps_per_cycle: int,
    binding: Any, training_exposures: tuple[Any, ...],
    expected_training_roster_sha256: str,
    candidate_run_id: str, candidate_artifact_sha256: str,
    predecessor_checkpoint_sha256: str, fence: int,
    **verification_inputs: Any,
) -> BoundMeasuredNeuralBatches:
    """Derive only from signed four-key development and actual TRAIN inventory.

    The separately supplied predecessor digest/fence are checked against the
    signed context and then covered by a schedule commitment. The commitment
    must be pinned outside the trainer; computing it here does not attest that
    any external authority has done so.
    """
    _run(candidate_run_id)
    _sha(candidate_artifact_sha256, "candidate artifact")
    _sha(predecessor_checkpoint_sha256, "predecessor checkpoint")
    if type(fence) is not int or not 0 <= fence < 2**63:
        raise ValueError("invalid monotonic fence")
    context = verification_inputs.get("context")
    if (getattr(context, "predecessor_lora_value_sha256", None)
            != predecessor_checkpoint_sha256):
        raise ValueError("candidate/predecessor identity differs from signed lineage")
    # Reexecute the signed-data verifier, not an untrusted feedback dataclass.
    planned = plan_signed_measured_sentiment_batches(
        declared_scope=declared_scope, training_cycles=training_cycles,
        steps_per_cycle=steps_per_cycle, binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        candidate_run_id=candidate_run_id,
        candidate_artifact_sha256=candidate_artifact_sha256,
        **verification_inputs,
    )
    if planned.source_signature_host_attested or planned.model_execution_authenticated:
        raise ValueError("source result cannot claim host-attested execution")
    order = _validate_batches(planned.batches, training_cycles[1], steps_per_cycle)
    digest = _canonical_digest(
        run_id=candidate_run_id,
        candidate_artifact_sha256=candidate_artifact_sha256,
        predecessor_checkpoint_sha256=predecessor_checkpoint_sha256,
        scope_fingerprint=declared_scope.fingerprint, fence=fence,
        source_event_id=planned.source_event_id,
        source_exam_id=planned.source_exam_id,
        weak_skill=planned.weak_skill, tactic=planned.tactic,
        steps_per_cycle=steps_per_cycle, cycle=training_cycles[1],
        ordered_visits=order,
    )
    return BoundMeasuredNeuralBatches(
        run_id=candidate_run_id,
        candidate_artifact_sha256=candidate_artifact_sha256,
        predecessor_checkpoint_sha256=predecessor_checkpoint_sha256,
        scope_fingerprint=declared_scope.fingerprint, fence=fence,
        source_event_id=planned.source_event_id,
        source_exam_id=planned.source_exam_id, weak_skill=planned.weak_skill,
        tactic=planned.tactic, steps_per_cycle=steps_per_cycle,
        cycle=training_cycles[1], batches=planned.batches, schedule_sha256=digest,
    )


def verify_precommitted_neural_batch_schedule(
    schedule: BoundMeasuredNeuralBatches, *,
    training_cycles: tuple[TrainingCycle, ...],
    expected_schedule_sha256: str,
    expected_run_id: str, expected_candidate_artifact_sha256: str,
    expected_predecessor_checkpoint_sha256: str,
    expected_scope_fingerprint: str, expected_fence: int,
) -> BoundMeasuredNeuralBatches:
    """Fail before model load when an externally pinned schedule mismatches."""
    # The caller has to pin the digest outside the trainer. A local dataclass
    # identity check is not evidence of such custody or of a past signature.
    if type(schedule) is not BoundMeasuredNeuralBatches:
        raise ValueError("typed source schedule required")
    _sha(expected_schedule_sha256, "precommitted schedule")
    _run(expected_run_id)
    _sha(expected_candidate_artifact_sha256, "candidate artifact")
    _sha(expected_predecessor_checkpoint_sha256, "predecessor checkpoint")
    _sha(expected_scope_fingerprint, "scope")
    if (type(expected_fence) is not int or not 0 <= expected_fence < 2**63
            or schedule.run_id != expected_run_id
            or schedule.candidate_artifact_sha256 != expected_candidate_artifact_sha256
            or schedule.predecessor_checkpoint_sha256 != expected_predecessor_checkpoint_sha256
            or schedule.scope_fingerprint != expected_scope_fingerprint
            or training_cycles[1].plan.scope_fingerprint != expected_scope_fingerprint
            or schedule.fence != expected_fence
            or schedule.arms != PREREG_ARMS
            or schedule.cycle != training_cycles[1]
            or schedule.host_signature_attested is not False
            or schedule.model_execution_authenticated is not False
            or schedule.graduation_authorized is not False):
        raise ValueError("schedule identity, control arms or authority flags mismatch")
    validate_cycles(training_cycles)
    visits = _validate_batches(schedule.batches, training_cycles[1],
                               schedule.steps_per_cycle)
    digest = _canonical_digest(
        run_id=schedule.run_id,
        candidate_artifact_sha256=schedule.candidate_artifact_sha256,
        predecessor_checkpoint_sha256=schedule.predecessor_checkpoint_sha256,
        scope_fingerprint=schedule.scope_fingerprint, fence=schedule.fence,
        source_event_id=schedule.source_event_id,
        source_exam_id=schedule.source_exam_id,
        weak_skill=schedule.weak_skill, tactic=schedule.tactic,
        steps_per_cycle=schedule.steps_per_cycle, cycle=schedule.cycle,
        ordered_visits=visits,
    )
    if digest != schedule.schedule_sha256 or digest != expected_schedule_sha256:
        raise ValueError("TRAIN schedule commitment differs from pinned signed plan")
    return schedule
