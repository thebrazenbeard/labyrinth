"""Opt-in, signed development FAIL -> Minotaur tactic + persistent Journey.

This compositor passes the *same* verified development event and pinned TRAIN
roster to two source-level consumers: a scored Minotaur diagnostic and a
checkpoint-preserving regenerated Journey. It can delegate the successor to
the existing opt-in HF pilot, which owns its own predecessor preflight and
host-supplied event-reservation callback.

It does NOT independently possess signing keys or event-reservation authority.
The candidate artifact SHA identifies file bytes while the Journey checkpoint
SHA identifies LoRA tensor values; their correspondence requires host custody.
Neither an authentic receipt nor these source tests prove live model execution,
causal learning, protected final examinations, or any graduation permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .hf_development_transition import run_journey_bound_successor
from .hf_pilot import PilotConfig
from .journey import JourneyState
from .measured_corridor_handoff import (
    MeasuredCorridorHandoff, advance_from_measured_development_failure,
)
from .measured_weakness_feedback import TrainingExposureRef
from .minotaur_development_feedback import (
    MinotaurDevelopmentFeedback, derive_minotaur_development_feedback,
)
from .signed_exam_lineage import SignedExamLineageBinding


@dataclass(frozen=True)
class MeasuredTaskmasterTransition:
    handoff: MeasuredCorridorHandoff
    feedback: MinotaurDevelopmentFeedback
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    host_event_consumption_verified: bool = False


def prepare_measured_taskmaster_transition(
    state: JourneyState,
    *,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> MeasuredTaskmasterTransition:
    """Derive one TRAIN-only Minotaur tactic and regenerating Maze from a score.

    The two derived values are intentionally re-verified from the same exact
    signed inputs and checked for identity before either is released. There
    are no input parameters for graduate status, arbitrary weak-skill labels,
    or private final-exam content. This function performs no stateful event
    consumption; a hosting runtime must prevent clone-state replay.
    """
    common = {
        "binding": binding,
        "training_exposures": training_exposures,
        "expected_training_roster_sha256": expected_training_roster_sha256,
        **verification_inputs,
    }
    handoff = advance_from_measured_development_failure(state, **common)
    feedback = derive_minotaur_development_feedback(**common)
    if (handoff.remediation != feedback.remediation
            or handoff.remediation.exam_id != feedback.exam_id
            or handoff.remediation.event_id != feedback.event_id
            or handoff.remediation.from_cycle != feedback.from_cycle
            or handoff.remediation.to_cycle != feedback.to_cycle
            or handoff.remediation.maze != handoff.journey.maze
            or feedback.candidate_artifact_sha256 !=
            verification_inputs["candidate_artifact_sha256"]
            or not feedback.source_signature_verified
            or handoff.graduation_authorized
            or feedback.graduation_authorized):
        raise ValueError("Minotaur feedback and Journey lineage are inconsistent")
    weak = set(handoff.remediation.weak_skills)
    if (set(obs.skill for obs in feedback.observations if obs.weak) != weak
            or any((obs.suggested_train_tactic is None) == obs.weak
                   for obs in feedback.observations)):
        raise ValueError("Minotaur tactic does not match measured weaknesses")
    return MeasuredTaskmasterTransition(handoff=handoff, feedback=feedback)


def run_measured_taskmaster_successor(
    config: PilotConfig,
    predecessor_run_dir: str | Path,
    reserve_development_event: Callable[[str], bool],
    *,
    state: JourneyState,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> tuple[MeasuredTaskmasterTransition, dict[str, Any]]:
    """Opt-in execution: source verification, then existing HF pilot boundary.

    ``reserve_development_event`` must be host-owned, durable and atomic;
    this module cannot verify that characteristic. The pilot still performs
    its own validation before any model/optimizer execution. The returned
    source flags remain FALSE even if an untrusted downstream report says PASS.
    """
    prepared = prepare_measured_taskmaster_transition(
        state,
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        **verification_inputs,
    )
    transition = verification_inputs["transition"]
    result = run_journey_bound_successor(
        config,
        verification_inputs["scope"],
        predecessor_run_dir,
        verification_inputs["context"],
        transition.receipt_bytes,
        transition.signature_hex,
        journey=prepared.handoff.journey,
        evaluator_precommitment_sha256=verification_inputs[
            "expected_development_roster_sha256"],
        evaluator_public_roster=transition.development_public_roster,
        independent_development_key=verification_inputs[
            "independent_transition_key"],
        reserve_development_event=reserve_development_event,
    )
    return prepared, result
