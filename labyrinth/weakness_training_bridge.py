"""Signed development weakness -> objective-conditioned TRAIN-only prompts.

This optional source bridge does not host the independent evaluator or retain
private exam answers. It verifies the signed development failure, requires the
active Journey to agree with that evidence and then uses only the approved
program-reasoning TRAIN generator. The returned views contain no trainer gold.

All signatures/precommitments require externally controlled keys and durable
one-shot reservation; source validation does not prove that deployment boundary,
physical model learning, transfer, or graduation. Other objectives (including
empathy) still lack an executable/qualified provider and fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import re
from typing import Any

from .journey import JourneyState, TRAINING, _validate_state as validate_journey
from .journey_challenges import TraineeRoomView
from .measured_weakness_feedback import (
    TrainingExposureRef, derive_measured_remediation,
)
from .objective_contract import ObjectiveContract, prepare_objective_program_room
from .regenerative_maze import MazePlan
from .signed_exam_lineage import SignedExamLineageBinding


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class MeasuredProgramTrainingAllocation:
    """Answerless *development* allocation, never an examination receipt."""

    maze: MazePlan
    views: tuple[TraineeRoomView, ...]
    weak_skills: tuple[str, ...]
    exam_id: str
    event_id: str
    objective_contract_sha256: str
    signed_binding_sha256: str
    training_roster_sha256: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    evaluator_host_independence_verified: bool = False

    def __post_init__(self) -> None:
        if (self.graduation_authorized is not False
                or self.model_execution_authenticated is not False
                or self.evaluator_host_independence_verified is not False):
            raise ValueError("public TRAIN allocation cannot claim independent qualification")


def allocate_signed_program_training(
    *,
    contract: ObjectiveContract,
    journey: JourneyState,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    expected_objective_contract_sha256: str,
    **verification_inputs: Any,
) -> MeasuredProgramTrainingAllocation:
    """Reverify signed FAIL and route its exact next Maze to public TRAIN views.

    A host must pin all signing keys, expose the independent scored result only
    through this verification seam, and durably consume the signed event once.
    The output never contains grade keys, private prompts, or model authority.
    """
    if type(contract) is not ObjectiveContract:
        raise ValueError("approved objective contract required")
    ObjectiveContract.__post_init__(contract)
    if (type(expected_objective_contract_sha256) is not str
            or not _SHA256.fullmatch(expected_objective_contract_sha256)
            or not hmac.compare_digest(expected_objective_contract_sha256,
                                       contract.fingerprint)):
        raise ValueError("operator-pinned objective contract fingerprint required")
    if (contract.scope.skills != ("program_reasoning",)
            or contract.scope.domain not in ("reasoning", "program_reasoning")
            or len(contract.targets) != 1
            or contract.targets[0].rubric_id != "program_structure_v1"):
        raise ValueError("no approved executable provider for declared objective")
    if verification_inputs.get("scope") != contract.scope:
        raise ValueError("signed development scope disagrees with objective contract")
    if type(journey) is not JourneyState:
        raise ValueError("active training JourneyState required")
    validate_journey(journey)
    if journey.phase != TRAINING or journey.scope != contract.scope or not journey.attempts:
        raise ValueError("signed remediation requires an active regenerated journey")

    remediation = derive_measured_remediation(
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        **verification_inputs,
    )
    context = verification_inputs["context"]
    transition = verification_inputs["transition"]
    latest = journey.attempts[-1]
    if (remediation.maze != journey.maze
            or latest.cycle_index != remediation.from_cycle
            or journey.maze.cycle_index != remediation.to_cycle
            or latest.exam_id != remediation.exam_id
            or latest.exam_status != "FAIL"
            or latest.weak_skills != remediation.weak_skills
            or journey.checkpoint_sha256 != context.predecessor_lora_value_sha256):
        raise ValueError("signed weakness or checkpoint does not match current journey")
    prior_ids = {case.case_id for case in training_exposures}
    prior_prompts = {case.prompt_sha256 for case in training_exposures}
    signed_train_pairs = {(case.case_id, case.prompt_sha256)
                          for case in training_exposures}
    journey_train_pairs = set(zip(journey.used_training_case_ids,
                                  journey.used_training_prompt_sha256, strict=True))
    if (signed_train_pairs != journey_train_pairs
            or len(training_exposures) != len(journey.used_training_case_ids)):
        raise ValueError("signed TRAIN exposure roster differs from current journey")
    exam_roster = transition.development_public_roster
    signed_exam_pairs = {(case.case_id, case.prompt_sha256) for case in exam_roster}
    latest_exam_pairs = set(zip(latest.exam_case_ids,
                                latest.exam_prompt_sha256, strict=True))
    historical_exam_pairs = set(zip(journey.sealed_exam_case_ids,
                                    journey.sealed_exam_prompt_sha256, strict=True))
    if (latest_exam_pairs != signed_exam_pairs
            or len(latest_exam_pairs) != len(exam_roster)
            or not signed_exam_pairs.issubset(historical_exam_pairs)
            or not journey.sealed_exam_ids
            or journey.sealed_exam_ids[-1] != latest.exam_id
            or not journey.exam_event_statuses
            or journey.exam_event_statuses[-1] != "FAIL"):
        raise ValueError("signed development exam differs from last journey exam")

    views = tuple(
        prepare_objective_program_room(contract, remediation.maze, room,
                                       journey=journey).trainee_view
        for room in remediation.maze.rooms
    )
    if (not views
            or len({view.challenge_id for view in views}) != len(views)
            or len({view.prompt_sha256 for view in views}) != len(views)
            or any(view.skill != "program_reasoning"
                   or not view.challenge_id.startswith("train-")
                   or view.challenge_id in prior_ids
                   or view.prompt_sha256 in prior_prompts
                   for view in views)):
        raise ValueError("new objective TRAIN rooms duplicate or escape approved scope")

    return MeasuredProgramTrainingAllocation(
        maze=remediation.maze, views=views,
        weak_skills=remediation.weak_skills, exam_id=remediation.exam_id,
        event_id=remediation.event_id,
        objective_contract_sha256=contract.fingerprint,
        signed_binding_sha256=sha256(binding.binding_bytes).hexdigest(),
        training_roster_sha256=remediation.roster_sha256,
    )
