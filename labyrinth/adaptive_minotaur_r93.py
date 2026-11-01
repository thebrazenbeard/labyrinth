"""Source-only multi-objective TRAIN scheduling after Minotaur development feedback.

An objective is never a generic prompt-to-task generator. Each lane must bind
an *already approved executable provider* (arithmetic, program reasoning, or
source-audit). A failed development decision moves its lane earlier in the
next public TRAIN schedule. Other lanes still receive their complete room
allocation: scheduling priority cannot erase required exposure.

This module executes no optimizer and owns no model identity, consent, host
lease, private evaluation data, memory state, or graduation authorization.
The JourneyState history and its sealed-case claims are only self-consistent
source data; independent examiner custody remains external.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

from .journey import JourneyState, TRAINING, _snapshot_journey_claims
from .journey_challenges import PreparedTrainingRoom, prepare_training_rooms
from .objective_challenge_provider_r92 import (
    _approved_scope as _approved_audit,
    prepare_objective_audit_room,
)
from .objective_contract import ObjectiveContract, prepare_objective_program_room
from .objective_learning_loop_r92 import _approved_objective as _approved_arithmetic


@dataclass(frozen=True)
class ObjectiveLane:
    """One declared objective and its currently active source JourneyState."""

    contract: ObjectiveContract
    journey: JourneyState


@dataclass(frozen=True)
class ScheduledTrainRoom:
    """Operator-side record; `prepared` contains TRAIN gold, not learner API."""

    objective_id: str
    contract_sha256: str
    cycle_index: int
    failed_development_predecessor: bool
    prepared: PreparedTrainingRoom

    @property
    def trainee_view(self):
        """Only this view may be sent to a model for inference."""
        return self.prepared.trainee_view


class AdaptiveTrainSchedule(NamedTuple):
    """No mutable authorization/status constructor field exists."""

    ordered_rooms: tuple[ScheduledTrainRoom, ...]
    objective_order: tuple[str, ...]

    @property
    def evidence_class(self) -> str:
        return "SOURCE_ONLY_UNATTESTED"

    def trainee_sequence(self) -> tuple[tuple[str, object], ...]:
        """Produce (objective ID, prompt-only view) pairs without TRAIN gold."""
        return tuple((entry.objective_id, entry.trainee_view)
                     for entry in self.ordered_rooms)


def _snapshot_lane(lane: ObjectiveLane) -> tuple[ObjectiveContract, JourneyState, bool]:
    if type(lane) is not ObjectiveLane:
        raise ValueError("exact ObjectiveLane required")
    if type(lane.contract) is not ObjectiveContract:
        raise ValueError("exact ObjectiveContract required")
    # Copies each primitive field through its canonical JSON parser, rather
    # than trusting the frozen dataclass wrapper after caller-side mutation.
    contract = ObjectiveContract.from_dict(lane.contract.to_dict())
    state = _snapshot_journey_claims(lane.journey)
    if state.phase != TRAINING or state.scope != contract.training_scope:
        raise ValueError("objective lane requires its own active TRAIN journey")

    failure = bool(state.attempts)
    if failure:
        predecessor = state.attempts[-1]
        if (predecessor.exam_status != "FAIL" or not predecessor.weak_skills
                or not predecessor.exam_case_ids
                or len(predecessor.exam_case_ids) != len(predecessor.exam_prompt_sha256)
                or predecessor.exam_id not in state.sealed_exam_ids
                or not set(predecessor.exam_case_ids).issubset(state.sealed_exam_case_ids)
                or not set(predecessor.exam_prompt_sha256).issubset(
                    state.sealed_exam_prompt_sha256)
                or state.exam_event_statuses[
                    state.sealed_exam_ids.index(predecessor.exam_id)
                ] != "FAIL"):
            raise ValueError("failed-development lane lacks linked sealed evidence")
    return contract, state, failure


def _approved_rooms(
    contract: ObjectiveContract, state: JourneyState,
) -> tuple[PreparedTrainingRoom, ...]:
    skill = state.scope.skills
    if skill == ("source_priority",):
        _approved_audit(contract)
        return tuple(prepare_objective_audit_room(contract, state.maze, room, journey=state)
                     for room in state.maze.rooms)
    if skill == ("program_reasoning",):
        return tuple(prepare_objective_program_room(contract, state.maze, room,
                                                    journey=state)
                     for room in state.maze.rooms)
    if skill == ("arithmetic",):
        _approved_arithmetic(contract)
        return prepare_training_rooms(state.maze)
    raise ValueError("no approved executable provider for objective lane")


def schedule_adaptive_training(
    lanes: tuple[ObjectiveLane, ...],
) -> AdaptiveTrainSchedule:
    """Prioritize verified-*shape* FAIL feedback across independent objectives.

    Every objective receives its complete current TRAIN maze. A failed lane
    goes first in each round; ties are canonical objective ID order. All case
    and prompt digests must be disjoint across lanes, including sealed history.
    This is scheduling for an operator, not training or an authorization token.
    """
    if type(lanes) is not tuple or not 2 <= len(lanes) <= 8:
        raise ValueError("2..8 immutable objective lanes required")
    snapshots = tuple(_snapshot_lane(lane) for lane in lanes)
    objective_ids = [contract.scope.objective_id for contract, _, _ in snapshots]
    scope_hashes = [state.scope.fingerprint for _, state, _ in snapshots]
    if len(set(objective_ids)) != len(objective_ids) or len(set(scope_hashes)) != len(scope_hashes):
        raise ValueError("objective lanes must have distinct identities and scopes")
    if sum(len(state.maze.rooms) for _, state, _ in snapshots) > 128:
        raise ValueError("portfolio exceeds bounded TRAIN capacity")
    forbidden_cases = set().union(*(
        set(state.used_training_case_ids) | set(state.sealed_exam_case_ids)
        for _, state, _ in snapshots
    ))
    forbidden_prompts = set().union(*(
        set(state.used_training_prompt_sha256) | set(state.sealed_exam_prompt_sha256)
        for _, state, _ in snapshots
    ))

    batches: list[tuple[str, str, int, bool, tuple[PreparedTrainingRoom, ...]]] = []
    used_cases: set[str] = set()
    used_prompts: set[str] = set()
    for contract, state, failed in snapshots:
        prepared = _approved_rooms(contract, state)
        if len(prepared) != len(state.maze.rooms):
            raise ValueError("approved provider omitted required TRAIN rooms")
        ids = [item.challenge.id for item in prepared]
        digests = [sha256(item.challenge.prompt.encode("utf-8")).hexdigest()
                   for item in prepared]
        if (len(set(ids)) != len(ids) or len(set(digests)) != len(digests)
                or set(ids) & used_cases or set(digests) & used_prompts):
            raise ValueError("TRAIN case or prompt overlaps another objective")
        if (set(ids) & forbidden_cases or set(digests) & forbidden_prompts):
            raise ValueError("previous TRAIN or development exam material cannot be reused")
        used_cases.update(ids)
        used_prompts.update(digests)
        batches.append((state.scope.objective_id, contract.fingerprint,
                        state.cycle_index, failed, prepared))

    # Deficit is categorical here: failed development vs no prior exam.
    # Quantitative deficit would require independently custodied scoring.
    batches.sort(key=lambda row: (not row[3], row[0]))
    ordered: list[ScheduledTrainRoom] = []
    for offset in range(max(len(row[4]) for row in batches)):
        for objective_id, contract_hash, cycle, failed, prepared in batches:
            if offset < len(prepared):
                ordered.append(ScheduledTrainRoom(
                    objective_id=objective_id,
                    contract_sha256=contract_hash,
                    cycle_index=cycle,
                    failed_development_predecessor=failed,
                    prepared=prepared[offset],
                ))
    return AdaptiveTrainSchedule(
        ordered_rooms=tuple(ordered),
        objective_order=tuple(row[0] for row in batches),
    )
