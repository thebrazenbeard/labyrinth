"""Fail-closed proposal for one signed development-FAIL Journey continuation.

This is a **source-only** integration seam. The host, not this process, must
atomically reserve each signed event, own the monotonic fence, independently
verify a host-owned signature, and pin the learner and state commitments.
Injected callables are *not* proof that such a host exists or is independent.
No neural execution, model update, private exam, or graduation happens here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Any, Callable

from .journey import JourneyState, TRAINING, _validate_state
from .measured_weakness_feedback import TrainingExposureRef
from .objective_contract import ObjectiveContract
from .signed_exam_lineage import SignedExamLineageBinding
from .weakness_training_bridge import (
    MeasuredProgramTrainingAllocation, allocate_signed_program_training,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE = re.compile(r"(?:[0-9a-f]{64}|[0-9a-f]{128})\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_RESERVATION_SCHEMA = "labyrinth.host-loop-reservation.v1"
_PROPOSAL_SCHEMA = "labyrinth.signed-journey-proposal.v1"


def _sha(value: str, label: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 commitment")


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def journey_state_commitment(journey: JourneyState) -> str:
    """Domain-separated identity of the *complete* current Journey state."""
    if type(journey) is not JourneyState:
        raise ValueError("exact JourneyState required")
    _validate_state(journey)
    return sha256(b"labyrinth.journey-state.v1\n" + _canonical(asdict(journey))).hexdigest()


@dataclass(frozen=True)
class HostLoopReservation:
    """Opaque host claim, requiring verification by an injected trusted provider.

    A caller-created instance is not a valid reservation on its own. The host
    must manage persistence, atomic consume-once, key custody and the fence.
    """

    schema: str
    event_id: str
    proposal_sha256: str
    fence: int
    host_id: str
    signature_hex: str

    def __post_init__(self) -> None:
        if self.schema != _RESERVATION_SCHEMA:
            raise ValueError("unsupported host reservation schema")
        for label, value in (("event_id", self.event_id), ("host_id", self.host_id)):
            if type(value) is not str or not _ID.fullmatch(value):
                raise ValueError(f"invalid reservation {label}")
        _sha(self.proposal_sha256, "proposal_sha256")
        # Common HMAC-SHA256 or Ed25519 widths; verification and algorithm
        # selection remain exclusively the injected host's responsibility.
        if type(self.signature_hex) is not str or not _SIGNATURE.fullmatch(self.signature_hex):
            raise ValueError("invalid host signature encoding")
        if type(self.fence) is not int or self.fence < 1:
            raise ValueError("positive integer host fence required")

    @property
    def signed_bytes(self) -> bytes:
        self.__post_init__()
        return _canonical({
            "schema": self.schema,
            "event_id": self.event_id,
            "proposal_sha256": self.proposal_sha256,
            "fence": self.fence,
            "host_id": self.host_id,
        })


@dataclass(frozen=True)
class SignedJourneyLoopPlan:
    """Public TRAIN-only continuation; no authority to complete or graduate."""

    allocation: MeasuredProgramTrainingAllocation
    reservation: HostLoopReservation
    learner_checkpoint_sha256: str
    journey_sha256: str
    proposal_sha256: str
    previous_cycle: int
    next_cycle: int
    host_fence: int
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    host_independence_verified: bool = False

    def __post_init__(self) -> None:
        if (self.graduation_authorized is not False
                or self.model_execution_authenticated is not False
                or self.host_independence_verified is not False):
            raise ValueError("a source-only loop cannot assert qualification or host independence")
        if type(self.allocation) is not MeasuredProgramTrainingAllocation:
            raise ValueError("signed TRAIN allocation required")
        MeasuredProgramTrainingAllocation.__post_init__(self.allocation)
        if type(self.reservation) is not HostLoopReservation:
            raise ValueError("host reservation required")
        self.reservation.__post_init__()
        for label, value in (("learner_checkpoint_sha256", self.learner_checkpoint_sha256),
                             ("journey_sha256", self.journey_sha256),
                             ("proposal_sha256", self.proposal_sha256)):
            _sha(value, label)
        if (type(self.previous_cycle) is not int or type(self.next_cycle) is not int
                or self.next_cycle != self.previous_cycle + 1
                or self.previous_cycle < 0):
            raise ValueError("continuation requires exactly one regenerated cycle")
        if (type(self.host_fence) is not int or self.host_fence != self.reservation.fence
                or self.reservation.event_id != self.allocation.event_id
                or not hmac.compare_digest(self.reservation.proposal_sha256,
                                           self.proposal_sha256)):
            raise ValueError("reservation and plan identities diverge")


@dataclass(frozen=True)
class SignedJourneyLoopProposal:
    """Validated, unreserved continuation; explicitly NOT host authority.

    This preview permits local full-state checks before a one-shot event is
    irreversibly consumed. A preview cannot execute training or be replayed as
    a reservation. Its signed counterpart is SignedJourneyLoopPlan.
    """

    allocation: MeasuredProgramTrainingAllocation
    proposal_bytes: bytes
    proposal_sha256: str
    learner_checkpoint_sha256: str
    journey_sha256: str
    previous_cycle: int
    next_cycle: int
    expected_fence: int

    def __post_init__(self) -> None:
        if type(self.allocation) is not MeasuredProgramTrainingAllocation:
            raise ValueError("signed TRAIN allocation required for proposal")
        self.allocation.__post_init__()
        for label, value in (("proposal_sha256", self.proposal_sha256),
                             ("learner_checkpoint_sha256", self.learner_checkpoint_sha256),
                             ("journey_sha256", self.journey_sha256)):
            _sha(value, label)
        if (type(self.proposal_bytes) is not bytes
                or not hmac.compare_digest(sha256(self.proposal_bytes).hexdigest(),
                                           self.proposal_sha256)
                or type(self.previous_cycle) is not int or self.previous_cycle < 0
                or type(self.next_cycle) is not int
                or self.next_cycle != self.previous_cycle + 1
                or type(self.expected_fence) is not int or self.expected_fence < 1):
            raise ValueError("invalid immutable journey proposal commitment or cycle")
        try:
            body = json.loads(self.proposal_bytes)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("invalid canonical journey proposal") from exc
        if (type(body) is not dict or _canonical(body) != self.proposal_bytes
                or body.get("schema") != _PROPOSAL_SCHEMA
                or body.get("event_id") != self.allocation.event_id
                or body.get("exam_id") != self.allocation.exam_id
                or body.get("contract_sha256") != self.allocation.objective_contract_sha256
                or body.get("signed_binding_sha256") != self.allocation.signed_binding_sha256
                or body.get("training_roster_sha256") != self.allocation.training_roster_sha256
                or body.get("expected_host_fence") != self.expected_fence
                or body.get("journey_sha256") != self.journey_sha256
                or body.get("learner_checkpoint_sha256") != self.learner_checkpoint_sha256
                or body.get("prior_cycle") != self.previous_cycle
                or body.get("regenerated_cycle") != self.next_cycle
                or body.get("weak_skills") != list(self.allocation.weak_skills)
                or body.get("rooms") != [
                    {"room_id": room.room_id, "skill": room.skill,
                     "case_seed": room.case_seed} for room in self.allocation.maze.rooms
                ]
                or body.get("answerless_train_pairs") != [
                    {"challenge_id": view.challenge_id,
                     "prompt_sha256": view.prompt_sha256,
                     "skill": view.skill} for view in self.allocation.views
                ]):
            raise ValueError("proposal identity diverges from signed TRAIN allocation")


def preview_signed_journey_loop(
    *,
    contract: ObjectiveContract,
    journey: JourneyState,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    expected_objective_contract_sha256: str,
    expected_journey_sha256: str,
    expected_learner_checkpoint_sha256: str,
    expected_fence: int,
    verification_inputs: dict[str, Any],
) -> SignedJourneyLoopProposal:
    """Pure, verified TRAIN proposal for pre-reservation identity checks.

    `verification_inputs` supplies the existing R10 independent development
    verifier. This has no host side effects and confers no authority.
    """
    if type(verification_inputs) is not dict:
        raise ValueError("independent verification inputs are mandatory")
    if type(journey) is not JourneyState:
        raise ValueError("active JourneyState required")
    current_commitment = journey_state_commitment(journey)
    _sha(expected_journey_sha256, "expected_journey_sha256")
    _sha(expected_learner_checkpoint_sha256, "expected_learner_checkpoint_sha256")
    if (not hmac.compare_digest(current_commitment, expected_journey_sha256)
            or not hmac.compare_digest(journey.checkpoint_sha256,
                                       expected_learner_checkpoint_sha256)):
        raise ValueError("operator-pinned journey or learner identity changed")
    if (type(expected_fence) is not int or expected_fence < 1
            or journey.phase != TRAINING or not journey.attempts
            or journey.cycle_index != journey.attempts[-1].cycle_index + 1):
        raise ValueError("valid regenerated active cycle and host fence required")

    # R10 owns signature verification, weakness measurement, contract scope,
    # case<->prompt fidelity, regeneration and answerless TRAIN allocation.
    allocation = allocate_signed_program_training(
        contract=contract,
        journey=journey,
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        expected_objective_contract_sha256=expected_objective_contract_sha256,
        **verification_inputs,
    )
    if allocation.maze != journey.maze or not allocation.views:
        raise ValueError("signed regenerated TRAIN maze changed after verification")
    proposal = {
        "schema": _PROPOSAL_SCHEMA,
        "event_id": allocation.event_id,
        "exam_id": allocation.exam_id,
        "scope_sha256": journey.scope.fingerprint,
        "contract_sha256": allocation.objective_contract_sha256,
        "journey_sha256": current_commitment,
        "learner_checkpoint_sha256": journey.checkpoint_sha256,
        "signed_binding_sha256": allocation.signed_binding_sha256,
        "training_roster_sha256": allocation.training_roster_sha256,
        "prior_cycle": journey.attempts[-1].cycle_index,
        "regenerated_cycle": journey.cycle_index,
        "expected_host_fence": expected_fence,
        "weak_skills": list(allocation.weak_skills),
        "rooms": [{"room_id": room.room_id, "skill": room.skill,
                   "case_seed": room.case_seed} for room in allocation.maze.rooms],
        "answerless_train_pairs": [
            {"challenge_id": view.challenge_id, "prompt_sha256": view.prompt_sha256,
             "skill": view.skill} for view in allocation.views
        ],
    }
    proposal_bytes = _canonical(proposal)
    proposal_sha256 = sha256(proposal_bytes).hexdigest()

    return SignedJourneyLoopProposal(
        allocation=allocation, proposal_bytes=proposal_bytes,
        proposal_sha256=proposal_sha256,
        learner_checkpoint_sha256=journey.checkpoint_sha256,
        journey_sha256=current_commitment,
        previous_cycle=journey.attempts[-1].cycle_index,
        next_cycle=journey.cycle_index, expected_fence=expected_fence,
    )


def reserve_signed_journey_loop(
    proposal: SignedJourneyLoopProposal,
    *,
    host_reserve_once: Callable[[bytes], HostLoopReservation],
    verify_host_reservation: Callable[[HostLoopReservation, bytes], bool],
) -> SignedJourneyLoopPlan:
    """Reserve a validated preview once through an externally trusted host.

    Source cannot attest host key custody, atomicity or replay resistance.
    If a post-reservation host identity verifier rejects, the event stays
    consumed and needs authorized reconciliation; never retry automatically.
    """
    if (not callable(host_reserve_once) or not callable(verify_host_reservation)):
        raise ValueError("injected host reservation and verifier are mandatory")
    if type(proposal) is not SignedJourneyLoopProposal:
        raise ValueError("exact verified Journey proposal required")
    proposal.__post_init__()

    # EXACTLY one call; a correct host must persistently reject replays or
    # collisions before signing. This source wrapper cannot enforce atomicity.
    reservation = host_reserve_once(proposal.proposal_bytes)
    if type(reservation) is not HostLoopReservation:
        raise ValueError("host returned no typed one-shot reservation")
    reservation.__post_init__()
    if (reservation.event_id != proposal.allocation.event_id
            or reservation.fence != proposal.expected_fence
            or not hmac.compare_digest(reservation.proposal_sha256, proposal.proposal_sha256)):
        raise ValueError("host reservation disagrees with proposal or fence")
    if verify_host_reservation(reservation, reservation.signed_bytes) is not True:
        raise ValueError("host signature verification failed")
    return SignedJourneyLoopPlan(
        allocation=proposal.allocation, reservation=reservation,
        learner_checkpoint_sha256=proposal.learner_checkpoint_sha256,
        journey_sha256=proposal.journey_sha256,
        proposal_sha256=proposal.proposal_sha256,
        previous_cycle=proposal.previous_cycle,
        next_cycle=proposal.next_cycle, host_fence=proposal.expected_fence,
    )


def prepare_signed_journey_loop(
    *,
    contract: ObjectiveContract,
    journey: JourneyState,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    expected_objective_contract_sha256: str,
    expected_journey_sha256: str,
    expected_learner_checkpoint_sha256: str,
    expected_fence: int,
    host_reserve_once: Callable[[bytes], HostLoopReservation],
    verify_host_reservation: Callable[[HostLoopReservation, bytes], bool],
    verification_inputs: dict[str, Any],
) -> SignedJourneyLoopPlan:
    """Backward-compatible one-shot proposal and signed reservation.

    R12 users retain their public API. Callers with additional local binding
    constraints must use preview_signed_journey_loop first, perform those
    checks, then reserve_signed_journey_loop precisely once.
    """
    if (not callable(host_reserve_once) or not callable(verify_host_reservation)
            or type(verification_inputs) is not dict):
        raise ValueError("injected host reservation and verifier are mandatory")
    preview = preview_signed_journey_loop(
        contract=contract, journey=journey, binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        expected_objective_contract_sha256=expected_objective_contract_sha256,
        expected_journey_sha256=expected_journey_sha256,
        expected_learner_checkpoint_sha256=expected_learner_checkpoint_sha256,
        expected_fence=expected_fence, verification_inputs=verification_inputs,
    )
    return reserve_signed_journey_loop(
        preview, host_reserve_once=host_reserve_once,
        verify_host_reservation=verify_host_reservation,
    )
