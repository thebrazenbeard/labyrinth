"""Opt-in signed FAIL -> regenerated TRAIN -> full-state CPU successor.

This is a real non-test caller of R11's signed Journey proposal and R8's
full-state development runner. It routes one exact answerless training roster
through a checkpointed neural update, rather than re-labeling a proposal as an
execution. All host authorization hooks remain externally supplied: no source
module can attest their custody, prevent an old database snapshot rollback, or
make a public development receipt a protected final examination.

The signed event is consumed only after pure signed-proposal and local roster
checks but before the full-state journal/worker starts. A failure in the
reservation-dependent host-binding verifier or afterward deliberately leaves
the event consumed and may leave an orphan checkpoint. Recovery requires a
separately authorized reconciliation; automatic replay is forbidden. Source
cannot preverify host custody or host identity without trusted host evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import re
from pathlib import Path
from typing import Any, Callable

from .fullstate_successor import (
    DevelopmentStepReceipt, FullStateSuccessorReceipt,
    run_fullstate_development_cycle,
)
from .hf_development_event_journal import FullStateRunBinding
from .journey import JourneyState, TRAINING
from .measured_weakness_feedback import TrainingExposureRef
from .objective_contract import ObjectiveContract
from .regenerative_maze import MazePlan
from .signed_exam_lineage import SignedExamLineageBinding
from .signed_journey_loop import (
    HostLoopReservation, SignedJourneyLoopPlan,
    preview_signed_journey_loop, reserve_signed_journey_loop,
)


_HEX = re.compile(r"[0-9a-f]{64}\Z")


class SignedJourneyExecutionError(ValueError):
    """One-shot signed development continuation was not authorized/executed."""


def maze_commitment_sha256(maze: MazePlan) -> str:
    """Bind the exact regenerated room descriptors to the full-state journal."""
    if type(maze) is not MazePlan:
        raise SignedJourneyExecutionError("exact MazePlan required")
    maze.__post_init__()
    canonical = (json.dumps(asdict(maze), sort_keys=True, separators=(",", ":"),
                            allow_nan=False, ensure_ascii=True) + "\n").encode()
    return sha256(b"labyrinth.signed-fullstate-maze.r12\n" + canonical).hexdigest()


def _unaliased(path: Path) -> bool:
    """Cheap pre-reservation path check; the checkpoint loader rechecks later."""
    if not isinstance(path, Path) or not path.is_absolute():
        return False
    try:
        return (path == path.resolve(strict=False)
                and not any(part.is_symlink() or
                            (hasattr(part, "is_junction") and part.is_junction())
                            for part in (path, *path.parents)))
    except (OSError, RuntimeError):
        return False


@dataclass(frozen=True)
class SignedJourneyExecutionReceipt:
    """Executed public TRAIN cycle, still without protected exam authority."""

    plan: SignedJourneyLoopPlan
    successor: FullStateSuccessorReceipt
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    host_independence_verified: bool = False

    def __post_init__(self) -> None:
        if type(self.plan) is not SignedJourneyLoopPlan:
            raise SignedJourneyExecutionError("signed plan is missing")
        self.plan.__post_init__()
        if type(self.successor) is not FullStateSuccessorReceipt:
            raise SignedJourneyExecutionError("full-state successor receipt is missing")
        if (self.graduation_authorized is not False
                or self.model_execution_authenticated is not False
                or self.host_independence_verified is not False
                or self.successor.graduation_authorized is not False
                or self.successor.physical_model_verified is not False
                or self.successor.examiner_custody_verified is not False
                or self.successor.status != "PROVISIONAL_SOURCE_ONLY"):
            raise SignedJourneyExecutionError("source-only execution cannot grant qualification")


def execute_signed_fullstate_journey(
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
    fullstate_binding: FullStateRunBinding,
    expected_source_tree_sha256: str,
    verify_host_binding: Callable[[SignedJourneyLoopPlan, FullStateRunBinding], bool],
    journal_db: Path,
    journal_state_key: bytes,
    initial_checkpoint_dir: Path,
    successor_checkpoint_dir: Path,
    model_factory: Callable[[], Any],
    optimizer_factory: Callable[[Any], Any],
    training_step: Callable[[Any, Any, tuple[tuple[str, str], ...], int], DevelopmentStepReceipt],
    verify_signed_event: Callable[[str, FullStateRunBinding], bool],
    attest_successor_bytes: Callable[[Path, Any], bool],
) -> SignedJourneyExecutionReceipt:
    """Run precisely one signed, host-reserved, answerless public TRAIN window.

    No default authority provider is accepted. A host must durably reserve the
    signed event, verify the distinct host full-state binding and checkpoint
    bytes, keep signing material outside the learner, and independently check
    model execution. An in-process True callback is only a test fixture, not
    physical or examiner qualification.
    """
    if any(not callable(fn) for fn in (
            host_reserve_once, verify_host_reservation, verify_host_binding,
            verify_signed_event, attest_successor_bytes, model_factory,
            optimizer_factory, training_step)):
        raise SignedJourneyExecutionError("external host authority and execution hooks required")
    if (type(fullstate_binding) is not FullStateRunBinding
            or type(journey) is not JourneyState
            or type(expected_source_tree_sha256) is not str
            or not _HEX.fullmatch(expected_source_tree_sha256)):
        raise SignedJourneyExecutionError("pinned full-state and source identities required")
    fullstate_binding.validate()
    if (type(journal_state_key) is not bytes or len(journal_state_key) < 32
            or not all(_unaliased(path) for path in
                       (journal_db, initial_checkpoint_dir, successor_checkpoint_dir))
            or not journal_db.is_file()
            or not initial_checkpoint_dir.is_dir()
            or not successor_checkpoint_dir.parent.is_dir()
            or successor_checkpoint_dir.exists()
            or successor_checkpoint_dir.is_symlink()):
        raise SignedJourneyExecutionError(
            "preflight host journal, key or successor destination before consuming event")
    if (journey.phase != TRAINING
            or fullstate_binding.objective_id != journey.scope.objective_id
            or not hmac.compare_digest(fullstate_binding.base_checkpoint_sha256,
                                       journey.baseline_checkpoint_sha256)
            or not hmac.compare_digest(fullstate_binding.prior_checkpoint_sha256,
                                       journey.checkpoint_sha256)
            or not hmac.compare_digest(fullstate_binding.source_tree_sha256,
                                       expected_source_tree_sha256)
            or not hmac.compare_digest(fullstate_binding.maze_commitment_sha256,
                                       maze_commitment_sha256(journey.maze))
            or fullstate_binding.initial_cursor != 0
            or not fullstate_binding.initial_exposures):
        raise SignedJourneyExecutionError(
            "full-state objective, learner, maze, source or cursor identity mismatch")

    # The independent evaluator and answerless allocation are pure. Check
    # the local full-state roster against them *before* an externally owned,
    # irreversible reservation is attempted. Do not recalculate the proposal
    # after checking: that would admit a check/reserve identity split.
    preview = preview_signed_journey_loop(
        contract=contract,
        journey=journey,
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        expected_objective_contract_sha256=expected_objective_contract_sha256,
        expected_journey_sha256=expected_journey_sha256,
        expected_learner_checkpoint_sha256=expected_learner_checkpoint_sha256,
        expected_fence=expected_fence,
        verification_inputs=verification_inputs,
    )
    expected_exposures = tuple((view.challenge_id, view.prompt_sha256)
                               for view in preview.allocation.views)
    if (fullstate_binding.initial_exposures != expected_exposures
            or len(expected_exposures) != len(preview.allocation.maze.rooms)):
        raise SignedJourneyExecutionError(
            "full-state exposure roster differs from signed TRAIN allocation")

    # Reserve once through the host; this is the first externally irreversible
    # operation. Host signature/fence and final full-state host-binding proof
    # must still pass before any model is loaded. The latter cannot be moved
    # before reservation: its public callback consumes a reservation-bearing
    # SignedJourneyLoopPlan, and source cannot synthesize trusted host proof.
    plan = reserve_signed_journey_loop(
        preview, host_reserve_once=host_reserve_once,
        verify_host_reservation=verify_host_reservation,
    )
    if verify_host_binding(plan, fullstate_binding) is not True:
        raise SignedJourneyExecutionError("host binding authority verification refused")

    def checked_host_event(event_id: str, run_binding: FullStateRunBinding) -> bool:
        if (type(event_id) is not str
                or not hmac.compare_digest(event_id, plan.allocation.event_id)
                or run_binding != fullstate_binding):
            return False
        return verify_signed_event(event_id, run_binding) is True

    successor = run_fullstate_development_cycle(
        binding=fullstate_binding,
        event_id=plan.allocation.event_id,
        journal_db=journal_db,
        state_key=journal_state_key,
        initial_checkpoint_dir=initial_checkpoint_dir,
        successor_checkpoint_dir=successor_checkpoint_dir,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        training_step=training_step,
        step_count=len(expected_exposures),
        verify_signed_event=checked_host_event,
        attest_successor_bytes=attest_successor_bytes,
    )
    if (type(successor) is not FullStateSuccessorReceipt
            or successor.run_id != fullstate_binding.run_id
            or successor.next_cursor != len(expected_exposures)
            or successor.fence != 0):
        raise SignedJourneyExecutionError("full-state successor receipt does not match signed window")
    return SignedJourneyExecutionReceipt(plan, successor)
