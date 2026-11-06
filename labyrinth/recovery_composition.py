"""Host-owned composition of a signed development event and full neural custody.

This adapter binds three *existing* substrates without giving any of them
additional authority: an upstream signature-verified development FAIL, the
MAC-fenced host journal, and the complete neural checkpoint loader.  The
``reservation_callback`` is suitable ONLY for an already authenticated
``run_journey_bound_successor`` invocation.  A callback call is not evidence
of a signature, an examination result, learning, or graduation.

The host must protect the state-MAC key and immutable checkpoint directories,
exclude old workers from the resumed process, and check the current journal
fence at **every** worker write and result boundary.  In particular, a stale
worker is not fenced merely because this adapter has claimed a new fence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .hf_development_event_journal import (
    claim_development_run_resume,
    commit_development_run_checkpoint,
    inspect_development_event,
    reserve_development_run,
)
from .neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointReceipt, load_full_checkpoint,
)


# The trusted implementation should close over its *isolated* model/AdamW and
# call neural_resume_checkpoint.load_full_checkpoint with all three arguments
# mapped to source, expected_sha256, expected_context.  No mock is trusted for
# physical qualification; injecting one only permits source-level tests.
TrustedCheckpointLoader = Callable[
    [Path, str, FullCheckpointContext], FullCheckpointReceipt
]


def neural_checkpoint_loader(*, model: Any, optimizer: Any) -> TrustedCheckpointLoader:
    """Bind the actual strict full-state loader to an isolated CPU worker.

    The supplied model and AdamW must be host-owned, and this factory does not
    itself authenticate an exam or enforce fences on later worker results.
    """
    def load(path: Path, digest: str, context: FullCheckpointContext) -> FullCheckpointReceipt:
        return load_full_checkpoint(
            path, model=model, optimizer=optimizer,
            expected_sha256=digest, expected_context=context,
        )

    return load


class RecoveryCompositionError(RuntimeError):
    """The event, full checkpoint, exposure, or journal fence did not compose."""


@dataclass(frozen=True)
class FencedCheckpointBinding:
    """A host commitment, not an authority to resume or graduate.

    ``journal_fence`` may exceed ``checkpoint_context.fence`` after a resume:
    the persisted checkpoint is immutable, while a new worker owns the newer
    fence.  Its *next* checkpoint must record that new fence.
    """

    event_id: str
    run_id: str
    run_commitment_sha256: str
    checkpoint_dir: Path
    checkpoint_sha256: str
    checkpoint_context: FullCheckpointContext
    journal_fence: int

    def validate(self) -> None:
        if type(self.checkpoint_context) is not FullCheckpointContext:
            raise RecoveryCompositionError("full checkpoint context required")
        self.checkpoint_context.validate()
        if self.checkpoint_context.run_id != self.run_id:
            raise RecoveryCompositionError("checkpoint run differs from reserved run")
        if not isinstance(self.checkpoint_dir, Path) or not self.checkpoint_dir.is_absolute():
            raise RecoveryCompositionError("absolute checkpoint directory required")
        if (type(self.journal_fence) is not int
                or not self.checkpoint_context.fence <= self.journal_fence < 2**63 - 2):
            raise RecoveryCompositionError("invalid or stale checkpoint fence")
        # Event, run, and digest syntax are also checked by the underlying
        # journal *before* mutation; keep a boundary check here as well.
        from .hf_development_event_journal import _DIGEST, _IDENTIFIER
        if (type(self.event_id) is not str or not _IDENTIFIER.fullmatch(self.event_id)
                or type(self.run_id) is not str or not _IDENTIFIER.fullmatch(self.run_id)
                or type(self.run_commitment_sha256) is not str
                or not _DIGEST.fullmatch(self.run_commitment_sha256)
                or type(self.checkpoint_sha256) is not str
                or not _DIGEST.fullmatch(self.checkpoint_sha256)):
            raise RecoveryCompositionError("invalid event or run/checkpoint commitment")


def _trusted_load(binding: FencedCheckpointBinding,
                  checkpoint_loader: TrustedCheckpointLoader) -> None:
    if not callable(checkpoint_loader):
        raise RecoveryCompositionError("trusted full checkpoint loader required")
    receipt = checkpoint_loader(
        binding.checkpoint_dir,
        binding.checkpoint_sha256,
        binding.checkpoint_context,
    )
    if (type(receipt) is not FullCheckpointReceipt
            or receipt.sha256 != binding.checkpoint_sha256
            or receipt.context != binding.checkpoint_context
            or receipt.optimizer != "AdamW"
            or receipt.graduation_authorized is not False
            or type(receipt.trainable_parameters) is not tuple
            or not receipt.trainable_parameters
            or any(type(name) is not str or not name for name in receipt.trainable_parameters)):
        raise RecoveryCompositionError("trusted loader did not attest full checkpoint identity")


def _assert_active_journal(binding: FencedCheckpointBinding, *,
                           journal_path: Path, state_key: bytes) -> None:
    state = inspect_development_event(
        journal_path, binding.event_id, state_key=state_key,
    )
    if (state.get("consumed") is not True
            or state.get("run_state") != "ACTIVE"
            or state.get("run_id") != binding.run_id
            or state.get("run_commitment_sha256") != binding.run_commitment_sha256
            or state.get("checkpoint_sha256") != binding.checkpoint_sha256
            or state.get("fence") != binding.journal_fence):
        raise RecoveryCompositionError("journal run/checkpoint/fence not ACTIVE or mismatched")


def reservation_callback(
    binding: FencedCheckpointBinding, *, journal_path: Path, state_key: bytes,
    checkpoint_loader: TrustedCheckpointLoader,
) -> Callable[[str], bool]:
    """Compose with the *already signature-verified* successor reservation API.

    Checkpoint verification precedes irreversible event consumption.  Returns
    a single-event callback and never provides a reusable event-only fallback.
    Duplicate consumption raises from the journal (fail-closed), never retries
    on a different run or clears/reissues the event.
    """
    binding.validate()
    if binding.journal_fence != 0 or binding.checkpoint_context.fence != 0:
        raise RecoveryCompositionError("initial reservation must start at fence zero")

    def reserve(verified_event_id: str) -> bool:
        if type(verified_event_id) is not str or verified_event_id != binding.event_id:
            raise RecoveryCompositionError("signed event differs from run-bound commitment")
        _trusted_load(binding, checkpoint_loader)
        fence = reserve_development_run(
            journal_path, binding.event_id, state_key=state_key,
            run_id=binding.run_id,
            run_commitment_sha256=binding.run_commitment_sha256,
            initial_checkpoint_sha256=binding.checkpoint_sha256,
        )
        if fence != 0:
            raise RecoveryCompositionError("initial run reservation returned invalid fence")
        return True

    return reserve


def restore_and_claim_same_run(
    binding: FencedCheckpointBinding, *, journal_path: Path, state_key: bytes,
    checkpoint_loader: TrustedCheckpointLoader,
) -> FencedCheckpointBinding:
    """Restore complete neural/optimizer/RNG state, then atomically fence resume.

    Caller must use an isolated stopped worker and must not run it if the claim
    fails.  The event stays consumed even if the process crashes after claim.
    """
    binding.validate()
    _assert_active_journal(binding, journal_path=journal_path, state_key=state_key)
    _trusted_load(binding, checkpoint_loader)
    new_fence = claim_development_run_resume(
        journal_path, binding.event_id, state_key=state_key,
        run_id=binding.run_id,
        run_commitment_sha256=binding.run_commitment_sha256,
        checkpoint_sha256=binding.checkpoint_sha256,
        expected_fence=binding.journal_fence,
    )
    if new_fence != binding.journal_fence + 1:
        raise RecoveryCompositionError("unexpected resumed journal fence")
    return replace(binding, journal_fence=new_fence)


def commit_fenced_checkpoint(
    current: FencedCheckpointBinding, *, next_checkpoint_dir: Path,
    next_checkpoint_sha256: str, next_context: FullCheckpointContext,
    journal_path: Path, state_key: bytes,
    checkpoint_loader: TrustedCheckpointLoader,
) -> FencedCheckpointBinding:
    """Commit one fully loaded checkpoint with monotonic consumed exposures.

    Pending (unconsumed) tasks may regenerate; consumed exposure IDs/hashes
    through the old cursor cannot disappear or change.  A full checkpoint
    (including optimizer/RNG) must be loaded before journal commit.  If a
    competing journal write wins after that load, discard this isolated worker
    and do not publish its writes/results.  This is a development cursor, not
    independent examination data or graduation.
    """
    current.validate()
    if type(next_context) is not FullCheckpointContext:
        raise RecoveryCompositionError("next full checkpoint context required")
    next_context.validate()
    if (next_context.run_id != current.run_id
            or next_context.objective_id != current.checkpoint_context.objective_id
            or next_context.base_checkpoint_sha256 != current.checkpoint_context.base_checkpoint_sha256
            or next_context.prior_checkpoint_sha256 != current.checkpoint_sha256
            or next_context.fence != current.journal_fence
            or next_context.cursor < current.checkpoint_context.cursor
            or next_context.exposures[:current.checkpoint_context.cursor]
            != current.checkpoint_context.exposures[:current.checkpoint_context.cursor]):
        raise RecoveryCompositionError("neural lineage, fence, cursor or consumed exposures changed")
    next_binding = replace(
        current, checkpoint_dir=next_checkpoint_dir,
        checkpoint_sha256=next_checkpoint_sha256,
        checkpoint_context=next_context,
    )
    next_binding.validate()
    if next_binding.checkpoint_sha256 == current.checkpoint_sha256:
        raise RecoveryCompositionError("checkpoint digest must advance")
    _assert_active_journal(current, journal_path=journal_path, state_key=state_key)
    _trusted_load(next_binding, checkpoint_loader)
    commit_development_run_checkpoint(
        journal_path, current.event_id, state_key=state_key,
        run_id=current.run_id,
        run_commitment_sha256=current.run_commitment_sha256,
        checkpoint_sha256=current.checkpoint_sha256,
        expected_fence=current.journal_fence,
        next_checkpoint_sha256=next_binding.checkpoint_sha256,
    )
    return next_binding
