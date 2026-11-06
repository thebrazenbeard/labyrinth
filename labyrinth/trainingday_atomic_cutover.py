"""R26 synthetic copy-on-write model/Minotaur/evidence cutover rehearsal.

This is a deterministic *in-memory* failure-injection model. No filesystem,
process, serving endpoint, model weights, graduation campaign, or evaluation
bank is changed. Hashes and previous-state receipts are caller declarations,
not independently authenticated artifacts. In particular, this module does
not establish a real multi-service atomic transaction or authorize a release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TRANSACTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DOMAIN = "labyrinth.trainingday.synthetic-cutover.v1"
_MAX_GENERATION = 2**63 - 2

# Each phase represents an independent place to inject a failure. The
# copy-on-write swap is the sole phase at which an active bundle changes.
CUTOVER_STEPS = (
    "VALIDATE_INPUT",
    "STAGE_MODEL",
    "STAGE_GATE",
    "STAGE_EVIDENCE",
    "PRECOMMIT_VERIFY",
    "ATOMIC_SWAP",
    "POST_SWAP_VERIFY",
    "FINALIZE_JOURNAL",
)


class CutoverInputError(ValueError):
    """An invalid declaration cannot enter the synthetic cutover model."""


@dataclass(frozen=True)
class ModelMinotaurBundle:
    """One logical state, never independently mutable serving/gate surfaces."""

    model_sha256: str
    adapter_sha256: str
    graduation_state_sha256: str
    evidence_index_sha256: str
    generation: int


@dataclass(frozen=True)
class CutoverCandidate:
    transaction_id: str
    expected_previous_commitment: str
    proposed: ModelMinotaurBundle
    graduation_handoff_sha256: str


@dataclass(frozen=True)
class RehearsalReceipt:
    status: str
    active: ModelMinotaurBundle
    retained_previous: ModelMinotaurBundle
    staged_candidate: ModelMinotaurBundle
    steps: tuple[str, ...]
    failure_step: str | None
    original_commitment: str
    active_commitment: str
    journal_sha256: str
    synthetic_rollback_preserved: bool
    physical_rollback_verified: bool = False
    host_custody_verified: bool = False
    deployment_allowed: bool = False
    promotion_allowed: bool = False


def _sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _validate_bundle(bundle: object) -> None:
    # A frozen dataclass subclass may override __getattribute__ and return
    # different hash fields between validation, staging and journal sealing.
    # The rehearsal accepts only its own plain immutable value type.
    if type(bundle) is not ModelMinotaurBundle:
        raise CutoverInputError("model/Minotaur bundle must be immutable data")
    if not all(_sha(v) for v in (
        bundle.model_sha256, bundle.adapter_sha256,
        bundle.graduation_state_sha256, bundle.evidence_index_sha256,
    )):
        raise CutoverInputError("malformed model, adapter, graduation or evidence digest")
    if type(bundle.generation) is not int or not 0 <= bundle.generation <= _MAX_GENERATION:
        raise CutoverInputError("invalid or overflowing generation")


def _snapshot_bundle(bundle: object) -> ModelMinotaurBundle:
    """Detach a caller-owned value before hashing or simulating a transition.

    Frozen dataclasses can still be changed via ``object.__setattr__`` in
    the same interpreter. A fresh exact-class value prevents subsequent
    caller mutation from changing our staged/journal representation.
    This is not an OS-owned immutable or authenticated snapshot.
    """
    if type(bundle) is not ModelMinotaurBundle:
        raise CutoverInputError("model/Minotaur bundle must be immutable data")
    first = (
        bundle.model_sha256,
        bundle.adapter_sha256,
        bundle.graduation_state_sha256,
        bundle.evidence_index_sha256,
        bundle.generation,
    )
    second = (
        bundle.model_sha256,
        bundle.adapter_sha256,
        bundle.graduation_state_sha256,
        bundle.evidence_index_sha256,
        bundle.generation,
    )
    if first != second:
        raise CutoverInputError("bundle changed during synthetic snapshot")
    return ModelMinotaurBundle(*first)


def _canonical_sha(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def bundle_commitment(bundle: ModelMinotaurBundle) -> str:
    """Canonical content commitment. It is NOT a signature or byte custody."""
    _validate_bundle(bundle)
    return _canonical_sha({"domain": _DOMAIN, "bundle": asdict(bundle)})


def _validate_candidate(previous: ModelMinotaurBundle,
                        candidate: object,
                        consumed_transaction_ids: object) -> None:
    if type(candidate) is not CutoverCandidate:
        raise CutoverInputError("invalid proposed cutover declaration")
    if (type(candidate.transaction_id) is not str
            or _TRANSACTION_ID.fullmatch(candidate.transaction_id) is None
            or not _sha(candidate.expected_previous_commitment)
            or not _sha(candidate.graduation_handoff_sha256)):
        raise CutoverInputError("malformed cutover transaction or evidence commitment")
    _validate_bundle(candidate.proposed)
    if candidate.expected_previous_commitment != bundle_commitment(previous):
        raise CutoverInputError("stale/tampered previous model/Minotaur/evidence state")
    next_state = candidate.proposed
    if next_state.generation != previous.generation + 1:
        raise CutoverInputError("cutover generation is stale or nonconsecutive")
    if (next_state.adapter_sha256 == previous.adapter_sha256
            or next_state.graduation_state_sha256 == previous.graduation_state_sha256
            or next_state.evidence_index_sha256 == previous.evidence_index_sha256):
        raise CutoverInputError("no-op or partially refreshed cutover evidence")
    if (type(consumed_transaction_ids) is not frozenset
            or len(consumed_transaction_ids) > 10_000
            or any(type(tx) is not str
                   or _TRANSACTION_ID.fullmatch(tx) is None
                   for tx in consumed_transaction_ids)):
        raise CutoverInputError("invalid or unbounded prior transaction IDs")
    if candidate.transaction_id in consumed_transaction_ids:
        raise CutoverInputError("cutover transaction replay detected")


def rehearse_cutover(
    previous: ModelMinotaurBundle,
    candidate: CutoverCandidate,
    *,
    injected_failure: str | None = None,
    recovery_policy: str = "rollback",
    consumed_transaction_ids: frozenset[str] = frozenset(),
) -> RehearsalReceipt:
    """Simulate a coordinated bundle swap and deterministic fail recovery.

    The logical model selector, Minotaur graduation state and evidence index
    are a *single frozen value*. There is no externally observable partially
    updated intermediate bundle. On any injected failure the caller receives
    an intact previous snapshot (rollback), or may select forward recovery
    only when all components were already atomically swapped. Neither choice
    is permission to perform these transitions on real endpoints.
    """
    # Freeze each caller declaration into *new* exact-class values before
    # performing any user-observable hash calculation. Otherwise a callback
    # can replace candidate.proposed after admission and produce a journal
    # claiming a different staged state than the simulated active state.
    if type(candidate) is not CutoverCandidate:
        raise CutoverInputError("invalid proposed cutover declaration")
    previous = _snapshot_bundle(previous)
    proposed_source = candidate.proposed
    proposed = _snapshot_bundle(proposed_source)
    candidate_fields = (
        candidate.transaction_id,
        candidate.expected_previous_commitment,
        candidate.graduation_handoff_sha256,
    )
    if (candidate.proposed is not proposed_source or candidate_fields != (
            candidate.transaction_id,
            candidate.expected_previous_commitment,
            candidate.graduation_handoff_sha256)):
        raise CutoverInputError("candidate changed during synthetic snapshot")
    candidate = CutoverCandidate(
        candidate_fields[0],
        candidate_fields[1],
        proposed,
        candidate_fields[2],
    )
    _validate_bundle(previous)
    _validate_candidate(previous, candidate, consumed_transaction_ids)
    # Never compare caller objects with phase/policy strings before type
    # checking: a custom __eq__ can impersonate any enum member and even
    # reach the JSON journal as a nonserializable object.
    if (injected_failure is not None
            and (type(injected_failure) is not str
                 or injected_failure not in CUTOVER_STEPS)):
        raise CutoverInputError("unknown failure injection phase")
    if type(recovery_policy) is not str or recovery_policy not in ("rollback", "forward"):
        raise CutoverInputError("unsupported recovery policy")

    # No separate writes to model selection, gate custody or evidence index:
    # preparing a new immutable bundle never mutates the one that is active.
    active = previous
    executed: list[str] = []
    failure_step: str | None = None
    swapped = False
    for step in CUTOVER_STEPS:
        executed.append(step)
        if step == "ATOMIC_SWAP":
            active = candidate.proposed
            swapped = True
        if step == injected_failure:
            failure_step = step
            break

    if failure_step is None:
        status = "SIMULATED_COMMIT_COMPLETE"
    elif swapped and recovery_policy == "forward":
        status = "SIMULATED_FORWARD_RECOVERY"
    else:
        # Rollback simultaneously restores the model selection, Minotaur gate
        # state and evidence index, and preserves the staged candidate solely
        # for auditable synthetic investigation.
        active = previous
        status = "SIMULATED_ROLLBACK_COMPLETE"

    prior_commit = bundle_commitment(previous)
    # Source-integrity assertion, not independent artifact custody. This
    # remains useful if future maintenance adds a mutable input boundary.
    if prior_commit != candidate.expected_previous_commitment:
        raise CutoverInputError("previous commitment changed during cutover rehearsal")
    active_commit = bundle_commitment(active)
    journal_sha = _canonical_sha({
        "domain": _DOMAIN,
        "tx_id": candidate.transaction_id,
        "prior": prior_commit,
        "proposal": bundle_commitment(candidate.proposed),
        "handoff": candidate.graduation_handoff_sha256,
        "steps": executed,
        "failure_step": failure_step,
        "recovery_policy": recovery_policy,
        "result": status,
        "active": active_commit,
    })
    return RehearsalReceipt(
        status, active, previous, candidate.proposed, tuple(executed),
        failure_step, prior_commit, active_commit, journal_sha,
        synthetic_rollback_preserved=(
            status == "SIMULATED_ROLLBACK_COMPLETE" and active == previous
        ),
    )
