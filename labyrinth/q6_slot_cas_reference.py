"""Reference-only Q6 exposure-slot CAS; NOT an independent host authority.

The separately signed campaign reservation in ``q6_governed_campaign_contract``
is necessary but insufficient: every planned TRAIN visit and EVAL observation
must be consumed *before* its prompt reaches a model. This reference describes
that per-slot transaction ordering, including lost acknowledgements and
completion readback, without pretending a Python in-memory lock can establish
rollback-resistant custody or an authenticated evaluator process.

Do not use this module to release private exam material, certify hardware,
promote a checkpoint, or authorize graduation. An owner-operated authority
must replace the in-memory ledger and externally anchor every transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from threading import RLock
from typing import Callable, TypeVar

from .q6_governed_campaign_contract import (
    Q6CampaignManifest, Q6CampaignReservationClient,
    Q6CampaignReservationIntent, Q6CampaignReservationReceipt,
    _MANIFEST_DOMAIN,
)


_SLOT_DOMAIN = b"labyrinth.q6-slot-reference.v1\0"
_EVENT_DOMAIN = b"labyrinth.q6-slot-event-reference.v1\0"
_HEAD_DOMAIN = b"labyrinth.q6-slot-head-reference.v1\0"
_HEX = frozenset("0123456789abcdef")
_MAX_INDEX = 2**63 - 1
_RESERVED = "RESERVED"
_REVEALED = "REVEALED_OR_CONSUMED"
_COMPLETED = "COMPLETED"
T = TypeVar("T")


class Q6SlotCasError(ValueError):
    """Invalid transition, stale expected head, or sealed-slot reuse."""


class Q6SlotOutcomeUnknown(Q6SlotCasError):
    """A transition might have committed; use read-only reconciliation."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _HEX for c in value):
        raise Q6SlotCasError(f"invalid {label}")
    return value


def _index(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < _MAX_INDEX:
        raise Q6SlotCasError(f"invalid {label}")
    return value


@dataclass(frozen=True)
class Q6SlotAddress:
    """Public, gold-free address of one preregistered exposure visit."""

    partition: str
    case_id: str
    arm: str
    phase: str
    visit: int
    slot_id: str
    prompt_sha256: str
    arithmetic_semantic_sha256: str


@dataclass(frozen=True)
class Q6SlotLedgerHead:
    epoch: int
    fence: int
    commit_index: int
    head_sha256: str
    reservation_id: str


@dataclass(frozen=True)
class Q6SlotTransition:
    slot_id: str
    event: str
    state: str
    slot_version: int
    epoch: int
    fence: int
    commit_index: int
    previous_head_sha256: str
    new_head_sha256: str
    reservation_id: str
    result_sha256: str | None = None
    execution_sha256: str | None = None
    checkpoint_sha256: str | None = None
    completion_nonce: str | None = None
    source_only_unqualified: bool = True
    model_execution_authenticated: bool = False
    individual_exposures_host_verified: bool = False
    graduation_authorized: bool = False


@dataclass(frozen=True)
class Q6SlotStatus:
    slot_id: str
    state: str
    slot_version: int
    begin_receipt: Q6SlotTransition | None
    complete_receipt: Q6SlotTransition | None
    source_only_unqualified: bool = True


class Q6SlotCasReference:
    """In-memory demonstration of per-exposure CAS ordering.

    Construction makes one verified *campaign* reservation using the existing
    operator-pinned Ed25519 client; it never accepts a self-issued campaign
    receipt. This is still a process-local teaching/reference implementation.
    All slots start in RESERVED; neither the full private question bank nor
    any answers are stored here. ``begin_exposure`` records the transition
    only in this Python object's memory before returning a reveal capability.
    """

    def __init__(self, *, client: Q6CampaignReservationClient,
                 intent: Q6CampaignReservationIntent,
                 manifest: Q6CampaignManifest,
                 after_commit_hook: Callable[[str], None] | None = None):
        if type(client) is not Q6CampaignReservationClient:
            raise Q6SlotCasError("verified Q6 campaign client required")
        if type(manifest) is not Q6CampaignManifest or type(intent) is not Q6CampaignReservationIntent:
            raise Q6SlotCasError("typed operator-pinned campaign required")
        # Validation and signature/readback happen before any local slot exists.
        manifest_payload = manifest.payload()
        # The owned canonical byte snapshot is the *only* source for future
        # case/slot metadata. Frozen case dataclasses are caller-mutable with
        # object.__setattr__ even after a signed reservation is read back.
        source_manifest = json.loads(manifest_payload)
        source_manifest_sha = sha256(_MANIFEST_DOMAIN + manifest_payload).hexdigest()
        if source_manifest_sha != intent.manifest_sha256:
            raise Q6SlotCasError("manifest snapshot differs from operator-pinned campaign")
        receipt = client.reserve(intent, manifest)
        if type(receipt) is not Q6CampaignReservationReceipt:
            raise Q6SlotCasError("untyped campaign reservation")
        if not hmac.compare_digest(manifest.payload(), manifest_payload):
            raise Q6SlotOutcomeUnknown("campaign changed during external reservation")
        if receipt.manifest_sha256 != source_manifest_sha:
            raise Q6SlotOutcomeUnknown("signed reservation differs from captured campaign")
        self._manifest_sha = source_manifest_sha
        self._campaign_payload = manifest_payload
        self._reservation_id = receipt.reservation_id
        self._epoch = receipt.epoch
        self._fence = receipt.fence
        self._commit_index = receipt.commit_index
        self._head_sha = receipt.new_head_sha256
        self._lock = RLock()
        self._after_commit_hook = after_commit_hook
        self._cases: dict[str, tuple[str, str, str]] = {}
        for case in source_manifest["cases"]:
            self._cases[case["case_id"]] = (
                case["partition"], case["prompt_sha256"],
                case["arithmetic_semantic_sha256"],
            )
        self._planned: dict[tuple[str, str, str, str], int] = {}
        for exposure in source_manifest["planned_exposures"]:
            key = (exposure["partition"], exposure["case_id"],
                   exposure["arm"], exposure["phase"])
            if key in self._planned:
                raise Q6SlotCasError("duplicate exposure scope")
            self._planned[key] = exposure["planned_visits"]
        # Slot history is held as immutable tuples and *never* as returned
        # frozen dataclass objects (object.__setattr__ could otherwise alter it).
        self._begun: dict[str, tuple[object, ...]] = {}
        self._completed: dict[str, tuple[object, ...]] = {}

    @property
    def campaign_sha256(self) -> str:
        return self._manifest_sha

    @property
    def source_only_unqualified(self) -> bool:
        return True

    def read_head(self) -> Q6SlotLedgerHead:
        with self._lock:
            return Q6SlotLedgerHead(self._epoch, self._fence,
                                    self._commit_index, self._head_sha,
                                    self._reservation_id)

    def address(self, partition: str, case_id: str, arm: str,
                phase: str, visit: int) -> Q6SlotAddress:
        """Derive an exact allowed visit; no private answer or prompt bytes."""
        _index(visit, "visit")
        key = (partition, case_id, arm, phase)
        count = self._planned.get(key)
        case = self._cases.get(case_id)
        if (count is None or visit >= count or case is None
                or case[0] != partition):
            raise Q6SlotCasError("slot absent from pinned campaign schedule")
        data = (self._manifest_sha, self._reservation_id, partition,
                case_id, arm, phase, visit, case[1], case[2])
        slot_id = sha256(_SLOT_DOMAIN + _canonical(data)).hexdigest()
        return Q6SlotAddress(partition, case_id, arm, phase, visit, slot_id,
                             case[1], case[2])

    def _slot(self, supplied: Q6SlotAddress) -> Q6SlotAddress:
        if type(supplied) is not Q6SlotAddress:
            raise Q6SlotCasError("typed source slot required")
        correct = self.address(supplied.partition, supplied.case_id,
                               supplied.arm, supplied.phase, supplied.visit)
        if supplied != correct:
            raise Q6SlotCasError("forged or substituted Q6 exposure address")
        return correct

    def _expected_head(self, expected: Q6SlotLedgerHead) -> None:
        if type(expected) is not Q6SlotLedgerHead:
            raise Q6SlotCasError("exact expected ledger head required")
        current = self.read_head()
        if current != expected:
            raise Q6SlotCasError("stale epoch/fence/commit/head; no exposure")

    def _event(self, *, slot_id: str, event: str, version: int,
               result_sha256: str | None = None,
               execution_sha256: str | None = None,
               checkpoint_sha256: str | None = None,
               completion_nonce: str | None = None) -> tuple[object, ...]:
        previous = self._head_sha
        index = self._commit_index + 1
        body = (self._manifest_sha, self._reservation_id, slot_id,
                event, version, self._epoch, self._fence, index,
                previous, result_sha256, execution_sha256,
                checkpoint_sha256, completion_nonce)
        current = sha256(_HEAD_DOMAIN + bytes.fromhex(previous)
                         + _EVENT_DOMAIN + _canonical(body)).hexdigest()
        self._commit_index = index
        self._head_sha = current
        return (slot_id, event, _REVEALED if event == "BEGIN" else _COMPLETED,
                version, self._epoch, self._fence, index, previous, current,
                self._reservation_id, result_sha256, execution_sha256,
                checkpoint_sha256, completion_nonce)

    @staticmethod
    def _transition(values: tuple[object, ...]) -> Q6SlotTransition:
        return Q6SlotTransition(*values)

    def _after_commit(self, event: str) -> None:
        if self._after_commit_hook is not None:
            try:
                self._after_commit_hook(event)
            except Exception as exc:
                # Store was already advanced; caller must inspect it rather
                # than issue a fresh slot nonce or re-show an EVAL case.
                raise Q6SlotOutcomeUnknown(
                    "postcommit acknowledgement unknown; read-only reconcile"
                ) from exc

    def begin_exposure(self, slot: Q6SlotAddress, *,
                       expected_head: Q6SlotLedgerHead,
                       expected_slot_version: int = 0) -> Q6SlotTransition:
        """Consume once *before* any prompt bytes or model callback."""
        # Always use the newly derived canonical address, never the caller's
        # frozen dataclass after validation; object.__setattr__ can race it.
        source_slot_id = self._slot(slot).slot_id
        _index(expected_slot_version, "expected slot version")
        with self._lock:
            self._expected_head(expected_head)
            if expected_slot_version != 0 or source_slot_id in self._begun:
                raise Q6SlotCasError("slot revealed/consumed; never re-expose")
            values = self._event(slot_id=source_slot_id, event="BEGIN", version=1)
            self._begun[source_slot_id] = values
        self._after_commit("BEGIN")
        return self._transition(values)

    def begin_then_call(self, slot: Q6SlotAddress, *,
                        expected_head: Q6SlotLedgerHead,
                        operation: Callable[[Q6SlotAddress], T]) -> tuple[Q6SlotTransition, T]:
        """Reference ordering: consume, then invoke a prompt-delivery callback.

        The caller must control prompt access outside this Python process;
        the source wrapper cannot stop someone directly calling another API.
        A callback failure leaves the slot consumed and unavailable for retry.
        """
        if not callable(operation):
            raise TypeError("model/prompt callback required")
        # Freeze an owned canonical address *before* committing BEGIN. An
        # after-commit hook may mutate even a frozen caller dataclass with
        # object.__setattr__, substituting another valid unconsumed slot.
        # Never reread the caller's address between BEGIN and prompt delivery.
        owned_slot = self._slot(slot)
        begin = self.begin_exposure(owned_slot, expected_head=expected_head)
        if begin.slot_id != owned_slot.slot_id:
            raise Q6SlotCasError("BEGIN receipt does not bind exposed slot")
        return begin, operation(owned_slot)

    def complete_exposure(self, slot: Q6SlotAddress, *,
                          expected_head: Q6SlotLedgerHead,
                          expected_slot_version: int,
                          completion_nonce: str,
                          result_sha256: str,
                          execution_sha256: str,
                          checkpoint_sha256: str) -> Q6SlotTransition:
        """Append one immutable score completion, with exact idempotent readback."""
        source_slot_id = self._slot(slot).slot_id
        _index(expected_slot_version, "expected slot version")
        if (type(completion_nonce) is not str or not completion_nonce
                or len(completion_nonce) > 128 or not completion_nonce.isascii()
                or not all(c.isalnum() or c in "_.:-" for c in completion_nonce)):
            raise Q6SlotCasError("invalid completion nonce")
        result_sha256 = _hex64(result_sha256, "result")
        execution_sha256 = _hex64(execution_sha256, "execution")
        checkpoint_sha256 = _hex64(checkpoint_sha256, "checkpoint")
        with self._lock:
            prior = self._completed.get(source_slot_id)
            if prior is not None:
                if (expected_slot_version == 1 and
                    prior[10:] == (result_sha256, execution_sha256,
                                   checkpoint_sha256, completion_nonce)):
                    return self._transition(prior)
                raise Q6SlotCasError("completion already pinned; changed outcome")
            self._expected_head(expected_head)
            if expected_slot_version != 1 or source_slot_id not in self._begun:
                raise Q6SlotCasError("only a revealed slot may be completed")
            values = self._event(
                slot_id=source_slot_id, event="COMPLETE", version=2,
                result_sha256=result_sha256,
                execution_sha256=execution_sha256,
                checkpoint_sha256=checkpoint_sha256,
                completion_nonce=completion_nonce,
            )
            self._completed[source_slot_id] = values
        self._after_commit("COMPLETE")
        return self._transition(values)

    def reconcile(self, slot: Q6SlotAddress) -> Q6SlotStatus:
        """Read-only local reference view; never interprets no record as freshness."""
        source_slot_id = self._slot(slot).slot_id
        with self._lock:
            begin = self._begun.get(source_slot_id)
            complete = self._completed.get(source_slot_id)
            return Q6SlotStatus(
                source_slot_id,
                _COMPLETED if complete is not None else _REVEALED if begin else _RESERVED,
                2 if complete else 1 if begin else 0,
                self._transition(begin) if begin else None,
                self._transition(complete) if complete else None,
            )

    def assert_external_head_matches(self, external_head: Q6SlotLedgerHead | None) -> None:
        """Explicitly deny trust after a missing/stale owner head readback.

        This only compares a *supplied* object. It does not authenticate who
        supplied it, cannot prevent snapshot rollback, and never sets a
        qualification flag. Actual deployment must query an independent host.
        """
        if external_head is None or type(external_head) is not Q6SlotLedgerHead:
            raise Q6SlotOutcomeUnknown("independent high-water witness missing")
        with self._lock:
            if self.read_head() != external_head:
                raise Q6SlotOutcomeUnknown("local ledger differs from external high-water")
