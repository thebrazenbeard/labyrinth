"""Opt-in provider-authenticated TRAIN reservation with an external epoch CAS.

This is a runnable, source-level *host-service candidate*, not a host service
installation. The administrator must isolate this process, its five secrets,
its database, and an independently administered monotonic anchor from the
learner. In-process callers with arbitrary Python/SQLite access can bypass it.
The supplied external adapter must durably CAS and authenticate requests;
tests use a synthetic adapter and establish no physical independence.

The external epoch is committed BEFORE the local SQLite projection. If a crash
occurs between them, the event remains spent at the remote authority and the
service refuses retry until an independently governed reconciliation. An old
MAC-valid SQLite snapshot cannot be used while the independent anchor remains
current, but rolling back both stores is outside this module's protection.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Protocol

from .signed_journey_loop import HostLoopReservation


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ZERO = "0" * 64
_MAX_EPOCH = 2**63 - 1
_MAX_EVENTS = 4096
_MAX_PROPOSAL = 1_000_000
_SCHEMA = "labyrinth.host-provider-authority.v1"
_REQUEST = b"labyrinth.host-provider-request.v1\0"
_ANCHOR = b"labyrinth.host-provider-anchor-receipt.v1\0"
_ADVANCE = b"labyrinth.host-provider-anchor-operation.v1\0"
_LOCAL = b"labyrinth.host-provider-local-state.v1\0"
_HOST = b"labyrinth.host-provider-reservation.v1\0"


class HostAuthorityError(ValueError):
    """An untrusted identity, transition, reservation, or local seal failed."""


class HostAuthorityHold(HostAuthorityError):
    """External state may have advanced; do not replay or authorize effects."""


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _digest(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _epoch(value: object) -> bool:
    return type(value) is int and 0 <= value < _MAX_EPOCH


def _secret(value: object) -> bool:
    return type(value) is bytes and len(value) >= 32


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _mac(key: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(key, domain + payload, sha256).hexdigest()


@dataclass(frozen=True)
class ProviderBinding:
    authority_id: str
    host_id: str
    provider_id: str
    provider_key_sha256: str
    policy_sha256: str

    def validate(self) -> None:
        if (not all(_id(v) for v in (self.authority_id, self.host_id, self.provider_id))
                or not _digest(self.provider_key_sha256)
                or not _digest(self.policy_sha256)):
            raise HostAuthorityError("invalid immutable provider/host/authority binding")

    @property
    def sha256(self) -> str:
        self.validate()
        return sha256(b"labyrinth.host-provider-binding.v1\0" + _json({
            "authority_id": self.authority_id, "host_id": self.host_id,
            "provider_id": self.provider_id,
            "provider_key_sha256": self.provider_key_sha256,
            "policy_sha256": self.policy_sha256,
        })).hexdigest()


@dataclass(frozen=True)
class ProviderRequest:
    authority_id: str
    provider_id: str
    binding_sha256: str
    event_id: str
    proposal_sha256: str
    expected_epoch: int
    signature_hex: str

    def signed_bytes(self) -> bytes:
        if (not all(_id(v) for v in (self.authority_id, self.provider_id, self.event_id))
                or not _digest(self.binding_sha256)
                or not _digest(self.proposal_sha256) or not _epoch(self.expected_epoch)):
            raise HostAuthorityError("malformed provider reservation request")
        return _json({"schema": _SCHEMA, "authority_id": self.authority_id,
                      "provider_id": self.provider_id,
                      "binding_sha256": self.binding_sha256, "event_id": self.event_id,
                      "proposal_sha256": self.proposal_sha256,
                      "expected_epoch": self.expected_epoch})


def sign_provider_request(binding: ProviderBinding, *, provider_key: bytes,
                          event_id: str, proposal_bytes: bytes,
                          expected_epoch: int) -> ProviderRequest:
    """Client-side HMAC request creation; key custody is not proven by this API."""
    binding.validate()
    if not _secret(provider_key) or not hmac.compare_digest(
            sha256(provider_key).hexdigest(), binding.provider_key_sha256):
        raise HostAuthorityError("wrong provider signing key")
    if type(proposal_bytes) is not bytes or not 0 < len(proposal_bytes) <= _MAX_PROPOSAL:
        raise HostAuthorityError("invalid proposal bytes")
    request = ProviderRequest(binding.authority_id, binding.provider_id,
                              binding.sha256, event_id,
                              sha256(proposal_bytes).hexdigest(), expected_epoch, "")
    return ProviderRequest(**{**request.__dict__, "signature_hex": _mac(
        provider_key, _REQUEST, request.signed_bytes())})


@dataclass(frozen=True)
class EpochReceipt:
    binding_sha256: str
    epoch: int
    history_sha256: str
    signature_hex: str

    def signed_bytes(self) -> bytes:
        if (not _digest(self.binding_sha256) or not _epoch(self.epoch)
                or not _digest(self.history_sha256)):
            raise HostAuthorityError("malformed external epoch receipt")
        return _json({"schema": _SCHEMA, "binding_sha256": self.binding_sha256,
                      "epoch": self.epoch, "history_sha256": self.history_sha256})


class ExternalProviderAnchor(Protocol):
    """Independently administered, durable authenticated CAS (not provided)."""

    def read(self, binding_sha256: str) -> EpochReceipt: ...

    def compare_and_swap(self, before: EpochReceipt, after: EpochReceipt,
                         event_id: str, proposal_sha256: str,
                         request_mac_hex: str) -> EpochReceipt: ...


def _history(previous: str, event_id: str, proposal_sha256: str) -> str:
    return sha256(b"labyrinth.host-provider-event-history.v1\0" + _json(
        [previous, event_id, proposal_sha256])).hexdigest()


def _exclusive_database_path(path: Path) -> bool:
    """Reject path aliases before opening a host-owned authority store.

    This is a local preflight, not a replacement for OS-enforced ownership or
    descriptor-level race protection in an installed host service.
    """
    if not isinstance(path, Path) or not path.is_absolute():
        return False
    try:
        if not path.parent.is_dir() or path != path.resolve(strict=False):
            return False
        for component in (path, *path.parents):
            if (component.is_symlink() or
                    (hasattr(component, "is_junction") and component.is_junction())):
                return False
        if path.exists():
            return path.is_file() and path.stat().st_nlink == 1
        return True
    except (OSError, RuntimeError, ValueError):
        return False


class HostProviderReservationService:
    """One pinned provider per source-level authority; remote CAS fences events.

    The constructor pins the *object identity* of the supplied remote adapter
    and the complete provider binding. Replacing either after construction is
    refused before another remote effect can be requested. This protects
    against accidental/reentrant source-level rebinding, **not** hostile code
    running in the same Python process or a provider under trainer control.
    Only a separately administered, independently monotonic host can provide
    those security guarantees.
    """

    __slots__ = (
        "db_path", "binding", "external", "_provider_key", "_state_key",
        "_host_key", "_anchor_key", "_anchor_request_key", "_pinned_keys",
        "_pinned_binding_sha256", "_pinned_db_path", "_pinned_external",
        "_pinned",
    )
    # Key assignment remains detectable at operation time for compatibility
    # with existing corruption tests; no rotated key can authorize an effect.
    _IMMUTABLE_FIELDS = frozenset((
        "db_path", "binding", "external", "_pinned_keys",
        "_pinned_binding_sha256", "_pinned_db_path", "_pinned_external",
        "_pinned",
    ))

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_pinned", False) and name in self._IMMUTABLE_FIELDS:
            raise HostAuthorityError("host authority construction binding is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, db_path: Path, binding: ProviderBinding, *,
                 provider_key: bytes, state_key: bytes, host_receipt_key: bytes,
                 anchor_receipt_key: bytes, anchor_request_key: bytes,
                 external: ExternalProviderAnchor):
        binding.validate()
        keys = (provider_key, state_key, host_receipt_key,
                anchor_receipt_key, anchor_request_key)
        if (not all(_secret(key) for key in keys)
                or len(set(keys)) != len(keys)
                or not hmac.compare_digest(sha256(provider_key).hexdigest(),
                                           binding.provider_key_sha256)):
            raise HostAuthorityError("five distinct pinned host/provider secrets required")
        if not _exclusive_database_path(db_path):
            raise HostAuthorityError("absolute unaliased host database required")
        if not callable(getattr(external, "read", None)) or not callable(
                getattr(external, "compare_and_swap", None)):
            raise HostAuthorityError("independent monotonic authority adapter required")
        self.db_path, self.binding, self.external = db_path, binding, external
        self._provider_key, self._state_key = provider_key, state_key
        self._host_key, self._anchor_key, self._anchor_request_key = (
            host_receipt_key, anchor_receipt_key, anchor_request_key)
        self._pinned_keys = tuple(sha256(key).hexdigest() for key in keys)
        self._pinned_binding_sha256 = binding.sha256
        self._pinned_db_path = db_path
        self._pinned_external = external
        self._pinned = True

    def _check_keys(self) -> None:
        # Object.__setattr__ can bypass Python-level immutability. Catch any
        # ordinary post-construction pointer/identity swap before a remote CAS.
        # This is an additional fail-closed tripwire, not host attestation.
        if (self.binding.sha256 != self._pinned_binding_sha256
                or self.db_path != self._pinned_db_path
                or self.external is not self._pinned_external):
            raise HostAuthorityHold("host authority provider or binding changed; no effects")
        keys = (self._provider_key, self._state_key, self._host_key,
                self._anchor_key, self._anchor_request_key)
        if tuple(sha256(key).hexdigest() for key in keys) != self._pinned_keys:
            raise HostAuthorityError("host/provider key changed after binding")

    def _read_remote(self) -> EpochReceipt:
        self._check_keys()
        try:
            receipt = self.external.read(self.binding.sha256)
        except Exception as exc:
            raise HostAuthorityHold("external authority unreadable; no effects") from exc
        self._check_keys()
        if type(receipt) is not EpochReceipt or not _digest(receipt.signature_hex):
            raise HostAuthorityHold("external authority returned malformed receipt")
        if (not hmac.compare_digest(receipt.binding_sha256, self.binding.sha256)
                or not hmac.compare_digest(receipt.signature_hex, _mac(
                    self._anchor_key, _ANCHOR, receipt.signed_bytes()))):
            raise HostAuthorityHold("external authority identity/signature mismatch")
        return receipt

    def _connection(self) -> sqlite3.Connection:
        self._check_keys()
        if not _exclusive_database_path(self.db_path):
            raise HostAuthorityError("authority store path changed or became aliased")
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def initialize(self) -> None:
        """Exclusive provisioning; never overwrites an existing/partial DB."""
        self._check_keys()
        if not _exclusive_database_path(self.db_path):
            raise HostAuthorityError("authority store path changed or became aliased")
        if self.db_path.exists() or self.db_path.is_symlink():
            raise HostAuthorityError("existing authority store cannot be reinitialized")
        genesis = self._read_remote()
        if genesis.epoch != 0 or genesis.history_sha256 != _ZERO:
            raise HostAuthorityHold("external authority is not at genesis")
        try:
            descriptor = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise HostAuthorityError("existing authority store cannot be reinitialized") from exc
        os.close(descriptor)
        with closing(self._connection()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""CREATE TABLE authority (
                    id INTEGER PRIMARY KEY CHECK(id=1), binding_sha TEXT NOT NULL,
                    epoch INTEGER NOT NULL, history_sha TEXT NOT NULL, state_mac TEXT NOT NULL)""")
                conn.execute("""CREATE TABLE consumed (
                    event_id TEXT PRIMARY KEY, proposal_sha TEXT NOT NULL,
                    epoch INTEGER NOT NULL UNIQUE, fence INTEGER NOT NULL,
                    host_signature TEXT NOT NULL)""")
                conn.execute("INSERT INTO authority VALUES (1, ?, 0, ?, ?)",
                             (self.binding.sha256, _ZERO, self._seal(0, _ZERO)))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _seal(self, epoch: int, history_sha: str) -> str:
        return _mac(self._state_key, _LOCAL, _json(
            [self.binding.sha256, epoch, history_sha]))

    def _load(self, conn: sqlite3.Connection) -> tuple[int, str]:
        try:
            rows = conn.execute("SELECT binding_sha, epoch, history_sha, state_mac "
                                "FROM authority").fetchall()
            events = conn.execute("SELECT event_id, proposal_sha, epoch, fence, "
                                  "host_signature FROM consumed ORDER BY epoch").fetchall()
        except sqlite3.Error as exc:
            raise HostAuthorityError("authority schema missing or damaged") from exc
        if len(rows) != 1 or len(rows[0]) != 4:
            raise HostAuthorityError("authority identity record invalid")
        identity, epoch, history_sha, state_mac = rows[0]
        if (identity != self.binding.sha256 or not _epoch(epoch)
                or not _digest(history_sha) or not _digest(state_mac)
                or not hmac.compare_digest(state_mac, self._seal(epoch, history_sha))
                or len(events) != epoch):
            raise HostAuthorityError("authority state seal or event count invalid")
        calculated = _ZERO
        for index, (event_id, proposal_sha, event_epoch, fence, signature) in enumerate(events, 1):
            if (not _id(event_id) or not _digest(proposal_sha)
                    or event_epoch != index or fence != index or not _digest(signature)):
                raise HostAuthorityError("malformed consumed event lineage")
            calculated = _history(calculated, event_id, proposal_sha)
            if not hmac.compare_digest(signature, self._host_signature(
                    event_id, proposal_sha, fence)):
                raise HostAuthorityError("consumed event receipt signature mismatch")
        if not hmac.compare_digest(calculated, history_sha):
            raise HostAuthorityError("consumed history tampered")
        return epoch, history_sha

    def _host_signature(self, event_id: str, proposal_sha: str, fence: int) -> str:
        receipt = HostLoopReservation("labyrinth.host-loop-reservation.v1", event_id,
                                      proposal_sha, fence, self.binding.host_id,
                                      "0" * 64)
        return _mac(self._host_key, _HOST + self.binding.sha256.encode("ascii") + b"\0",
                    receipt.signed_bytes)

    def verify_host_receipt(self, receipt: HostLoopReservation, payload: bytes) -> bool:
        """Source MAC verifier; caller key custody is not host attestation."""
        try:
            self._check_keys()
            return (type(receipt) is HostLoopReservation
                    and receipt.host_id == self.binding.host_id
                    and payload == receipt.signed_bytes
                    and hmac.compare_digest(receipt.signature_hex, _mac(
                        self._host_key,
                        _HOST + self.binding.sha256.encode("ascii") + b"\0",
                        payload)))
        except (ValueError, TypeError):
            return False

    def _validate_proposal(self, request: ProviderRequest, data: bytes) -> int:
        if (type(request) is not ProviderRequest
                or request.authority_id != self.binding.authority_id
                or request.provider_id != self.binding.provider_id
                or request.binding_sha256 != self.binding.sha256
                or not _digest(request.signature_hex)):
            raise HostAuthorityError("wrong authenticated provider identity")
        if not hmac.compare_digest(request.signature_hex, _mac(
                self._provider_key, _REQUEST, request.signed_bytes())):
            raise HostAuthorityError("provider admission signature invalid")
        if (type(data) is not bytes or not 0 < len(data) <= _MAX_PROPOSAL
                or not hmac.compare_digest(sha256(data).hexdigest(), request.proposal_sha256)):
            raise HostAuthorityError("provider proposal changed after signing")
        try:
            record = json.loads(data)
            fields = {"schema", "event_id", "exam_id", "scope_sha256",
                      "contract_sha256", "journey_sha256",
                      "learner_checkpoint_sha256", "signed_binding_sha256",
                      "training_roster_sha256", "prior_cycle", "regenerated_cycle",
                      "expected_host_fence", "weak_skills", "rooms",
                      "answerless_train_pairs"}
            if (type(record) is not dict or record.keys() != fields
                    or _json(record) != data
                    or record.get("schema") != "labyrinth.signed-journey-proposal.v1"
                    or record.get("event_id") != request.event_id
                    or type(record.get("expected_host_fence")) is not int
                    or not _epoch(request.expected_epoch + 1)
                    or record["expected_host_fence"] != request.expected_epoch + 1
                    or not all(_digest(record[key]) for key in (
                        "scope_sha256", "contract_sha256", "journey_sha256",
                        "learner_checkpoint_sha256", "signed_binding_sha256",
                        "training_roster_sha256"))
                    or not _id(record["exam_id"])
                    or type(record["prior_cycle"]) is not int
                    or record["prior_cycle"] < 0
                    or type(record["regenerated_cycle"]) is not int
                    or record["regenerated_cycle"] != record["prior_cycle"] + 1
                    or type(record["rooms"]) is not list or not record["rooms"]
                    or type(record["answerless_train_pairs"]) is not list
                    or not record["answerless_train_pairs"]
                    or type(record["weak_skills"]) is not list
                    or not record["weak_skills"]):
                raise ValueError("unsigned or mismatched TRAIN proposal")
            if (any(type(skill) is not str or not _id(skill)
                    for skill in record["weak_skills"])
                    or len(set(record["weak_skills"])) != len(record["weak_skills"])):
                raise ValueError("invalid declared weakness skills")
            permitted_skills = set(record["weak_skills"])
            room_ids = []
            room_skills = []
            for room in record["rooms"]:
                if (type(room) is not dict
                        or room.keys() != {"room_id", "skill", "case_seed"}
                        or not _id(room["room_id"]) or not _id(room["skill"])
                        or room["skill"] not in permitted_skills
                        or type(room["case_seed"]) is not int
                        or room["case_seed"] < 0):
                    raise ValueError("malformed or answer-bearing TRAIN room")
                room_ids.append(room["room_id"])
                room_skills.append(room["skill"])
            if len(set(room_ids)) != len(room_ids):
                raise ValueError("duplicate TRAIN room identities")
            challenge_ids = []
            view_skills = []
            prompt_digests = []
            for item in record["answerless_train_pairs"]:
                if (type(item) is not dict
                        or item.keys() != {"challenge_id", "prompt_sha256", "skill"}
                        or not _id(item["challenge_id"])
                        or not _digest(item["prompt_sha256"])
                        or not _id(item["skill"])
                        or item["skill"] not in permitted_skills):
                    raise ValueError("malformed or answer-bearing TRAIN view")
                challenge_ids.append(item["challenge_id"])
                view_skills.append(item["skill"])
                prompt_digests.append(item["prompt_sha256"])
            if (len(set(challenge_ids)) != len(challenge_ids)
                    or len(set(prompt_digests)) != len(prompt_digests)
                    or room_skills != view_skills):
                raise ValueError("duplicate or mismatched TRAIN challenge identities")
        except (ValueError, TypeError, UnicodeDecodeError, OverflowError) as exc:
            raise HostAuthorityError("noncanonical or mismatched TRAIN proposal") from exc
        return record["expected_host_fence"]

    def reserve(self, request: ProviderRequest, proposal_bytes: bytes) -> HostLoopReservation:
        """Provider HMAC -> remote epoch CAS -> local consumed transaction.

        A remote postcommit timeout requires manual reconciliation. This method
        does not launch training or authenticate an actual learner checkpoint.
        """
        self._check_keys()
        fence = self._validate_proposal(request, proposal_bytes)
        with closing(self._connection()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                epoch, history_sha = self._load(conn)
                if epoch >= _MAX_EVENTS:
                    raise HostAuthorityError("bounded event reservation capacity exhausted")
                if request.expected_epoch != epoch:
                    raise HostAuthorityError("stale provider epoch or replay")
                if conn.execute("SELECT 1 FROM consumed WHERE event_id=?",
                                (request.event_id,)).fetchone():
                    raise HostAuthorityError("signed event was already consumed")
                before = self._read_remote()
                if (before.epoch != epoch
                        or not hmac.compare_digest(before.history_sha256, history_sha)):
                    raise HostAuthorityHold("external epoch differs from local state; rollback/reconcile")
                next_history = _history(history_sha, request.event_id, request.proposal_sha256)
                after = EpochReceipt(self.binding.sha256, epoch + 1, next_history, "")
                command = _json([before.binding_sha256, before.epoch,
                                 before.history_sha256, after.epoch,
                                 after.history_sha256, request.event_id,
                                 request.proposal_sha256])
                command_mac = _mac(self._anchor_request_key, _ADVANCE, command)
                try:
                    acknowledged = self.external.compare_and_swap(
                        before, after, request.event_id,
                        request.proposal_sha256, command_mac)
                except Exception as exc:
                    raise HostAuthorityHold("remote CAS outcome unknown; reconcile, never retry") from exc
                if (type(acknowledged) is not EpochReceipt
                        or not _digest(acknowledged.signature_hex)
                        or (acknowledged.binding_sha256, acknowledged.epoch,
                            acknowledged.history_sha256) != (
                                after.binding_sha256, after.epoch, after.history_sha256)
                        or not hmac.compare_digest(acknowledged.signature_hex,
                            _mac(self._anchor_key, _ANCHOR, acknowledged.signed_bytes()))):
                    raise HostAuthorityHold("remote CAS acknowledgment untrusted")
                confirmed = self._read_remote()
                if confirmed != acknowledged:
                    raise HostAuthorityHold("remote postcommit readback differs; reconcile")
                signature = self._host_signature(request.event_id,
                                                  request.proposal_sha256, fence)
                try:
                    conn.execute("INSERT INTO consumed VALUES (?, ?, ?, ?, ?)",
                                 (request.event_id, request.proposal_sha256,
                                  epoch + 1, fence, signature))
                    conn.execute("UPDATE authority SET epoch=?, history_sha=?, state_mac=? "
                                 "WHERE id=1 AND epoch=? AND history_sha=?",
                                 (epoch + 1, next_history,
                                  self._seal(epoch + 1, next_history), epoch, history_sha))
                    if conn.execute("SELECT changes()").fetchone()[0] != 1:
                        raise HostAuthorityHold("local projection lost CAS; remote already advanced")
                    conn.commit()
                except Exception as exc:
                    raise HostAuthorityHold("remote advanced but local projection failed") from exc
            except Exception:
                conn.rollback()
                raise
        receipt = HostLoopReservation(
            "labyrinth.host-loop-reservation.v1", request.event_id,
            request.proposal_sha256, fence, self.binding.host_id, signature)
        if self._read_remote() != acknowledged:
            raise HostAuthorityHold("external epoch changed after local commit")
        return receipt

    def reconcile(self, event_id: str) -> HostLoopReservation | None:
        """Read-only exact-current receipt lookup; never reauthorizes training."""
        self._check_keys()
        if not _id(event_id):
            raise HostAuthorityError("invalid event")
        with closing(self._connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            epoch, history = self._load(conn)
            remote = self._read_remote()
            if remote.epoch != epoch or remote.history_sha256 != history:
                raise HostAuthorityHold("reconciliation requires matching external state")
            row = conn.execute("SELECT proposal_sha, fence, host_signature "
                               "FROM consumed WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                return None
            return HostLoopReservation("labyrinth.host-loop-reservation.v1",
                                       event_id, row[0], row[1],
                                       self.binding.host_id, row[2])
