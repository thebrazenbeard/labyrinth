"""SOURCE-ONLY reference core for a separately hosted Journey authority.

This is executable *service-side policy and persistence logic*, not a host
service, listener, authenticated API, deployment, or authority qualification.
The service process must be operated outside the trainer's filesystem,
administrator, signing-key and rollback domains. Its transport must supply a
verified peer principal from OS/mTLS credentials, NOT a trainer JSON field.
The provisioning roster and verified legacy migration witness must likewise
come from an independent host operator. This module cannot attest any of
those boundary conditions; it treats them as **unverified caller premises**.

Unlike a request-digest-only adapter, this core receives the complete R17
Journey binding, validates it against a service-provisioned roster, and only
then atomically consumes a globally unique event, exam and operation nonce.
It persists a signed receipt before returning an ACK, permitting read-only
lost-ACK reconciliation. SQLite here is durable *against normal process
restart*, not resistant to rollback by a principal controlling its volume.

Only DEVELOPMENT_TRAIN_ONLY reservation is supported. No model training,
graduation, host lease, external monotonic anchor or physical qualification
is asserted by this implementation.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Sequence

from .external_monotonic_authority_contract import (
    ExternalAuthorityHead, ExternalReservationReceipt,
    ExternalReservationRequest, next_head_digest, _identifier, _digest,
)
from .journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, verify_journey_external_request,
)


_SERVICE_SCHEMA = "labyrinth.source-only-journey-authority-core.v1"
_ROSTER_DOMAIN = b"labyrinth.source-only-journey-authority-roster.v1\0"
_INITIAL_DOMAIN = b"labyrinth.source-only-journey-authority-initial.v1\0"
_RESERVATION_DOMAIN = b"labyrinth.source-only-journey-reservation.v1\0"


class AuthorityCoreError(ValueError):
    """Source contract, service provisioning or persisted state invalid."""


class AuthorityCoreConflict(AuthorityCoreError):
    """The authenticated identity, source pins or atomic CAS were refused."""


class AuthorityCoreOutcomeUnknown(AuthorityCoreError):
    """Storage may have committed: reconcile a receipt; never blindly retrain."""


@dataclass(frozen=True)
class HostProvisionedJourney:
    """Operator-provisioned source pins; NOT proof of the operator or witness.

    An actual independently hosted service must validate migration custody
    *before* constructing this record. The two pins below prevent accidental
    source mismatch, not a caller lying about witness verification.
    """

    binding: JourneyExternalAuthorityBinding
    authenticated_principal_id: str
    operator_pinned_ledger_root_sha256: str
    operator_pinned_migration_witness_sha256: str
    initial_authority_epoch: int
    initial_fence: int

    def validate(self) -> None:
        if type(self) is not HostProvisionedJourney or type(self.binding) is not JourneyExternalAuthorityBinding:
            raise AuthorityCoreError("exact typed operator-provisioned Journey required")
        try:
            self.binding.validate()
            _identifier(self.authenticated_principal_id, "verified peer principal")
            _digest(self.operator_pinned_ledger_root_sha256, "operator legacy ledger root")
            _digest(self.operator_pinned_migration_witness_sha256, "operator migration witness")
        except ValueError as exc:
            raise AuthorityCoreError("invalid operator-provisioned Journey") from exc
        if (self.operator_pinned_ledger_root_sha256
                != self.binding.consumed_event_ledger_root_sha256
                or self.operator_pinned_migration_witness_sha256
                != self.binding.migration_witness_sha256):
            raise AuthorityCoreError("Journey legacy history differs from operator-provisioned pins")
        if (type(self.initial_authority_epoch) is not int
                or not 1 <= self.initial_authority_epoch < 2**63 - 1
                or type(self.initial_fence) is not int
                or not 0 <= self.initial_fence < 2**63 - 1):
            raise AuthorityCoreError("invalid operator-provisioned epoch/fence")

    def canonical_record(self) -> dict[str, object]:
        self.validate()
        return {
            "binding": self.binding.payload().decode("ascii"),
            "principal_id": self.authenticated_principal_id,
            "ledger_root_sha256": self.operator_pinned_ledger_root_sha256,
            "migration_witness_sha256": self.operator_pinned_migration_witness_sha256,
            "epoch": self.initial_authority_epoch,
            "fence": self.initial_fence,
        }


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _roster_digest(roster: Sequence[HostProvisionedJourney]) -> str:
    return sha256(_ROSTER_DOMAIN + (_json([
        record.canonical_record() for record in sorted(
            roster, key=lambda row: row.binding.run_id)
    ]) + "\n").encode("ascii")).hexdigest()


def _initial_head(record: HostProvisionedJourney) -> str:
    return sha256(_INITIAL_DOMAIN + record.binding.digest().encode("ascii")
                  + b"\0" + str(record.initial_authority_epoch).encode("ascii")
                  + b"\0" + str(record.initial_fence).encode("ascii")).hexdigest()


def _public_signing_key(private_key: bytes):
    if type(private_key) is not bytes or len(private_key) != 32:
        raise AuthorityCoreError("service signing seed must be exactly 32 bytes")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        signer = Ed25519PrivateKey.from_private_bytes(private_key)
        public = signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    except (ImportError, ValueError, TypeError) as exc:
        raise AuthorityCoreError("Ed25519 service signing is unavailable") from exc
    return signer, public


class SourceOnlyJourneyAuthorityCore:
    """Transactional policy core requiring independently authenticated inputs.

    No transport is included. `authenticated_principal_id` is a trusted server
    adapter injection in this SOURCE test seam, never a client-supplied claim.
    The constructor is an operator-only provisioning entry point; exposing it
    to a trainer would invalidate the trust model. A separate host must secure
    the key, database, roster, process identity, filesystem and migration.
    """

    def __init__(self, *, database_path: Path, provider_id: str,
                 service_signing_key: bytes,
                 provisioned_journeys: tuple[HostProvisionedJourney, ...]):
        if not isinstance(database_path, Path) or not database_path.parent.is_dir():
            raise AuthorityCoreError("existing host-operated database parent required")
        try:
            self.provider_id = _identifier(provider_id, "authority provider")
        except ValueError as exc:
            raise AuthorityCoreError("invalid authority provider") from exc
        if (type(provisioned_journeys) is not tuple or not provisioned_journeys
                or any(type(row) is not HostProvisionedJourney for row in provisioned_journeys)):
            raise AuthorityCoreError("nonempty exact typed host provisioning roster required")
        seen = set()
        for row in provisioned_journeys:
            row.validate()
            if row.binding.provider_id != self.provider_id:
                raise AuthorityCoreError("provisioned Journey has foreign authority provider")
            if row.binding.run_id in seen:
                raise AuthorityCoreError("duplicate provisioned run")
            seen.add(row.binding.run_id)
        self._database_path = database_path
        self._roster = {row.binding.run_id: row for row in provisioned_journeys}
        self._roster_sha256 = _roster_digest(provisioned_journeys)
        self._signer, public_key = _public_signing_key(service_signing_key)
        self.public_key = public_key  # Public inspection only, never a trust pin.
        self._public_key_sha256 = sha256(public_key).hexdigest()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self._database_path), isolation_level=None,
                             timeout=10.0)
        db.execute("PRAGMA busy_timeout = 10000")
        db.execute("PRAGMA synchronous = FULL")
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS authority_config (
                      id INTEGER PRIMARY KEY CHECK (id = 1),
                      schema TEXT NOT NULL, provider_id TEXT NOT NULL,
                      public_key_sha256 TEXT NOT NULL, roster_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS journey_runs (
                      run_id TEXT PRIMARY KEY, binding_payload TEXT NOT NULL,
                      principal_id TEXT NOT NULL, provider_id TEXT NOT NULL,
                      epoch INTEGER NOT NULL, fence INTEGER NOT NULL,
                      commit_index INTEGER NOT NULL, head_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS reservations (
                      reservation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                      event_id TEXT NOT NULL UNIQUE, exam_id TEXT NOT NULL UNIQUE,
                      operation_nonce TEXT NOT NULL UNIQUE,
                      request_sha256 TEXT NOT NULL UNIQUE,
                      receipt_json TEXT NOT NULL,
                      FOREIGN KEY(run_id) REFERENCES journey_runs(run_id)
                    );
                """)
                # Lock after schema creation, which executescript commits.
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute(
                    "SELECT schema, provider_id, public_key_sha256, roster_sha256 "
                    "FROM authority_config WHERE id=1").fetchone()
                expected = (_SERVICE_SCHEMA, self.provider_id,
                            self._public_key_sha256, self._roster_sha256)
                if existing is None:
                    db.execute("INSERT INTO authority_config VALUES (1, ?, ?, ?, ?)", expected)
                    for row in self._roster.values():
                        db.execute(
                            "INSERT INTO journey_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (row.binding.run_id, row.binding.payload().decode("ascii"),
                             row.authenticated_principal_id, self.provider_id,
                             row.initial_authority_epoch, row.initial_fence, 0,
                             _initial_head(row)))
                elif existing != expected:
                    raise AuthorityCoreConflict(
                        "existing service key, provider or host provisioning roster differs")
                db.commit()
        except sqlite3.Error as exc:
            raise AuthorityCoreOutcomeUnknown("host storage initialization unverified") from exc

    def read_head(self, run_id: str) -> ExternalAuthorityHead:
        try:
            _identifier(run_id, "run")
            if run_id not in self._roster:
                raise AuthorityCoreConflict("unprovisioned run")
            with closing(self._connect()) as db:
                row = db.execute(
                    "SELECT provider_id, epoch, fence, commit_index, head_sha256 "
                    "FROM journey_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise AuthorityCoreConflict("missing host-provisioned run")
            head = ExternalAuthorityHead(row[0], run_id, row[1], row[2], row[3], row[4])
            head.validate()
            return head
        except sqlite3.Error as exc:
            raise AuthorityCoreOutcomeUnknown("host authority head unavailable") from exc

    def read_receipt(self, run_id: str, event_id: str,
                     operation_nonce: str) -> ExternalReservationReceipt | None:
        """Read-only recovery after lost ACK. Caller must verify host key/pins."""
        for value, field in ((run_id, "run"), (event_id, "event"),
                             (operation_nonce, "operation nonce")):
            _identifier(value, field)
        try:
            with closing(self._connect()) as db:
                row = db.execute(
                    "SELECT receipt_json FROM reservations "
                    "WHERE run_id=? AND event_id=? AND operation_nonce=?",
                    (run_id, event_id, operation_nonce)).fetchone()
            if row is None:
                return None
            receipt = self._verified_stored_receipt(row[0])
            if receipt.run_id != run_id or receipt.event_id != event_id:
                raise AuthorityCoreOutcomeUnknown("stored receipt identity inconsistent")
            return receipt
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise AuthorityCoreOutcomeUnknown("signed receipt readback unavailable") from exc

    def _verified_stored_receipt(self, data: str) -> ExternalReservationReceipt:
        """Check host DB bytes before using them for idempotent reconciliation."""
        try:
            from cryptography.exceptions import InvalidSignature
            result = ExternalReservationReceipt(**json.loads(data))
            self._signer.public_key().verify(
                bytes.fromhex(result.signature_hex), result.signed_payload())
            return result
        except (ValueError, TypeError, InvalidSignature) as exc:
            raise AuthorityCoreOutcomeUnknown("stored signed receipt corrupted") from exc

    def reserve_journey(self, binding: JourneyExternalAuthorityBinding,
                        request: ExternalReservationRequest, *,
                        authenticated_principal_id: str) -> ExternalReservationReceipt:
        """Source-only service-side reservation; never call from trainer code.

        Duplicate exact request returns the existing durable receipt, not a
        second reservation. A changed request/nonce remains a hard conflict.
        """
        if (type(binding) is not JourneyExternalAuthorityBinding
                or type(request) is not ExternalReservationRequest):
            raise AuthorityCoreError("exact Journey binding and reservation request required")
        try:
            binding.validate()
            request.validate()
            _identifier(authenticated_principal_id, "authenticated principal")
        except ValueError as exc:
            raise AuthorityCoreError("invalid Journey authority request") from exc
        pin = self._roster.get(binding.run_id)
        if (pin is None or authenticated_principal_id != pin.authenticated_principal_id
                or binding.payload() != pin.binding.payload()
                or request.provider_id != self.provider_id):
            raise AuthorityCoreConflict("full Journey binding or authenticated principal not provisioned")
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT binding_payload, principal_id, provider_id, epoch, fence, "
                    "commit_index, head_sha256 FROM journey_runs WHERE run_id=?",
                    (binding.run_id,)).fetchone()
                if (row is None or row[0] != binding.payload().decode("ascii")
                        or row[1] != authenticated_principal_id
                        or row[2] != self.provider_id):
                    raise AuthorityCoreConflict("persisted host source pins changed")

                previous = db.execute(
                    "SELECT request_sha256, receipt_json FROM reservations "
                    "WHERE operation_nonce=?", (request.operation_nonce,)).fetchone()
                if previous is not None:
                    if previous[0] != request.digest():
                        raise AuthorityCoreConflict("operation nonce reused with different intent")
                    result = self._verified_stored_receipt(previous[1])
                    if (result.run_id != binding.run_id
                            or result.event_id != binding.event_id
                            or result.provider_id != self.provider_id
                            or result.request_sha256 != request.digest()):
                        raise AuthorityCoreConflict("operation nonce collides across runs")
                    db.commit()
                    return result

                head = ExternalAuthorityHead(row[2], binding.run_id,
                                             row[3], row[4], row[5], row[6])
                if head != request.expected_head():
                    raise AuthorityCoreConflict("stale or substituted authority CAS head")
                try:
                    verify_journey_external_request(
                        binding, head, request,
                        operation_nonce=request.operation_nonce)
                except ValueError as exc:
                    raise AuthorityCoreConflict("request differs from complete Journey preimage") from exc

                reservation_id = sha256(
                    _RESERVATION_DOMAIN + request.digest().encode("ascii")
                    + b"\0" + request.operation_nonce.encode("ascii")).hexdigest()
                unsigned = ExternalReservationReceipt(
                    provider_id=self.provider_id, run_id=binding.run_id,
                    event_id=binding.event_id, reservation_id=reservation_id,
                    request_sha256=request.digest(), epoch=head.epoch,
                    fence=head.fence, commit_index=head.commit_index + 1,
                    previous_head_sha256=head.head_sha256,
                    new_head_sha256=next_head_digest(request, reservation_id),
                    signature_hex="",
                )
                receipt = replace(
                    unsigned,
                    signature_hex=self._signer.sign(unsigned.signed_payload()).hex())
                # UNIQUE(event_id), UNIQUE(exam_id) and UNIQUE(operation_nonce)
                # all apply across run identities in this one host database.
                db.execute(
                    "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (reservation_id, binding.run_id, binding.event_id,
                     binding.exam_id, request.operation_nonce, request.digest(),
                     _json(asdict(receipt))))
                updated = db.execute(
                    "UPDATE journey_runs SET commit_index=?, head_sha256=? "
                    "WHERE run_id=? AND epoch=? AND fence=? AND commit_index=? "
                    "AND head_sha256=?",
                    (head.commit_index + 1, receipt.new_head_sha256,
                     binding.run_id, head.epoch, head.fence,
                     head.commit_index, head.head_sha256)).rowcount
                if updated != 1:
                    raise AuthorityCoreConflict("atomic head CAS failed")
                db.commit()
                return receipt
        except sqlite3.IntegrityError as exc:
            raise AuthorityCoreConflict(
                "event, exam, operation nonce or request already globally consumed") from exc
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            raise AuthorityCoreOutcomeUnknown(
                "host transaction outcome uncertain; reconcile before any effect") from exc
