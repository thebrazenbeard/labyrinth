"""Runnable SQLite CAS port for the Labyrinth external-reservation contract.

This is a *storage/service prototype*, NOT evidence of an independent host.
Operator bootstrap, intent admission, DB custody and the Ed25519 signing key
must be separated from the trainer by a real OS principal/service boundary.
A copied DB and copied signing key can authorize the same event twice.

The model process must never receive this class, its signer, the DB file, or
its operator methods in a qualification deployment. A protected transport and
host-managed API will be required before any physical release.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Callable, Iterator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict,
    ExternalAuthorityError,
    ExternalAuthorityHead,
    ExternalReservationReceipt,
    ExternalReservationRequest,
    next_head_digest,
)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


class SQLiteReservationPort:
    """One SQLite state store implementing the *source* ExternalAuthorityPort.

    Transactions serialize concurrent writers using BEGIN IMMEDIATE. A
    separate privileged operator first provisions the run head and admits the
    complete expected request. Trainer-chosen but structurally valid requests
    are rejected unless the operator had already pinned their exact bytes.

    Signing uses an injected 64-byte Ed25519 signature function rather than
    storing a private key in SQLite. That function MUST live only in the
    authority's separate trust domain for any physical claim.
    """

    def __init__(self, *, db_path: str | Path, provider_id: str,
                 sign_bytes: Callable[[bytes], bytes],
                 pinned_ed25519_public_key: bytes):
        if type(provider_id) is not str or not _ID.fullmatch(provider_id):
            raise ExternalAuthorityError("invalid authority provider")
        if not callable(sign_bytes):
            raise TypeError("host-held signing function required")
        if type(pinned_ed25519_public_key) is not bytes or len(pinned_ed25519_public_key) != 32:
            raise ExternalAuthorityError("exact pinned Ed25519 public key required")
        if type(db_path) is not str and not isinstance(db_path, Path):
            raise TypeError("explicit SQLite path required")
        self._db_path = Path(db_path)
        if not self._db_path.is_absolute():
            raise ExternalAuthorityError("absolute SQLite path required")
        for component in (self._db_path, *self._db_path.parents):
            if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
                raise ExternalAuthorityError("SQLite path may not traverse symlinks or junctions")
        if not self._db_path.parent.is_dir():
            raise ExternalAuthorityError("database parent must exist")
        self._provider = provider_id
        self._sign = sign_bytes
        self._public_key = Ed25519PublicKey.from_public_bytes(pinned_ed25519_public_key)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._db_path), timeout=5.0, isolation_level=None,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ExternalAuthorityError("unknown authority storage schema")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS heads (
                    provider_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    fence INTEGER NOT NULL,
                    commit_index INTEGER NOT NULL,
                    head_sha256 TEXT NOT NULL,
                    PRIMARY KEY (provider_id, run_id)
                );
                CREATE TABLE IF NOT EXISTS pinned_intents (
                    provider_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    exam_id TEXT NOT NULL,
                    operation_nonce TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_payload BLOB NOT NULL,
                    PRIMARY KEY (provider_id, run_id, event_id),
                    UNIQUE (provider_id, run_id, exam_id),
                    UNIQUE (provider_id, run_id, operation_nonce)
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    provider_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    exam_id TEXT NOT NULL,
                    operation_nonce TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    PRIMARY KEY (provider_id, run_id, event_id),
                    UNIQUE (provider_id, run_id, exam_id),
                    UNIQUE (provider_id, run_id, operation_nonce)
                );
                PRAGMA user_version=1;
            """)
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _head(row: tuple[object, ...] | None) -> ExternalAuthorityHead:
        if row is None:
            raise ExternalAuthorityConflict("unknown externally provisioned run")
        head = ExternalAuthorityHead(*row)
        head.validate()
        return head

    def _load_head(self, connection: sqlite3.Connection,
                   run_id: str) -> ExternalAuthorityHead:
        row = connection.execute(
            "SELECT provider_id, run_id, epoch, fence, commit_index, "
            "head_sha256 FROM heads WHERE provider_id=? AND run_id=?",
            (self._provider, run_id),
        ).fetchone()
        return self._head(row)

    def operator_bootstrap_run(self, head: ExternalAuthorityHead) -> None:
        """ADMIN-ONLY provisioning; no trainer-controlled access is implied."""
        if type(head) is not ExternalAuthorityHead:
            raise ExternalAuthorityError("typed initial head required")
        head.validate()
        if head.provider_id != self._provider:
            raise ExternalAuthorityConflict("foreign provider head")
        try:
            with self._write() as connection:
                connection.execute(
                    "INSERT INTO heads VALUES (?, ?, ?, ?, ?, ?)",
                    (head.provider_id, head.run_id, head.epoch, head.fence,
                     head.commit_index, head.head_sha256),
                )
        except sqlite3.IntegrityError as exc:
            raise ExternalAuthorityConflict("run already provisioned") from exc

    def operator_stage_intent(self, request: ExternalReservationRequest) -> None:
        """ADMIN-ONLY exact intent admission before any model execution."""
        if type(request) is not ExternalReservationRequest:
            raise ExternalAuthorityError("typed intent required")
        request.validate()
        if request.provider_id != self._provider:
            raise ExternalAuthorityConflict("foreign provider intent")
        try:
            with self._write() as connection:
                if self._load_head(connection, request.run_id) != request.expected_head():
                    raise ExternalAuthorityConflict("cannot pin intent to stale head")
                connection.execute(
                    "INSERT INTO pinned_intents VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (request.provider_id, request.run_id, request.event_id,
                     request.exam_id, request.operation_nonce,
                     request.digest(), request.payload()),
                )
        except sqlite3.IntegrityError as exc:
            raise ExternalAuthorityConflict("duplicate admitted event/exam/nonce") from exc

    def read_head(self, run_id: str) -> ExternalAuthorityHead:
        if type(run_id) is not str or not _ID.fullmatch(run_id):
            raise ExternalAuthorityError("invalid run id")
        connection = self._connect()
        try:
            return self._load_head(connection, run_id)
        finally:
            connection.close()

    def read_receipt(self, run_id: str,
                     event_id: str) -> ExternalReservationReceipt | None:
        """Read-only reconciliation after an ambiguous/lost acknowledgment."""
        for label, value in (("run", run_id), ("event", event_id)):
            if type(value) is not str or not _ID.fullmatch(value):
                raise ExternalAuthorityError(f"invalid {label} id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT receipt_json FROM reservations WHERE provider_id=? "
                "AND run_id=? AND event_id=?",
                (self._provider, run_id, event_id),
            ).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            receipt = ExternalReservationReceipt(**value)
            receipt.signed_payload()
            return receipt
        finally:
            connection.close()

    def reserve_once(self, request: ExternalReservationRequest) -> ExternalReservationReceipt:
        if type(request) is not ExternalReservationRequest:
            raise ExternalAuthorityError("typed reservation required")
        request.validate()
        if request.provider_id != self._provider:
            raise ExternalAuthorityConflict("foreign provider reservation")
        try:
            with self._write() as connection:
                pinned = connection.execute(
                    "SELECT request_sha256, request_payload FROM pinned_intents "
                    "WHERE provider_id=? AND run_id=? AND event_id=?",
                    (self._provider, request.run_id, request.event_id),
                ).fetchone()
                if (pinned is None or pinned[0] != request.digest()
                        or pinned[1] != request.payload()):
                    raise ExternalAuthorityConflict("unadmitted or altered host intent")
                if self._load_head(connection, request.run_id) != request.expected_head():
                    raise ExternalAuthorityConflict("stale external monotonic head")
                # Protected by the transaction and UNIQUE constraints. A
                # crashed/lost-ACK caller cannot consume the same event twice.
                duplicate = connection.execute(
                    "SELECT 1 FROM reservations WHERE provider_id=? "
                    "AND run_id=? AND (event_id=? OR exam_id=? OR operation_nonce=?)",
                    (self._provider, request.run_id, request.event_id,
                     request.exam_id, request.operation_nonce),
                ).fetchone()
                if duplicate:
                    raise ExternalAuthorityConflict("event/exam/nonce already consumed")
                reservation_id = "reservation-" + secrets.token_hex(16)
                unsigned = ExternalReservationReceipt(
                    provider_id=request.provider_id, run_id=request.run_id,
                    event_id=request.event_id, reservation_id=reservation_id,
                    request_sha256=request.digest(), epoch=request.expected_epoch,
                    fence=request.expected_fence,
                    commit_index=request.expected_commit_index + 1,
                    previous_head_sha256=request.expected_head_sha256,
                    new_head_sha256=next_head_digest(request, reservation_id),
                    signature_hex="",
                )
                signature = self._sign(unsigned.signed_payload())
                if type(signature) is not bytes or len(signature) != 64:
                    raise ExternalAuthorityError("host signer returned invalid Ed25519 signature")
                try:
                    self._public_key.verify(signature, unsigned.signed_payload())
                except InvalidSignature as exc:
                    raise ExternalAuthorityError("host signer does not match pinned authority key") from exc
                receipt = replace(unsigned, signature_hex=signature.hex())
                connection.execute(
                    "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (request.provider_id, request.run_id, request.event_id,
                     request.exam_id, request.operation_nonce,
                     request.digest(), json.dumps(asdict(receipt), sort_keys=True)),
                )
                cursor = connection.execute(
                    "UPDATE heads SET commit_index=?, head_sha256=? "
                    "WHERE provider_id=? AND run_id=? AND epoch=? AND fence=? "
                    "AND commit_index=? AND head_sha256=?",
                    (receipt.commit_index, receipt.new_head_sha256,
                     request.provider_id, request.run_id, request.expected_epoch,
                     request.expected_fence, request.expected_commit_index,
                     request.expected_head_sha256),
                )
                if cursor.rowcount != 1:
                    raise ExternalAuthorityConflict("lost atomic head compare-and-swap")
                return receipt
        except sqlite3.IntegrityError as exc:
            raise ExternalAuthorityConflict("duplicate reservation blocked by store") from exc
