"""SQLite-backed *candidate* sink for atomically visible checkpoint bytes.

The service owns both the current epoch/fence and published artifact bytes in
ONE SQLite transaction. Workers can stage bytes but cannot publish directly
through its public API. A stale fence is checked under BEGIN IMMEDIATE before
the artifact and authority head become visible together. Each database has
one immutable source-level sink-id/key commitment; a second service with a
different key or sink identity cannot claim the existing run via this API.

This is runnable local source, NOT independently hosted authority: anyone
with write access to this process/database or signing key can replace the
database with an older valid snapshot, forge state, or bypass its API.
No network authentication, protected deployment, external monotonic minimum,
host-enforced file permissions, or production checkpoint call-site is supplied.
Therefore it cannot establish Q8 by itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import hmac
from pathlib import Path
import sqlite3
from typing import Iterator

from .atomic_artifact_publication import (
    ArtifactPublicationError, PublicationIntent, PublicationReceipt,
    SinkConflict, canonical_receipt_payload,
)


_GENESIS_DIGEST = "0" * 64


@dataclass(frozen=True)
class VisibleArtifact:
    receipt: PublicationReceipt
    payload: bytes


class SQLiteAtomicPublicationSink:
    """Real DB transaction boundary behind the existing publication protocol.

    Separate handles share one SQLite database. The opening administrator
    reserves a run via register_run; a worker may only stage and request a
    CAS publication. These roles are logical only until deployed on separately
    administered infrastructure with enforced access controls.
    """

    def __init__(self, db_path: Path, *, sink_id: str, receipt_key: bytes):
        if (not isinstance(db_path, Path) or type(sink_id) is not str
                or type(receipt_key) is not bytes or len(receipt_key) < 32):
            raise ArtifactPublicationError("pinned sink path, identity and key required")
        # Reuse the canonical ID validation instead of inventing another grammar.
        PublicationIntent(sink_id, "validation", "validation", "validation",
                          "validation", 1, 0, _GENESIS_DIGEST, "f" * 64).validate()
        self.db_path = db_path
        self.sink_id = sink_id
        self._receipt_key = receipt_key
        self._key_commitment = hmac.new(
            receipt_key,
            b"labyrinth.sqlite-publication-sink-identity.v1\0" + sink_id.encode("ascii"),
            sha256,
        ).hexdigest()
        # Existing database must be opened in place; never delete or replace it.
        with self._connect(require_identity=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""CREATE TABLE IF NOT EXISTS run_authority (
                    run_id TEXT PRIMARY KEY,
                    epoch INTEGER NOT NULL,
                    fence INTEGER NOT NULL,
                    current_sha TEXT NOT NULL,
                    current_operation TEXT
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS staged (
                    staged_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload BLOB NOT NULL
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS publications (
                    operation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    staged_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    fence INTEGER NOT NULL,
                    predecessor_sha TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    transaction_id TEXT NOT NULL,
                    signature_hex TEXT NOT NULL
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS sink_identity (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    sink_id TEXT NOT NULL,
                    key_commitment TEXT NOT NULL
                )""")
                existing = conn.execute(
                    "SELECT sink_id, key_commitment FROM sink_identity "
                    "WHERE singleton=1"
                ).fetchone()
                if existing is None:
                    # Never silently take control of an existing populated
                    # database whose original signing identity is unknown.
                    if any(conn.execute(
                        f"SELECT 1 FROM {table} LIMIT 1"
                    ).fetchone() for table in (
                        "run_authority", "staged", "publications",
                    )):
                        raise SinkConflict("populated unbound sink requires explicit migration")
                    conn.execute(
                        "INSERT INTO sink_identity VALUES (1, ?, ?)",
                        (sink_id, self._key_commitment),
                    )
                self._assert_db_binding(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _assert_db_binding(self, conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute(
                "SELECT sink_id, key_commitment FROM sink_identity WHERE singleton=1"
            ).fetchone()
        except sqlite3.Error as exc:
            raise SinkConflict("sink identity metadata is missing or damaged") from exc
        if (row is None or row != (self.sink_id, self._key_commitment)
                or conn.execute("SELECT COUNT(*) FROM sink_identity").fetchone()[0] != 1):
            raise SinkConflict("database is bound to another sink identity or receipt key")

    @contextmanager
    def _connect(self, *, require_identity: bool = True) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=5000")
            if require_identity:
                self._assert_db_binding(conn)
            yield conn
        finally:
            conn.close()

    def _begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        self._assert_db_binding(conn)

    def register_run(self, run_id: str, *, epoch: int = 1,
                     fence: int = 0, current_sha256: str = _GENESIS_DIGEST) -> None:
        """Administrator-only in a real deployment (not enforced in-process)."""
        PublicationIntent(self.sink_id, "register", run_id, "artifact",
                          "staged", epoch, fence, current_sha256,
                          "f" * 64 if current_sha256 != "f" * 64 else "e" * 64).validate()
        with self._connect() as conn:
            try:
                self._begin(conn)
                conn.execute(
                    "INSERT INTO run_authority VALUES (?, ?, ?, ?, NULL)",
                    (run_id, epoch, fence, current_sha256),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def advance_fence(self, run_id: str, *, expected_epoch: int,
                      expected_fence: int, new_epoch: int,
                      new_fence: int) -> None:
        """Administrator-owned monotonic transition; no artifact is published."""
        if (any(type(value) is not int for value in (
                expected_epoch, expected_fence, new_epoch, new_fence))
                or not 1 <= expected_epoch < 2**63 - 1
                or not 0 <= expected_fence < 2**63 - 1
                or not 1 <= new_epoch < 2**63 - 1
                or not 0 <= new_fence < 2**63 - 1
                or new_epoch != expected_epoch + 1
                or new_fence != expected_fence + 1):
            raise SinkConflict("invalid monotonic epoch/fence transition")
        with self._connect() as conn:
            try:
                self._begin(conn)
                changed = conn.execute(
                    "UPDATE run_authority SET epoch=?, fence=? "
                    "WHERE run_id=? AND epoch=? AND fence=?",
                    (new_epoch, new_fence, run_id, expected_epoch, expected_fence),
                ).rowcount
                if changed != 1:
                    raise SinkConflict("stale administrator fence")
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def stage(self, *, run_id: str, staged_object_id: str, payload: bytes) -> str:
        """Accept private staged bytes, with no public visibility."""
        if type(payload) is not bytes or not payload:
            raise ArtifactPublicationError("nonempty immutable staged bytes required")
        digest = sha256(payload).hexdigest()
        PublicationIntent(self.sink_id, "stage", run_id, "artifact",
                          staged_object_id, 1, 0, _GENESIS_DIGEST, digest).validate()
        with self._connect() as conn:
            try:
                self._begin(conn)
                conn.execute("INSERT INTO staged VALUES (?, ?, ?, ?)",
                             (staged_object_id, run_id, digest, payload))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return digest

    def _signed_receipt(self, intent: PublicationIntent, tx_id: str) -> PublicationReceipt:
        unsigned = PublicationReceipt(
            intent=intent, transaction_id=tx_id,
            current_sha256=intent.staged_sha256, durable=True,
            signature_hex="pending",
        )
        signature = hmac.new(self._receipt_key,
                             canonical_receipt_payload(unsigned), sha256).hexdigest()
        return PublicationReceipt(intent, tx_id, intent.staged_sha256, True, signature)

    def _verify_saved_receipt(self, receipt: PublicationReceipt) -> None:
        receipt.validate()
        expected = hmac.new(self._receipt_key,
                            canonical_receipt_payload(receipt), sha256).hexdigest()
        if not hmac.compare_digest(expected, receipt.signature_hex):
            raise ArtifactPublicationError("published receipt authentication failed")

    def commit_if_current(self, intent: PublicationIntent) -> PublicationReceipt:
        """Publish bytes and move authority head in one SQLite transaction."""
        if type(intent) is not PublicationIntent:
            raise ArtifactPublicationError("typed publication intent required")
        intent.validate()
        if intent.sink_id != self.sink_id:
            raise SinkConflict("wrong sink identity")
        tx_id = "pub-" + sha256(intent.operation_id.encode("ascii")).hexdigest()[:24]
        receipt = self._signed_receipt(intent, tx_id)
        with self._connect() as conn:
            try:
                self._begin(conn)
                # Replays are rejected even if the run moved or a blob remains.
                if conn.execute(
                    "SELECT 1 FROM publications WHERE operation_id=?",
                    (intent.operation_id,),
                ).fetchone():
                    raise SinkConflict("already committed operation")
                authority = conn.execute(
                    "SELECT epoch, fence, current_sha FROM run_authority WHERE run_id=?",
                    (intent.run_id,),
                ).fetchone()
                if authority != (intent.epoch, intent.fence,
                                 intent.expected_current_sha256):
                    raise SinkConflict("stale epoch, fence, or checkpoint")
                staged = conn.execute(
                    "SELECT run_id, digest, payload FROM staged WHERE staged_id=?",
                    (intent.staged_object_id,),
                ).fetchone()
                if (staged is None or staged[0] != intent.run_id
                        or staged[1] != intent.staged_sha256
                        or sha256(staged[2]).hexdigest() != intent.staged_sha256):
                    raise SinkConflict("wrong run or staged-byte commitment")
                # The published BLOB and current authority identity become
                # visible in the *same* COMMIT; no intermediate rename is used.
                conn.execute(
                    "INSERT INTO publications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (intent.operation_id, intent.run_id, intent.artifact_id,
                     intent.staged_object_id, intent.epoch, intent.fence,
                     intent.expected_current_sha256, intent.staged_sha256,
                     staged[2], tx_id, receipt.signature_hex),
                )
                conn.execute(
                    "UPDATE run_authority SET current_sha=?, current_operation=? "
                    "WHERE run_id=?",
                    (intent.staged_sha256, intent.operation_id, intent.run_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return receipt

    def read_current(self, run_id: str) -> VisibleArtifact | None:
        """Read only the artifact currently approved by the committed head."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT p.operation_id, p.artifact_id, p.staged_id,
                       p.epoch, p.fence, p.predecessor_sha, p.digest,
                       p.payload, p.transaction_id, p.signature_hex
                FROM run_authority AS a
                JOIN publications AS p ON p.operation_id=a.current_operation
                WHERE a.run_id=? AND p.run_id=a.run_id
                  AND p.digest=a.current_sha
            """, (run_id,)).fetchone()
        if row is None:
            return None
        intent = PublicationIntent(
            self.sink_id, row[0], run_id, row[1], row[2],
            row[3], row[4], row[5], row[6],
        )
        receipt = PublicationReceipt(intent, row[8], row[6], True, row[9])
        self._verify_saved_receipt(receipt)
        if sha256(row[7]).hexdigest() != receipt.current_sha256:
            raise ArtifactPublicationError("published bytes do not match receipt")
        return VisibleArtifact(receipt, row[7])

    def reconcile_receipt(self, operation_id: str) -> PublicationReceipt | None:
        """Ambiguous postcommit timeout: lookup; never blindly retry."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT run_id, artifact_id, staged_id, epoch, fence,
                       predecessor_sha, digest, transaction_id, signature_hex
                FROM publications WHERE operation_id=?
            """, (operation_id,)).fetchone()
        if row is None:
            return None
        intent = PublicationIntent(
            self.sink_id, operation_id, row[0], row[1], row[2],
            row[3], row[4], row[5], row[6],
        )
        receipt = PublicationReceipt(intent, row[7], row[6], True, row[8])
        self._verify_saved_receipt(receipt)
        return receipt


def verify_sink_receipt(payload: bytes, signature_hex: str, expected_sink_id: str,
                        *, pinned_sink_id: str, pinned_receipt_key: bytes) -> bool:
    """Testable MAC verifier; key deployment/ownership is NOT qualified."""
    if (type(payload) is not bytes or type(signature_hex) is not str
            or type(pinned_sink_id) is not str or type(expected_sink_id) is not str
            or expected_sink_id != pinned_sink_id
            or type(pinned_receipt_key) is not bytes or len(pinned_receipt_key) < 32):
        return False
    expected = hmac.new(pinned_receipt_key, payload, sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)
