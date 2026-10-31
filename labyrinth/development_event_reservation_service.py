"""Real SQLite one-shot reservation *candidate* for signed development events.

The exact four-key scored-exam lineage is verified before a transaction
consumes an event. A SQLite BEGIN IMMEDIATE transaction binds the reserved
event, run/candidate/scope/cycle, and epoch/fence in one durable record. A
second handle, duplicate event, reissued exam identity or stale fence cannot
reserve the same development transition through the API. A postcommit lost
acknowledgment must be reconciled, never blindly retried.

This is **source/local SQLite evidence**, not a host-owned authorization
service. The in-process caller can possess verification keys or modify the
database; no deployed role isolation, external monotonic minimum, protected
signing keys or independent model-execution custody is established. Restoring
an older valid database snapshot really DOES permit replay, as tested.
Therefore this module cannot qualify Q8 or grant private graduation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterator

from .signed_exam_lineage import (
    SignedExamLineageBinding, verify_exam_lineage_binding,
)


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_KEY_DOMAIN = b"labyrinth.sqlite-development-event-sink-identity.v1\0"
_RECEIPT_DOMAIN = b"labyrinth.sqlite-development-event-reservation.v1\0"
_MAX_COUNTER = 2**63 - 1


class ReservationConflict(ValueError):
    """The authenticated event was already consumed or the fence is stale."""


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _counter(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < _MAX_COUNTER:
        raise ReservationConflict("invalid bounded authority epoch/fence")
    return value


@dataclass(frozen=True)
class DevelopmentEventReservationReceipt:
    sink_id: str
    binding_signer_id: str
    event_id: str
    exam_id: str
    run_id: str
    candidate_artifact_sha256: str
    scope_fingerprint: str
    to_cycle: int
    binding_sha256: str
    epoch: int
    fence: int
    reservation_id: str
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    host_event_consumption_verified: bool = False


def _receipt_payload(receipt: DevelopmentEventReservationReceipt) -> bytes:
    if type(receipt) is not DevelopmentEventReservationReceipt:
        raise ValueError("typed reservation receipt required")
    for name in ("sink_id", "binding_signer_id", "event_id", "exam_id",
                 "run_id", "reservation_id"):
        _identifier(getattr(receipt, name), name)
    for name in ("candidate_artifact_sha256", "scope_fingerprint",
                 "binding_sha256"):
        _sha(getattr(receipt, name), name)
    _counter(receipt.to_cycle)
    _counter(receipt.epoch, minimum=1)
    _counter(receipt.fence)
    if (receipt.graduation_authorized is not False
            or receipt.model_execution_authenticated is not False
            or receipt.host_event_consumption_verified is not False):
        raise ValueError("reservation cannot claim graduation, execution or host custody")
    fields = (
        "sink_id", "binding_signer_id", "event_id", "exam_id", "run_id",
        "candidate_artifact_sha256", "scope_fingerprint", "to_cycle",
        "binding_sha256", "epoch", "fence", "reservation_id",
        "graduation_authorized", "model_execution_authenticated",
        "host_event_consumption_verified",
    )
    return (json.dumps({name: getattr(receipt, name) for name in fields},
                       sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("ascii")


class SQLiteDevelopmentEventReservationSink:
    """Cooperative local service; role separations require external hosting."""

    def __init__(self, db_path: Path, *, sink_id: str,
                 expected_binding_signer_id: str,
                 independent_binding_key: bytes, receipt_key: bytes):
        if not isinstance(db_path, Path):
            raise ValueError("pinned SQLite path required")
        self.sink_id = _identifier(sink_id, "sink ID")
        self.expected_binding_signer_id = _identifier(
            expected_binding_signer_id, "binding signer")
        if (type(independent_binding_key) is not bytes
                or type(receipt_key) is not bytes
                or len(independent_binding_key) < 32 or len(receipt_key) < 32
                or hmac.compare_digest(independent_binding_key, receipt_key)):
            raise ValueError("distinct independently pinned signing keys required")
        self.db_path = db_path
        self._binding_key = independent_binding_key
        self._receipt_key = receipt_key
        self._binding_commit = hmac.new(independent_binding_key,
            _KEY_DOMAIN + b"binding\0" + self.expected_binding_signer_id.encode("ascii"),
            sha256).hexdigest()
        self._receipt_commit = hmac.new(receipt_key,
            _KEY_DOMAIN + b"receipt\0" + self.sink_id.encode("ascii"),
            sha256).hexdigest()
        # Initialization is serialized even for two simultaneous fresh handles.
        with self._connect(require_identity=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""CREATE TABLE IF NOT EXISTS event_sink_identity (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    sink_id TEXT NOT NULL, signer_id TEXT NOT NULL,
                    binding_key_commitment TEXT NOT NULL,
                    receipt_key_commitment TEXT NOT NULL
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS event_run_fences (
                    run_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL,
                    fence INTEGER NOT NULL,
                    expected_candidate_sha TEXT NOT NULL,
                    expected_scope_sha TEXT NOT NULL,
                    expected_to_cycle INTEGER NOT NULL
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS development_event_reservations (
                    event_id TEXT PRIMARY KEY, exam_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, candidate_sha TEXT NOT NULL,
                    scope_sha TEXT NOT NULL, to_cycle INTEGER NOT NULL,
                    binding_sha TEXT NOT NULL, epoch INTEGER NOT NULL,
                    fence INTEGER NOT NULL, reservation_id TEXT NOT NULL,
                    signature_hex TEXT NOT NULL,
                    UNIQUE(run_id, exam_id, scope_sha, to_cycle)
                )""")
                existing = conn.execute("""SELECT sink_id, signer_id,
                    binding_key_commitment, receipt_key_commitment
                    FROM event_sink_identity WHERE singleton=1""").fetchone()
                if existing is None:
                    if (conn.execute("SELECT 1 FROM event_run_fences LIMIT 1").fetchone()
                            or conn.execute(
                                "SELECT 1 FROM development_event_reservations LIMIT 1"
                            ).fetchone()):
                        raise ReservationConflict(
                            "populated unbound event database cannot be reclaimed")
                    conn.execute("INSERT INTO event_sink_identity VALUES (1, ?, ?, ?, ?)",
                                 self._identity())
                self._assert_identity(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _identity(self) -> tuple[str, str, str, str]:
        # Do not trust the immutable-at-construction commitment alone if a
        # cooperative in-process caller accidentally replaces a key attribute.
        # This is not a defense against arbitrary code capable of rewriting
        # both keys and commitments or mutating SQLite directly.
        actual_binding = hmac.new(self._binding_key,
            _KEY_DOMAIN + b"binding\0" + self.expected_binding_signer_id.encode("ascii"),
            sha256).hexdigest()
        actual_receipt = hmac.new(self._receipt_key,
            _KEY_DOMAIN + b"receipt\0" + self.sink_id.encode("ascii"),
            sha256).hexdigest()
        if (not hmac.compare_digest(actual_binding, self._binding_commit)
                or not hmac.compare_digest(actual_receipt, self._receipt_commit)):
            raise ReservationConflict("in-process sink key no longer matches pinned identity")
        return (self.sink_id, self.expected_binding_signer_id,
                self._binding_commit, self._receipt_commit)

    def _assert_identity(self, conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute("""SELECT sink_id, signer_id,
                binding_key_commitment, receipt_key_commitment
                FROM event_sink_identity WHERE singleton=1""").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM event_sink_identity").fetchone()[0]
        except sqlite3.Error as exc:
            raise ReservationConflict("event sink identity missing or malformed") from exc
        if row != self._identity() or count != 1:
            raise ReservationConflict("database bound to a different event sink or signer key")

    @contextmanager
    def _connect(self, *, require_identity: bool = True) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=5.0)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=5000")
            if require_identity:
                self._assert_identity(conn)
            yield conn
        finally:
            conn.close()

    def _begin(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        self._assert_identity(conn)

    def register_run(self, run_id: str, *,
                     expected_candidate_artifact_sha256: str,
                     expected_scope_fingerprint: str,
                     expected_to_cycle: int,
                     epoch: int = 1, fence: int = 0) -> None:
        """Pin the operator's candidate/scope/cycle as well as authority fence.

        This is only a *logical* administrator: independent source-of-truth
        measurement, signing-key custody and role enforcement are still absent.
        """
        _identifier(run_id, "candidate run")
        _counter(epoch, minimum=1)
        _counter(fence)
        _sha(expected_candidate_artifact_sha256, "pinned candidate artifact")
        _sha(expected_scope_fingerprint, "pinned scope")
        _counter(expected_to_cycle)
        with self._connect() as conn:
            try:
                self._begin(conn)
                conn.execute("INSERT INTO event_run_fences VALUES (?, ?, ?, ?, ?, ?)",
                             (run_id, epoch, fence,
                              expected_candidate_artifact_sha256,
                              expected_scope_fingerprint, expected_to_cycle))
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise ReservationConflict("run already registered") from exc
            except Exception:
                conn.rollback()
                raise

    def advance_fence(self, run_id: str, *, expected_epoch: int,
                      expected_fence: int, new_epoch: int, new_fence: int,
                      new_expected_candidate_artifact_sha256: str,
                      new_expected_scope_fingerprint: str,
                      new_expected_to_cycle: int) -> None:
        """Logical admin transition with new independently pinned candidate."""
        _identifier(run_id, "candidate run")
        for value, minimum in ((expected_epoch, 1), (expected_fence, 0),
                               (new_epoch, 1), (new_fence, 0)):
            _counter(value, minimum=minimum)
        if new_epoch != expected_epoch + 1 or new_fence != expected_fence + 1:
            raise ReservationConflict("nonmonotonic authority transition")
        _sha(new_expected_candidate_artifact_sha256, "next candidate")
        _sha(new_expected_scope_fingerprint, "next scope")
        _counter(new_expected_to_cycle)
        with self._connect() as conn:
            try:
                self._begin(conn)
                changed = conn.execute("""UPDATE event_run_fences
                    SET epoch=?, fence=?, expected_candidate_sha=?,
                        expected_scope_sha=?, expected_to_cycle=?
                    WHERE run_id=? AND epoch=? AND fence=?""",
                    (new_epoch, new_fence,
                     new_expected_candidate_artifact_sha256,
                     new_expected_scope_fingerprint, new_expected_to_cycle,
                     run_id, expected_epoch, expected_fence)).rowcount
                if changed != 1:
                    raise ReservationConflict("stale epoch or fence")
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _authenticated_body(self, binding: SignedExamLineageBinding,
                            verification_inputs: dict[str, Any]) -> dict[str, Any]:
        # The existing scored Journey compositor needs the four-key input
        # bundle. Accept a repeated binding-key parameter only if it is
        # byte-identical to the sink's pinned key; never let an alternate
        # caller-supplied verifier silently replace sink identity.
        supplied_key = verification_inputs.get("independent_binding_key", self._binding_key)
        if (type(supplied_key) is not bytes
                or not hmac.compare_digest(supplied_key, self._binding_key)):
            raise ValueError("caller binding verifier differs from pinned sink key")
        verification_inputs = {
            key: value for key, value in verification_inputs.items()
            if key != "independent_binding_key"
        }
        body = verify_exam_lineage_binding(
            binding, independent_binding_key=self._binding_key,
            **verification_inputs,
        )
        if body["binding_signer_id"] != self.expected_binding_signer_id:
            raise ValueError("wrong independently pinned exam-binding signer")
        return body

    def _make_receipt(self, body: dict[str, Any], *, epoch: int,
                      fence: int, binding_sha: str) -> DevelopmentEventReservationReceipt:
        reservation_id = "res-" + sha256(
            (body["event_id"] + ":" + binding_sha).encode("ascii")
        ).hexdigest()[:32]
        unsigned = DevelopmentEventReservationReceipt(
            sink_id=self.sink_id,
            binding_signer_id=self.expected_binding_signer_id,
            event_id=body["event_id"], exam_id=body["exam_id"],
            run_id=body["candidate_run_id"],
            candidate_artifact_sha256=body["candidate_artifact_sha256"],
            scope_fingerprint=body["scope_fingerprint"],
            to_cycle=body["to_cycle"], binding_sha256=binding_sha,
            epoch=epoch, fence=fence, reservation_id=reservation_id,
            signature_hex="pending",
        )
        signature = hmac.new(self._receipt_key,
                             _RECEIPT_DOMAIN + _receipt_payload(unsigned), sha256).hexdigest()
        return DevelopmentEventReservationReceipt(
            **{**unsigned.__dict__, "signature_hex": signature}
        )

    def _verify_receipt(self, receipt: DevelopmentEventReservationReceipt) -> None:
        if (receipt.sink_id != self.sink_id
                or receipt.binding_signer_id != self.expected_binding_signer_id):
            raise ReservationConflict("reservation receipt belongs to another sink")
        signature = hmac.new(self._receipt_key,
                             _RECEIPT_DOMAIN + _receipt_payload(receipt), sha256).hexdigest()
        if (type(receipt.signature_hex) is not str
                or not hmac.compare_digest(signature, receipt.signature_hex)):
            raise ReservationConflict("reservation receipt signature invalid")

    def _before_commit(self) -> None:
        """Testable fault injection: failure here rolls back the whole insert."""

    def _after_commit(self) -> None:
        """Testable postcommit lost acknowledgment: reconciliation is required."""

    def reserve_authenticated_binding(
        self, binding: SignedExamLineageBinding, *, epoch: int, fence: int,
        **verification_inputs: Any,
    ) -> DevelopmentEventReservationReceipt:
        """Consume one fully verified scored development lineage once.

        A passed MAC is source verification, not independent model execution.
        Raising after COMMIT does NOT roll back the consumed event; reconcile.
        """
        _counter(epoch, minimum=1)
        _counter(fence)
        body = self._authenticated_body(binding, verification_inputs)
        binding_sha = sha256(binding.binding_bytes).hexdigest()
        receipt = self._make_receipt(body, epoch=epoch, fence=fence,
                                     binding_sha=binding_sha)
        with self._connect() as conn:
            try:
                self._begin(conn)
                current = conn.execute(
                    """SELECT epoch, fence, expected_candidate_sha,
                        expected_scope_sha, expected_to_cycle
                        FROM event_run_fences WHERE run_id=?""",
                    (receipt.run_id,),
                ).fetchone()
                if current != (epoch, fence, receipt.candidate_artifact_sha256,
                               receipt.scope_fingerprint, receipt.to_cycle):
                    raise ReservationConflict(
                        "stale fence, wrong candidate/scope/cycle or unregistered run")
                if conn.execute("SELECT 1 FROM development_event_reservations "
                                "WHERE event_id=?", (receipt.event_id,)).fetchone():
                    raise ReservationConflict("signed development event already consumed")
                conn.execute("""INSERT INTO development_event_reservations VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                        receipt.event_id, receipt.exam_id, receipt.run_id,
                        receipt.candidate_artifact_sha256,
                        receipt.scope_fingerprint, receipt.to_cycle,
                        receipt.binding_sha256, receipt.epoch, receipt.fence,
                        receipt.reservation_id, receipt.signature_hex,
                    ))
                self._before_commit()
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise ReservationConflict("reissued exam/cycle or duplicate event") from exc
            except Exception:
                conn.rollback()
                raise
        self._after_commit()
        return receipt

    def reconcile(self, event_id: str) -> DevelopmentEventReservationReceipt | None:
        """Read signed durable outcome following uncertain process/transport ACK."""
        _identifier(event_id, "development event")
        with self._connect() as conn:
            row = conn.execute("""SELECT exam_id, run_id, candidate_sha,
                scope_sha, to_cycle, binding_sha, epoch, fence, reservation_id,
                signature_hex FROM development_event_reservations WHERE event_id=?""",
                (event_id,)).fetchone()
        if row is None:
            return None
        receipt = DevelopmentEventReservationReceipt(
            sink_id=self.sink_id, binding_signer_id=self.expected_binding_signer_id,
            event_id=event_id, exam_id=row[0], run_id=row[1],
            candidate_artifact_sha256=row[2], scope_fingerprint=row[3],
            to_cycle=row[4], binding_sha256=row[5], epoch=row[6],
            fence=row[7], reservation_id=row[8], signature_hex=row[9],
        )
        self._verify_receipt(receipt)
        return receipt

    def callback_for_binding(
        self, binding: SignedExamLineageBinding, *, epoch: int,
        fence: int, **verification_inputs: Any,
    ) -> Callable[[str], bool]:
        """LOW-LEVEL adapter to HF's legacy event-ID-only callback seam.

        **Do not inject this callback into an arbitrary HF run.** The HF seam
        supplies only ``event_id``; a different run sharing that event string
        could invoke this callback and consume the *wrong* bound candidate.
        Use ``run_bound_measured_successor`` so the same signed identity is
        supplied to both the measured Journey and reservation sink. Even that
        opt-in source path is not a host-owned service, authenticated model
        execution, or a mandatory enforcement point.
        """
        expected = self._authenticated_body(binding, verification_inputs)["event_id"]

        def reserve_event(observed: str) -> bool:
            if type(observed) is not str or observed != expected:
                return False
            try:
                self.reserve_authenticated_binding(
                    binding, epoch=epoch, fence=fence, **verification_inputs,
                )
            except ReservationConflict:
                return False
            return True

        return reserve_event

    def run_bound_measured_successor(
        self, config: Any, predecessor_run_dir: str | Path, *,
        epoch: int, fence: int, state: Any,
        binding: SignedExamLineageBinding,
        training_exposures: Any,
        expected_training_roster_sha256: str,
        **verification_inputs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Bind the *same* signed event to the Journey and one-shot callback.

        The existing HF callback signature remains event-ID-only and can be
        bypassed elsewhere; this is a safer opt-in call site, not a migration
        or an independently operated authorization server.
        """
        from .measured_taskmaster_transition import run_measured_taskmaster_successor

        _counter(epoch, minimum=1)
        _counter(fence)
        body = self._authenticated_body(binding, verification_inputs)
        if (state.scope.fingerprint != body["scope_fingerprint"]
                or state.cycle_index + 1 != body["to_cycle"]
                or verification_inputs["context"].predecessor_lora_value_sha256
                != state.checkpoint_sha256):
            raise ValueError("measured Journey differs from pinned event lineage")
        # Read-only early rejection; the actual consumption path performs the
        # same comparison under BEGIN IMMEDIATE to close the race.
        with self._connect() as conn:
            current = conn.execute("""SELECT epoch, fence, expected_candidate_sha,
                expected_scope_sha, expected_to_cycle FROM event_run_fences
                WHERE run_id=?""", (body["candidate_run_id"],)).fetchone()
        if current != (epoch, fence, body["candidate_artifact_sha256"],
                       body["scope_fingerprint"], body["to_cycle"]):
            raise ReservationConflict("measured successor is not current pinned run")
        callback = self.callback_for_binding(
            binding, epoch=epoch, fence=fence, **verification_inputs)
        return run_measured_taskmaster_successor(
            config, predecessor_run_dir, callback,
            state=state, binding=binding,
            training_exposures=training_exposures,
            expected_training_roster_sha256=expected_training_roster_sha256,
            **verification_inputs,
        )
