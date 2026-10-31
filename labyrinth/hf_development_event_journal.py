"""Optional host-owned durable journal for signed development-FAIL events.

``run_exam_bound_successor`` deliberately accepts a host reservation callback;
this module offers one SQLite/WAL-backed implementation of that callback.
The journal does *not* verify a signature, train a model, independently inspect
neural artifacts, or grant graduation. The operator must protect its directory
and its separate state-MAC secret from the learner and prevent database rollback.

Reservation is committed before returning ``True``. A crash after that point
leaves the event consumed forever, even if no successor was started. An operator
may attach one bounded, explicitly *unverified* reconciliation observation,
but neither that observation nor any error releases the event for replay.

An opt-in run-bound reservation can additionally fence recovery attempts to one
immutable logical run and a committed checkpoint identity. This journal cannot
inspect checkpoint bytes or prove optimizer/RNG restoration: the trusted host
must validate complete checkpoint contents and enforce the returned fence at
every worker/result boundary. Legacy event-only reservations cannot be upgraded.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


_DOMAIN = b"labyrinth.hf-development-event-journal.v1\x00"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TABLE_SQL = """CREATE TABLE journal (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL CHECK (version >= 0),
    events_json TEXT NOT NULL,
    state_mac TEXT NOT NULL
)"""
_MAX_EVENTS = 4096
_MAX_JSON_BYTES = 2 * 1024 * 1024
_OBSERVATIONS = frozenset(("NOT_STARTED", "PARTIAL_ARTIFACT", "REPORTED_RESULT", "UNKNOWN"))
_LEGACY_FIELDS = frozenset(("observation", "artifact_sha256", "review_ref"))
_RUN_FIELDS = _LEGACY_FIELDS | frozenset((
    "run_id", "run_commitment_sha256", "checkpoint_sha256", "fence",
    "run_state", "result_sha256",
))


class DevelopmentJournalError(RuntimeError):
    """Journal is missing, invalid, busy, or its custody seal cannot be read."""


def _key_check(state_key: bytes) -> None:
    if type(state_key) is not bytes or len(state_key) < 32:
        raise ValueError("a separate 32+ byte host journal secret is required")


def _path_check(db_path: Path) -> Path:
    if not isinstance(db_path, Path) or not db_path.is_absolute():
        raise ValueError("an absolute host-owned journal database Path is required")
    if not db_path.parent.is_dir() or db_path.parent.is_symlink():
        raise ValueError("a preexisting non-symlink journal directory is required")
    if db_path.is_symlink() or (db_path.exists() and not db_path.is_file()):
        raise ValueError("invalid journal database file type")
    return db_path


def _event_check(event_id: str) -> None:
    if type(event_id) is not str or not _IDENTIFIER.fullmatch(event_id):
        raise ValueError("invalid signed development event identifier")


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    db.execute("PRAGMA busy_timeout = 5000")
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = FULL")
    return db


def _encode(events: dict[str, dict[str, Any]]) -> str:
    return json.dumps({"events": events}, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mac(state_key: bytes, version: int, serialized: str) -> str:
    return hmac.new(state_key, _DOMAIN + version.to_bytes(8, "big") +
                    b"\x00" + serialized.encode("utf-8"), hashlib.sha256).hexdigest()


def _parse(serialized: str) -> dict[str, dict[str, Any]]:
    try:
        if type(serialized) is not str or len(serialized.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("oversized journal state")
        state = json.loads(serialized)
        if type(state) is not dict or state.keys() != {"events"}:
            raise ValueError("unexpected journal schema")
        events = state["events"]
        if type(events) is not dict or len(events) > _MAX_EVENTS:
            raise ValueError("invalid event registry")
        run_ids: set[str] = set()
        for event_id, record in events.items():
            _event_check(event_id)
            if type(record) is not dict or record.keys() not in (_LEGACY_FIELDS, _RUN_FIELDS):
                raise ValueError("malformed event entry")
            observation = record["observation"]
            if observation is not None and observation not in _OBSERVATIONS:
                raise ValueError("invalid observation")
            digest = record["artifact_sha256"]
            review = record["review_ref"]
            if observation is None:
                if digest is not None or review is not None:
                    raise ValueError("unreconciled event has unexpected evidence")
            elif (type(digest) is not str or not _DIGEST.fullmatch(digest)
                  or type(review) is not str or not _IDENTIFIER.fullmatch(review)):
                raise ValueError("invalid reconciliation metadata")
            if record.keys() == _RUN_FIELDS:
                if (type(record["run_id"]) is not str
                        or not _IDENTIFIER.fullmatch(record["run_id"])
                        or type(record["run_commitment_sha256"]) is not str
                        or not _DIGEST.fullmatch(record["run_commitment_sha256"])
                        or type(record["checkpoint_sha256"]) is not str
                        or not _DIGEST.fullmatch(record["checkpoint_sha256"])
                        or type(record["fence"]) is not int
                        or not 0 <= record["fence"] < 2**63 - 1
                        or record["run_state"] not in ("ACTIVE", "COMPLETED")):
                    raise ValueError("invalid run-bound reservation")
                if record["run_id"] in run_ids:
                    raise ValueError("duplicate logical run custody")
                run_ids.add(record["run_id"])
                result = record["result_sha256"]
                if ((record["run_state"] == "ACTIVE" and result is not None)
                        or (record["run_state"] == "COMPLETED" and
                            (type(result) is not str or not _DIGEST.fullmatch(result)))):
                    raise ValueError("invalid run completion")
        return events
    except (ValueError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise DevelopmentJournalError("journal state malformed or tampered") from exc


def _load(db: sqlite3.Connection, state_key: bytes) -> tuple[int, dict[str, dict[str, Any]]]:
    row = db.execute("SELECT version, events_json, state_mac FROM journal WHERE id=1").fetchone()
    if row is None:
        raise DevelopmentJournalError("journal not initialized")
    version, serialized, saved_mac = row
    if (type(version) is not int or not 0 <= version < 2**63 - 1
            or type(serialized) is not str or type(saved_mac) is not str
            or not hmac.compare_digest(_mac(state_key, version, serialized), saved_mac)):
        raise DevelopmentJournalError("journal seal invalid or wrong host secret")
    return version, _parse(serialized)


def _save(db: sqlite3.Connection, version: int, events: dict[str, dict[str, Any]],
          state_key: bytes) -> None:
    serialized = _encode(events)
    if len(serialized.encode("utf-8")) > _MAX_JSON_BYTES:
        raise DevelopmentJournalError("journal size budget exhausted")
    db.execute("UPDATE journal SET version=?, events_json=?, state_mac=? WHERE id=1",
               (version + 1, serialized, _mac(state_key, version + 1, serialized)))


def initialize_development_event_journal(db_path: Path, *, state_key: bytes) -> None:
    """Exclusively initialize once; never overwrite a preexisting/partial DB."""
    _key_check(state_key)
    path = _path_check(db_path)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DevelopmentJournalError("existing journal cannot be reinitialized") from exc
    os.close(descriptor)
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        db.execute(_TABLE_SQL)
        serialized = _encode({})
        db.execute("INSERT INTO journal VALUES (1, 0, ?, ?)",
                   (serialized, _mac(state_key, 0, serialized)))
        db.commit()
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("failed to initialize journal; preserve partial DB") from exc
    finally:
        if db is not None:
            db.close()


def reserve_development_event(db_path: Path, event_id: str, *, state_key: bytes) -> bool:
    """Atomically consume a development event ID across processes.

    The caller must already have authenticated and preflighted its signed FAIL.
    Returns ``True`` only after a durable transaction; duplicate returns
    ``False`` without modifying state. Any failure raises and must stop training.
    """
    _key_check(state_key)
    path = _path_check(db_path)
    _event_check(event_id)
    if not path.is_file():
        raise DevelopmentJournalError("journal not initialized")
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        version, events = _load(db, state_key)
        if event_id in events:
            db.rollback()
            return False
        if len(events) >= _MAX_EVENTS:
            raise DevelopmentJournalError("journal event budget exhausted")
        events[event_id] = {"observation": None, "artifact_sha256": None, "review_ref": None}
        _save(db, version, events, state_key)
        db.commit()
        return True
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("atomic journal reservation unavailable") from exc
    finally:
        if db is not None:
            db.close()


def _run_identity_check(run_id: str, run_commitment_sha256: str,
                        checkpoint_sha256: str) -> None:
    if (type(run_id) is not str or not _IDENTIFIER.fullmatch(run_id)
            or type(run_commitment_sha256) is not str
            or not _DIGEST.fullmatch(run_commitment_sha256)
            or type(checkpoint_sha256) is not str
            or not _DIGEST.fullmatch(checkpoint_sha256)):
        raise ValueError("invalid run identity or checkpoint commitment")


def reserve_development_run(
    db_path: Path, event_id: str, *, state_key: bytes, run_id: str,
    run_commitment_sha256: str, initial_checkpoint_sha256: str,
) -> int:
    """Consume once while binding an event to a precommitted logical run.

    The host must first authenticate the signed FAIL and verify a *complete*
    initial checkpoint (including optimizer, RNG and cursor) at this digest.
    Return fence 0 on first reservation; duplicates, including the same run,
    fail closed and require the separately fenced recovery operation.
    """
    _key_check(state_key)
    path = _path_check(db_path)
    _event_check(event_id)
    _run_identity_check(run_id, run_commitment_sha256, initial_checkpoint_sha256)
    if not path.is_file():
        raise DevelopmentJournalError("journal not initialized")
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        version, events = _load(db, state_key)
        if event_id in events:
            raise DevelopmentJournalError("event already consumed; use fenced recovery")
        # One logical run is allowed one event-owned fence domain for its
        # entire lifetime, including after completion. The transaction's
        # BEGIN IMMEDIATE serializes competing reservations across processes.
        if any(record.get("run_id") == run_id for record in events.values()):
            raise DevelopmentJournalError("run already reserved; use fenced recovery")
        if len(events) >= _MAX_EVENTS:
            raise DevelopmentJournalError("journal event budget exhausted")
        events[event_id] = {
            "observation": None, "artifact_sha256": None, "review_ref": None,
            "run_id": run_id, "run_commitment_sha256": run_commitment_sha256,
            "checkpoint_sha256": initial_checkpoint_sha256, "fence": 0,
            "run_state": "ACTIVE", "result_sha256": None,
        }
        _save(db, version, events, state_key)
        db.commit()
        return 0
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("atomic run reservation unavailable") from exc
    finally:
        if db is not None:
            db.close()


def _mutate_development_run(
    db_path: Path, event_id: str, *, state_key: bytes, run_id: str,
    run_commitment_sha256: str, checkpoint_sha256: str, expected_fence: int,
    action: str, next_sha256: str | None = None,
) -> int:
    _key_check(state_key)
    path = _path_check(db_path)
    _event_check(event_id)
    _run_identity_check(run_id, run_commitment_sha256, checkpoint_sha256)
    if type(expected_fence) is not int or not 0 <= expected_fence < 2**63 - 2:
        raise ValueError("invalid expected run fence")
    if action not in ("RESUME", "CHECKPOINT", "COMPLETE"):
        raise ValueError("invalid run transition")
    if action != "RESUME" and (type(next_sha256) is not str or
                                   not _DIGEST.fullmatch(next_sha256)):
        raise ValueError("invalid successor checkpoint or result digest")
    if not path.is_file():
        raise DevelopmentJournalError("journal not initialized")
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        version, events = _load(db, state_key)
        record = events.get(event_id)
        if record is None or record.keys() != _RUN_FIELDS:
            raise DevelopmentJournalError("run reservation missing or legacy-only")
        if (record["run_id"] != run_id or
                record["run_commitment_sha256"] != run_commitment_sha256 or
                record["checkpoint_sha256"] != checkpoint_sha256 or
                record["fence"] != expected_fence):
            raise DevelopmentJournalError("run identity, checkpoint or fence mismatch")
        # An operator observation is explicitly unverified and cannot mutate
        # run custody, fence state, or same-run recovery authority. Only a
        # completed run is terminal. The event remains consumed throughout.
        if record["run_state"] != "ACTIVE":
            raise DevelopmentJournalError("run completed; recovery denied")
        if action == "RESUME":
            record["fence"] += 1
        elif action == "CHECKPOINT":
            if next_sha256 == checkpoint_sha256:
                raise DevelopmentJournalError("checkpoint must advance")
            record["checkpoint_sha256"] = next_sha256
        else:
            record["run_state"] = "COMPLETED"
            record["result_sha256"] = next_sha256
        _save(db, version, events, state_key)
        db.commit()
        return record["fence"]
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("atomic run transition unavailable") from exc
    finally:
        if db is not None:
            db.close()


def claim_development_run_resume(
    db_path: Path, event_id: str, *, state_key: bytes, run_id: str,
    run_commitment_sha256: str, checkpoint_sha256: str, expected_fence: int,
) -> int:
    """Atomically increment the fence for this *same* run and checkpoint.

    Only a trusted host may call this after verifying durable full checkpoint
    bytes. Every worker write/result must be rejected unless its fence is the
    current value; otherwise a paused former worker can race the resumed one.
    A claim does not attest learning or release the signed event.
    """
    return _mutate_development_run(
        db_path, event_id, state_key=state_key, run_id=run_id,
        run_commitment_sha256=run_commitment_sha256,
        checkpoint_sha256=checkpoint_sha256, expected_fence=expected_fence,
        action="RESUME",
    )


def commit_development_run_checkpoint(
    db_path: Path, event_id: str, *, state_key: bytes, run_id: str,
    run_commitment_sha256: str, checkpoint_sha256: str, expected_fence: int,
    next_checkpoint_sha256: str,
) -> None:
    """Commit one validated full checkpoint identity at the current fence."""
    _mutate_development_run(
        db_path, event_id, state_key=state_key, run_id=run_id,
        run_commitment_sha256=run_commitment_sha256,
        checkpoint_sha256=checkpoint_sha256, expected_fence=expected_fence,
        action="CHECKPOINT", next_sha256=next_checkpoint_sha256,
    )


def complete_development_run(
    db_path: Path, event_id: str, *, state_key: bytes, run_id: str,
    run_commitment_sha256: str, checkpoint_sha256: str, expected_fence: int,
    result_sha256: str,
) -> None:
    """Seal a reported run result, without claiming qualification/graduation."""
    _mutate_development_run(
        db_path, event_id, state_key=state_key, run_id=run_id,
        run_commitment_sha256=run_commitment_sha256,
        checkpoint_sha256=checkpoint_sha256, expected_fence=expected_fence,
        action="COMPLETE", next_sha256=result_sha256,
    )


def inspect_development_event(db_path: Path, event_id: str, *, state_key: bytes) -> dict[str, Any]:
    """Read one event; operator observations are not attested model outcomes."""
    _key_check(state_key)
    path = _path_check(db_path)
    _event_check(event_id)
    if not path.is_file():
        raise DevelopmentJournalError("journal not initialized")
    db = None
    try:
        db = _connect(path)
        _version, events = _load(db, state_key)
        if event_id not in events:
            return {"event_id": event_id, "consumed": False,
                    "observation": None, "operator_observation_only": True}
        return {"event_id": event_id, "consumed": True,
                **events[event_id], "operator_observation_only": True}
    except sqlite3.Error as exc:
        raise DevelopmentJournalError("unable to inspect journal") from exc
    finally:
        if db is not None:
            db.close()


def reconcile_development_event(
    db_path: Path, event_id: str, *, state_key: bytes,
    observation: str, artifact_sha256: str, review_ref: str,
) -> None:
    """Persist one host observation; *never* unconsume or authorize replay.

    The caller's external review is not verified by this module. An artifact
    hash alone does not prove source, checkpoint continuity, or learning.
    """
    _key_check(state_key)
    path = _path_check(db_path)
    _event_check(event_id)
    if (type(observation) is not str or observation not in _OBSERVATIONS
            or type(artifact_sha256) is not str or not _DIGEST.fullmatch(artifact_sha256)
            or type(review_ref) is not str or not _IDENTIFIER.fullmatch(review_ref)):
        raise ValueError("invalid bounded operator reconciliation")
    if not path.is_file():
        raise DevelopmentJournalError("journal not initialized")
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        version, events = _load(db, state_key)
        if event_id not in events:
            raise DevelopmentJournalError("cannot reconcile an unreserved event")
        if events[event_id]["observation"] is not None:
            raise DevelopmentJournalError("event already reconciled; never replace observation")
        # Preserve the run-bound fields if present. Replacing the entire entry
        # with the legacy three-field shape silently erased the immutable run
        # and checkpoint commitments after a partial-artifact observation.
        record = dict(events[event_id])
        record.update({"observation": observation,
                       "artifact_sha256": artifact_sha256, "review_ref": review_ref})
        events[event_id] = record
        _save(db, version, events, state_key)
        db.commit()
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("unable to persist reconciliation") from exc
    finally:
        if db is not None:
            db.close()
