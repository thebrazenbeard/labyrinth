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
        for event_id, record in events.items():
            _event_check(event_id)
            if (type(record) is not dict or record.keys() !=
                    {"observation", "artifact_sha256", "review_ref"}):
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
        events[event_id] = {"observation": observation,
                            "artifact_sha256": artifact_sha256, "review_ref": review_ref}
        _save(db, version, events, state_key)
        db.commit()
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise DevelopmentJournalError("unable to persist reconciliation") from exc
    finally:
        if db is not None:
            db.close()
