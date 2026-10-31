"""Atomic custodian-owned persistence for Minotaur graduation decisions.

This is one part of an independent evaluator, **not** an evaluator itself:
it cannot verify a checkpoint ran, generate truly private problems, or make a
neural critic trustworthy. No signing key is stored in SQLite; an operator
must keep independent evaluator and state-MAC secrets off trainee machines.

A preinitialized, policy-bound, HMAC-protected SQLite WAL record makes each
authenticated decision and consumed-attempt update atomic across processes.
Deleting or replacing the database, changing the external keys, or rolling
back a whole previously signed database is outside this module's protection;
these require separately governed filesystem custody/monotonic anchoring.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3

from .graduation_gate import (
    GateDecision, GateState, GraduationPolicy, _policy_fingerprint,
    decide_graduation,
)


_STATE_DOMAIN = b"labyrinth.minotaur.atomic-custody.v1\x00"
_TABLE_SQL = """CREATE TABLE campaign (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    policy_fingerprint TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 0),
    serialized_state TEXT NOT NULL,
    state_mac TEXT NOT NULL
)"""


class CustodyError(RuntimeError):
    """The trusted state is missing, corrupt, locked, or not policy-bound."""


def _valid_key(state_key: bytes) -> None:
    if type(state_key) is not bytes or len(state_key) < 32:
        raise ValueError("state custody requires a separate 32-byte secret")


def _valid_path(db_path: Path) -> Path:
    if not isinstance(db_path, Path) or not db_path.is_absolute():
        raise ValueError("explicit absolute custodian database Path required")
    for part in (db_path, *db_path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("symlink or junction in custodian database path")
    if not db_path.parent.is_dir() or db_path.parent.is_symlink():
        raise ValueError("preexisting non-symlink custodian directory required")
    if db_path.is_symlink() or (db_path.exists() and not db_path.is_file()):
        raise ValueError("invalid custodian database type")
    return db_path


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    db.execute("PRAGMA busy_timeout = 5000")
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = FULL")
    return db


def _serialize(state: GateState) -> str:
    record = asdict(state)
    for name in ("consumed_run_ids", "consumed_suite_sha256s", "consumed_eval_case_ids",
                 "consumed_eval_prompt_commitments"):
        record[name] = sorted(record[name])
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse(serialized: str) -> GateState:
    try:
        record = json.loads(serialized)
        exact = {"next_attempt", "consumed_run_ids", "consumed_suite_sha256s",
                 "consumed_eval_case_ids", "consumed_eval_prompt_commitments",
                 "graduated", "policy_fingerprint"}
        if type(record) is not dict or record.keys() != exact:
            raise ValueError("unexpected state schema")
        for field in ("consumed_run_ids", "consumed_suite_sha256s",
                      "consumed_eval_case_ids", "consumed_eval_prompt_commitments"):
            values = record[field]
            if (type(values) is not list or any(type(x) is not str for x in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid set representation")
            record[field] = frozenset(values)
        return GateState(**record)
    except (ValueError, TypeError, KeyError) as exc:
        raise CustodyError("tampered or malformed trusted state") from exc


def _mac(state_key: bytes, fingerprint: str, version: int, serialized: str) -> str:
    message = (_STATE_DOMAIN + fingerprint.encode("ascii") + b"\x00"
               + version.to_bytes(8, "big") + b"\x00" + serialized.encode("utf-8"))
    return hmac.new(state_key, message, hashlib.sha256).hexdigest()


def initialize_campaign(db_path: Path, policy: GraduationPolicy, *, state_key: bytes) -> None:
    """Create a new exclusive custodian campaign; never reset an old campaign.

    The operator must secure the containing directory, back up the DB, and
    enforce access-control and rollback protection outside this module.
    """
    _valid_key(state_key)
    path = _valid_path(db_path)
    if not isinstance(policy, GraduationPolicy):
        raise TypeError("expected frozen GraduationPolicy")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CustodyError("campaign already exists; never reset admission history") from exc
    else:
        os.close(descriptor)

    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        db.execute(_TABLE_SQL)
        fingerprint = _policy_fingerprint(policy)
        serialized = _serialize(GateState(policy_fingerprint=fingerprint))
        db.execute("INSERT INTO campaign VALUES (1, ?, 0, ?, ?)",
                   (fingerprint, serialized, _mac(state_key, fingerprint, 0, serialized)))
        db.commit()
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise CustodyError("unable to atomically initialize custody") from exc
    finally:
        if db is not None:
            db.close()


def decide_campaign(
    db_path: Path,
    policy: GraduationPolicy,
    receipt_bytes: bytes | None,
    signature_hex: str | None,
    *,
    independent_evaluator_key: bytes,
    state_key: bytes,
) -> GateDecision:
    """Consume one signed exam decision transactionally and return its status.

    The external evaluator must actually run models, own the private oracle,
    and sign valid evidence. This wrapper authenticates and persists the
    outcome, but does not establish that external trust on its own.
    """
    _valid_key(state_key)
    path = _valid_path(db_path)
    if not path.is_file():
        raise CustodyError("campaign must be initialized before accepting receipts")
    if type(independent_evaluator_key) is not bytes or len(independent_evaluator_key) < 32:
        raise ValueError("independent evaluator key is missing")
    if hmac.compare_digest(independent_evaluator_key, state_key):
        raise ValueError("evaluator and state custody must use different secrets")
    if not isinstance(policy, GraduationPolicy):
        raise TypeError("expected frozen GraduationPolicy")

    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT policy_fingerprint, version, serialized_state, state_mac "
                         "FROM campaign WHERE id = 1").fetchone()
        if row is None:
            raise CustodyError("trusted campaign record missing")
        fingerprint, version, serialized, stored_mac = row
        if (type(version) is not int or not 0 <= version < 2**63 - 1
                or type(fingerprint) is not str or type(serialized) is not str
                or type(stored_mac) is not str
                or fingerprint != _policy_fingerprint(policy)
                or not hmac.compare_digest(
                    _mac(state_key, fingerprint, version, serialized), stored_mac)):
            raise CustodyError("state seal invalid or campaign bound to another policy")
        state = _parse(serialized)
        if state.policy_fingerprint != fingerprint:
            raise CustodyError("trusted policy binding corrupted")
        result = decide_graduation(policy, state, receipt_bytes, signature_hex,
                                   independent_evaluator_key=independent_evaluator_key)
        if result.disposition == "BLOCKED":
            db.rollback()
            return result
        if result.state != state:
            updated = _serialize(result.state)
            next_version = version + 1
            db.execute("UPDATE campaign SET version=?, serialized_state=?, state_mac=? WHERE id=1",
                       (next_version, updated, _mac(state_key, fingerprint, next_version, updated)))
        db.commit()
        return result
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise CustodyError("atomic custody transaction unavailable") from exc
    finally:
        if db is not None:
            db.close()
