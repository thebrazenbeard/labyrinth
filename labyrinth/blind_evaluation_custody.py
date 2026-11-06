"""One-shot, evaluator-side *development* examination custody.

The private oracle and signing keys must live on a genuinely separate
evaluator host. This module only authenticates source-level custody claims;
it cannot attest that a model produced the supplied predictions, that an
exam was blind, or that external filesystem/key custody is independent.

The resulting aggregate-only receipt is NEVER a graduation credential.
Formal final authority remains in ``graduation_gate``/``graduation_state``.
There is deliberately no route from this module to training data or keys.

Each private assessment gets a fresh, exclusive SQLite ledger. The state is
MAC-sealed and transitions atomically from PENDING to CONSUMED. Whole-database
rollback or deletion requires separately protected custodian storage and a
monotonic external anchor; no local protocol can prevent an attacker who owns
the evaluator's storage and both secrets from recreating an assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Mapping
import unicodedata


SCHEMA = "labyrinth.blind-development-evidence.v1"
_PUBLIC_DOMAIN = b"labyrinth.blind-development-public.v1\0"
_GOLD_DOMAIN = b"labyrinth.blind-development-gold.v1\0"
_RECEIPT_DOMAIN = b"labyrinth.blind-development-result.v1\0"
_STATE_DOMAIN = b"labyrinth.blind-development-state.v1\0"
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.safetensors\Z")
_MAX_CASES = 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


class BlindCustodyError(RuntimeError):
    """A private assessment is missing, invalid, replayed, or not authentic."""


@dataclass(frozen=True)
class PrivateExamCase:
    case_id: str
    skill: str
    split: str  # heldout or transfer, never train
    prompt: str
    structure_key: str
    gold_answer: str


@dataclass(frozen=True)
class SignedAssessment:
    receipt_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False


def _identifier(value: Any, label: str) -> None:
    if type(value) is not str or not _IDENT.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _digest(value: Any, label: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _key(value: Any, label: str) -> None:
    if type(value) is not bytes or len(value) < 32:
        raise ValueError(f"{label} requires an independently held 32-byte key")


def _keys(evaluator_key: bytes, state_key: bytes) -> None:
    _key(evaluator_key, "evaluator")
    _key(state_key, "state custody")
    if hmac.compare_digest(evaluator_key, state_key):
        raise ValueError("evaluator and state keys must be distinct")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _exact_json(data: bytes) -> Any:
    if type(data) is not bytes or len(data) > 1024 * 1024:
        raise BlindCustodyError("invalid receipt bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for name, value in pairs:
            if name in result:
                raise BlindCustodyError("duplicate receipt field")
            result[name] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=unique)
        if _canonical(value) != data:
            raise BlindCustodyError("noncanonical receipt encoding")
        return value
    except (ValueError, UnicodeError, TypeError) as exc:
        raise BlindCustodyError("malformed receipt encoding") from exc


def _validated_cases(cases: tuple[PrivateExamCase, ...]) -> tuple[PrivateExamCase, ...]:
    if type(cases) is not tuple or not 4 <= len(cases) <= _MAX_CASES:
        raise ValueError("private exam needs 4..1024 frozen cases")
    seen: set[str] = set()
    prompt_commitments: set[str] = set()
    structure_splits: dict[tuple[str, str], str] = {}
    groups: dict[tuple[str, str], int] = {}
    for case in cases:
        if type(case) is not PrivateExamCase:
            raise ValueError("expected exact PrivateExamCase")
        _identifier(case.case_id, "case identifier")
        _identifier(case.skill, "skill identifier")
        _identifier(case.structure_key, "structure key")
        if (case.case_id in seen or case.split not in ("heldout", "transfer")
                or type(case.prompt) is not str or not 1 <= len(case.prompt) <= 4096
                or type(case.gold_answer) is not str
                or not 1 <= len(case.gold_answer) <= 128):
            raise ValueError("duplicate, incomplete, or TRAIN/private exam case")
        seen.add(case.case_id)
        normalized_prompt = " ".join(unicodedata.normalize(
            "NFKC", case.prompt).casefold().split())
        if not normalized_prompt:
            raise ValueError("blank normalized private exam prompt")
        commitment = sha256(normalized_prompt.encode("utf-8")).hexdigest()
        if commitment in prompt_commitments:
            raise ValueError("duplicate normalized private exam prompt")
        prompt_commitments.add(commitment)
        family = (case.skill, case.structure_key)
        previous_split = structure_splits.setdefault(family, case.split)
        if previous_split != case.split:
            raise ValueError("structure family cannot cross exam splits")
        groups[(case.split, case.skill)] = groups.get((case.split, case.skill), 0) + 1
    if ({split for split, _ in groups} != {"heldout", "transfer"}
            or any(count < 2 for count in groups.values())):
        raise ValueError("each exam split/skill group needs >=2 private cases")
    skills = {skill for _, skill in groups}
    if any(("heldout", skill) not in groups or ("transfer", skill) not in groups
           for skill in skills):
        raise ValueError("each skill must have both splits")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def public_exam_questions(cases: tuple[PrivateExamCase, ...]) -> tuple[dict[str, str], ...]:
    """Return prompts for examination delivery, never answer keys or gold seal."""
    return tuple({"case_id": case.case_id, "skill": case.skill,
                  "split": case.split, "prompt": case.prompt,
                  "structure_key": case.structure_key}
                 for case in _validated_cases(cases))


def _public_commitment(cases: tuple[PrivateExamCase, ...]) -> str:
    public = [(case.case_id, case.skill, case.split,
               sha256(case.prompt.encode("utf-8")).hexdigest(), case.structure_key)
              for case in _validated_cases(cases)]
    return sha256(_PUBLIC_DOMAIN + _canonical(public)).hexdigest()


def _gold_seal(cases: tuple[PrivateExamCase, ...], evaluator_key: bytes) -> str:
    _key(evaluator_key, "evaluator")
    gold = [(case.case_id, case.gold_answer) for case in _validated_cases(cases)]
    return hmac.new(evaluator_key, _GOLD_DOMAIN + _canonical(gold), sha256).hexdigest()


def _safe_path(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("an explicit absolute evaluator Path is required")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise BlindCustodyError("symlink or junction in evaluator path")
    if not path.parent.is_dir() or (path.exists() and not path.is_file()):
        raise BlindCustodyError("invalid evaluator custody path")
    return path


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


def _mac(state_key: bytes, *, version: int, state: str, record: str,
         receipt: bytes | None, signature: str | None) -> str:
    payload = _canonical({"version": version, "state": state, "record": record,
                          "receipt_hex": receipt.hex() if receipt else None,
                          "signature": signature})
    return hmac.new(state_key, _STATE_DOMAIN + payload, sha256).hexdigest()


def initialize_assessment(
    db_path: Path, cases: tuple[PrivateExamCase, ...], *,
    run_id: str, objective_id: str, checkpoint_sha256: str,
    evaluator_key: bytes, state_key: bytes,
) -> str:
    """Precommit one assessor-owned exam. Never reinitialize a consumed run."""
    _keys(evaluator_key, state_key)
    _identifier(run_id, "run ID")
    _identifier(objective_id, "objective ID")
    _digest(checkpoint_sha256, "checkpoint digest")
    path = _safe_path(db_path)
    record = _canonical({"run_id": run_id, "objective_id": objective_id,
                         "checkpoint_sha256": checkpoint_sha256,
                         "public_commitment": _public_commitment(cases),
                         "gold_seal": _gold_seal(cases, evaluator_key)}).decode("utf-8")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    except FileExistsError as exc:
        raise BlindCustodyError("assessment already exists; no reset") from exc
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE assessment (
            id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL,
            state TEXT NOT NULL, record TEXT NOT NULL,
            receipt BLOB, signature TEXT, state_mac TEXT NOT NULL)""")
        mac = _mac(state_key, version=0, state="PENDING", record=record,
                   receipt=None, signature=None)
        db.execute("INSERT INTO assessment VALUES (1,0,'PENDING',?,NULL,NULL,?)",
                   (record, mac))
        db.commit()
        return json.loads(record)["public_commitment"]
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise BlindCustodyError("could not initialize atomic evaluator state") from exc
    finally:
        if db is not None:
            db.close()


def _artifact_digest(root: Path, name: str) -> str:
    if not isinstance(root, Path) or not root.is_absolute() or not root.is_dir():
        raise ValueError("absolute, existing evaluator artifact root required")
    if type(name) is not str or not _ARTIFACT.fullmatch(name):
        raise ValueError("bounded safetensors artifact filename required")
    path = root / name
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise BlindCustodyError("symlink or junction artifact is forbidden")
    try:
        # Pin the digest to one opened file descriptor. O_NOFOLLOW closes the
        # last-component symlink race on hosts that provide it. Other hosts
        # still require independent evaluator-host filesystem isolation.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_ARTIFACT_BYTES:
                raise BlindCustodyError("missing or invalid checkpoint artifact")
            digest = sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        path_after = path.lstat()
        def identity(info: os.stat_result) -> tuple[int, int, int, int]:
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        if identity(before) != identity(after) or identity(before) != identity(path_after):
            raise BlindCustodyError("checkpoint changed during digest readback")
        return digest.hexdigest()
    except OSError as exc:
        raise BlindCustodyError("checkpoint artifact could not be securely opened") from exc


def _read_state(db: sqlite3.Connection, state_key: bytes) -> tuple[str, str]:
    row = db.execute("SELECT version,state,record,receipt,signature,state_mac "
                     "FROM assessment WHERE id=1").fetchone()
    if row is None:
        raise BlindCustodyError("evaluator state record missing")
    version, state, record, receipt, signature, stored_mac = row
    if (type(version) is not int or version not in (0, 1)
            or state not in ("PENDING", "CONSUMED")
            or type(record) is not str or type(stored_mac) is not str
            or not hmac.compare_digest(_mac(
                state_key, version=version, state=state, record=record,
                receipt=receipt, signature=signature), stored_mac)):
        raise BlindCustodyError("evaluator state seal invalid")
    if (state == "PENDING" and (version != 0 or receipt is not None or signature is not None)):
        raise BlindCustodyError("pending evaluator state is malformed")
    if (state == "CONSUMED" and (version != 1 or type(receipt) is not bytes
                                or type(signature) is not str)):
        raise BlindCustodyError("consumed evaluator state is malformed")
    return state, record


def score_assessment(
    db_path: Path, cases: tuple[PrivateExamCase, ...],
    responses: Mapping[str, str], *, artifact_root: Path, artifact_name: str,
    evaluator_key: bytes, state_key: bytes,
) -> SignedAssessment:
    """Atomically consume one private development exam, with no case-level leak.

    This method accepts model responses from an external inference worker; it
    does *not* attest that the pinned checkpoint actually produced them.
    """
    _keys(evaluator_key, state_key)
    ordered = _validated_cases(cases)
    if (type(responses) is not dict or set(responses) != {case.case_id for case in ordered}
            or any(type(value) is not str or len(value) > 128 for value in responses.values())):
        raise ValueError("responses must cover exactly the committed exam IDs")
    path = _safe_path(db_path)
    if not path.is_file():
        raise BlindCustodyError("assessment ledger missing")
    observed = _artifact_digest(artifact_root, artifact_name)
    db = None
    try:
        db = _connect(path)
        db.execute("BEGIN IMMEDIATE")
        state, record_json = _read_state(db, state_key)
        if state != "PENDING":
            raise BlindCustodyError("assessment already consumed")
        record = _exact_json(record_json.encode("utf-8"))
        if (type(record) is not dict or record.keys() != {
                "run_id", "objective_id", "checkpoint_sha256",
                "public_commitment", "gold_seal"}):
            raise BlindCustodyError("invalid evaluator precommit record")
        if (record["public_commitment"] != _public_commitment(cases)
                or not hmac.compare_digest(record["gold_seal"],
                                           _gold_seal(cases, evaluator_key))):
            raise BlindCustodyError("exam commitment or private oracle seal mismatch")
        if observed != record["checkpoint_sha256"]:
            raise BlindCustodyError("checkpoint artifact digest differs from precommit")
        groups: dict[tuple[str, str], list[int]] = {}
        for case in ordered:
            aggregate = groups.setdefault((case.split, case.skill), [0, 0])
            aggregate[0] += 1
            aggregate[1] += int(hmac.compare_digest(responses[case.case_id],
                                                    case.gold_answer))
        summary = [dict(split=split, skill=skill, total=count,
                        correct=correct) for (split, skill), (count, correct)
                   in sorted(groups.items())]
        receipt = _canonical({"schema": SCHEMA, "status": "DEVELOPMENT_SCORED",
                              "run_id": record["run_id"],
                              "objective_id": record["objective_id"],
                              "checkpoint_sha256": observed,
                              "public_exam_sha256": record["public_commitment"],
                              "groups": summary, "graduation_authorized": False,
                              "model_execution_authenticated": False})
        signature = hmac.new(evaluator_key, _RECEIPT_DOMAIN + receipt,
                             sha256).hexdigest()
        updated_mac = _mac(state_key, version=1, state="CONSUMED",
                           record=record_json, receipt=receipt, signature=signature)
        db.execute("UPDATE assessment SET version=1,state='CONSUMED',receipt=?,"
                   "signature=?,state_mac=? WHERE id=1",
                   (receipt, signature, updated_mac))
        db.commit()
        return SignedAssessment(receipt, signature)
    except sqlite3.Error as exc:
        if db is not None:
            db.rollback()
        raise BlindCustodyError("atomic evaluator custody unavailable") from exc
    finally:
        if db is not None:
            db.close()


def verify_assessment(
    receipt_bytes: bytes, signature_hex: str, *, evaluator_key: bytes,
    run_id: str, checkpoint_sha256: str,
) -> dict[str, Any]:
    """Authenticate a *development-only* receipt, never grant graduation."""
    _key(evaluator_key, "evaluator")
    _identifier(run_id, "run ID")
    _digest(checkpoint_sha256, "checkpoint digest")
    if type(receipt_bytes) is not bytes or not 0 < len(receipt_bytes) <= 1024 * 1024:
        raise BlindCustodyError("invalid development receipt bytes")
    if type(signature_hex) is not str or not _DIGEST.fullmatch(signature_hex):
        raise BlindCustodyError("malformed development receipt signature")
    expected = hmac.new(evaluator_key, _RECEIPT_DOMAIN + receipt_bytes,
                        sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_hex):
        raise BlindCustodyError("invalid development receipt signature")
    receipt = _exact_json(receipt_bytes)
    required = {"schema", "status", "run_id", "objective_id",
                "checkpoint_sha256", "public_exam_sha256", "groups",
                "graduation_authorized", "model_execution_authenticated"}
    if (type(receipt) is not dict or receipt.keys() != required
            or receipt["schema"] != SCHEMA or receipt["status"] != "DEVELOPMENT_SCORED"
            or receipt["run_id"] != run_id
            or receipt["checkpoint_sha256"] != checkpoint_sha256
            or receipt["graduation_authorized"] is not False
            or receipt["model_execution_authenticated"] is not False
            or type(receipt["groups"]) is not list):
        raise BlindCustodyError("signed development receipt identity invalid")
    _identifier(receipt["objective_id"], "objective ID")
    _digest(receipt["public_exam_sha256"], "public exam digest")
    if not 2 <= len(receipt["groups"]) <= _MAX_CASES:
        raise BlindCustodyError("invalid signed aggregate roster")
    seen_groups: set[tuple[str, str]] = set()
    for group in receipt["groups"]:
        if (type(group) is not dict or group.keys() != {"split", "skill", "total", "correct"}
                or group["split"] not in ("heldout", "transfer")
                or type(group["total"]) is not int
                or type(group["correct"]) is not int
                or not 2 <= group["total"] <= _MAX_CASES
                or not 0 <= group["correct"] <= group["total"]):
            raise BlindCustodyError("invalid signed aggregate group")
        _identifier(group["skill"], "skill ID")
        group_id = (group["split"], group["skill"])
        if group_id in seen_groups:
            raise BlindCustodyError("duplicate signed aggregate group")
        seen_groups.add(group_id)
    if {split for split, _ in seen_groups} != {"heldout", "transfer"}:
        raise BlindCustodyError("missing signed split group")
    return receipt
