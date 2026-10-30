"""Evaluator-invoked *development* inference with bounded source provenance.

An evaluator-owned runner calls a pinned, module-level loader, invokes its
predictor on private prompts (never private answers), then atomically consumes
the existing blind-development assessment. No caller-provided response map is
accepted. A separate HMAC trace binds the inference invocation to its scored
receipt; neither result is a graduation credential.

The source-file digest and same-process invocation are NOT cryptographic proof
that the loader read the claimed model bytes or that host/key isolation exists.
The actual model's execution remains unauthenticated until an independent
runtime attestation binds its executable, inputs, weights, and process.

Before the private questions reach a predictor, an exclusive, never-reused
reservation file is created next to the MAC-sealed assessment. A process crash
leaves that assessment spent rather than allowing another prompt exposure.
Deleting/restoring that file remains a privileged-host rollback threat.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable

from labyrinth import blind_evaluation_custody as blind


SCHEMA = "labyrinth.evaluator-invoked-development.v1"
_DOMAIN = b"labyrinth.evaluator-invoked-development.v1\0"
_RESERVATION_DOMAIN = b"labyrinth.evaluator-invocation-reserved.v1\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SOURCE_BYTES = 2 * 1024 * 1024


class InferenceCustodyError(RuntimeError):
    """The loader, artifact, predictor, or signed trace failed custody checks."""


@dataclass(frozen=True)
class PinnedLoader:
    source_file: Path
    source_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class SignedEvaluatorInvocation:
    assessment: blind.SignedAssessment
    trace_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False


def _validate_digest(value: Any, name: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _source_digest(source_file: Path) -> str:
    if (not isinstance(source_file, Path) or not source_file.is_absolute()
            or source_file.suffix != ".py"):
        raise ValueError("loader source must be an absolute Python file Path")
    for part in (source_file, *source_file.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise InferenceCustodyError("linked loader source or ancestor")
    try:
        fd = os.open(source_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_SOURCE_BYTES:
                raise InferenceCustodyError("loader source is absent or oversized")
            digest = sha256()
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        final = source_file.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if identity(before) != identity(after) or identity(before) != identity(final):
            raise InferenceCustodyError("loader source changed during readback")
        return digest.hexdigest()
    except OSError as exc:
        raise InferenceCustodyError("loader source unavailable") from exc


def _verified_loader(loader: Callable[..., Any], identity: PinnedLoader) -> None:
    if type(identity) is not PinnedLoader:
        raise ValueError("independently pinned loader identity required")
    _validate_digest(identity.source_sha256, "loader source")
    _validate_digest(identity.checkpoint_sha256, "checkpoint")
    if (not inspect.isfunction(loader) or loader.__closure__ is not None
            or getattr(loader, "__code__", None) is None
            or not loader.__name__.isidentifier()
            or loader.__qualname__ != loader.__name__
            or getattr(sys.modules.get(loader.__module__), loader.__name__, None) is not loader):
        raise InferenceCustodyError("top-level nonclosure Python loader required")
    if Path(loader.__code__.co_filename).absolute() != identity.source_file:
        raise InferenceCustodyError("loader function originates from another source")
    if not hmac.compare_digest(_source_digest(identity.source_file),
                               identity.source_sha256):
        raise InferenceCustodyError("loader source bytes differ from pinned SHA-256")


def _require_precommitted_pending_assessment(
    db_path: Path, cases: tuple[blind.PrivateExamCase, ...], *,
    run_id: str, checkpoint_sha256: str, state_key: bytes,
) -> None:
    """Check signed run/candidate/roster before invoking a private model."""
    path = blind._safe_path(db_path)
    if not path.is_file():
        raise blind.BlindCustodyError("precommitted evaluator ledger absent")
    db = blind._connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        state, record_json = blind._read_state(db, state_key)
        record = blind._exact_json(record_json.encode("utf-8"))
        if (state != "PENDING" or type(record) is not dict
                or record.get("run_id") != run_id
                or record.get("checkpoint_sha256") != checkpoint_sha256
                or record.get("public_commitment") != blind._public_commitment(cases)):
            raise blind.BlindCustodyError("no pending run-bound private examination")
    finally:
        db.rollback()
        db.close()


def _reserve_once_before_private_inference(
    db_path: Path, cases: tuple[blind.PrivateExamCase, ...], *,
    run_id: str, checkpoint_sha256: str, loader_source_sha256: str,
    state_key: bytes,
) -> None:
    """Exclusive, irrevocable reservation: crash must NOT permit exam replay.

    This separate local file is deliberately never removed, even on invalid
    predictor output. It is a process-level, filesystem-backed one-shot latch,
    NOT a host rollback-resistant monotonic anchor or power-loss guarantee.
    """
    ledger = blind._safe_path(db_path)
    reservation = ledger.with_name(ledger.name + ".inference-reserved")
    # The parent was checked in _safe_path; O_EXCL prevents a concurrent
    # invocation or symlink at the final pathname from gaining ownership.
    payload = _canonical({
        "schema": "labyrinth.evaluator-invocation-reservation.v1",
        "run_id": run_id, "checkpoint_sha256": checkpoint_sha256,
        "loader_source_sha256": loader_source_sha256,
        "public_exam_sha256": blind._public_commitment(cases),
        "state": "RESERVED_NO_REPLAY",
    })
    seal = hmac.new(state_key, _RESERVATION_DOMAIN + payload, sha256).hexdigest()
    data = _canonical({"payload": payload.decode("utf-8"), "state_mac": seal})
    try:
        fd = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise InferenceCustodyError("private inference already reserved; no replay") from exc
    except OSError as exc:
        raise InferenceCustodyError("cannot reserve evaluator-owned inference") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # Never delete the partial reservation: a crash at this point cannot
        # safely distinguish whether private prompts may have been exposed.
        raise InferenceCustodyError("inference reservation could not be persisted") from exc


def execute_private_development_exam(
    db_path: Path, cases: tuple[blind.PrivateExamCase, ...], *,
    artifact_root: Path, artifact_name: str, pinned_loader: PinnedLoader,
    loader: Callable[[Path, str], Any], evaluator_key: bytes, state_key: bytes,
    run_id: str,
) -> SignedEvaluatorInvocation:
    """Invoke evaluator-owned prediction and score one existing sealed exam.

    ``loader`` is an operator-trusted, independently source-pinned callable.
    Its returned model must have ``predict(prompt: str) -> str``. Source tests
    can inject a mock predictor, but this never authenticates physical inference.
    The signed assessment and trace explicitly withhold graduation authority.
    """
    blind._keys(evaluator_key, state_key)
    blind._identifier(run_id, "run ID")
    _verified_loader(loader, pinned_loader)
    measured = blind._artifact_digest(artifact_root, artifact_name)
    if measured != pinned_loader.checkpoint_sha256:
        raise InferenceCustodyError("candidate bytes differ from pinned checkpoint")
    _require_precommitted_pending_assessment(
        db_path, cases, run_id=run_id,
        checkpoint_sha256=measured, state_key=state_key,
    )
    _reserve_once_before_private_inference(
        db_path, cases, run_id=run_id,
        checkpoint_sha256=measured,
        loader_source_sha256=pinned_loader.source_sha256,
        state_key=state_key,
    )
    questions = blind.public_exam_questions(cases)

    predictor = loader(artifact_root, artifact_name)
    if predictor is None or not callable(getattr(predictor, "predict", None)):
        raise InferenceCustodyError("loader failed to construct a predictor")
    responses: dict[str, str] = {}
    for question in questions:
        result = predictor.predict(question["prompt"])
        if type(result) is not str or not 0 < len(result) <= 128:
            raise InferenceCustodyError("predictor returned invalid answer")
        responses[question["case_id"]] = result
    if blind._artifact_digest(artifact_root, artifact_name) != measured:
        raise InferenceCustodyError("checkpoint changed during private inference")

    assessment = blind.score_assessment(
        db_path, cases, responses, artifact_root=artifact_root,
        artifact_name=artifact_name, evaluator_key=evaluator_key,
        state_key=state_key,
    )
    verified = blind.verify_assessment(
        assessment.receipt_bytes, assessment.signature_hex,
        evaluator_key=evaluator_key, run_id=run_id,
        checkpoint_sha256=measured,
    )
    trace = _canonical({
        "schema": SCHEMA, "run_id": run_id,
        "checkpoint_sha256": measured,
        "loader_source_sha256": pinned_loader.source_sha256,
        "assessment_sha256": sha256(assessment.receipt_bytes).hexdigest(),
        "assessment_signature": assessment.signature_hex,
        "public_exam_sha256": verified["public_exam_sha256"],
        "question_count": len(questions),
        "groups": verified["groups"],
        "evaluator_invoked_predictor": True,
        "caller_responses_accepted": False,
        "model_execution_authenticated": False,
        "graduation_authorized": False,
    })
    signature = hmac.new(evaluator_key, _DOMAIN + trace, sha256).hexdigest()
    return SignedEvaluatorInvocation(assessment, trace, signature)


def verify_invocation(
    result: SignedEvaluatorInvocation, *, evaluator_key: bytes,
    run_id: str, checkpoint_sha256: str, loader_source_sha256: str,
) -> dict[str, Any]:
    """Verify trace plus scored receipt; always development-only authority."""
    blind._key(evaluator_key, "evaluator")
    blind._identifier(run_id, "run ID")
    _validate_digest(checkpoint_sha256, "checkpoint")
    _validate_digest(loader_source_sha256, "loader")
    if (type(result) is not SignedEvaluatorInvocation
            or result.graduation_authorized is not False
            or type(result.trace_bytes) is not bytes
            or type(result.signature_hex) is not str
            or not _DIGEST.fullmatch(result.signature_hex)):
        raise InferenceCustodyError("invalid signed evaluator invocation")
    expected = hmac.new(evaluator_key, _DOMAIN + result.trace_bytes,
                        sha256).hexdigest()
    if not hmac.compare_digest(expected, result.signature_hex):
        raise InferenceCustodyError("invocation signature mismatch")
    trace = blind._exact_json(result.trace_bytes)
    required = {"schema", "run_id", "checkpoint_sha256", "loader_source_sha256",
                "assessment_sha256", "assessment_signature", "public_exam_sha256",
                "question_count", "groups", "evaluator_invoked_predictor",
                "caller_responses_accepted", "model_execution_authenticated",
                "graduation_authorized"}
    if (type(trace) is not dict or trace.keys() != required
            or trace["schema"] != SCHEMA or trace["run_id"] != run_id
            or trace["checkpoint_sha256"] != checkpoint_sha256
            or trace["loader_source_sha256"] != loader_source_sha256
            or trace["graduation_authorized"] is not False
            or trace["model_execution_authenticated"] is not False
            or trace["evaluator_invoked_predictor"] is not True
            or trace["caller_responses_accepted"] is not False
            or type(trace["question_count"]) is not int
            or not 4 <= trace["question_count"] <= 1024):
        raise InferenceCustodyError("signed inference trace identity invalid")
    verified = blind.verify_assessment(
        result.assessment.receipt_bytes, result.assessment.signature_hex,
        evaluator_key=evaluator_key, run_id=run_id,
        checkpoint_sha256=checkpoint_sha256,
    )
    if (trace["assessment_sha256"] != sha256(result.assessment.receipt_bytes).hexdigest()
            or trace["assessment_signature"] != result.assessment.signature_hex
            or trace["public_exam_sha256"] != verified["public_exam_sha256"]
            or trace["groups"] != verified["groups"]
            or trace["question_count"] != sum(g["total"] for g in verified["groups"])):
        raise InferenceCustodyError("invocation not bound to scored assessment")
    return trace
