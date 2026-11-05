"""SOURCE-ONLY VN1 -> Lappy private TRAIN dispatch contract, with local WAL.

No network or host execution exists here. A job identifies immutable W0
artifacts and approved TRAIN work by hash, never by embedded gold, prompts,
credentials or arbitrary worker-side absolute paths. The WAL records an
intent *before* any future transport send; uncertain effects forbid retries.

Hashes establish local consistency, not worker identity, independent custody,
idle-host attestation, neural learning or transport authentication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


JOB_SCHEMA = "labyrinth.private-worker-job.v1"
WAL_SCHEMA = "labyrinth.private-worker-wal.v1"
RECEIPT_SCHEMA = "labyrinth.private-worker-result.v1"
_DOMAIN = b"labyrinth-private-worker-v1\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_NONCE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_RELATIVE = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+){0,7}\Z", re.ASCII)
_WINDOWS_RESERVED = frozenset(("CON", "PRN", "AUX", "NUL")) | frozenset(
    f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
)
_MAX_FILES = 128
_MAX_JOB_BYTES = 64 * 1024
_MAX_EVENTS = 16
_MAX_FILE_BYTES = 1 << 40
_ALLOWED_ROLES = frozenset((
    "weight_shard", "weight_index", "model_config", "tokenizer", "adapter_weight", "adapter_config",
))


def _canonical(value: Any) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(raw) > _MAX_JOB_BYTES:
        raise ValueError("private dispatch payload exceeds 64 KiB")
    return raw


def _sha(value: Any, label: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact SHA-256 digest")


def _id(value: Any, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _hash(value: Any, label: str) -> str:
    return hashlib.sha256(_DOMAIN + label.encode() + b"\0" + _canonical(value)).hexdigest()


@dataclass(frozen=True)
class ArtifactPart:
    """Relative file identity, never a host filesystem instruction."""

    path: str
    role: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (type(self.path) is not str or len(self.path) > 240
                or _RELATIVE.fullmatch(self.path) is None
                or any(segment in (".", "..") or segment.endswith(".") for segment in self.path.split("/"))
                or any(segment.split(".", 1)[0].upper() in _WINDOWS_RESERVED
                       for segment in self.path.split("/"))
                or type(self.role) is not str or self.role not in _ALLOWED_ROLES
                or type(self.size_bytes) is not int or not 1 <= self.size_bytes <= _MAX_FILE_BYTES):
            raise ValueError("invalid bounded model artifact role/path/size")
        _sha(self.sha256, "artifact")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256,
                "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ModelSnapshot:
    """W0 closure: all weight shards, index, config, tokenizer, adapter, receipts."""

    entrance_manifest_sha256: str
    w0_state_sha256: str
    artifacts: tuple[ArtifactPart, ...]
    adapter_status: str

    def __post_init__(self) -> None:
        _sha(self.entrance_manifest_sha256, "entrance manifest")
        _sha(self.w0_state_sha256, "W0 state")
        if (type(self.artifacts) is not tuple or not 3 <= len(self.artifacts) <= _MAX_FILES
                or any(type(part) is not ArtifactPart for part in self.artifacts)
                or type(self.adapter_status) is not str
                or self.adapter_status not in ("ADAPTER_PRESENT", "BASE_ONLY_NOT_ADAPTED")):
            raise ValueError("invalid W0 artifact closure")
        for part in self.artifacts:
            ArtifactPart.__post_init__(part)
        paths = [part.path.lower() for part in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate or case-colliding W0 artifact paths")
        if paths != sorted(paths):
            raise ValueError("W0 artifact closure must use canonical path order")
        roles = [part.role for part in self.artifacts]
        if ("weight_shard" not in roles or "model_config" not in roles
                or "tokenizer" not in roles):
            raise ValueError("W0 requires weights, config and tokenizer")
        shard_count = roles.count("weight_shard")
        if shard_count > 1 and roles.count("weight_index") != 1:
            raise ValueError("sharded W0 requires a single weight index")
        if roles.count("model_config") != 1 or roles.count("weight_index") > 1:
            raise ValueError("W0 has ambiguous metadata")
        has_adapter = "adapter_weight" in roles or "adapter_config" in roles
        if (has_adapter != (self.adapter_status == "ADAPTER_PRESENT")
                or (has_adapter and (roles.count("adapter_weight") != 1
                                     or roles.count("adapter_config") != 1))):
            raise ValueError("W0 adapter files/status disagree")
        if any((part.role == "weight_shard" and not part.path.endswith((".safetensors", ".gguf")))
               or (part.role == "weight_index" and not part.path.endswith(".index.json"))
               or (part.role == "model_config" and part.path.split("/")[-1] != "config.json")
               or (part.role == "adapter_weight" and part.path.split("/")[-1] != "adapter_model.safetensors")
               or (part.role == "adapter_config" and part.path.split("/")[-1] != "adapter_config.json")
               or (part.role == "tokenizer" and part.path.split("/")[-1] not in (
                   "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt",
                   "tokenizer_config.json", "merges.txt", "special_tokens_map.json"))
               for part in self.artifacts):
            raise ValueError("W0 artifact filename does not match declared role")
        if len({(part.role, part.sha256) for part in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate W0 artifact content within a role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entrance_manifest_sha256": self.entrance_manifest_sha256,
            "w0_state_sha256": self.w0_state_sha256,
            "adapter_status": self.adapter_status,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @property
    def closure_sha256(self) -> str:
        return _hash(self.to_dict(), "w0-closure")


@dataclass(frozen=True)
class ApprovedTrainWork:
    objective_id: str
    scope_sha256: str
    approved_train_manifest_sha256: str
    train_source_receipt_sha256: str
    predecessor_checkpoint_sha256: str
    predecessor_worker_receipt_sha256: str | None
    cycle_index: int
    chamber_count: int
    train_split: str = "train"

    def __post_init__(self) -> None:
        _id(self.objective_id, "objective")
        for label in ("scope_sha256", "approved_train_manifest_sha256",
                      "train_source_receipt_sha256", "predecessor_checkpoint_sha256"):
            _sha(getattr(self, label), label)
        if self.predecessor_worker_receipt_sha256 is not None:
            _sha(self.predecessor_worker_receipt_sha256, "previous worker receipt")
        if (type(self.cycle_index) is not int or not 0 <= self.cycle_index <= 7
                or type(self.chamber_count) is not int or not 1 <= self.chamber_count <= 128
                or self.train_split != "train"
                or (self.cycle_index == 0) != (self.predecessor_worker_receipt_sha256 is None)):
            raise ValueError("dispatch permits only bounded, approved TRAIN chambers")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "objective_id", "scope_sha256", "approved_train_manifest_sha256",
            "train_source_receipt_sha256", "predecessor_checkpoint_sha256",
            "predecessor_worker_receipt_sha256", "cycle_index", "chamber_count", "train_split",
        )}


@dataclass(frozen=True)
class IdleGpuLease:
    """Operator claim only; no real host-idle or GPU observation is performed."""

    lease_nonce: str
    worker_host: str
    issued_unix_s: int
    expires_unix_s: int
    min_vram_mib: int
    min_ram_mib: int
    min_disk_free_mib: int
    cuda_required: bool = True
    idle_attestation: str = "SOURCE_ONLY_OPERATOR_DECLARATION"

    def __post_init__(self) -> None:
        if type(self.lease_nonce) is not str or _NONCE.fullmatch(self.lease_nonce) is None:
            raise ValueError("idle lease needs a fixed 128-bit nonce")
        if self.worker_host != "Lappy" or self.idle_attestation != "SOURCE_ONLY_OPERATOR_DECLARATION":
            raise ValueError("unqualified or wrong-host lease")
        if (type(self.issued_unix_s) is not int or type(self.expires_unix_s) is not int
                or not 0 < self.issued_unix_s < self.expires_unix_s
                or self.expires_unix_s - self.issued_unix_s > 600):
            raise ValueError("idle resource lease must last at most 600 seconds")
        for field in ("min_vram_mib", "min_ram_mib", "min_disk_free_mib"):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= 1_048_576:
                raise ValueError("invalid bounded resource requirement")
        if self.cuda_required is not True:
            raise ValueError("this worker lane is reserved for CUDA TRAIN work")

    def valid_at(self, unix_s: int) -> bool:
        return type(unix_s) is int and self.issued_unix_s <= unix_s < self.expires_unix_s

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "lease_nonce", "worker_host", "issued_unix_s", "expires_unix_s",
            "min_vram_mib", "min_ram_mib", "min_disk_free_mib",
            "cuda_required", "idle_attestation",
        )}


@dataclass(frozen=True)
class PrivateTrainJob:
    snapshot: ModelSnapshot
    work: ApprovedTrainWork
    lease: IdleGpuLease
    attempt_nonce: str
    coordinator_host: str = "VN1"
    worker_host: str = "Lappy"
    route: str = "PRIVATE_LAN_OR_TAILSCALE_FUTURE"
    schema: str = JOB_SCHEMA

    def __post_init__(self) -> None:
        if (type(self.snapshot) is not ModelSnapshot
                or type(self.work) is not ApprovedTrainWork
                or type(self.lease) is not IdleGpuLease
                or type(self.attempt_nonce) is not str or _NONCE.fullmatch(self.attempt_nonce) is None
                or self.coordinator_host != "VN1" or self.worker_host != "Lappy"
                or self.route != "PRIVATE_LAN_OR_TAILSCALE_FUTURE" or self.schema != JOB_SCHEMA
                or self.lease.worker_host != self.worker_host
                or (self.work.cycle_index == 0
                    and self.work.predecessor_checkpoint_sha256 != self.snapshot.w0_state_sha256)):
            raise ValueError("invalid VN1/Lappy TRAIN job or W0 predecessor")
        self.snapshot.__post_init__()
        self.work.__post_init__()
        self.lease.__post_init__()
        _canonical(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "coordinator_host": self.coordinator_host,
            "worker_host": self.worker_host, "route": self.route,
            "snapshot": self.snapshot.to_dict(), "work": self.work.to_dict(),
            "lease": self.lease.to_dict(), "attempt_nonce": self.attempt_nonce,
        }

    @property
    def idempotency_key(self) -> str:
        # Stable across attempt nonce/lease changes, binding a single approved
        # TRAIN work unit to W0 and designated hosts. An uncertain attempt is
        # never bypassed with a fresh job nonce for the same work.
        return _hash({
            "snapshot": self.snapshot.closure_sha256,
            "work": self.work.to_dict(),
            "coordinator": self.coordinator_host, "worker": self.worker_host,
        }, "idempotency")

    @property
    def job_id(self) -> str:
        return _hash(self.to_dict(), "job-id")


@dataclass(frozen=True)
class WorkerOutputReceipt:
    """Unsigned source-level response; physical saved weights not verified here."""

    job_id: str
    idempotency_key: str
    lease_nonce: str
    worker_host: str
    predecessor_state_sha256: str
    output_state_sha256: str
    output_manifest_sha256: str
    completed_train_chambers: int
    classification: str = "WORKER_SELF_REPORTED_UNVERIFIED"
    schema: str = RECEIPT_SCHEMA

    def validate_for(self, job: PrivateTrainJob) -> None:
        if type(job) is not PrivateTrainJob:
            raise ValueError("exact job required")
        for field in ("job_id", "idempotency_key", "predecessor_state_sha256",
                      "output_state_sha256", "output_manifest_sha256"):
            _sha(getattr(self, field), field)
        if (self.schema != RECEIPT_SCHEMA
                or self.classification != "WORKER_SELF_REPORTED_UNVERIFIED"
                or self.job_id != job.job_id or self.idempotency_key != job.idempotency_key
                or self.lease_nonce != job.lease.lease_nonce or self.worker_host != job.worker_host
                or self.predecessor_state_sha256 != job.work.predecessor_checkpoint_sha256
                or self.output_state_sha256 == self.predecessor_state_sha256
                or type(self.completed_train_chambers) is not int
                or self.completed_train_chambers != job.work.chamber_count):
            raise ValueError("worker receipt is inconsistent, replayed or not a completed TRAIN unit")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "schema", "job_id", "idempotency_key", "lease_nonce", "worker_host",
            "predecessor_state_sha256", "output_state_sha256", "output_manifest_sha256",
            "completed_train_chambers", "classification",
        )}


class LocalDispatchWAL:
    """No-overwrite intent journal; source-only coordination, never a sender.

    A prepared idempotency key is reserved by its first file. Every transition
    creates a separate durable file. A missing/corrupt sequence fails closed.
    Reopening the exact job is allowed for readback, but no resubmit exists
    after DISPATCH_INTENT_DURABLE or UNKNOWN_EFFECT.
    """

    _TRANSITIONS = {
        None: frozenset(("PREPARED",)),
        "PREPARED": frozenset(("DISPATCH_INTENT_DURABLE", "ABORTED")),
        "DISPATCH_INTENT_DURABLE": frozenset(("ACK_RECEIVED", "UNKNOWN_EFFECT")),
        "ACK_RECEIVED": frozenset(("OUTPUT_RECORDED", "UNKNOWN_EFFECT")),
        "OUTPUT_RECORDED": frozenset(("COMPLETED",)),
        "UNKNOWN_EFFECT": frozenset(),
        "ABORTED": frozenset(),
        "COMPLETED": frozenset(),
    }

    def __init__(self, job: PrivateTrainJob, directory: Path):
        if type(job) is not PrivateTrainJob:
            raise ValueError("exact bounded TRAIN job required")
        job.__post_init__()
        if (not isinstance(directory, Path) or not directory.is_absolute()
                or directory.is_symlink() or not directory.is_dir()
                or directory != directory.resolve(strict=True)):
            raise ValueError("WAL requires an existing canonical, non-symlink local directory")
        self.job = job
        self.directory = directory
        self.events = self._readback()

    def _path(self, index: int) -> Path:
        return self.directory / f"dispatch-{self.job.idempotency_key}-{index:02d}.json"

    @staticmethod
    def _event_hash(event: dict[str, Any]) -> str:
        return _hash({key: value for key, value in event.items() if key != "event_sha256"}, "wal-event")

    def _readback(self) -> list[dict[str, Any]]:
        paths = sorted(self.directory.glob(f"dispatch-{self.job.idempotency_key}-*.json"))
        if len(paths) > _MAX_EVENTS:
            raise ValueError("WAL event budget exceeded")
        events = []
        previous_hash = "0" * 64
        state = None
        for index, path in enumerate(paths):
            if (path != self._path(index) or path.is_symlink() or not path.is_file()
                    or not 0 < path.stat().st_size <= 4096):
                raise ValueError("missing or unsafe WAL event slot")
            pairs = json.loads(path.read_bytes(), object_pairs_hook=self._unique_pairs)
            if (type(pairs) is not dict or set(pairs) != {
                "schema", "index", "job_id", "idempotency_key", "state",
                "parent_sha256", "receipt_sha256", "worker_receipt", "event_sha256",
            } or pairs["schema"] != WAL_SCHEMA or type(pairs["index"]) is not int
                    or pairs["index"] != index or pairs["job_id"] != self.job.job_id
                    or pairs["idempotency_key"] != self.job.idempotency_key
                    or pairs["parent_sha256"] != previous_hash
                    or pairs["state"] not in self._TRANSITIONS[state]
                    or pairs["event_sha256"] != self._event_hash(pairs)):
                raise ValueError("WAL is corrupt, forked or cross-job replayed")
            if pairs["receipt_sha256"] is not None:
                _sha(pairs["receipt_sha256"], "WAL receipt")
            if (pairs["state"] == "OUTPUT_RECORDED") != (pairs["receipt_sha256"] is not None):
                raise ValueError("WAL result binding is missing or unexpected")
            if pairs["state"] == "OUTPUT_RECORDED":
                if type(pairs["worker_receipt"]) is not dict:
                    raise ValueError("WAL lacks actual worker output receipt")
                try:
                    output = WorkerOutputReceipt(**pairs["worker_receipt"])
                    output.validate_for(self.job)
                except (TypeError, ValueError) as exc:
                    raise ValueError("WAL worker output receipt cannot be replay-verified") from exc
                if _hash(output.to_dict(), "worker-result") != pairs["receipt_sha256"]:
                    raise ValueError("WAL worker receipt content differs from sealed digest")
            elif pairs["worker_receipt"] is not None:
                raise ValueError("worker receipt cannot be smuggled into another WAL phase")
            previous_hash = pairs["event_sha256"]
            state = pairs["state"]
            events.append(pairs)
        return events

    @staticmethod
    def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate WAL field")
            value[key] = item
        return value

    @property
    def state(self) -> str | None:
        return self.events[-1]["state"] if self.events else None

    def _advance(self, next_state: str, *, receipt: WorkerOutputReceipt | None = None) -> None:
        self.events = self._readback()
        if next_state not in self._TRANSITIONS[self.state] or len(self.events) >= _MAX_EVENTS:
            raise ValueError("illegal or ambiguous worker effect transition")
        if (next_state == "OUTPUT_RECORDED") != (receipt is not None):
            raise ValueError("output receipt commitment is required only for OUTPUT_RECORDED")
        event = {
            "schema": WAL_SCHEMA, "index": len(self.events),
            "job_id": self.job.job_id, "idempotency_key": self.job.idempotency_key,
            "state": next_state,
            "parent_sha256": self.events[-1]["event_sha256"] if self.events else "0" * 64,
            "receipt_sha256": _hash(receipt.to_dict(), "worker-result") if receipt else None,
            "worker_receipt": receipt.to_dict() if receipt else None,
        }
        event["event_sha256"] = self._event_hash(event)
        raw = _canonical(event) + b"\n"
        path = self._path(len(self.events))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # An incomplete event file is intentionally left for readback to
            # reject; silently deleting it could re-open ambiguous dispatch.
            raise
        # Directory metadata persistence is platform-dependent; attempt a
        # best-effort sync where supported, without claiming host custody.
        try:
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        self.events = self._readback()

    def prepare(self) -> None:
        self._advance("PREPARED")

    def record_send_intent(self, *, unix_s: int) -> None:
        if not self.job.lease.valid_at(unix_s):
            raise ValueError("idle/GPU lease expired or has not begun")
        self._advance("DISPATCH_INTENT_DURABLE")

    def acknowledge(self, *, job_id: str, idempotency_key: str, lease_nonce: str) -> None:
        if (job_id, idempotency_key, lease_nonce) != (
            self.job.job_id, self.job.idempotency_key, self.job.lease.lease_nonce,
        ):
            raise ValueError("cross-job or stale-lease acknowledgement")
        self._advance("ACK_RECEIVED")

    def record_output(self, receipt: WorkerOutputReceipt) -> None:
        if type(receipt) is not WorkerOutputReceipt:
            raise ValueError("exact output receipt required")
        receipt.validate_for(self.job)
        self._advance("OUTPUT_RECORDED", receipt=receipt)

    def complete(self) -> None:
        self._advance("COMPLETED")

    def mark_unknown_effect(self) -> None:
        self._advance("UNKNOWN_EFFECT")

    def abort_before_send(self) -> None:
        self._advance("ABORTED")

    def retry_authorized(self) -> bool:
        self.events = self._readback()
        # A PREPARED job has no recorded external effect. Once intent is
        # durable, a future sender must reconcile; never blind-resubmit.
        return self.state == "PREPARED"


__all__ = [
    "ArtifactPart", "ModelSnapshot", "ApprovedTrainWork", "IdleGpuLease",
    "PrivateTrainJob", "WorkerOutputReceipt", "LocalDispatchWAL",
    "JOB_SCHEMA", "WAL_SCHEMA", "RECEIPT_SCHEMA",
]
