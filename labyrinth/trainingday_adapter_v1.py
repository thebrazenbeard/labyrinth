"""Source-bound, fail-closed Labyrinth handoff to TrainingDay artifacts.

This is an *inspector*, not a physical Qwen trainer or a graduation authority.
Even perfect local hashes cannot authenticate the model's learning trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, NoReturn


SCHEMA = "labyrinth.trainingday-source-handoff.v1"
TRAINER_HEAD = "2b83430dd36076c5f65624709968e372b170ea36"
TRAINER_ENTRYPOINT = "trainingday.lowmem_experiment.one_step_kernel"
OFFICIAL_BASE_ID = "Qwen/Qwen3.5-4B"
OFFICIAL_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z", re.ASCII)
_MAX_MANIFEST = 256 * 1024
_MAX_RECEIPT = 256 * 1024
_MAX_ARTIFACT = 64 * 1024 ** 3
_MAX_ITEMS = 128
_ROLES = ("base", "tokenizer", "adapter", "optimizer", "scheduler", "rng", "data")
_DENIED = frozenset({".py", ".sh", ".exe", ".ps1", ".bat", ".cmd", ".pkl", ".dll", ".so"})


class TrainingDayHandoffError(ValueError):
    """Local handoff artifacts or independent expected hashes disagree."""


class TrainingDayPhysicalExecutionHold(RuntimeError):
    """Upstream source has no qualified physical launch authority."""


@dataclass(frozen=True)
class TrainingDayInspection:
    checkpoint_id: str
    step: int
    manifest_sha256: str
    file_count: int
    trainer_head: str = TRAINER_HEAD
    classification: str = "SOURCE_ONLY_LOCAL_MOCK_FILE_INTEGRITY"
    neural_update_verified: bool = False
    full_resume_verified: bool = False
    graduation_authorized: bool = False


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TrainingDayHandoffError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(data: bytes) -> object:
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              TrainingDayHandoffError("nonfinite JSON constant")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDayHandoffError("malformed handoff JSON") from exc


def _relative(name: str) -> Path:
    if (type(name) is not str or not 1 <= len(name) <= 200 or "\\" in name
            or "\x00" in name or name.startswith("/") or ":" in name):
        raise TrainingDayHandoffError("artifact path is not a bounded relative POSIX path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise TrainingDayHandoffError("relative artifact path traverses or aliases parent")
    if Path(parts[-1]).suffix.lower() in _DENIED:
        raise TrainingDayHandoffError("executable or unsafe pickle is not a handoff artifact")
    return Path(*parts)


def _stream_file(root: Path, relative: str, *, max_size: int = _MAX_ARTIFACT) -> tuple[str, int]:
    path = root / _relative(relative)
    # No symlink in any parent component, not just the final open target.
    cursor = root
    for part in _relative(relative).parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink() or not cursor.is_dir():
            raise TrainingDayHandoffError("artifact parent is missing or a symlink")
    if path.is_symlink():
        raise TrainingDayHandoffError("symlink artifact refused")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise TrainingDayHandoffError("handoff artifact is missing or unreadable") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_size:
            raise TrainingDayHandoffError("handoff artifact is not a bounded nonempty file")
        h = hashlib.sha256()
        size = 0
        with os.fdopen(os.dup(fd), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
                size += len(block)
                if size > max_size:
                    raise TrainingDayHandoffError("handoff artifact exceeds byte budget")
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise TrainingDayHandoffError("artifact changed during streaming hash")
        return h.hexdigest(), size
    finally:
        os.close(fd)


def inspect_trainingday_handoff(
    root: str | Path, manifest_name: str, *, trusted_sha256: Mapping[str, str],
) -> TrainingDayInspection:
    """Read bytes and caller-*independently* pinned hashes; accept mock custody only.

    `trusted_sha256` must be supplied through an owner-controlled channel. A
    declaration copied from the untrusted manifest would prove nothing.
    This function deliberately never accepts a genuine neural-update claim.
    """
    original_root = Path(root)
    if original_root.is_symlink():
        raise TrainingDayHandoffError("handoff root cannot be a symlink")
    root = original_root.resolve(strict=True)
    if not root.is_dir():
        raise TrainingDayHandoffError("handoff root must be a canonical directory")
    if not isinstance(trusted_sha256, Mapping):
        raise TrainingDayHandoffError("caller-controlled expected artifact hashes required")
    digest, size = _stream_file(root, manifest_name, max_size=_MAX_MANIFEST)
    if trusted_sha256.get(manifest_name) != digest:
        raise TrainingDayHandoffError("manifest lacks independent SHA-256 pin")
    manifest_bytes = (root / _relative(manifest_name)).read_bytes()
    if len(manifest_bytes) != size or hashlib.sha256(manifest_bytes).hexdigest() != digest:
        raise TrainingDayHandoffError("manifest changed after streamed readback")
    manifest = _load_json(manifest_bytes)
    if type(manifest) is not dict or set(manifest) != {
            "schema", "source", "model", "authority", "checkpoint", "components",
            "files", "worker_receipt", "reload_receipt", "result_receipt"}:
        raise TrainingDayHandoffError("handoff manifest schema is incomplete")
    if manifest["schema"] != SCHEMA or manifest["source"] != {
            "repository": "thebrazenbeard/model-training",
            "head": TRAINER_HEAD, "entrypoint": TRAINER_ENTRYPOINT,
            "status": "SOURCE_ONLY_HARD_HOLD"}:
        raise TrainingDayHandoffError("trainer source authority or head mismatch")
    if manifest["model"] != {
            "id": OFFICIAL_BASE_ID, "revision": OFFICIAL_BASE_REVISION,
            "loader_class": "Qwen3_5ForCausalLM"}:
        raise TrainingDayHandoffError("official pretrained model identity mismatch")
    if manifest["authority"] != {
            "physical_training_authorized": False, "network": "disabled",
            "data_scope": "PUBLIC_TRAIN_ONLY", "budget_usd": 0}:
        raise TrainingDayHandoffError("handoff attempts to assert physical launch authority")
    checkpoint = manifest["checkpoint"]
    if (type(checkpoint) is not dict or set(checkpoint) != {
            "id", "step", "neural_update_verified", "full_resume_verified",
            "graduation_authorized"}
            or type(checkpoint["id"]) is not str or not _ID.fullmatch(checkpoint["id"])
            or type(checkpoint["step"]) is not int or checkpoint["step"] < 0
            or any(checkpoint[key] is not False for key in (
                "neural_update_verified", "full_resume_verified", "graduation_authorized"))):
        raise TrainingDayHandoffError("checkpoint falsely asserts neural quality or authority")
    components = manifest["components"]
    if (type(components) is not dict or set(components) != set(_ROLES)
            or any(type(components[role]) is not list or not components[role]
                   or len(components[role]) > _MAX_ITEMS
                   or any(type(name) is not str for name in components[role])
                   for role in _ROLES)):
        raise TrainingDayHandoffError("incomplete model or trainer-state component inventory")
    files = manifest["files"]
    if type(files) is not dict:
        raise TrainingDayHandoffError("file digest roster required")
    receipt_names = [manifest[key] for key in (
        "worker_receipt", "reload_receipt", "result_receipt")]
    names = [name for role in _ROLES for name in components[role]] + receipt_names
    if (len(names) > _MAX_ITEMS or len(set(names)) != len(names)
            or set(files) != set(names) or len(trusted_sha256) != len(names) + 1):
        raise TrainingDayHandoffError("duplicate, unlisted, or untrusted handoff artifact")
    for role in _ROLES:
        expected = components[role]
        if role == "base" and not any(name.endswith(".safetensors") for name in expected):
            raise TrainingDayHandoffError("official weight shard missing")
        if role == "tokenizer" and not any(name.endswith("tokenizer.json") for name in expected):
            raise TrainingDayHandoffError("official tokenizer missing")
        if role == "adapter" and not any(name.endswith(".safetensors") for name in expected):
            raise TrainingDayHandoffError("trainable adapter missing")
    for name in names:
        record = files.get(name)
        if (type(record) is not dict or set(record) != {"sha256", "bytes"}
                or type(record["sha256"]) is not str or not _SHA.fullmatch(record["sha256"])
                or type(record["bytes"]) is not int or record["bytes"] <= 0):
            raise TrainingDayHandoffError("artifact expected digest entry invalid")
        observed_sha, observed_size = _stream_file(
            root, name, max_size=_MAX_RECEIPT if name in receipt_names else _MAX_ARTIFACT)
        if (observed_sha, observed_size) != (record["sha256"], record["bytes"]):
            raise TrainingDayHandoffError("manifest file digest disagrees with physical bytes")
        if trusted_sha256.get(name) != observed_sha:
            raise TrainingDayHandoffError("owner-trusted hash disagrees with physical bytes")
    for label, name in (("LOCAL_MOCK_WORKER", manifest["worker_receipt"]),
                        ("LOCAL_MOCK_RELOAD", manifest["reload_receipt"]),
                        ("LOCAL_MOCK_RESULT", manifest["result_receipt"])):
        receipt_bytes = (root / _relative(name)).read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != files[name]["sha256"]:
            raise TrainingDayHandoffError("receipt changed after streamed readback")
        receipt = _load_json(receipt_bytes)
        if (type(receipt) is not dict or receipt.get("kind") != label
                or receipt.get("checkpoint_id") != checkpoint["id"]
                or receipt.get("step") != checkpoint["step"]
                or receipt.get("artifact_sha256") != {
                    artifact: files[artifact]["sha256"]
                    for artifact in names if artifact not in receipt_names}
                or (label == "LOCAL_MOCK_RELOAD" and receipt.get("fresh_process") is not True)
                or receipt.get("neural_update_verified") is not False):
            raise TrainingDayHandoffError("mock receipt cannot authenticate neural training")
    return TrainingDayInspection(
        checkpoint["id"], checkpoint["step"], digest, len(names))


def invoke_trainingday_trainer(*_args: object, **_kwargs: object) -> NoReturn:
    """Live source's physical launch authority denies this direct call."""
    raise TrainingDayPhysicalExecutionHold(
        "model-training@" + TRAINER_HEAD + ": SOURCE_ONLY_HARD_HOLD. "
        "one_step_kernel(base_binding_path) has no qualified direct physical launch authority, "
        "and this adapter cannot override it.")


__all__ = [
    "SCHEMA", "TRAINER_HEAD", "TRAINER_ENTRYPOINT", "OFFICIAL_BASE_ID",
    "OFFICIAL_BASE_REVISION", "TrainingDayHandoffError", "TrainingDayPhysicalExecutionHold",
    "TrainingDayInspection", "inspect_trainingday_handoff", "invoke_trainingday_trainer",
]
