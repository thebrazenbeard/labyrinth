"""Recoverable *source-level* adapter/head checkpoint at a Maze cycle boundary.

This opt-in envelope composes the existing pickle-free full-state CPU snapshot.
It binds actual trainable adapter/head bytes and the current frozen encoder
bytes separately from the caller's pinned pretrained-base *artifact* digest.
AdamW moments and step counters are mandatory, not optional continuation data.

The write-once directory is published by an atomic no-replace rename after
both payload files and this manifest have been synced. On supported POSIX
filesystems, the parent directory is also synced before a successful receipt
or an ``after_publish`` crash. The distinct ``before_sync`` seam publishes a
name whose power-loss durability is UNKNOWN. Recovery MUST use a digest
independently pinned by the calling journal/authority.

Nothing here establishes publisher authenticity of base weights, a true latest
checkpoint, external monotonic CAS, host ownership, Q8 antirollback, pretrained
effectiveness, identity/memory change authority, or graduation. In particular,
the previous receipt is caller-supplied source evidence, not a protected fence.
Windows has a no-replace directory rename, but this implementation lacks a
Windows directory-durability primitive and fails closed for checkpoint writes.
Even on POSIX, actual storage power-loss durability is a physical gate requiring
filesystem/hardware fault testing; no such Q8 proof is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

from .neural_resume_checkpoint import (
    FullCheckpointContextV2, FullCheckpointError, _context_json,
    _decode_tree, _dependencies, _MAX_JSON_BYTES, _MAX_TENSOR_BYTES,
    _safe_snapshot_dir, _strict_json,
    _trainable_and_groups, load_full_checkpoint, save_full_checkpoint,
)


SCHEMA = "labyrinth.neural-cycle-boundary.r92.v2"
_DOMAIN = b"labyrinth.neural-cycle-boundary.r92.v2\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_STAGES = ("trained", "regeneration_ready")
_FILES = frozenset(("cycle_manifest.json", "full_state"))
_MAX_MANIFEST = 64 * 1024
_ZERO = "0" * 64


class CycleCheckpointError(ValueError):
    """Malformed, unsafe, incomplete, stale, or incompatible checkpoint."""


class InjectedCycleCrash(RuntimeError):
    """Deterministic test-only power-loss simulation at a durability barrier."""


@dataclass(frozen=True)
class CycleBoundaryReceipt:
    manifest_sha256: str
    run_id: str
    cycle_index: int
    stage: str
    base_artifact_sha256: str
    frozen_base_sha256: str
    adapter_head_sha256: str
    full_state_sha256: str
    predecessor_manifest_sha256: str
    completed_optimizer_steps: int
    cursor: int
    optimizer_moments_sha256: str
    graduation_authorized: bool = False


def _digest(value: Any) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _receipt_digest(data: bytes) -> str:
    return sha256(_DOMAIN + data).hexdigest()


def _validate_transition(previous: CycleBoundaryReceipt | None, *,
                         context: FullCheckpointContextV2, stage: str,
                         steps: int, base: str, frozen: str,
                         adapter: str, moments: str) -> str:
    if type(context) is not FullCheckpointContextV2:
        raise CycleCheckpointError("v2 immediate-predecessor context required")
    try:
        context.validate()
    except ValueError as exc:
        raise CycleCheckpointError("invalid full-state checkpoint context") from exc
    if stage not in _STAGES or type(stage) is not str:
        raise CycleCheckpointError("invalid cycle stage")
    if type(steps) is not int or steps < 1 or steps > context.cursor:
        raise CycleCheckpointError("completed optimizer step count invalid")
    if not all(_digest(item) for item in (base, frozen, adapter, moments)):
        raise CycleCheckpointError("explicit base/frozen/adapter/AdamW SHA-256 pins required")
    if context.base_checkpoint_sha256 != base:
        raise CycleCheckpointError("context and pretrained artifact pin disagree")
    if previous is None:
        if (context.cycle_index != 0 or stage != "trained"
                or context.immediate_predecessor_sha256 != context.prior_checkpoint_sha256):
            raise CycleCheckpointError("first boundary must be cycle-zero trained")
        return _ZERO
    if (type(previous) is not CycleBoundaryReceipt
            or not all(_digest(item) for item in
                       (previous.manifest_sha256, previous.adapter_head_sha256,
                        previous.optimizer_moments_sha256))):
        raise CycleCheckpointError("prior boundary receipt required")
    if (previous.run_id != context.run_id
            or previous.base_artifact_sha256 != base
            or previous.frozen_base_sha256 != frozen
            # The substrate reserves cycle-zero's immediate predecessor for
            # the initial anchor even at its second (regeneration) stage.
            # The envelope's predecessor_manifest_sha256 chains those two
            # stages. Later stages use the actual previous full-state SHA.
            or context.immediate_predecessor_sha256 != (
                context.prior_checkpoint_sha256 if context.cycle_index == 0
                else previous.full_state_sha256)
            or previous.cursor > context.cursor
            or previous.completed_optimizer_steps > steps):
        raise CycleCheckpointError("base, predecessor or optimizer cursor continuity mismatch")
    allowed = ((previous.cycle_index, previous.stage, context.cycle_index, stage)
               in ((context.cycle_index, "trained", context.cycle_index, "regeneration_ready"),
                   (context.cycle_index - 1, "regeneration_ready", context.cycle_index, "trained")))
    if not allowed:
        raise CycleCheckpointError("cycle stage must advance exactly once")
    if ((stage == "regeneration_ready" and steps != previous.completed_optimizer_steps)
            or (stage == "trained" and steps <= previous.completed_optimizer_steps)):
        raise CycleCheckpointError("optimizer step count contradicts stage")
    if (stage == "regeneration_ready"
            and (context.cursor != previous.cursor
                 or adapter != previous.adapter_head_sha256
                 or moments != previous.optimizer_moments_sha256)):
        raise CycleCheckpointError(
            "regeneration cannot consume exposure or mutate adapter/head or AdamW moments")
    return previous.manifest_sha256


def _typed_members_digest(members: list[tuple[str, Any]], torch: Any) -> str:
    """Hash typed names/shapes/bytes, not Python hash or file path identities."""
    if not members:
        raise CycleCheckpointError("empty frozen base or adapter/head inventory")
    hasher = sha256(b"labyrinth.typed-tensors.r92.v1\0")
    for name, tensor in sorted(members):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.is_sparse or tensor.is_quantized):
            raise CycleCheckpointError("only dense CPU tensor identities are supported")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise CycleCheckpointError("nonfinite model tensor")
        # Reshape first: AdamW's scalar step tensor is zero-dimensional, and
        # dtype-reinterpreting view() requires at least one dimension.
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        info = _canonical({"name": name, "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "bytes": len(raw)})
        hasher.update(len(info).to_bytes(8, "big"))
        hasher.update(info)
        hasher.update(raw)
    return hasher.hexdigest()


def _tensor_digest(model: Any, torch: Any, *, frozen: bool) -> str:
    members: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        if bool(parameter.requires_grad) != frozen:
            members.append(("parameter:" + name, parameter))
    if frozen:
        members.extend(("buffer:" + name, buffer) for name, buffer in model.named_buffers())
    return _typed_members_digest(members, torch)


def _assert_complete_adamw(model: Any, optimizer: Any, steps: int, torch: Any,
                           *, state_dict: dict[str, Any] | None = None) -> None:
    try:
        named, groups = _trainable_and_groups(model, optimizer, torch)
    except FullCheckpointError as exc:
        raise CycleCheckpointError("incompatible CPU AdamW parameter inventory") from exc
    state = optimizer.state_dict() if state_dict is None else state_dict
    if (type(state) is not dict or set(state) != {"state", "param_groups"}
            or type(state["state"]) is not dict or type(state["param_groups"]) is not list
            or len(state["param_groups"]) != len(groups)):
        raise CycleCheckpointError("complete AdamW state dictionary required")
    matched: dict[int, str] = {}
    for item, names in zip(state["param_groups"], groups):
        if (type(item) is not dict or type(item.get("params")) is not list
                or len(item["params"]) != len(names) or item.get("amsgrad") is not False):
            raise CycleCheckpointError("unsupported AdamW group shape/options")
        for identifier, name in zip(item["params"], names):
            if type(identifier) is not int or identifier in matched:
                raise CycleCheckpointError("duplicate AdamW state parameter id")
            matched[identifier] = name
    if set(state["state"]) != set(matched):
        raise CycleCheckpointError("missing/extra AdamW optimizer parameter state")
    for identifier, name in matched.items():
        snapshot = state["state"][identifier]
        if type(snapshot) is not dict or set(snapshot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CycleCheckpointError("missing mandatory AdamW moments or step")
        step = snapshot["step"]
        if (not isinstance(step, torch.Tensor) or step.device.type != "cpu"
                or step.numel() != 1 or not bool(torch.isfinite(step).all())
                or step.item() != steps):
            raise CycleCheckpointError("AdamW step counter mismatch")
        parameter = named[name]
        for key in ("exp_avg", "exp_avg_sq"):
            value = snapshot[key]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise CycleCheckpointError("missing or invalid AdamW moment tensor")


def _optimizer_moments_digest(model: Any, optimizer: Any, steps: int, torch: Any,
                              *, state_dict: dict[str, Any] | None = None) -> str:
    """Digest actual AdamW step/moment tensor bytes by stable parameter name.

    The full-state manifest SHA includes RNG and serialization metadata, so it
    cannot serve as the regeneration invariance witness. This distinct digest
    is derived from all parameter-mapped AdamW moment tensors and step values.
    """
    _assert_complete_adamw(model, optimizer, steps, torch, state_dict=state_dict)
    try:
        _, groups = _trainable_and_groups(model, optimizer, torch)
    except FullCheckpointError as exc:
        raise CycleCheckpointError("incompatible AdamW parameter inventory") from exc
    state = optimizer.state_dict() if state_dict is None else state_dict
    members: list[tuple[str, Any]] = []
    for group, names in zip(state["param_groups"], groups):
        for identifier, name in zip(group["params"], names):
            snapshot = state["state"][identifier]
            members.extend((f"adamw:{name}:{key}", snapshot[key])
                           for key in ("step", "exp_avg", "exp_avg_sq"))
    return _typed_members_digest(members, torch)


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        raise CycleCheckpointError("directory fsync unavailable; no durable checkpoint receipt")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_noreplace(staging: Path, destination: Path) -> None:
    """No POSIX rename-over-existing race, including an existing empty dir."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise CycleCheckpointError("atomic no-replace rename unavailable")
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        if operation(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise CycleCheckpointError("atomic no-replace publish failed") from OSError(error, os.strerror(error))
    elif os.name == "nt":
        # Windows os.rename refuses to replace an existing directory.
        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise CycleCheckpointError("atomic no-replace publish failed") from exc
    else:
        raise CycleCheckpointError("atomic no-replace rename unsupported on this platform")


def _publish_and_sync(staging: Path, destination: Path,
                      failure_at: str | None) -> None:
    """Only acknowledge publication after syncing the containing directory."""
    _publish_noreplace(staging, destination)
    _crash(failure_at, "before_sync")  # Visible name; durability not proven.
    _sync_directory(destination.parent)
    _crash(failure_at, "after_publish")  # Synced name; receipt may be lost.


def _check_root(path: str | Path, *, existing: bool) -> Path:
    root = Path(path).absolute()
    # Existing substrate's path traversal and symlink checks are reused.
    if not existing:
        try:
            _safe_snapshot_dir(root, existing=False)
        except FullCheckpointError as exc:
            raise CycleCheckpointError("unsafe or occupied checkpoint destination") from exc
    if existing:
        for part in (root, *root.parents):
            if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
                raise CycleCheckpointError("checkpoint symlink/junction forbidden")
        if not root.is_dir() or {p.name for p in root.iterdir()} != _FILES:
            raise CycleCheckpointError("cycle snapshot missing or contains unexpected files")
        manifest = root / "cycle_manifest.json"
        try:
            if not stat.S_ISREG(manifest.lstat().st_mode) or manifest.is_symlink():
                raise CycleCheckpointError("cycle manifest is not a regular file")
        except FileNotFoundError as exc:
            raise CycleCheckpointError("cycle manifest missing") from exc
    return root


def _receipt(obj: dict[str, Any], digest: str) -> CycleBoundaryReceipt:
    return CycleBoundaryReceipt(digest, obj["run_id"], obj["cycle_index"], obj["stage"],
                                obj["base_artifact_sha256"], obj["frozen_base_sha256"],
                                obj["adapter_head_sha256"], obj["full_state_sha256"],
                                obj["predecessor_manifest_sha256"],
                                obj["completed_optimizer_steps"], obj["cursor"],
                                obj["optimizer_moments_sha256"])


def _crash(failure_at: str | None, barrier: str) -> None:
    if failure_at == barrier:
        raise InjectedCycleCrash("simulated crash " + barrier)


def _readback_digests(state_dir: Path, *, model: Any, optimizer: Any,
                      context: FullCheckpointContextV2, full_sha: str,
                      steps: int, torch: Any, load_file: Any) -> tuple[str, str]:
    """Independently verify the serialized model and moment bytes before trust.

    This intentionally reads the published full-state format (including the
    RNG tensor inventory) instead of hashing the live optimizer twice or
    using the full-state SHA, which can vary with unrelated RNG metadata.
    """
    from .neural_resume_checkpoint import _manifest_digest, SCHEMA_V2
    try:
        root = _safe_snapshot_dir(state_dir, existing=True)
        manifest_path = root / "manifest.json"
        if not 0 < manifest_path.stat().st_size <= _MAX_JSON_BYTES:
            raise CycleCheckpointError("invalid full-state manifest length")
        raw_state = manifest_path.read_bytes()
        if _manifest_digest(raw_state, schema=SCHEMA_V2) != full_sha:
            raise CycleCheckpointError("full-state checkpoint differs from cycle pin")
        state_obj = _strict_json(raw_state)
        named, groups = _trainable_and_groups(model, optimizer, torch)
        if (state_obj.get("schema") != SCHEMA_V2
                or state_obj.get("context") != _context_json(context)
                or state_obj.get("optimizer") != "AdamW"
                or state_obj.get("trainable_names") != sorted(named)
                or state_obj.get("optimizer_groups") != groups):
            raise CycleCheckpointError("full-state model/optimizer/context disagreement")
        tensor_path = root / "state.safetensors"
        if not 0 < tensor_path.stat().st_size <= _MAX_TENSOR_BYTES:
            raise CycleCheckpointError("invalid full-state tensor length")
        if sha256(tensor_path.read_bytes()).hexdigest() != state_obj.get("tensor_sha256"):
            raise CycleCheckpointError("full-state tensor bytes mismatch")
        saved_tensors = load_file(str(tensor_path), device="cpu")
        used: set[str] = set()
        saved_model = _decode_tree(state_obj["model_tree"], saved_tensors, used)
        saved_optimizer = _decode_tree(state_obj["optimizer_tree"], saved_tensors, used)
        _decode_tree(state_obj["rng_tree"], saved_tensors, used)
        if used != set(saved_tensors):
            raise CycleCheckpointError("unattested full-state tensors")
        if type(saved_model) is not dict or set(saved_model) != set(named):
            raise CycleCheckpointError("saved adapter/head inventory mismatch")
        for name, value in saved_model.items():
            parameter = named[name]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.shape != parameter.shape or value.dtype != parameter.dtype):
                raise CycleCheckpointError("saved adapter/head tensor incompatible with model")
        adapter = _typed_members_digest(
            [("parameter:" + name, value) for name, value in saved_model.items()], torch)
        moments = _optimizer_moments_digest(
            model, optimizer, steps, torch, state_dict=saved_optimizer)
        return adapter, moments
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, FullCheckpointError) as exc:
        if isinstance(exc, CycleCheckpointError):
            raise
        raise CycleCheckpointError("invalid saved model/optimizer/RNG state") from exc


def save_cycle_checkpoint(
    destination: str | Path, *, model: Any, optimizer: Any,
    context: FullCheckpointContextV2, stage: str,
    base_artifact_sha256: str, expected_frozen_base_sha256: str,
    completed_optimizer_steps: int,
    previous: CycleBoundaryReceipt | None = None,
    failure_at: str | None = None,
) -> CycleBoundaryReceipt:
    """Atomically commit a unique immutable directory; never reuse its name.

    ``failure_at`` is a literal test seam: ``after_state``, ``after_manifest``,
    ``before_sync``, or ``after_publish``. ``before_sync`` leaves publication
    durability uncertain; ``after_publish`` simulates lost acknowledgment after
    the parent directory has been fsynced. Never retry into the same name.
    """
    if failure_at not in (None, "after_state", "after_manifest",
                          "before_sync", "after_publish"):
        raise ValueError("unknown crash injection barrier")
    if os.name != "posix" or sys.platform != "linux":
        raise CycleCheckpointError(
            "durable no-replace checkpoint publication not supported on this platform")
    root = _check_root(destination, existing=False)
    torch, load_file, _ = _dependencies()
    frozen = _tensor_digest(model, torch, frozen=True)
    if frozen != expected_frozen_base_sha256:
        raise CycleCheckpointError("frozen pretrained base tensor digest mismatch")
    adapter = _tensor_digest(model, torch, frozen=False)
    moments = _optimizer_moments_digest(model, optimizer, completed_optimizer_steps, torch)
    prior_sha = _validate_transition(previous, context=context, stage=stage,
                                     steps=completed_optimizer_steps,
                                     base=base_artifact_sha256,
                                     frozen=expected_frozen_base_sha256,
                                     adapter=adapter, moments=moments)
    temporary = Path(tempfile.mkdtemp(prefix=".labyrinth-cycle-uncommitted-",
                                      dir=str(root.parent)))
    try:
        full = save_full_checkpoint(temporary / "full_state", model=model,
                                    optimizer=optimizer, context=context)
        if (_tensor_digest(model, torch, frozen=True) != frozen
                or _tensor_digest(model, torch, frozen=False) != adapter
                or _optimizer_moments_digest(
                    model, optimizer, completed_optimizer_steps, torch) != moments):
            raise CycleCheckpointError("model or AdamW mutated while saving cycle boundary")
        saved_adapter, saved_moments = _readback_digests(
            temporary / "full_state", model=model, optimizer=optimizer,
            context=context, full_sha=full.sha256, steps=completed_optimizer_steps,
            torch=torch, load_file=load_file)
        if (saved_adapter != adapter or saved_moments != moments):
            raise CycleCheckpointError("full-state readback differs from live boundary")
        _crash(failure_at, "after_state")
        obj = {"schema": SCHEMA, "run_id": context.run_id,
               "cycle_index": context.cycle_index, "stage": stage,
               "base_artifact_sha256": base_artifact_sha256,
               "frozen_base_sha256": frozen, "adapter_head_sha256": adapter,
               "optimizer_moments_sha256": moments,
               "full_state_sha256": full.sha256,
               "predecessor_manifest_sha256": prior_sha,
               "completed_optimizer_steps": completed_optimizer_steps,
               "cursor": context.cursor, "optimizer": "AdamW",
               "optimizer_state_required": True, "graduation_authorized": False,
               "change_class": "model_component_update",
               "state_scope": "local_adapter_head_and_optimizer_only",
               "memory_or_identity_write_authorized": False}
        data = _canonical(obj)
        if len(data) > _MAX_MANIFEST:
            raise CycleCheckpointError("cycle manifest size limit")
        with (temporary / "cycle_manifest.json").open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _sync_directory(temporary)
        _crash(failure_at, "after_manifest")
        _publish_and_sync(temporary, root, failure_at)
        return _receipt(obj, _receipt_digest(data))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_cycle_checkpoint(
    source: str | Path, *, model: Any, optimizer: Any,
    expected_context: FullCheckpointContextV2,
    independently_pinned_manifest_sha256: str,
    expected_base_artifact_sha256: str,
    expected_frozen_base_sha256: str,
    expected_predecessor_manifest_sha256: str,
    expected_stage: str, completed_optimizer_steps: int,
) -> CycleBoundaryReceipt:
    """Preflight pinned bytes/state before restoring any trainable parameters.

    A failed underlying restore may have mutated state: discard the process.
    Pin provenance/latest-stage selection remains the caller's responsibility.
    """
    root = _check_root(source, existing=True)
    if not all(_digest(s) for s in (independently_pinned_manifest_sha256,
                                   expected_base_artifact_sha256,
                                   expected_frozen_base_sha256,
                                   expected_predecessor_manifest_sha256)):
        raise CycleCheckpointError("independent expected SHA-256 pins required")
    manifest_file = root / "cycle_manifest.json"
    if not 0 < manifest_file.stat().st_size <= _MAX_MANIFEST:
        raise CycleCheckpointError("cycle manifest exceeds bound")
    raw = manifest_file.read_bytes()
    if _receipt_digest(raw) != independently_pinned_manifest_sha256:
        raise CycleCheckpointError("cycle manifest differs from independent pin")
    try:
        obj = _strict_json(raw)
    except FullCheckpointError as exc:
        raise CycleCheckpointError("noncanonical cycle manifest") from exc
    expected_keys = {"schema", "run_id", "cycle_index", "stage",
                     "base_artifact_sha256", "frozen_base_sha256",
                     "adapter_head_sha256", "optimizer_moments_sha256",
                     "full_state_sha256",
                     "predecessor_manifest_sha256", "completed_optimizer_steps",
                     "cursor", "optimizer", "optimizer_state_required",
                     "graduation_authorized", "change_class", "state_scope",
                     "memory_or_identity_write_authorized"}
    if (set(obj) != expected_keys or obj["schema"] != SCHEMA
            or obj["optimizer"] != "AdamW" or obj["optimizer_state_required"] is not True
            or obj["graduation_authorized"] is not False
            or obj["change_class"] != "model_component_update"
            or obj["state_scope"] != "local_adapter_head_and_optimizer_only"
            or obj["memory_or_identity_write_authorized"] is not False
            or not _digest(obj["adapter_head_sha256"])
            or not _digest(obj["optimizer_moments_sha256"])
            or not _digest(obj["full_state_sha256"])
            or type(expected_context) is not FullCheckpointContextV2):
        raise CycleCheckpointError("invalid cycle schema or optimizer commitment")
    try:
        expected_context.validate()
    except ValueError as exc:
        raise CycleCheckpointError("invalid expected full-state context") from exc
    if (obj["run_id"] != expected_context.run_id
            or type(obj["cycle_index"]) is not int
            or obj["cycle_index"] != expected_context.cycle_index
            or obj["stage"] != expected_stage or expected_stage not in _STAGES
            or obj["base_artifact_sha256"] != expected_base_artifact_sha256
            or obj["frozen_base_sha256"] != expected_frozen_base_sha256
            or obj["predecessor_manifest_sha256"] != expected_predecessor_manifest_sha256
            or type(obj["completed_optimizer_steps"]) is not int
            or obj["completed_optimizer_steps"] != completed_optimizer_steps
            or type(obj["cursor"]) is not int or obj["cursor"] != expected_context.cursor):
        raise CycleCheckpointError("expected base, lineage, stage, or cursor mismatch")
    torch, load_file, _ = _dependencies()
    if _tensor_digest(model, torch, frozen=True) != expected_frozen_base_sha256:
        raise CycleCheckpointError("restoration attempted against wrong frozen base")
    state_dir = root / "full_state"
    saved_adapter, saved_moments = _readback_digests(
        state_dir, model=model, optimizer=optimizer,
        context=expected_context, full_sha=obj["full_state_sha256"],
        steps=completed_optimizer_steps, torch=torch, load_file=load_file)
    if (saved_adapter != obj["adapter_head_sha256"]
            or saved_moments != obj["optimizer_moments_sha256"]):
        raise CycleCheckpointError("saved model/AdamW bytes disagree with manifest")
    try:
        load_full_checkpoint(state_dir, model=model, optimizer=optimizer,
                             expected_sha256=obj["full_state_sha256"],
                             expected_context=expected_context)
        _assert_complete_adamw(model, optimizer, completed_optimizer_steps, torch)
        if (_tensor_digest(model, torch, frozen=False) != obj["adapter_head_sha256"]
                or _tensor_digest(model, torch, frozen=True) != expected_frozen_base_sha256
                or _optimizer_moments_digest(
                    model, optimizer, completed_optimizer_steps, torch)
                != obj["optimizer_moments_sha256"]):
            raise CycleCheckpointError("restored adapter or frozen base identity mismatch")
    except Exception as exc:
        raise CycleCheckpointError("restore failed; discard process") from exc
    return _receipt(obj, independently_pinned_manifest_sha256)
