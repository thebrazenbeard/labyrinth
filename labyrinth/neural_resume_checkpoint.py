"""Experimental CPU neural *full-state* checkpoint custody.

Unlike adapter-only continuation, a checkpoint here retains trainable tensors,
AdamW moments, Python/NumPy/Torch RNG, a training cursor and exposure sequence.
It uses safetensors and strictly typed JSON; no pickle is read. A supplied,
previously trusted digest and journal fence are required when loading.

This is a checkpoint substrate, not an integrated HF runner or an authority to
replay a consumed development event, graduate, or deploy a model. The caller
must bind the returned digest to the MAC-sealed development journal and enforce
the fence on every worker output. CPU execution only is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import tempfile
from typing import Any


SCHEMA = "labyrinth.full-neural-checkpoint.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_TENSOR_BYTES = 1024 * 1024 * 1024
_FILES = frozenset(("manifest.json", "state.safetensors"))


class FullCheckpointError(ValueError):
    """A snapshot is absent, invalid, incomplete, unsafe, or mismatched."""


@dataclass(frozen=True)
class FullCheckpointContext:
    run_id: str
    objective_id: str
    base_checkpoint_sha256: str
    prior_checkpoint_sha256: str
    fence: int
    cursor: int
    exposures: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if (type(self.run_id) is not str or not _IDENTIFIER.fullmatch(self.run_id)
                or type(self.objective_id) is not str
                or not _IDENTIFIER.fullmatch(self.objective_id)):
            raise ValueError("invalid checkpoint run or objective identifier")
        if (type(self.base_checkpoint_sha256) is not str
                or not _DIGEST.fullmatch(self.base_checkpoint_sha256)
                or type(self.prior_checkpoint_sha256) is not str
                or not _DIGEST.fullmatch(self.prior_checkpoint_sha256)):
            raise ValueError("invalid checkpoint lineage digest")
        if type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1:
            raise ValueError("invalid checkpoint fence")
        if (type(self.exposures) is not tuple or len(self.exposures) > 100000
                or type(self.cursor) is not int
                or not 0 <= self.cursor <= len(self.exposures)):
            raise ValueError("invalid checkpoint cursor or exposure sequence")
        seen_occurrence_ids: set[str] = set()
        for exposure in self.exposures:
            if (type(exposure) is not tuple or len(exposure) != 2
                    or type(exposure[0]) is not str
                    or not _IDENTIFIER.fullmatch(exposure[0])
                    or type(exposure[1]) is not str
                    or not _DIGEST.fullmatch(exposure[1])):
                raise ValueError("invalid checkpoint training exposure")
            # A case may be practiced repeatedly, but every exposure record
            # represents a distinct training occurrence/cursor position.
            if exposure[0] in seen_occurrence_ids:
                raise ValueError("duplicate checkpoint training exposure occurrence")
            seen_occurrence_ids.add(exposure[0])


@dataclass(frozen=True)
class FullCheckpointReceipt:
    sha256: str
    context: FullCheckpointContext
    trainable_parameters: tuple[str, ...]
    optimizer: str = "AdamW"
    graduation_authorized: bool = False


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _strict_json(data: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FullCheckpointError("duplicate checkpoint JSON field")
            result[key] = value
        return result

    if len(data) > _MAX_JSON_BYTES:
        raise FullCheckpointError("checkpoint manifest exceeds byte limit")
    try:
        result = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise FullCheckpointError("malformed checkpoint manifest JSON") from exc
    if type(result) is not dict or _json_bytes(result) != data:
        raise FullCheckpointError("checkpoint manifest is not canonical JSON")
    return result


def _reject_symlink_components(path: Path) -> None:
    current = path.absolute()
    for part in (current, *current.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise FullCheckpointError("symlink or junction checkpoint path is forbidden")


def _safe_snapshot_dir(path: str | Path, *, existing: bool) -> Path:
    root = Path(path).absolute()
    _reject_symlink_components(root)
    if existing:
        if not root.is_dir():
            raise FullCheckpointError("checkpoint directory missing or non-directory")
        if {entry.name for entry in root.iterdir()} != _FILES:
            raise FullCheckpointError("checkpoint directory contains missing or unexpected files")
        for name in _FILES:
            item = root / name
            if item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode):
                raise FullCheckpointError("checkpoint contains symlink or nonregular file")
    else:
        if root.exists() or root.is_symlink():
            raise FullCheckpointError("checkpoint destination already exists")
        if not root.parent.is_dir():
            raise FullCheckpointError("checkpoint parent directory is missing")
    return root


def _dependencies():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise FullCheckpointError("optional torch+safetensors dependencies are required") from exc
    return torch, load_file, save_file


def _trainable_and_groups(model: Any, optimizer: Any, torch: Any):
    if optimizer.__class__.__name__ != "AdamW":
        raise FullCheckpointError("only AdamW full-state checkpoints are supported")
    named = {name: param for name, param in model.named_parameters()
             if param.requires_grad}
    if not named or len(named) > 10000:
        raise FullCheckpointError("invalid trainable-parameter inventory")
    if any(param.device.type != "cpu" for param in model.parameters()):
        raise FullCheckpointError("CPU-only neural checkpoint; CUDA unsupported")
    by_id = {id(param): name for name, param in named.items()}
    groups: list[list[str]] = []
    seen: set[str] = set()
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = by_id.get(id(parameter))
            if name is None or name in seen:
                raise FullCheckpointError("optimizer references untracked or duplicate parameter")
            names.append(name)
            seen.add(name)
        groups.append(names)
    if seen != set(named):
        raise FullCheckpointError("optimizer omits a trainable parameter")
    return named, groups


def _encode_tree(value: Any, tensors: dict[str, Any], torch: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise FullCheckpointError("checkpoint state exceeds nesting bound")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise FullCheckpointError("checkpoint tensor must reside on CPU")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FullCheckpointError("nonfinite neural checkpoint tensor")
        if len(tensors) >= 20000:
            raise FullCheckpointError("checkpoint exceeds tensor count limit")
        key = f"tensor_{len(tensors):06d}"
        tensors[key] = value.detach().contiguous().clone()
        return {"type": "tensor", "key": key}
    if value is None or type(value) in (str, bool, int):
        if type(value) is str and len(value) > 8192:
            raise FullCheckpointError("checkpoint string exceeds length bound")
        return {"type": "scalar", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise FullCheckpointError("nonfinite optimizer scalar")
        return {"type": "scalar", "value": value}
    if type(value) in (list, tuple):
        return {"type": "tuple" if type(value) is tuple else "list",
                "items": [_encode_tree(item, tensors, torch, depth + 1) for item in value]}
    if type(value) is dict:
        if len(value) > 100000:
            raise FullCheckpointError("checkpoint mapping exceeds entry bound")
        entries = []
        for key, item in value.items():
            if type(key) not in (str, int):
                raise FullCheckpointError("unsupported optimizer mapping key")
            entries.append([_encode_tree(key, tensors, torch, depth + 1),
                            _encode_tree(item, tensors, torch, depth + 1)])
        return {"type": "dict", "items": entries}
    raise FullCheckpointError("unsupported checkpoint state type")


def _decode_tree(value: Any, tensors: dict[str, Any], used: set[str], depth: int = 0) -> Any:
    if depth > 32 or type(value) is not dict:
        raise FullCheckpointError("invalid checkpoint state encoding")
    kind = value.get("type")
    if kind == "tensor" and set(value) == {"type", "key"}:
        key = value["key"]
        if type(key) is not str or key not in tensors or key in used:
            raise FullCheckpointError("missing or duplicate checkpoint tensor")
        used.add(key)
        return tensors[key].clone()
    if kind == "scalar" and set(value) == {"type", "value"}:
        scalar = value["value"]
        if scalar is None or type(scalar) in (str, bool, int):
            return scalar
        if type(scalar) is float and math.isfinite(scalar):
            return scalar
        raise FullCheckpointError("invalid checkpoint scalar")
    if kind in ("list", "tuple") and set(value) == {"type", "items"}:
        items = value["items"]
        if type(items) is not list or len(items) > 100000:
            raise FullCheckpointError("invalid checkpoint sequence")
        result = [_decode_tree(item, tensors, used, depth + 1) for item in items]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict" and set(value) == {"type", "items"}:
        items = value["items"]
        if type(items) is not list or len(items) > 100000:
            raise FullCheckpointError("invalid checkpoint mapping")
        result: dict[Any, Any] = {}
        for item in items:
            if type(item) is not list or len(item) != 2:
                raise FullCheckpointError("invalid checkpoint mapping pair")
            key = _decode_tree(item[0], tensors, used, depth + 1)
            if type(key) not in (str, int) or key in result:
                raise FullCheckpointError("invalid or duplicate checkpoint mapping key")
            result[key] = _decode_tree(item[1], tensors, used, depth + 1)
        return result
    raise FullCheckpointError("unsupported checkpoint state encoding")


def _numpy_rng_snapshot() -> Any:
    try:
        import numpy as np
    except ImportError:
        return None
    name, keys, position, has_gauss, cached = np.random.get_state()
    return [name, keys.astype("uint32").tolist(), position, has_gauss, cached]


def _numpy_rng_tuple(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    try:
        import numpy as np
    except ImportError as exc:
        raise FullCheckpointError("NumPy RNG state cannot be restored without NumPy") from exc
    if (type(snapshot) is not list or len(snapshot) != 5
            or snapshot[0] != "MT19937" or type(snapshot[1]) is not list
            or len(snapshot[1]) != 624):
        raise FullCheckpointError("invalid NumPy RNG snapshot")
    state = (snapshot[0], np.array(snapshot[1], dtype="uint32"),
             snapshot[2], snapshot[3], snapshot[4])
    # Validate against an isolated instance, not the process-global generator.
    try:
        np.random.RandomState().set_state(state)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FullCheckpointError("invalid NumPy RNG state") from exc
    return state


def _context_json(context: FullCheckpointContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "objective_id": context.objective_id,
        "base_checkpoint_sha256": context.base_checkpoint_sha256,
        "prior_checkpoint_sha256": context.prior_checkpoint_sha256,
        "fence": context.fence,
        "cursor": context.cursor,
        "exposures": [list(exposure) for exposure in context.exposures],
    }


def _manifest_digest(manifest_bytes: bytes) -> str:
    return sha256(b"labyrinth.full-neural-checkpoint.v1\0" + manifest_bytes).hexdigest()


def save_full_checkpoint(
    destination: str | Path, *, model: Any, optimizer: Any,
    context: FullCheckpointContext,
) -> FullCheckpointReceipt:
    """Atomically publish one new CPU checkpoint; never overwrite an existing one."""
    if type(context) is not FullCheckpointContext:
        raise ValueError("expected full checkpoint context")
    context.validate()
    root = _safe_snapshot_dir(destination, existing=False)
    torch, _, save_file = _dependencies()
    named, groups = _trainable_and_groups(model, optimizer, torch)
    tensors: dict[str, Any] = {}
    model_tree = _encode_tree({name: p.detach() for name, p in sorted(named.items())},
                              tensors, torch)
    optimizer_tree = _encode_tree(optimizer.state_dict(), tensors, torch)
    rng_tree = _encode_tree({"torch": torch.get_rng_state(),
                             "python": random.getstate()}, tensors, torch)
    numpy_rng = _numpy_rng_snapshot()

    temporary = Path(tempfile.mkdtemp(prefix=".labyrinth-full-checkpoint-",
                                      dir=str(root.parent)))
    try:
        tensor_path = temporary / "state.safetensors"
        save_file(tensors, str(tensor_path))
        if not 0 < tensor_path.stat().st_size <= _MAX_TENSOR_BYTES:
            raise FullCheckpointError("checkpoint tensor bytes exceed bound")
        with tensor_path.open("rb") as stream:
            tensor_digest = sha256(stream.read()).hexdigest()
        manifest = {
            "schema": SCHEMA,
            "context": _context_json(context),
            "optimizer": "AdamW",
            "optimizer_groups": groups,
            "trainable_names": sorted(named),
            "model_tree": model_tree,
            "optimizer_tree": optimizer_tree,
            "rng_tree": rng_tree,
            "numpy_rng": numpy_rng,
            "tensor_sha256": tensor_digest,
        }
        data = _json_bytes(manifest)
        if len(data) > _MAX_JSON_BYTES:
            raise FullCheckpointError("checkpoint manifest exceeds byte limit")
        with (temporary / "manifest.json").open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # No shared writer is allowed to target this path; the caller's journal
        # fence and unique directory reservation are the external race controls.
        if root.exists():
            raise FullCheckpointError("checkpoint destination already exists")
        os.rename(temporary, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return FullCheckpointReceipt(_manifest_digest(data), context,
                                 tuple(sorted(named)))


def load_full_checkpoint(
    source: str | Path, *, model: Any, optimizer: Any,
    expected_sha256: str, expected_context: FullCheckpointContext,
) -> FullCheckpointReceipt:
    """Verify complete byte/lineage custody before restoring trainable state."""
    root = _safe_snapshot_dir(source, existing=True)
    if type(expected_sha256) is not str or not _DIGEST.fullmatch(expected_sha256):
        raise ValueError("expected trusted checkpoint SHA-256")
    if type(expected_context) is not FullCheckpointContext:
        raise ValueError("expected full checkpoint context")
    expected_context.validate()
    manifest_path = root / "manifest.json"
    if not 0 < manifest_path.stat().st_size <= _MAX_JSON_BYTES:
        raise FullCheckpointError("checkpoint manifest byte limit violated")
    data = manifest_path.read_bytes()
    if not _DIGEST.fullmatch(_manifest_digest(data)) or _manifest_digest(data) != expected_sha256:
        raise FullCheckpointError("checkpoint digest differs from trusted identity")
    manifest = _strict_json(data)
    if (set(manifest) != {
        "schema", "context", "optimizer", "optimizer_groups", "trainable_names",
        "model_tree", "optimizer_tree", "rng_tree", "numpy_rng", "tensor_sha256",
    } or manifest["schema"] != SCHEMA
            or manifest["optimizer"] != "AdamW"
            or manifest["context"] != _context_json(expected_context)):
        raise FullCheckpointError("checkpoint context, schema, or optimizer mismatch")
    tensor_path = root / "state.safetensors"
    if not 0 < tensor_path.stat().st_size <= _MAX_TENSOR_BYTES:
        raise FullCheckpointError("checkpoint tensor byte limit violated")
    if sha256(tensor_path.read_bytes()).hexdigest() != manifest["tensor_sha256"]:
        raise FullCheckpointError("checkpoint tensor digest mismatch")

    torch, load_file, _ = _dependencies()
    named, groups = _trainable_and_groups(model, optimizer, torch)
    if manifest["optimizer_groups"] != groups or manifest["trainable_names"] != sorted(named):
        raise FullCheckpointError("checkpoint parameter groups or inventory mismatch")
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except (OSError, ValueError, RuntimeError) as exc:
        raise FullCheckpointError("invalid safetensors checkpoint") from exc
    used: set[str] = set()
    model_state = _decode_tree(manifest["model_tree"], tensors, used)
    optimizer_state = _decode_tree(manifest["optimizer_tree"], tensors, used)
    rng_state = _decode_tree(manifest["rng_tree"], tensors, used)
    if used != set(tensors):
        raise FullCheckpointError("extra unattested tensor in checkpoint")
    if (type(model_state) is not dict or set(model_state) != set(named)
            or type(optimizer_state) is not dict
            or set(optimizer_state) != {"state", "param_groups"}
            or type(rng_state) is not dict or set(rng_state) != {"torch", "python"}):
        raise FullCheckpointError("incomplete neural state")
    for name, param in named.items():
        saved = model_state[name]
        if (not isinstance(saved, torch.Tensor) or saved.shape != param.shape
                or saved.dtype != param.dtype or saved.device.type != "cpu"):
            raise FullCheckpointError("checkpoint trainable tensor incompatible with model")
    if (not isinstance(rng_state["torch"], torch.Tensor)
            or rng_state["torch"].dtype != torch.uint8
            or rng_state["torch"].shape != torch.get_rng_state().shape
            or not isinstance(rng_state["python"], tuple)):
        raise FullCheckpointError("invalid saved CPU RNG state")
    # Validate both non-Torch RNG states before modifying model parameters.
    try:
        random.Random().setstate(rng_state["python"])
    except (TypeError, ValueError) as exc:
        raise FullCheckpointError("invalid Python RNG state") from exc
    numpy_state = _numpy_rng_tuple(manifest["numpy_rng"])
    try:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(model_state[name])
        optimizer.load_state_dict(optimizer_state)
        torch.set_rng_state(rng_state["torch"])
        random.setstate(rng_state["python"])
        if numpy_state is not None:
            import numpy as np
            np.random.set_state(numpy_state)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise FullCheckpointError("checkpoint restoration failed; discard this process") from exc
    return FullCheckpointReceipt(expected_sha256, expected_context,
                                 tuple(sorted(named)))
