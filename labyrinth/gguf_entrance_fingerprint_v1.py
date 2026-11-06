"""Read-only GGUF base/LoRA inventory: byte identity, not HF trainability.

Streaming whole-file SHA-256 and bounded GGUF v3 LITTLE-ENDIAN header
inspection, using only Python stdlib. No model load, conversion or mutation.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any


SCHEMA = "labyrinth.gguf-entrance-fingerprint.v1"
_DOMAIN = b"labyrinth.gguf-entrance-fingerprint.v1\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_META_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\Z", re.ASCII)
_MAX_SIZE = 64 * 1024**3
_MAX_HEADER = 64 * 1024**2
_MAX_KVS = 8192
_MAX_TENSORS = 131072
_MAX_ARRAY_ELEMENTS = 1_000_000
_KEYS = frozenset((
    "general.architecture", "general.name", "general.type",
    "general.file_type", "general.quantization_version",
    "general.alignment", "adapter.type", "adapter.lora.alpha", "adapter.alpha",
))
_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
         10: 8, 11: 8, 12: 8}
_STRUCT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}


class GGUFEntranceError(ValueError):
    """GGUF bytes, metadata or externally pinned identity are unqualified."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash(value: Any) -> str:
    return sha256(_DOMAIN + _canonical(value)).hexdigest()


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _safe_file(path: Path) -> os.stat_result:
    if (not isinstance(path, Path) or not path.is_absolute()
            or path.suffix.casefold() != ".gguf" or ".." in path.parts):
        raise GGUFEntranceError("canonical absolute .gguf file path required")
    for part in (path, *path.parents):
        if _linked(part):
            raise GGUFEntranceError("GGUF file or ancestor is a symlink/junction")
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or not 32 <= info.st_size <= _MAX_SIZE
                or path != path.resolve(strict=True)):
            raise GGUFEntranceError("GGUF must be bounded canonical regular file")
    except OSError as exc:
        raise GGUFEntranceError("GGUF model file unavailable") from exc
    return info


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.offset = 0

    def take(self, count: int) -> bytes:
        if type(count) is not int or count < 0 or self.offset + count > _MAX_HEADER:
            raise GGUFEntranceError("GGUF metadata/header exceeds bounded parser budget")
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFEntranceError("truncated GGUF metadata or tensor header")
        self.offset += count
        return data

    def skip(self, count: int) -> None:
        while count:
            consumed = min(count, 1024 * 1024)
            self.take(consumed)
            count -= consumed

    def scalar(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self, *, maximum: int = 4 * 1024 * 1024) -> str:
        length = self.scalar("Q")
        if length > maximum:
            raise GGUFEntranceError("GGUF metadata string exceeds bounded length")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeError as exc:
            raise GGUFEntranceError("GGUF metadata string is not valid UTF-8") from exc

    def value(self, kind: int, *, depth: int = 0, collect: bool = False) -> Any:
        if kind == 8:
            return self.string() if collect else self._skip_string()
        if kind == 9:
            if depth >= 2:
                raise GGUFEntranceError("GGUF metadata arrays exceed nesting limit")
            subtype = self.scalar("I")
            length = self.scalar("Q")
            if length > _MAX_ARRAY_ELEMENTS or subtype not in {*_SIZE, 8, 9}:
                raise GGUFEntranceError("invalid GGUF array subtype or element count")
            if subtype in _SIZE:
                self.skip(length * _SIZE[subtype])
            else:
                for _ in range(length):
                    self.value(subtype, depth=depth + 1, collect=False)
            return {"array_type": subtype, "length": length} if collect else None
        if kind not in _STRUCT:
            raise GGUFEntranceError("unsupported GGUF metadata scalar type")
        if collect:
            return self.scalar(_STRUCT[kind])
        self.skip(_SIZE[kind])
        return None

    def _skip_string(self):
        length = self.scalar("Q")
        if length > 4 * 1024 * 1024:
            raise GGUFEntranceError("GGUF string element exceeds budget")
        self.skip(length)
        return None


def _header(path: Path, total_bytes: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        reader = _Reader(stream)
        if reader.take(4) != b"GGUF":
            raise GGUFEntranceError("not a GGUF little-endian magic header")
        version = reader.scalar("I")
        tensors = reader.scalar("Q")
        kv_count = reader.scalar("Q")
        if version != 3 or not 1 <= tensors <= _MAX_TENSORS or kv_count > _MAX_KVS:
            raise GGUFEntranceError("only bounded GGUF v3 with physical tensors is inspected")
        observed_keys: set[str] = set()
        summary: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string(maximum=65535)
            if not key or len(key) > 65535 or _META_KEY.fullmatch(key) is None:
                raise GGUFEntranceError("invalid GGUF metadata key")
            if key in observed_keys:
                raise GGUFEntranceError("duplicate GGUF metadata key")
            observed_keys.add(key)
            kind = reader.scalar("I")
            value = reader.value(kind, collect=key in _KEYS)
            if key in _KEYS:
                if type(value) is str and len(value) > 512:
                    value = "LONG_VALUE_NOT_RECORDED"
                summary[key] = {"metadata_type": kind, "value": value}
        names: set[str] = set()
        types: Counter[int] = Counter()
        offsets: set[int] = set()
        for _ in range(tensors):
            name = reader.string(maximum=64)
            dimensions = reader.scalar("I")
            if (not name or name in names or not 1 <= dimensions <= 4
                    or any(ord(ch) < 32 for ch in name)):
                raise GGUFEntranceError("invalid GGUF tensor index")
            names.add(name)
            for _ in range(dimensions):
                if reader.scalar("Q") == 0:
                    raise GGUFEntranceError("zero GGUF tensor dimension")
            ggml_type = reader.scalar("I")
            offset = reader.scalar("Q")
            if ggml_type > 1024 or offset in offsets:
                raise GGUFEntranceError("invalid or overlapping GGUF tensor offsets")
            offsets.add(offset)
            types[ggml_type] += 1
        alignment = summary.get("general.alignment", {}).get("value", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096:
            raise GGUFEntranceError("invalid GGUF tensor data alignment")
        payload_offset = reader.offset + ((-reader.offset) % alignment)
        if (payload_offset >= total_bytes or max(offsets) >= total_bytes - payload_offset
                or any(offset % alignment for offset in offsets)):
            raise GGUFEntranceError("GGUF tensor index points beyond aligned binary payload")
        header_bytes = reader.offset
    # Header digest includes the tensor index but not padding or tensor data.
    h = sha256()
    with path.open("rb") as stream:
        remaining = header_bytes
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise GGUFEntranceError("GGUF header bytes changed while rereading")
            h.update(chunk)
            remaining -= len(chunk)
    return {"format_version": version, "tensor_count": tensors,
            "metadata_count": kv_count, "tensor_data_offset": payload_offset,
            "header_sha256": h.hexdigest(),
            "tensor_ggml_types": {str(k): v for k, v in sorted(types.items())},
            "metadata": summary}


def fingerprint_gguf_file(path: Path) -> dict[str, Any]:
    """Stream and hash every byte; inspect at most 64MiB of metadata/index."""
    before = _safe_file(path)
    parsed = _header(path, before.st_size)
    digest = sha256()
    count = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                             | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as input_stream:
            opened = os.fstat(input_stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev, before.st_ino, before.st_size
            ):
                raise GGUFEntranceError("GGUF identity changed during file opening")
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                count += len(chunk)
                if count > _MAX_SIZE:
                    raise GGUFEntranceError("GGUF streamed bytes exceed limit")
                digest.update(chunk)
        after = _safe_file(path)
    except OSError as exc:
        raise GGUFEntranceError("GGUF bytes unavailable during hash") from exc
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or count != before.st_size:
        raise GGUFEntranceError("GGUF changed while streaming full-file SHA-256")
    return {"path": str(path), "bytes": count,
            "sha256": digest.hexdigest(), **parsed}


def capture_gguf_entrance(
    *, base_path: Path, lora_adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Capture whole GGUF contents, never infer a trainable HF genotype."""
    if lora_adapter_path is not None and base_path == lora_adapter_path:
        raise GGUFEntranceError("GGUF base and LoRA cannot be the same physical file")
    base = fingerprint_gguf_file(base_path)
    if (base["metadata"].get("general.type", {}).get("value") == "adapter"
            or base["metadata"].get("adapter.type", {}).get("value") == "lora"):
        raise GGUFEntranceError("GGUF base role points at declared LoRA adapter")
    adapter = fingerprint_gguf_file(lora_adapter_path) if lora_adapter_path else None
    if adapter is None:
        pairing = "NO_ADAPTER"
    else:
        base_arch = base["metadata"].get("general.architecture", {}).get("value")
        adapter_arch = adapter["metadata"].get("general.architecture", {}).get("value")
        adapter_kind = adapter["metadata"].get("adapter.type", {}).get("value")
        general_kind = adapter["metadata"].get("general.type", {}).get("value")
        if adapter_kind != "lora" or general_kind != "adapter":
            pairing = "UNVERIFIED_ADAPTER_FORMAT"
        elif base_arch is not None and adapter_arch is not None and base_arch != adapter_arch:
            pairing = "DECLARED_ARCHITECTURE_MISMATCH"
        elif base_arch is not None and base_arch == adapter_arch:
            pairing = "DECLARED_ARCHITECTURE_MATCH_ONLY"
        else:
            pairing = "UNKNOWN_ARCHITECTURE_PAIRING"
    data = {
        "schema": SCHEMA, "evidence": "LOCAL_GGUF_BYTES_AND_METADATA_ONLY",
        "base": base, "lora_adapter": adapter,
        "pairing_status": pairing,
        "gguf_model_loaded": False, "hf_trainable_genotype_proven": False,
        "adapter_applied_proven": False, "model_identity_proven": False,
        "origin_attested": False, "neural_improvement_proven": False,
        "graduation_authorized": False,
    }
    return {**data, "manifest_sha256": _hash(data)}


def verify_gguf_entrance(
    expected: dict[str, Any], *, trusted_manifest_sha256: str,
) -> dict[str, Any]:
    """Compare physical bytes against an externally pinned expected digest."""
    if (type(expected) is not dict or type(trusted_manifest_sha256) is not str
            or _SHA.fullmatch(trusted_manifest_sha256) is None
            or set(expected) != {
                "schema", "evidence", "base", "lora_adapter", "pairing_status",
                "gguf_model_loaded", "hf_trainable_genotype_proven",
                "adapter_applied_proven", "model_identity_proven", "origin_attested",
                "neural_improvement_proven", "graduation_authorized", "manifest_sha256",
            }
            or expected["schema"] != SCHEMA
            or expected["evidence"] != "LOCAL_GGUF_BYTES_AND_METADATA_ONLY"
            or any(expected[key] is not False for key in (
                "gguf_model_loaded", "hf_trainable_genotype_proven",
                "adapter_applied_proven", "model_identity_proven", "origin_attested",
                "neural_improvement_proven", "graduation_authorized",
            ))):
        raise GGUFEntranceError("expected GGUF receipt has invalid or inflated authority")
    previous = {k: v for k, v in expected.items() if k != "manifest_sha256"}
    if (type(expected["manifest_sha256"]) is not str
            or _SHA.fullmatch(expected["manifest_sha256"]) is None
            or not hmac.compare_digest(_hash(previous), expected["manifest_sha256"])
            or not hmac.compare_digest(expected["manifest_sha256"], trusted_manifest_sha256)):
        raise GGUFEntranceError("GGUF receipt differs from separately trusted commitment")
    # Exact paths are claims, not provenance. Reopen and rehash every byte.
    try:
        fresh = capture_gguf_entrance(
            base_path=Path(expected["base"]["path"]),
            lora_adapter_path=(Path(expected["lora_adapter"]["path"])
                               if expected["lora_adapter"] is not None else None),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GGUFEntranceError("invalid path or changed GGUF physical bytes") from exc
    if fresh != expected:
        raise GGUFEntranceError("GGUF physical pair differs from pinned entrance receipt")
    return fresh


__all__ = [
    "GGUFEntranceError", "fingerprint_gguf_file",
    "capture_gguf_entrance", "verify_gguf_entrance",
]
