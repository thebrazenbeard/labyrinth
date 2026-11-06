"""Fail-closed byte inventory of a *local* pretrained HuggingFace model.

The manifest pins the bytes of selected model weights, configuration and
tokenizer files under one non-symlink directory. Its digest is only meaningful
when a separate trusted custodian pins the digest before subsequent training.
Neither a directory name nor a self-reported model ID attests publisher origin,
pretraining, license, model semantics, or whether a later worker used the model.

This is read-only source tooling: no downloads, deserialization of model weights,
credential access, checkpoint mutation, graduation or deployment.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


SCHEMA = "labyrinth.pretrained-model-byte-manifest.v1"
_DOMAIN = b"labyrinth.pretrained-model-byte-manifest.v1\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_WEIGHT_SHARD = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")
_SUPPORTED_WEIGHT_FOLDERS = frozenset(("", "0_Transformer"))
_OTHER_WEIGHT_EXTENSIONS = (
    ".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt",
    ".h5", ".msgpack", ".onnx", ".ot", ".pdparams",
)
_CONFIG = frozenset((
    "config.json", "generation_config.json", "preprocessor_config.json",
    "sentence_bert_config.json", "config_sentence_transformers.json", "modules.json",
    "model.safetensors.index.json",
))
_TOKENIZER = frozenset((
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
    "spiece.model", "tokenizer.model", "sentencepiece.bpe.model",
))
_TOKENIZER_CORE = frozenset((
    "tokenizer.json", "vocab.txt", "vocab.json", "spiece.model",
    "tokenizer.model", "sentencepiece.bpe.model",
))
_MAX_FILES = 512
_MAX_ENTRIES = 8192
_MAX_FILE_BYTES = 32 * 1024**3
_MAX_TOTAL_BYTES = 128 * 1024**3
_CHUNK_BYTES = 1024 * 1024
_MANIFEST_FIELDS = frozenset((
    "schema", "model_id_claim", "revision_claim", "cache_revision_hint",
    "evidence_class", "origin_attested", "graduation_authorized", "files",
    "file_count", "total_bytes", "manifest_sha256",
))


class ModelProvenanceError(ValueError):
    """Local model bytes or purported custody receipt are unsafe/mismatched."""


def _claimed_id(value: Any, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or len(value) > 260 or not value:
        raise ValueError(f"invalid {field}")
    components = value.split("/")
    if not 1 <= len(components) <= 2 or any(
        not _IDENTIFIER.fullmatch(component) or component in (".", "..")
        for component in components
    ):
        raise ValueError(f"invalid {field}")
    return value


def _is_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _safe_root(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ModelProvenanceError("model root must be a path")
    raw = Path(path)
    if ".." in raw.parts:
        raise ModelProvenanceError("model root traversal is forbidden")
    root = raw.absolute()
    for part in (root, *root.parents):
        if _is_link(part):
            raise ModelProvenanceError("symlink or junction model root is forbidden")
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ModelProvenanceError("missing model root") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ModelProvenanceError("model root must be a directory")
    return root


def _kind(name: str) -> str | None:
    if name == "model.safetensors" or _WEIGHT_SHARD.fullmatch(name):
        return "weights"
    if name in _CONFIG:
        return "config"
    if name in _TOKENIZER:
        return "tokenizer"
    return None


def _unsupported_weight_candidate(name: str) -> bool:
    """Recognize files that a different model loader might use as weights.

    These are deliberately rejected rather than silently omitted from a
    byte-identity claim. This is not an exhaustive detector of arbitrary
    custom formats; loaders must still bind their actual opened artifacts.
    """
    lowered = name.casefold()
    if lowered.endswith(_OTHER_WEIGHT_EXTENSIONS):
        return True
    if lowered.endswith(".bin.index.json") or lowered.endswith(".safetensors.index.json"):
        return True
    return False


def _valid_relative(name: Any) -> bool:
    return (type(name) is str and 0 < len(name) <= 1024
            and not name.startswith("/") and "\\" not in name
            and not any(ord(char) < 32 for char in name)
            and all(part not in ("", ".", "..") for part in name.split("/")))


def _require_complete_weight_sets(entries: list[dict[str, Any]]) -> None:
    """Reject partial or ambiguously duplicated standard safetensors shards."""
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["kind"] != "weights":
            continue
        path = entry["path"]
        folder, _, name = path.rpartition("/")
        group = groups.setdefault(folder, {"single": False, "total": None, "indices": set()})
        if name == "model.safetensors":
            group["single"] = True
            continue
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
        if match is None:
            raise ModelProvenanceError("unsupported safetensors shard name")
        index, count = int(match.group(1)), int(match.group(2))
        if (not 1 <= index <= count <= _MAX_FILES
                or (group["total"] is not None and group["total"] != count)):
            raise ModelProvenanceError("inconsistent safetensors shard set")
        group["total"] = count
        group["indices"].add(index)
    for group in groups.values():
        if group["single"] and group["total"] is not None:
            raise ModelProvenanceError("ambiguous single and sharded safetensors model")
        if group["total"] is not None and group["indices"] != set(range(1, group["total"] + 1)):
            raise ModelProvenanceError("incomplete safetensors shard set")
    if any(folder not in _SUPPORTED_WEIGHT_FOLDERS for folder in groups):
        raise ModelProvenanceError("unsupported model weight folder location")
    if len(groups) > 1:
        raise ModelProvenanceError("ambiguous multiple model weight folders")


def _validate_expected_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    if type(expected) is not dict or set(expected) != _MANIFEST_FIELDS:
        raise ModelProvenanceError("invalid expected model inventory receipt")
    if (expected["schema"] != SCHEMA or expected["evidence_class"] != "LOCAL_BYTES_ONLY"
            or expected["origin_attested"] is not False
            or expected["graduation_authorized"] is not False):
        raise ModelProvenanceError("model receipt overclaims provenance or authority")
    try:
        _claimed_id(expected["model_id_claim"], "model ID")
        _claimed_id(expected["revision_claim"], "revision claim", optional=True)
    except ValueError as exc:
        raise ModelProvenanceError("invalid model origin claim") from exc
    hint = expected["cache_revision_hint"]
    if (hint is not None
            and (type(hint) is not str
                 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", hint))):
        raise ModelProvenanceError("invalid cache revision hint")
    records = expected["files"]
    count = expected["file_count"]
    total = expected["total_bytes"]
    if (type(records) is not list or not 1 <= len(records) <= _MAX_FILES
            or type(count) is not int or count != len(records)
            or type(total) is not int or not 0 < total <= _MAX_TOTAL_BYTES):
        raise ModelProvenanceError("invalid expected model file collection")
    paths: list[str] = []
    computed_total = 0
    for item in records:
        if type(item) is not dict or set(item) != {"path", "kind", "size", "sha256"}:
            raise ModelProvenanceError("invalid expected model file record")
        name = item["path"]
        if (not _valid_relative(name) or item["kind"] != _kind(name.rsplit("/", 1)[-1])
                or type(item["size"]) is not int
                or not 0 < item["size"] <= _MAX_FILE_BYTES
                or type(item["sha256"]) is not str
                or not _DIGEST.fullmatch(item["sha256"])):
            raise ModelProvenanceError("unsafe expected model file record")
        paths.append(name)
        computed_total += item["size"]
    if paths != sorted(set(paths)) or computed_total != total:
        raise ModelProvenanceError("invalid expected model inventory ordering or total")
    _require_complete_weight_sets(records)
    digest = expected["manifest_sha256"]
    if type(digest) is not str or not _DIGEST.fullmatch(digest):
        raise ModelProvenanceError("invalid model manifest SHA-256")
    return {name: value for name, value in expected.items() if name != "manifest_sha256"}


def _hash_regular_file(path: Path) -> tuple[str, int]:
    if _is_link(path):
        raise ModelProvenanceError("symlink or junction model file is forbidden")
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_FILE_BYTES:
            raise ModelProvenanceError("model file is not bounded regular data")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino, opened.st_size)
                    != (before.st_dev, before.st_ino, before.st_size)):
                raise ModelProvenanceError("model file changed during opening")
            digest = sha256()
            count = 0
            for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                count += len(block)
                if count > _MAX_FILE_BYTES:
                    raise ModelProvenanceError("model file exceeds byte limit")
                digest.update(block)
        after = path.lstat()
    except OSError as exc:
        raise ModelProvenanceError("unreadable or changed model file") from exc
    comparable = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )
    if _is_link(path) or comparable(before) != comparable(after) or count != before.st_size:
        raise ModelProvenanceError("model file changed during byte inventory")
    return digest.hexdigest(), count


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(payload: dict[str, Any]) -> str:
    return sha256(_DOMAIN + _canonical_bytes(payload)).hexdigest()


def _cache_revision_hint(root: Path) -> str | None:
    name = root.name
    if (root.parent.name == "snapshots"
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", name)):
        return name
    return None


def capture_pretrained_model_manifest(
    root: str | Path, *, model_id_claim: str,
    revision_claim: str | None = None,
) -> dict[str, Any]:
    """Hash safe local model files into a deterministic, non-authoritative receipt.

    Supported base weights must be at the model root (or the canonical
    ``0_Transformer`` subtree), named ``model.safetensors`` or standard
    ``model-00001-of-00002.safetensors`` shards. Other recognizable loadable
    weight formats and additional weight locations fail closed rather than
    being ignored. A config and tokenizer vocabulary are required. Other
    unselected descriptive files are not authenticated.
    Standard symlink-based HF snapshots must first be materialized safely into
    a non-symlink immutable directory; this method deliberately fails closed.
    """
    model_id = _claimed_id(model_id_claim, "model ID")
    revision = _claimed_id(revision_claim, "revision claim", optional=True)
    safe = _safe_root(root)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    visited_entries = 0

    def onerror(error: OSError) -> None:
        raise ModelProvenanceError("unreadable model directory") from error

    for dirname, dirs, files in os.walk(safe, topdown=True, followlinks=False,
                                        onerror=onerror):
        dirs.sort()
        files.sort()
        visited_entries += len(dirs) + len(files)
        if visited_entries > _MAX_ENTRIES:
            raise ModelProvenanceError("model directory exceeds entry limit")
        parent = Path(dirname)
        for label in (*dirs, *files):
            if _is_link(parent / label):
                raise ModelProvenanceError("symlink or junction in model directory")
        for label in files:
            kind = _kind(label)
            if kind is None:
                if _unsupported_weight_candidate(label):
                    raise ModelProvenanceError(
                        f"unsupported or unselected model weight candidate: {label}"
                    )
                continue
            target = parent / label
            relative = target.relative_to(safe).as_posix()
            if relative in seen or not _valid_relative(relative):
                raise ModelProvenanceError("invalid model-relative file path")
            seen.add(relative)
            digest, size = _hash_regular_file(target)
            total += size
            if len(entries) >= _MAX_FILES or total > _MAX_TOTAL_BYTES:
                raise ModelProvenanceError("model inventory exceeds byte or file limit")
            entries.append({"path": relative, "kind": kind, "size": size, "sha256": digest})

    entries.sort(key=lambda item: item["path"])
    _require_complete_weight_sets(entries)
    names = {item["path"] for item in entries}
    if not any(item["kind"] == "weights" for item in entries):
        raise ModelProvenanceError("missing safetensors model weights")
    if "config.json" not in names and "0_Transformer/config.json" not in names:
        raise ModelProvenanceError("missing model config.json")
    if not any(item["path"].rsplit("/", 1)[-1] in _TOKENIZER_CORE
               for item in entries):
        raise ModelProvenanceError("missing tokenizer vocabulary")
    payload = {
        "schema": SCHEMA,
        "model_id_claim": model_id,
        "revision_claim": revision,
        "cache_revision_hint": _cache_revision_hint(safe),
        "evidence_class": "LOCAL_BYTES_ONLY",
        "origin_attested": False,
        "graduation_authorized": False,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": total,
    }
    return {**payload, "manifest_sha256": _fingerprint(payload)}


def verify_pretrained_model_manifest(
    root: str | Path, expected: Mapping[str, Any], *,
    trusted_manifest_sha256: str,
) -> dict[str, Any]:
    """Rehash files against a *separately trusted* prior manifest digest.

    A self-recomputed manifest hash has no custody value. The caller must
    protect ``trusted_manifest_sha256`` outside the model/training directory
    and independently establish when it was committed.
    """
    if type(trusted_manifest_sha256) is not str or not _DIGEST.fullmatch(trusted_manifest_sha256):
        raise ModelProvenanceError("missing trusted manifest SHA-256")
    payload = _validate_expected_manifest(expected)
    digest = expected["manifest_sha256"]
    if not hmac.compare_digest(_fingerprint(payload), digest):
        raise ModelProvenanceError("model manifest self-hash mismatch")
    if not hmac.compare_digest(digest, trusted_manifest_sha256):
        raise ModelProvenanceError("model manifest differs from trusted commitment")
    observed = capture_pretrained_model_manifest(
        root, model_id_claim=expected.get("model_id_claim"),
        revision_claim=expected.get("revision_claim"),
    )
    if observed != expected:
        raise ModelProvenanceError("model bytes or inventory differ from trusted manifest")
    return observed
