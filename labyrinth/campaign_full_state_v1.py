"""Source-only physical genome capture for experimental Labyrinth campaigns.

This verifies bytes, not genuine neural learning, trainer restore, or graduation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from .entrance_state_v1 import _file_sha, _inventory, _source_dir


SCHEMA = "labyrinth.campaign-full-state.v1"
_DOMAIN = b"labyrinth.campaign-full-state.v1\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROLES = ("base", "adapter", "optimizer", "scheduler", "rng")
_FAMILIES = frozenset({
    ("hf_safetensors_lora", "lora_adapter"),
    ("hf_safetensors_full", "full_weight_edit"),
})
_MAX_JSON_BYTES = 256 * 1024
_AUX_BUDGETS = {"optimizer": 64 * 1024**3,
                "scheduler": 8 * 1024**3, "rng": 8 * 1024**3}
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1,
                "F8_E5M2": 1, "F8_E4M3FN": 1, "F8_E5M2FNUZ": 1,
                "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4,
                "I64": 8, "U64": 8, "F64": 8}


def _canonical(value: Any) -> bytes:
    result = json.dumps(value, ensure_ascii=True, sort_keys=True,
                        allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(result) > _MAX_JSON_BYTES:
        raise ValueError("campaign manifest exceeds size limit")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(_DOMAIN + _canonical(value)).hexdigest()


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate campaign JSON key")
        result[key] = value
    return result


def _cursor(path: Path) -> dict[str, Any]:
    if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
            or not path.is_file() or path != path.resolve(strict=True)):
        raise ValueError("data cursor must be a canonical absolute local file")
    digest, size = _file_sha(path)
    if size > 1024 * 1024:
        raise ValueError("data cursor is too large")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _safetensors_names(path: Path) -> set[str]:
    """Parse the standard bounded tensor header, not just a filename suffix."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors file lacks a tensor header")
        header_size = int.from_bytes(prefix, "little")
        size = path.stat().st_size
        if not 2 <= header_size <= 16 * 1024 * 1024 or size <= 8 + header_size:
            raise ValueError("safetensors header size or tensor bytes invalid")
        header = json.loads(stream.read(header_size), object_pairs_hook=_unique_json)
    if type(header) is not dict or not header:
        raise ValueError("safetensors header is not a tensor mapping")
    if "__metadata__" in header:
        metadata = header.pop("__metadata__")
        if (type(metadata) is not dict
                or any(type(key) is not str or type(value) is not str
                       for key, value in metadata.items())):
            raise ValueError("safetensors metadata invalid")
    if not header or len(header) > 100000:
        raise ValueError("safetensors requires bounded tensor entries")
    ranges: list[tuple[int, int]] = []
    for name, entry in header.items():
        if (type(name) is not str or not name or type(entry) is not dict
                or set(entry) != {"dtype", "shape", "data_offsets"}
                or type(entry["dtype"]) is not str
                or entry["dtype"] not in _DTYPE_BYTES
                or type(entry["shape"]) is not list or not 1 <= len(entry["shape"]) <= 16
                or any(type(dim) is not int or not 1 <= dim <= 2**40
                       for dim in entry["shape"])
                or type(entry["data_offsets"]) is not list
                or len(entry["data_offsets"]) != 2
                or any(type(offset) is not int or offset < 0
                       for offset in entry["data_offsets"])):
            raise ValueError("safetensors tensor entry is malformed")
        start, end = entry["data_offsets"]
        count = 1
        for dimension in entry["shape"]:
            count *= dimension
        if (start >= end or end - start != count * _DTYPE_BYTES[entry["dtype"]]):
            raise ValueError("safetensors data offset or size mismatch")
        ranges.append((start, end))
    current = 0
    for start, end in sorted(ranges):
        if start != current:
            raise ValueError("safetensors gaps or overlapping tensor data")
        current = end
    if current != size - 8 - header_size:
        raise ValueError("safetensors uncommitted trailing tensor bytes")
    return set(header)


def _check_base(base: dict[str, Any]) -> None:
    """Verify complete local HF safetensors shards and pinned metadata."""
    files = base["files"]
    shards = {name for name in files if name.endswith(".safetensors")}
    if (not shards or any(name.endswith(".gguf") for name in files)
            or "config.json" not in files
            or not any(name in files for name in (
                "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt"))):
        raise ValueError("HF safetensors requires actual weights, config and tokenizer")
    tensors = {
        name: _safetensors_names(Path(base["root"]) / name)
        for name in shards
    }
    index_name = "model.safetensors.index.json"
    if index_name not in files:
        if len(shards) != 1:
            raise ValueError("multi-shard HF weights require an index")
        return
    index = json.loads((Path(base["root"]) / index_name).read_bytes(),
                       object_pairs_hook=_unique_json)
    if (type(index) is not dict or type(index.get("weight_map")) is not dict
            or not index["weight_map"]
            or any(type(name) is not str or not name
                   or type(shard) is not str or shard not in shards
                   for name, shard in index["weight_map"].items())
            or set(index["weight_map"].values()) != shards):
        raise ValueError("HF index references missing or unindexed shards")
    if (sum(len(names) for names in tensors.values())
            != len(set().union(*tensors.values()))
            or set(index["weight_map"]) != set().union(*tensors.values())
            or any(tensor not in tensors[shard]
                   for tensor, shard in index["weight_map"].items())):
        raise ValueError("HF safetensors index does not map physical tensor names")


def _training_inventory(directory: Path, role: str) -> dict[str, Any]:
    """Reuse entrance's streaming digests without its 512 MiB pilot-only cap."""
    directory = _source_dir(directory, role)
    entries = sorted(directory.rglob("*"))
    if (not entries or len(entries) > 128
            or any(item.is_symlink() or not (item.is_dir() or item.is_file())
                   for item in entries)):
        raise ValueError(f"{role} state requires bounded regular non-symlink files")
    files = [item for item in entries if item.is_file()]
    if not files or sum(item.stat().st_size for item in files) > _AUX_BUDGETS[role]:
        raise ValueError(f"{role} state exceeds explicit full-campaign byte budget")
    records = {}
    for item in files:
        digest, size = _file_sha(item)
        records[item.relative_to(directory).as_posix()] = {"sha256": digest, "bytes": size}
    return {"root": str(directory), "files": records}


def _parts(*, base_dir: Path, adapter_dir: Path | None, optimizer_dir: Path,
           scheduler_dir: Path, rng_dir: Path, data_cursor_path: Path,
           family: str, mutation_kind: str) -> dict[str, Any]:
    if (type(family) is not str or type(mutation_kind) is not str
            or (family, mutation_kind) not in _FAMILIES):
        raise ValueError("unsupported campaign family or mutation kind")
    if (mutation_kind == "lora_adapter") != (adapter_dir is not None):
        raise ValueError("LoRA requires an adapter; full-weight-edit must omit it")
    for label, source in (("base", base_dir), ("optimizer", optimizer_dir),
                          ("scheduler", scheduler_dir), ("rng", rng_dir)):
        if not isinstance(source, Path):
            raise ValueError(f"required campaign {label} directory missing")
    value = {
        "family": family, "mutation_kind": mutation_kind,
        "base": _inventory(base_dir, kind="base"),
        "adapter": _inventory(adapter_dir, kind="adapter") if adapter_dir else None,
        "optimizer": _training_inventory(optimizer_dir, "optimizer"),
        "scheduler": _training_inventory(scheduler_dir, "scheduler"),
        "rng": _training_inventory(rng_dir, "rng"),
        "data_cursor": _cursor(data_cursor_path),
    }
    if sum(len(value[role]["files"]) if value[role] else 0 for role in _ROLES) > 256:
        raise ValueError("campaign has too many constituent files")
    _check_base(value["base"])
    if value["adapter"] is not None:
        for name in value["adapter"]["files"]:
            if name.endswith(".safetensors"):
                _safetensors_names(Path(value["adapter"]["root"]) / name)
    return value


def _content(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": manifest["family"], "mutation_kind": manifest["mutation_kind"],
        "files": {role: manifest[role]["files"] if manifest[role] else None
                  for role in _ROLES},
        "data_cursor": {key: manifest["data_cursor"][key]
                        for key in ("sha256", "bytes")},
    }


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return _digest({key: item for key, item in manifest.items()
                    if key != "manifest_sha256"})


def _verify_memory(manifest: dict[str, Any]) -> None:
    if (type(manifest) is not dict
            or set(manifest) != {"schema", "evidence", "family", "mutation_kind",
                                "base", "adapter", "optimizer", "scheduler", "rng",
                                "data_cursor", "generation_index", "parent_state_sha256",
                                "origin_state_sha256", "state_sha256", "manifest_sha256",
                                "restorable_training_state_proven", "neural_update_proven",
                                "graduation_authorized"}
            or manifest["schema"] != SCHEMA
            or manifest["evidence"] != "SOURCE_ONLY_LOCAL_FILE_CONTENT_INTEGRITY"
            or any(manifest[name] is not False for name in (
                "restorable_training_state_proven", "neural_update_proven",
                "graduation_authorized"))
            or type(manifest["generation_index"]) is not int
            or manifest["generation_index"] < 0
            or (manifest["generation_index"] == 0)
               != (manifest["parent_state_sha256"] is None)
            or any(type(manifest[name]) is not str
                   or _SHA.fullmatch(manifest[name]) is None
                   for name in ("state_sha256", "origin_state_sha256"))
            or (manifest["parent_state_sha256"] is not None
                and (type(manifest["parent_state_sha256"]) is not str
                     or _SHA.fullmatch(manifest["parent_state_sha256"]) is None))
            or manifest["manifest_sha256"] != _manifest_hash(manifest)):
        raise ValueError("invalid campaign manifest schema, claims or ancestry")
    live = _parts(
        base_dir=Path(manifest["base"]["root"]),
        adapter_dir=Path(manifest["adapter"]["root"]) if manifest["adapter"] else None,
        optimizer_dir=Path(manifest["optimizer"]["root"]),
        scheduler_dir=Path(manifest["scheduler"]["root"]),
        rng_dir=Path(manifest["rng"]["root"]),
        data_cursor_path=Path(manifest["data_cursor"]["path"]),
        family=manifest["family"], mutation_kind=manifest["mutation_kind"],
    )
    if any(live[name] != manifest[name] for name in (*_ROLES, "data_cursor")):
        raise ValueError("campaign physical files differ from pinned full genome")
    if manifest["state_sha256"] != _digest(_content(manifest)):
        raise ValueError("campaign content digest mismatch")
    if (manifest["generation_index"] == 0
            and manifest["origin_state_sha256"] != manifest["state_sha256"]):
        raise ValueError("founder state does not bind its own origin")


def capture_campaign_state(
    *, base_dir: Path, adapter_dir: Path | None, optimizer_dir: Path,
    scheduler_dir: Path, rng_dir: Path, data_cursor_path: Path,
    family: str, mutation_kind: str,
    parent_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a physical genome, not an attested update or restored trainer."""
    if parent_manifest is not None:
        _verify_memory(parent_manifest)
    parts = _parts(
        base_dir=base_dir, adapter_dir=adapter_dir, optimizer_dir=optimizer_dir,
        scheduler_dir=scheduler_dir, rng_dir=rng_dir, data_cursor_path=data_cursor_path,
        family=family, mutation_kind=mutation_kind,
    )
    manifest = {
        "schema": SCHEMA, "evidence": "SOURCE_ONLY_LOCAL_FILE_CONTENT_INTEGRITY",
        **parts,
        "generation_index": (parent_manifest["generation_index"] + 1
                             if parent_manifest else 0),
        "parent_state_sha256": (parent_manifest["state_sha256"]
                                if parent_manifest else None),
        "origin_state_sha256": (parent_manifest["origin_state_sha256"]
                                if parent_manifest else None),
        "restorable_training_state_proven": False,
        "neural_update_proven": False, "graduation_authorized": False,
    }
    manifest["state_sha256"] = _digest(_content(manifest))
    if parent_manifest is None:
        manifest["origin_state_sha256"] = manifest["state_sha256"]
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return manifest


def save_campaign_state_once(manifest: dict[str, Any], destination: Path) -> str:
    """Atomically publish a write-once manifest; never copy weight bytes."""
    _verify_memory(manifest)
    if (not isinstance(destination, Path) or not destination.is_absolute()
            or destination.is_symlink() or destination.exists()
            or destination.parent.is_symlink() or not destination.parent.is_dir()
            or destination.parent != destination.parent.resolve(strict=True)):
        raise ValueError("campaign manifest needs new canonical absolute destination")
    content = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".campaign-staged-", dir=destination.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.link(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return manifest["manifest_sha256"]


def _verify_local(path: Path) -> dict[str, Any]:
    if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
            or not path.is_file() or path != path.resolve(strict=True)
            or not 0 < path.stat().st_size <= _MAX_JSON_BYTES):
        raise ValueError("campaign manifest requires canonical bounded local file")
    manifest = json.loads(path.read_bytes(), object_pairs_hook=_unique_json)
    _verify_memory(manifest)
    return {
        "schema": SCHEMA, "status": "LOCAL_FILE_READBACK_ONLY",
        "state_sha256": manifest["state_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "pid": os.getpid(),
        "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "restorable_training_state_proven": False,
        "neural_update_proven": False, "graduation_authorized": False,
    }


def verify_campaign_state(path: Path, *, fresh_process: bool = True) -> dict[str, Any]:
    """Rehash all claimed constituent bytes; optionally fork new Python PID."""
    if type(fresh_process) is not bool:
        raise ValueError("fresh_process must be boolean")
    before = _verify_local(path)
    if not fresh_process:
        return before
    process = subprocess.run(
        [sys.executable, "-m", "labyrinth.campaign_full_state_v1", "--verify", str(path)],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
        timeout=180, check=False,
    )
    if process.returncode or len(process.stdout) > 4096:
        raise ValueError("fresh campaign verifier refused physical readback")
    result = json.loads(process.stdout, object_pairs_hook=_unique_json)
    after = _verify_local(path)
    if (type(result) is not dict or set(result) != set(before)
            or type(result.get("pid")) is not int or result["pid"] == os.getpid()
            or result != {**before, "pid": result["pid"]}
            or result != {**after, "pid": result["pid"]}):
        raise ValueError("fresh campaign readback disagrees with original bytes")
    return result


def verify_generation_ancestry(parent_path, sibling_paths, champion_path, next_entry_path):
    """Verify claimed branch lineage without inventing training causality.

    Paths are write-once manifests. All actual referenced files are rehashed
    by a new Python interpreter per manifest, then by this interpreter again.
    This cannot prove that a sibling *started* from its claimed parent; that
    requires the trainer's separately journaled pre-update checkpoint receipt.
    """
    if (type(sibling_paths) is not tuple or not 2 <= len(sibling_paths) <= 8
            or any(not isinstance(path, Path) for path in
                   (parent_path, champion_path, next_entry_path, *sibling_paths))
            or len(set(sibling_paths)) != len(sibling_paths)
            or champion_path not in sibling_paths):
        raise ValueError("ancestry needs distinct siblings and one selected champion")

    def read(path: Path) -> dict[str, Any]:
        verify_campaign_state(path, fresh_process=True)
        return json.loads(path.read_bytes(), object_pairs_hook=_unique_json)

    parent = read(parent_path)
    siblings = tuple(read(path) for path in sibling_paths)
    champion = siblings[sibling_paths.index(champion_path)]
    next_entry = read(next_entry_path)
    content_hashes = [child["state_sha256"] for child in siblings]
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError("two siblings have identical full-state contents")
    if any(
        child["family"] != parent["family"]
        or child["mutation_kind"] != parent["mutation_kind"]
        or child["generation_index"] != parent["generation_index"] + 1
        or child["parent_state_sha256"] != parent["state_sha256"]
        or child["origin_state_sha256"] != parent["origin_state_sha256"]
        or child["state_sha256"] == parent["state_sha256"]
        for child in siblings
    ):
        raise ValueError("sibling physical ancestry or mutation family disagrees")
    if parent["mutation_kind"] == "lora_adapter":
        if any(child["base"]["files"] != parent["base"]["files"]
               or child["adapter"]["files"] == parent["adapter"]["files"]
               for child in siblings):
            raise ValueError("LoRA siblings must retain exact base and change adapter bytes")
    else:
        def metadata(manifest: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in manifest["base"]["files"].items()
                    if not key.endswith(".safetensors")
                    and key != "model.safetensors.index.json"}

        if any(metadata(child) != metadata(parent)
               or {key: value for key, value in child["base"]["files"].items()
                   if key.endswith(".safetensors")}
               == {key: value for key, value in parent["base"]["files"].items()
                   if key.endswith(".safetensors")}
               for child in siblings):
            raise ValueError("full-weight siblings must preserve tokenizer/config and alter weights")
    if (next_entry["state_sha256"] != champion["state_sha256"]
            or next_entry["generation_index"] != champion["generation_index"] + 1
            or next_entry["parent_state_sha256"] != champion["state_sha256"]
            or next_entry["origin_state_sha256"] != parent["origin_state_sha256"]
            or next_entry["family"] != champion["family"]
            or next_entry["mutation_kind"] != champion["mutation_kind"]):
        raise ValueError("next generation entry is not exact selected champion state")
    return {
        "status": "SOURCE_VERIFIED_LINEAGE_NOT_CAUSAL_TRAINING",
        "parent_state_sha256": parent["state_sha256"],
        "origin_state_sha256": parent["origin_state_sha256"],
        "sibling_state_sha256s": tuple(content_hashes),
        "champion_state_sha256": champion["state_sha256"],
        "next_entry_state_sha256": next_entry["state_sha256"],
        "neural_update_proven": False, "restorable_training_state_proven": False,
        "graduation_authorized": False,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Read-only physical genome verifier")
    parser.add_argument("--verify", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_verify_local(args.verify), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
