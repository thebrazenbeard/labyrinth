"""Write-once local entrance identity for model-weight and capability baselines.

The large immutable model tensors are *referenced*, never copied. Digests are
streamed and checked again in a fresh process. This is operator-local
integrity/provenance, not a trusted anti-rollback witness or evidence that
any capability improved. An independently custodied private examination is
entirely outside this source module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA = "labyrinth.model-entrance-state.v1"
_DOMAIN = b"labyrinth.model-entrance-state.v1\0"
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_BASE_WEIGHT_SUFFIXES = (".safetensors", ".gguf")
_META_NAMES = {"config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "vocab.json", "vocab.txt", "special_tokens_map.json", "generation_config.json",
               "model.safetensors.index.json", "merges.txt"}
_MAX_SMALL_STATE_BYTES = 512 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024


def _canonical(value: Any) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("entrance manifest exceeds 64 KiB")
    return payload


def _digest(value: Any, label: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact SHA-256 hex digest")


def _source_dir(source: Path, label: str) -> Path:
    if not isinstance(source, Path) or not source.is_absolute():
        raise ValueError(f"{label} requires an absolute local Path")
    if source.is_symlink() or not source.is_dir() or source != source.resolve(strict=True):
        raise ValueError(f"{label} must be a real non-symlink local directory")
    return source


def _file_sha(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("only regular non-symlink model/state files may be captured")
    before = path.stat()
    if before.st_size <= 0:
        raise ValueError("model/state file cannot be empty")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    ):
        raise RuntimeError("model/state bytes changed while capturing entrance")
    return digest.hexdigest(), before.st_size


def _inventory(source: Path, *, kind: str) -> dict[str, Any]:
    source = _source_dir(source, kind)
    entries = sorted(source.rglob("*"))
    if not entries or len(entries) > 128:
        raise ValueError(f"{kind} source file count is invalid")
    if any(item.is_symlink() or not (item.is_file() or item.is_dir()) for item in entries):
        raise ValueError(f"{kind} source contains a symlink or special file")
    files = [item for item in entries if item.is_file()]
    if kind == "base":
        if (not any(item.name.endswith(_BASE_WEIGHT_SUFFIXES) for item in files)
                or not any(item.name == "config.json" for item in files)
                or not any(item.name in {"tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt"}
                           for item in files)
                or any(item.name not in _META_NAMES and
                       not item.name.endswith(_BASE_WEIGHT_SUFFIXES) for item in files)):
            raise ValueError("base requires immutable weight, config and tokenizer files")
    elif kind == "adapter":
        if (not any(item.name == "adapter_model.safetensors" for item in files)
                or not any(item.name == "adapter_config.json" for item in files)):
            raise ValueError("adapter needs an actual saved LoRA safetensors/config pair")
    else:
        if sum(item.stat().st_size for item in files) > _MAX_SMALL_STATE_BYTES:
            raise ValueError(f"{kind} auxiliary-state budget exceeded")
    records = {}
    for item in files:
        digest, size = _file_sha(item)
        records[item.relative_to(source).as_posix()] = {"sha256": digest, "bytes": size}
    return {"root": str(source), "files": records}


def _baseline(observations: Any) -> list[dict[str, Any]]:
    if type(observations) is not list or not 1 <= len(observations) <= 64:
        raise ValueError("entrance needs one to 64 scored TRAIN-safe baseline observations")
    approved = []
    used: set[str] = set()
    for entry in observations:
        if (type(entry) is not dict
                or set(entry) != {"case_id", "skill", "prompt", "prompt_sha256", "score", "max_score", "scorer"}
                or any(type(entry[field]) is not str for field in (
                    "case_id", "skill", "prompt", "prompt_sha256", "scorer"))
                or any(_ID.fullmatch(entry[field]) is None for field in ("case_id", "skill", "scorer"))
                or not 0 < len(entry["prompt"].encode("utf-8")) <= 512
                or hashlib.sha256(entry["prompt"].encode("utf-8")).hexdigest() != entry["prompt_sha256"]
                or type(entry["score"]) is not int or type(entry["max_score"]) is not int
                or entry["max_score"] < 1 or entry["max_score"] > 100
                or not 0 <= entry["score"] <= entry["max_score"]
                or entry["case_id"] in used or entry["prompt_sha256"] in used):
            raise ValueError("invalid or non-TRAIN-safe baseline evidence row")
        # Only explicitly designated local development prompts are accepted.
        # A private holdout/protected exam must not enter this baseline.
        if not entry["case_id"].startswith("devbaseline-"):
            raise ValueError("baseline case must be development-only, never private examination")
        used.add(entry["case_id"])
        used.add(entry["prompt_sha256"])
        approved.append(dict(entry))
    return approved


def _state_digest(base: dict[str, Any], adapter: Any) -> str:
    """The model-parameter identity only; excludes optimizer/scheduler/RNG."""
    payload = {"base_files": base["files"],
               "adapter_files": adapter["files"] if type(adapter) is dict else None}
    return hashlib.sha256(_DOMAIN + _canonical(payload)).hexdigest()


def _captured_training_state_digest(
    base: dict[str, Any], adapter: Any, auxiliary: dict[str, Any],
) -> str:
    """Commit to all captured bytes, without claiming the files are restorable."""
    payload = {"parameter_state_sha256": _state_digest(base, adapter),
               "auxiliary_files": {kind: inventory["files"] if inventory else None
                                   for kind, inventory in sorted(auxiliary.items())}}
    return hashlib.sha256(_DOMAIN + _canonical(payload)).hexdigest()


def _manifest_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_DOMAIN + _canonical({key: item for key, item in value.items()
                                               if key != "manifest_sha256"})).hexdigest()


def build_entrance_state(
    *, base_dir: Path, adapter_dir: Path | None,
    optimizer_dir: Path | None = None, scheduler_dir: Path | None = None,
    rng_dir: Path | None = None, scope_sha256: str, source_tree_sha256: str,
    source_receipt_path: Path, source_receipt_sha256: str,
    baseline_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hash existing local bytes; never initialize, infer, train or copy them."""
    for label, value in (("scope", scope_sha256), ("source tree", source_tree_sha256),
                         ("source receipt", source_receipt_sha256)):
        _digest(value, label)
    if (not isinstance(source_receipt_path, Path) or not source_receipt_path.is_absolute()
            or source_receipt_path != source_receipt_path.resolve(strict=True)
            or _file_sha(source_receipt_path)[0] != source_receipt_sha256):
        raise ValueError("entrance source receipt bytes differ from pinned identity")
    base = _inventory(base_dir, kind="base")
    adapter = _inventory(adapter_dir, kind="adapter") if adapter_dir is not None else None
    auxiliary = {
        "optimizer": _inventory(optimizer_dir, kind="optimizer") if optimizer_dir is not None else None,
        "scheduler": _inventory(scheduler_dir, kind="scheduler") if scheduler_dir is not None else None,
        "rng": _inventory(rng_dir, kind="rng") if rng_dir is not None else None,
    }
    entrance = {
        "schema": SCHEMA, "evidence": "OPERATOR_LOCAL_CONTENT_INTEGRITY_ONLY",
        "scope_sha256": scope_sha256, "source_tree_sha256": source_tree_sha256,
        "source_receipt_sha256": source_receipt_sha256,
        "source_receipt_path": str(source_receipt_path),
        "base": base, "adapter": adapter, "auxiliary_state": auxiliary,
        "state_gaps": [label + "_NOT_CAPTURED" for label, source in (
            ("ADAPTER", adapter), ("OPTIMIZER", auxiliary["optimizer"]),
            ("SCHEDULER", auxiliary["scheduler"]), ("RNG", auxiliary["rng"]),
        ) if source is None],
        "w0_state_sha256": _state_digest(base, adapter),
        "w0_captured_training_state_sha256": _captured_training_state_digest(base, adapter, auxiliary),
        "training_safe_baseline": {
            "classification": "SELF_REPORTED_LOCAL_DEVELOPMENT_DIAGNOSTIC",
            "checkpoint_state_sha256": _state_digest(base, adapter),
            "partition": "TRAIN_SAFE_BASELINE_NOT_PRIVATE_HOLDOUT",
            "observations": _baseline(baseline_observations),
            "execution_witnessed": False,
        },
        "promotion_authorized": False, "neural_improvement_proven": False,
    }
    entrance["manifest_sha256"] = _manifest_hash(entrance)
    return entrance


def save_entrance_state_once(entrance: dict[str, Any], destination: Path) -> str:
    """Atomic, no-overwrite manifest publication; model files remain in place."""
    if not isinstance(destination, Path) or not destination.is_absolute():
        raise ValueError("entrance manifest destination must be absolute")
    if destination.is_symlink() or destination.exists():
        raise FileExistsError("entrance manifest is write-once and already exists")
    if (destination.parent.is_symlink() or not destination.parent.is_dir()
            or destination.parent != destination.parent.resolve(strict=True)):
        raise ValueError("entrance manifest parent must be an existing real directory")
    if not isinstance(entrance, dict) or entrance.get("manifest_sha256") != _manifest_hash(entrance):
        raise ValueError("entrance manifest checksum invalid")
    payload = json.dumps(entrance, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".entrance-staged-", dir=destination.parent)
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return entrance["manifest_sha256"]


def verify_entrance_state(path: Path) -> dict[str, Any]:
    """Fresh-process readback checks every referenced file, never just JSON."""
    if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
            or not path.is_file() or not 0 < path.stat().st_size <= 128 * 1024):
        raise ValueError("entrance report path must be a bounded non-symlink local file")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate entrance report key")
            value[key] = item
        return value

    manifest = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if (type(manifest) is not dict or manifest.get("schema") != SCHEMA
            or manifest.get("manifest_sha256") != _manifest_hash(manifest)
            or manifest.get("promotion_authorized") is not False
            or manifest.get("neural_improvement_proven") is not False
            or manifest.get("evidence") != "OPERATOR_LOCAL_CONTENT_INTEGRITY_ONLY"
            or set(manifest) != {
                "schema", "evidence", "scope_sha256", "source_tree_sha256",
                "source_receipt_sha256", "source_receipt_path", "base", "adapter",
                "auxiliary_state", "state_gaps", "w0_state_sha256",
                "w0_captured_training_state_sha256",
                "training_safe_baseline", "promotion_authorized",
                "neural_improvement_proven", "manifest_sha256",
            }):
        raise ValueError("entrance state identity or evidence classification changed")
    for label in ("scope_sha256", "source_tree_sha256", "source_receipt_sha256"):
        _digest(manifest.get(label), label)
    receipt_text = manifest.get("source_receipt_path")
    if type(receipt_text) is not str or not 1 <= len(receipt_text) <= 4096:
        raise ValueError("entrance receipt must have a bounded canonical path")
    receipt_path = Path(receipt_text)
    if (not receipt_path.is_absolute() or receipt_path.is_symlink()
            or receipt_path != receipt_path.resolve(strict=True)):
        raise ValueError("entrance receipt path is not canonical or contains a symlink")
    if _file_sha(receipt_path)[0] != manifest["source_receipt_sha256"]:
        raise ValueError("source receipt bytes changed since model entrance")
    base = _inventory(Path(manifest["base"]["root"]), kind="base")
    if base != manifest["base"]:
        raise ValueError("immutable base model/tokenizer bytes changed since entrance")
    adapter_value = manifest["adapter"]
    adapter = (_inventory(Path(adapter_value["root"]), kind="adapter")
               if adapter_value is not None else None)
    if adapter != adapter_value:
        raise ValueError("trainable adapter bytes changed since entrance")
    for label, source in manifest["auxiliary_state"].items():
        observed = _inventory(Path(source["root"]), kind=label) if source is not None else None
        if observed != source:
            raise ValueError(f"{label} state changed since entrance")
    expected_gaps = [label + "_NOT_CAPTURED" for label, source in (
        ("ADAPTER", adapter), ("OPTIMIZER", manifest["auxiliary_state"].get("optimizer")),
        ("SCHEDULER", manifest["auxiliary_state"].get("scheduler")),
        ("RNG", manifest["auxiliary_state"].get("rng")),
    ) if source is None]
    profile = manifest["training_safe_baseline"]
    if (set(manifest["auxiliary_state"]) != {"optimizer", "scheduler", "rng"}
            or manifest["state_gaps"] != expected_gaps
            or type(profile) is not dict
            or set(profile) != {"classification", "checkpoint_state_sha256", "partition",
                                "observations", "execution_witnessed"}
            or profile["classification"] != "SELF_REPORTED_LOCAL_DEVELOPMENT_DIAGNOSTIC"
            or profile["partition"] != "TRAIN_SAFE_BASELINE_NOT_PRIVATE_HOLDOUT"
            or profile["execution_witnessed"] is not False):
        raise ValueError("entrance capability or state-gap evidence claim invalid")
    if (_state_digest(base, adapter) != manifest["w0_state_sha256"]
            or _captured_training_state_digest(base, adapter, manifest["auxiliary_state"])
               != manifest["w0_captured_training_state_sha256"]
            or profile["checkpoint_state_sha256"] != manifest["w0_state_sha256"]
            or _baseline(profile["observations"]) != profile["observations"]):
        raise ValueError("entrance W0 and TRAIN-safe capability baseline unbound")
    return {"schema": SCHEMA, "status": "LOCAL_BYTES_MATCH_ENTRANCE",
            "w0_state_sha256": manifest["w0_state_sha256"],
            "w0_captured_training_state_sha256": manifest["w0_captured_training_state_sha256"],
            "all_declared_auxiliary_files_captured": not manifest["state_gaps"],
            # A real trainer restore/replay test is required; arbitrary files
            # that merely hash correctly cannot prove a restorable optimizer.
            "full_resume_state_available": False,
            "capability_claim": "SELF_REPORTED_LOCAL_BASELINE_ONLY",
            "graduation_authorized": False}


def compare_adapter_to_w0(entrance: dict[str, Any], successor_adapter_dir: Path) -> dict[str, Any]:
    """Identify changed adapter bytes; this does not establish actual learning."""
    if entrance.get("schema") != SCHEMA or entrance.get("manifest_sha256") != _manifest_hash(entrance):
        raise ValueError("invalid entrance state")
    if _inventory(Path(entrance["base"]["root"]), kind="base") != entrance["base"]:
        raise ValueError("immutable W0 base model differs before successor comparison")
    successor = _inventory(successor_adapter_dir, kind="adapter")
    digest = _state_digest(entrance["base"], successor)
    return {"w0_state_sha256": entrance["w0_state_sha256"],
            "wt_state_sha256": digest, "bytes_changed": digest != entrance["w0_state_sha256"],
            "learning_demonstrated": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify existing EntranceState bytes in a fresh process")
    parser.add_argument("verify", choices=("verify",))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_entrance_state(Path(args.manifest)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
