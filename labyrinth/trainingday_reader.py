"""Verified streaming reader for the *training* lane of a TRAININGDAY corpus.

Evaluation shards are never opened or exposed by this API. Read-only by design.
"""
import hashlib
import json
from pathlib import Path

SCHEMA = "vera.trainingday.corpus.v1"


def _inside(root, relative):
    if not isinstance(relative, str) or relative.startswith(("/", "\\")) or "\\" in relative:
        raise ValueError("invalid shard path")
    target = (root / relative).resolve(strict=False)
    if root != target and root not in target.parents:
        raise ValueError("path escapes corpus")
    if not target.is_file() or target.is_symlink():
        raise ValueError("missing or symlink corpus file")
    return target


def _digest(path):
    hash_state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_state.update(block)
    return hash_state.hexdigest()


def iter_training_rows(corpus_root):
    """Yield only train rows after checking manifest, licenses and shard hashes.

    Does not trust caller-provided source strings, trust unverified code answers,
    open evaluation data, or replace model weights. A downstream trainer must
    still enforce licensing and independent evaluation policy.
    """
    root = Path(corpus_root).resolve(strict=True)
    manifest_file = _inside(root, "manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "CORPUS_BUILT_NO_MODEL_TRAINING":
        raise ValueError("unsupported corpus manifest")
    licenses = manifest.get("licenses")
    if not isinstance(licenses, list) or not licenses:
        raise ValueError("license manifest absent")
    for license in licenses:
        target = _inside(root, license["path"])
        if _digest(target) != license["sha256"]:
            raise ValueError("license checksum mismatch")
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise ValueError("shard manifest absent")
    training = sorted((s for s in entries if s["split"] == "train"), key=lambda x: x["path"])
    paths = {s["path"] for s in training}
    actual = {p.relative_to(root).as_posix() for p in (root / "train").glob("*.jsonl")}
    if paths != actual:
        raise ValueError("missing or undeclared training shards")
    if not training:
        raise ValueError("train corpus empty")
    total = 0
    for shard in training:
        relative = shard["path"]
        if not relative.startswith("train/") or not relative.endswith(".jsonl"):
            raise ValueError("train shard path not confined to train")
        file = _inside(root, relative)
        if file.stat().st_size != shard["bytes"] or _digest(file) != shard["sha256"]:
            raise ValueError("shard checksum mismatch")
        count = 0
        with file.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (row.get("kind") not in ("code", "verified_reasoning")
                        or row.get("source_repo") == manifest["heldout_repo"]
                        or not isinstance(row.get("text"), str)):
                    raise ValueError("unsupported or leaked training row")
                count += 1
                total += 1
                yield row
        if count != shard["count"]:
            raise ValueError("shard row count mismatch")
    if total != manifest["counts"]["train"]:
        raise ValueError("manifest train count mismatch")
