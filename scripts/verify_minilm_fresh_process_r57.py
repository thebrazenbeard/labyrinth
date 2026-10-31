"""Offline, separately invokable MiniLM checkpoint *byte* preflight.

The caller must stage six specifically named artifacts and provide pins from an
independent source. This code reads and hashes bytes; it NEVER loads a model,
scores a private exam, authenticates the pin custodian, or grants any effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Any


SCHEMA = "labyrinth.minilm-fresh-process-byte-readback.v1"
STATUS = "OFFLINE_STAGED_ARTIFACT_BYTES_MATCH_SOURCE_ONLY_UNATTESTED"
MAX_MANIFEST_BYTES = 65536
ROLE_PATHS = MappingProxyType({
    "adapter": "adapter/adapter_model.safetensors",
    "classifier_head": "head.safetensors",
    "optimizer_state": "optimizer-state/state.safetensors",
    "rng_state": "optimizer-state/rng.json",
    "tokenizer": "tokenizer/tokenizer.json",
    "frozen_base": "base/model.safetensors",
})
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")


class ArtifactVerificationError(ValueError):
    """A proposed artifact manifest or independently supplied pin is invalid."""


def _object_pairs_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ArtifactVerificationError("duplicate JSON field")
        obj[key] = value
    return obj


def _require_non_symlink_path(path: Path, *, directory: bool, label: str) -> os.stat_result:
    """Reject symlinks at *every* lexical component, including ancestors.

    This is an offline, static-layout preflight. Component lstat checks do not
    establish custody or an atomic snapshot against concurrent directory edits.
    """
    if not isinstance(path, Path):
        raise ArtifactVerificationError(f"{label} path must be a Path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    if ".." in absolute.parts:
        raise ArtifactVerificationError(f"{label} parent traversal forbidden")
    current = Path(absolute.anchor)
    try:
        observed = current.lstat()
        if not stat.S_ISDIR(observed.st_mode):
            raise ArtifactVerificationError(f"{label} filesystem anchor is not a directory")
        for index, part in enumerate(absolute.parts[1:]):
            current = current / part
            observed = current.lstat()
            if stat.S_ISLNK(observed.st_mode):
                raise ArtifactVerificationError(f"{label} symlink path component forbidden")
            is_final = index == len(absolute.parts) - 2
            if not is_final and not stat.S_ISDIR(observed.st_mode):
                raise ArtifactVerificationError(f"{label} ancestor must be a directory")
        expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_kind(observed.st_mode):
            raise ArtifactVerificationError(f"{label} has incorrect filesystem kind")
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot inspect {label} path components") from exc
    return observed


def _load_canonical_manifest(manifest_path: Path) -> tuple[dict[str, Any], bytes]:
    observed_stat = _require_non_symlink_path(manifest_path, directory=False, label="manifest")
    if observed_stat.st_size > MAX_MANIFEST_BYTES:
        raise ArtifactVerificationError("oversized manifest")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(manifest_path, flags)
        with os.fdopen(fd, "rb") as source:
            opened_stat = os.fstat(source.fileno())
            if (not stat.S_ISREG(opened_stat.st_mode)
                    or (opened_stat.st_dev, opened_stat.st_ino)
                    != (observed_stat.st_dev, observed_stat.st_ino)):
                raise ArtifactVerificationError("manifest replaced during file read")
            raw = source.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ArtifactVerificationError("cannot read pinned manifest") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ArtifactVerificationError("oversized manifest")
    try:
        parsed = json.loads(raw.decode("utf-8"),
                            object_pairs_hook=_object_pairs_no_duplicate_keys,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ArtifactVerificationError("non-finite JSON value")))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ArtifactVerificationError("invalid canonical JSON manifest") from exc
    if type(parsed) is not dict:
        raise ArtifactVerificationError("manifest root must be an object")
    canonical = (json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if canonical != raw:
        raise ArtifactVerificationError("noncanonical manifest encoding")
    return parsed, raw


def _expected_pin(value: object, pattern: re.Pattern[str], label: str) -> str:
    if type(value) is not str or not pattern.fullmatch(value):
        raise ArtifactVerificationError(f"missing or malformed independent {label} pin")
    return value


def _read_role_bytes(root: Path, relative: str) -> tuple[str, int]:
    # Paths are fixed in source, rather than supplied for unrestricted reads.
    parts = Path(relative).parts
    if not parts or any(part in (".", "..") for part in parts):
        raise ArtifactVerificationError("invalid artifact path")
    cursor = root / relative
    observed_stat = _require_non_symlink_path(cursor, directory=False, label="artifact")
    digest = hashlib.sha256()
    count = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(cursor, flags)
        with os.fdopen(fd, "rb") as source:
            opened_stat = os.fstat(source.fileno())
            if (not stat.S_ISREG(opened_stat.st_mode)
                    or (opened_stat.st_dev, opened_stat.st_ino)
                    != (observed_stat.st_dev, observed_stat.st_ino)):
                raise ArtifactVerificationError("artifact replaced during file read")
            while block := source.read(1024 * 1024):
                count += len(block)
                digest.update(block)
    except OSError as exc:
        raise ArtifactVerificationError("cannot read pinned artifact") from exc
    return digest.hexdigest(), count


def verify_staged_bundle(
    *, bundle_root: Path, manifest_path: Path,
    expected_manifest_sha256: str, expected_source_head: str,
    expected_source_tree: str, expected_run_id: str,
) -> dict[str, Any]:
    """Read staged files with exact external pins; return NO permissions.

    The expected pins are *caller supplied*. Their genuinely independent
    provenance cannot be established here. The process identity and all six
    byte hashes are diagnostic, not a host-attestation or model-evaluation seal.
    """
    manifest_pin = _expected_pin(expected_manifest_sha256, _SHA64, "manifest")
    head_pin = _expected_pin(expected_source_head, _SHA40, "source head")
    tree_pin = _expected_pin(expected_source_tree, _SHA40, "source tree")
    run_pin = _expected_pin(expected_run_id, _RUN, "run")
    _require_non_symlink_path(bundle_root, directory=True, label="bundle root")
    manifest, raw = _load_canonical_manifest(manifest_path)
    if hashlib.sha256(raw).hexdigest() != manifest_pin:
        raise ArtifactVerificationError("manifest bytes differ from independent commitment")
    if set(manifest) != {"schema", "source_head", "source_tree", "run_id",
                         "trainer_pid", "files"}:
        raise ArtifactVerificationError("missing or privileged manifest field")
    if manifest["schema"] != SCHEMA:
        raise ArtifactVerificationError("unsupported manifest schema")
    if (type(manifest["source_head"]) is not str or manifest["source_head"] != head_pin
            or type(manifest["source_tree"]) is not str or manifest["source_tree"] != tree_pin
            or type(manifest["run_id"]) is not str or manifest["run_id"] != run_pin):
        raise ArtifactVerificationError("source/run identity diverges from independently pinned values")
    if type(manifest["trainer_pid"]) is not int or manifest["trainer_pid"] <= 0:
        raise ArtifactVerificationError("invalid declared trainer process identity")
    files = manifest["files"]
    if type(files) is not dict or set(files) != set(ROLE_PATHS):
        raise ArtifactVerificationError("all six pinned artifact roles are required")
    observations = {}
    for role, fixed_relative in ROLE_PATHS.items():
        claim = files[role]
        if type(claim) is not dict or set(claim) != {"path", "sha256", "size_bytes"}:
            raise ArtifactVerificationError("invalid artifact role commitment")
        if type(claim["path"]) is not str or claim["path"] != fixed_relative:
            raise ArtifactVerificationError("artifact role path changed")
        _expected_pin(claim["sha256"], _SHA64, "artifact SHA256")
        if type(claim["size_bytes"]) is not int or claim["size_bytes"] <= 0:
            raise ArtifactVerificationError("invalid pinned artifact size")
        observed_sha, observed_size = _read_role_bytes(bundle_root, fixed_relative)
        if observed_sha != claim["sha256"] or observed_size != claim["size_bytes"]:
            raise ArtifactVerificationError("artifact bytes differ from pinned role commitment")
        observations[role] = {"sha256": observed_sha, "size_bytes": observed_size}
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "source_head": head_pin,
        "source_tree": tree_pin,
        "run_id": run_pin,
        "manifest_sha256": manifest_pin,
        "evaluator_pid": os.getpid(),
        "different_pid_from_declared_trainer": os.getpid() != manifest["trainer_pid"],
        "files": observations,
        "model_loaded": False,
        "optimizer_executed": False,
        "independent_evaluator_custody_verified": False,
        "private_evaluation_verified": False,
        "can_authorize_training": False,
        "can_authorize_graduation": False,
        "can_authorize_deployment": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-run-id", required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_staged_bundle(
            bundle_root=args.bundle_root, manifest_path=args.manifest_file,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_source_head=args.expected_source_head,
            expected_source_tree=args.expected_source_tree,
            expected_run_id=args.expected_run_id,
        )
    except ArtifactVerificationError as exc:
        print(f"offline source-only preflight refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
