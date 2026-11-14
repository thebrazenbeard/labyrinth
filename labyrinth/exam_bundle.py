"""Source-only staged examiner bundle with one ready-directory publication.

Prepares the existing original private/public/precommit protocol in a sibling
staging directory, checks its bytes and finite public mechanics, writes a
strict manifest, then publishes the directory in one same-volume rename.

This gives an observable ready/not-ready boundary for normal execution; it
does NOT promise power-loss durability, a race-free ACL sandbox, independent
examiner identity, model execution, weight attestation or neural learning.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .blind_structural_exam import MAX_BYTES, _read_bundle, prepare_exam
from .live_model_probe import SHA, _canonical, _file_bytes, _load, _write_once

SCHEMA = "labyrinth.blind-mechanics-staged-bundle.v1"
FILES = frozenset({"private.json", "public.json", "precommit.json", "manifest.json"})
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _safe_ancestors(path: Path) -> None:
    """Reject known static redirecting ancestors; not a TOCTOU-safe handle."""
    for parent in path.parents:
        if parent.is_symlink() or (
                getattr(parent.lstat(), "st_file_attributes", 0) & REPARSE):
            raise ValueError("exam bundle parent is a symlink or reparse point")


def _original_bytes(directory: Path):
    raw_private = _file_bytes(directory / "private.json", MAX_BYTES)
    raw_public = _file_bytes(directory / "public.json", MAX_BYTES)
    raw_precommit = _file_bytes(directory / "precommit.json", MAX_BYTES)
    original = _load(raw_precommit)
    if type(original) is not dict:
        raise ValueError("original source precommit is invalid")
    receipt = original.get("receipt_sha256")
    if type(receipt) is not str or SHA.fullmatch(receipt) is None:
        raise ValueError("original source receipt SHA is invalid")
    _read_bundle(directory / "private.json", directory / "public.json",
                 directory / "precommit.json", receipt)
    return raw_private, raw_public, raw_precommit, original


def verify_exam_bundle(bundle, *, expected_manifest_sha256):
    """Validate exactly one complete ready directory with caller-held SHA."""
    if (type(expected_manifest_sha256) is not str
            or SHA.fullmatch(expected_manifest_sha256) is None):
        raise ValueError("expected original exam manifest SHA pin is invalid")
    root = Path(bundle)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("ready exam bundle directory is missing or redirected")
    _safe_ancestors(root)
    if {p.name for p in root.iterdir()} != FILES:
        raise ValueError("ready exam bundle must contain exactly four files")
    manifest_bytes = _file_bytes(root / "manifest.json", MAX_BYTES)
    if _sha(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("original exam manifest differs from caller-held SHA pin")
    manifest = _load(manifest_bytes)
    required = {"schema", "exam_id", "case_count", "status",
                "private_sha256", "public_sha256", "precommit_sha256",
                "precommit_receipt_sha256", "independent_custody_verified",
                "neural_learning_verified"}
    if (type(manifest) is not dict or set(manifest) != required
            or manifest.get("schema") != SCHEMA
            or manifest.get("status") != "SOURCE_ONLY_READY_NOT_INDEPENDENT"
            or manifest.get("independent_custody_verified") is not False
            or manifest.get("neural_learning_verified") is not False
            or type(manifest.get("case_count")) is not int
            or not 1 <= manifest["case_count"] <= 16
            or manifest_bytes != _canonical(manifest) + b"\n"):
        raise ValueError("invalid original exam bundle manifest")
    raw_private, raw_public, raw_precommit, plan = _original_bytes(root)
    if (manifest["private_sha256"] != _sha(raw_private)
            or manifest["public_sha256"] != _sha(raw_public)
            or manifest["precommit_sha256"] != _sha(raw_precommit)
            or manifest["precommit_receipt_sha256"] != plan["receipt_sha256"]
            or manifest["exam_id"] != plan["exam_id"]
            or manifest["case_count"] != plan["case_count"]):
        raise ValueError("bundle source files differ from original verified manifest")
    return {
        "status": "SOURCE_ONLY_VERIFIED_NOT_INDEPENDENT",
        "exam_id": plan["exam_id"],
        "case_count": plan["case_count"],
        "manifest_sha256": expected_manifest_sha256,
        "precommit_receipt_sha256": plan["receipt_sha256"],
        "independent_custody_verified": False,
        "neural_learning_verified": False,
        "verification_class": "LOCAL_SOURCE_ONLY_NOT_INDEPENDENT",
    }


def prepare_exam_bundle(destination, *, count=8, rng=None, worlds=None):
    """Stage complete original exam files and publish by sibling rename.

    Uses only a newly-created staging directory. Cleans that directory after
    an ordinary failure, without deleting any preexisting destination.
    Stage remnants may remain after a crash; do not treat those as ready.
    """
    ready = Path(destination)
    if ready.exists() or ready.is_symlink():
        raise ValueError("ready bundle destination must be new")
    if not ready.parent.is_dir():
        raise FileNotFoundError("ready bundle parent directory does not exist")
    _safe_ancestors(ready)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{ready.name}.stage-", dir=ready.parent))
    published = False
    try:
        prepared = prepare_exam(
            stage / "private.json", stage / "public.json",
            stage / "precommit.json", count=count, rng=rng, worlds=worlds)
        raw_private, raw_public, raw_precommit, plan = _original_bytes(stage)
        manifest = {
            "schema": SCHEMA, "exam_id": plan["exam_id"],
            "case_count": plan["case_count"],
            "status": "SOURCE_ONLY_READY_NOT_INDEPENDENT",
            "private_sha256": _sha(raw_private),
            "public_sha256": _sha(raw_public),
            "precommit_sha256": _sha(raw_precommit),
            "precommit_receipt_sha256": plan["receipt_sha256"],
            "independent_custody_verified": False,
            "neural_learning_verified": False,
        }
        _write_once(stage / "manifest.json", manifest)
        manifest_pin = _sha(_canonical(manifest) + b"\n")
        verify_exam_bundle(stage, expected_manifest_sha256=manifest_pin)
        if ready.exists() or ready.is_symlink():
            raise FileExistsError("ready bundle was created during staging")
        os.rename(stage, ready)
        published = True
        return {
            "schema": SCHEMA,
            "status": "SOURCE_ONLY_READY_NOT_INDEPENDENT",
            "exam_id": prepared["exam_id"],
            "case_count": prepared["case_count"],
            "manifest_sha256": manifest_pin,
            "precommit_receipt_sha256": plan["receipt_sha256"],
            "independent_custody_verified": False,
            "neural_learning_verified": False,
        }
    finally:
        if not published and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out-bundle", required=True)
    prepare.add_argument("--count", type=int, default=8)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_exam_bundle(args.out_bundle, count=args.count)
    else:
        result = verify_exam_bundle(
            args.bundle,
            expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
