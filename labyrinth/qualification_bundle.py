"""Deterministic *local* qualification-bundle byte inventory.

The runner's source_tree_sha256 only identifies runtime source files. This
separate inventory includes the qualification driver, CI definitions and test
suite, so a prior captured manifest can detect changes to those files too.

The digest is NOT a signature, an attestation, a claim that tests ran, proof
of neural learning, or evidence of independent evaluator custody. External
model weights, operator-supplied scope files and host state are not included.
A consumer must retain a trusted previous manifest outside the mutable
checkout to use ``verify_qualification_bundle`` meaningfully.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "labyrinth.qualification-bundle.v1"
_REQUIRED = (
    ".github/workflows/ci.yml",
    ".github/workflows/minotaur-neural-cpu.yml",
    "pyproject.toml",
    "scripts/make_tiny_hf_fixture.py",
    "scripts/minotaur_ci.py",
    "scripts/qualify_hf_regenerative.py",
    "tests/minotaur_required_tests.txt",
)
_COLLECTIONS = (
    "labyrinth/*.py",
    "tests/test_*.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _root_path(repo_root: Path | str | None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    if root.is_symlink() or not root.is_dir():
        raise ValueError("qualification root must be an existing non-symlink directory")
    return root.resolve(strict=True)


def _safe_contents(path: Path, root: Path) -> bytes:
    # Fail closed for accidental symlinked files or directories. This is a
    # best-effort snapshot of the checkout, not an adversarial filesystem race
    # defense or a replacement for independently retained custody records.
    parent = path.parent
    while parent != root:
        if parent.is_symlink():
            raise ValueError("non-regular or symlink qualification path")
        if not parent.is_relative_to(root):
            raise ValueError("qualification path escapes root")
        parent = parent.parent
    if path.is_symlink() or not path.is_file():
        raise ValueError("non-regular or symlink qualification file")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("qualification file escapes root")
    return path.read_bytes()


def capture_qualification_bundle(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Inventory current byte contents without Git, network or file writes.

    The same checkout contents produce an identical bundle across machines;
    absolute paths, timestamps and working-tree metadata are intentionally
    excluded. The required paths and collections form an explicit versioned
    coverage policy. A missing required file or collection fails closed.
    """
    root = _root_path(repo_root)
    paths = {name: root / name for name in _REQUIRED}
    for name in _REQUIRED:
        if not paths[name].is_file():
            raise ValueError(f"missing qualification file: {name}")
    for pattern in _COLLECTIONS:
        matched = sorted(root.glob(pattern))
        if not matched:
            raise ValueError(f"missing qualification collection: {pattern}")
        for path in matched:
            name = path.relative_to(root).as_posix()
            paths[name] = path

    files = []
    for name, path in sorted(paths.items()):
        contents = _safe_contents(path, root)
        files.append({"path": name, "bytes": len(contents), "sha256": _sha(contents)})
    payload = {
        "schema": SCHEMA,
        "required_files": list(_REQUIRED),
        "collections": list(_COLLECTIONS),
        "files": files,
    }
    return {**payload, "bundle_sha256": _sha(_canonical(payload))}


def verify_qualification_bundle(expected: Mapping[str, Any],
                                repo_root: Path | str | None = None) -> bool:
    """Compare a previously captured manifest to current checkout bytes.

    A re-created manifest supplied by the same untrusted process is not
    independent evidence. Invalid input or inaccessible files fail closed.
    """
    if not isinstance(expected, Mapping):
        return False
    try:
        return dict(expected) == capture_qualification_bundle(repo_root)
    except (OSError, ValueError):
        return False
