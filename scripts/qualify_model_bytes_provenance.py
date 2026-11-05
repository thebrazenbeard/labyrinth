"""Opt-in, offline byte-custody wrapper for the public MiniLM CPU pilot.

It verifies a separately committed local safetensors/config/tokenizer manifest
*before and after* launching the existing pilot with a local model directory.
It binds the pilot's report bytes and exact runner source hash to that window.

This is NOT a cryptographic attestation of what the model loader opened: a
mutable model directory can change and change back between the two checks.
The trusted manifest SHA must be custodied outside the worker's control.
The source-level wrapper does not authenticate publisher origin, model
semantics, independent examination, or graduation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable

from labyrinth.pretrained_model_provenance import (
    ModelProvenanceError, verify_pretrained_model_manifest,
)
from scripts.qualify_minilm_cpu import SCHEMA as PILOT_SCHEMA, plan_output, validate_fixtures


SCHEMA = "labyrinth.minilm-local-byte-window.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON = 4 * 1024 * 1024
_REPO = Path(__file__).resolve().parents[1]
_RUNNER = Path(__file__).with_name("qualify_minilm_cpu.py")


class ModelByteWindowError(ValueError):
    """A model, runner, or development-result provenance precondition failed."""


def _digest_file(path: Path, *, limit: int = _MAX_JSON) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ModelByteWindowError("missing, linked or oversized artifact")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    """Parse exactly the bytes later identified by the JSON artifact digest."""
    if path.is_symlink():
        raise ModelByteWindowError("linked JSON artifact")
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON:
            raise ModelByteWindowError("missing or oversized JSON artifact")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                             | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    before.st_dev, before.st_ino, before.st_size):
                raise ModelByteWindowError("JSON artifact changed while opening")
            data = stream.read(_MAX_JSON + 1)
        after = path.lstat()
        if (len(data) != before.st_size or len(data) > _MAX_JSON
                or path.is_symlink()
                or (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)):
            raise ModelByteWindowError("JSON artifact changed while reading")
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelByteWindowError("unreadable JSON artifact") from exc
    if type(value) is not dict:
        raise ModelByteWindowError("artifact must be a JSON object")
    return value, hashlib.sha256(data).hexdigest()


def _require_output_root(root: Path) -> None:
    if (not root.is_dir() or root.is_symlink()
            or bool(getattr(root, "is_junction", lambda: False)())
            or root.resolve(strict=True) != root):
        raise ModelByteWindowError("unsafe runner output directory")


def _verify_model(root: Path, manifest: dict[str, Any], trusted: str) -> None:
    try:
        verify_pretrained_model_manifest(
            root, manifest, trusted_manifest_sha256=trusted,
        )
    except (ModelProvenanceError, ValueError, OSError) as exc:
        raise ModelByteWindowError("local model bytes differ from trusted manifest") from exc


def _check_report(report: dict[str, Any], *, model_root: Path,
                  runner_sha: str, seed: int, steps: int, rate: float) -> None:
    expected = {
        "schema": PILOT_SCHEMA,
        "model_id": str(model_root),
        "harness_sha256": runner_sha,
        "seed": seed,
        "steps_per_cycle": steps,
        "learning_rate": rate,
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "private_final_exam": "NOT_RUN",
        "graduation_authorized": False,
        "physical_model_byte_custody": "SELF_REPORTED",
    }
    if any(type(report.get(key)) is not type(value) or report[key] != value
           for key, value in expected.items()):
        raise ModelByteWindowError("pilot report fails exact runner/configuration binding")
    if report.get("learning_qualification") != "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION":
        raise ModelByteWindowError("pilot report overclaims learning qualification")
    if not isinstance(report.get("arms"), dict) or set(report["arms"]) != {
        "correct", "zero_update", "inverted"
    }:
        raise ModelByteWindowError("missing matched development arms")


def run_bound_pilot(
    *, model_root: str | Path, manifest_json: str | Path,
    trusted_manifest_sha256: str, output_parent: str | Path,
    run_name: str, seed: int = 20261030, steps_per_cycle: int = 8,
    learning_rate: float = .002, execute: bool = False,
    _invoke: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run only after preflight; write an exclusive binding receipt on success.

    ``_invoke`` is a test seam; actual command-line operation always uses a
    fresh subprocess. The caller must separately protect the trusted digest.
    """
    if type(trusted_manifest_sha256) is not str or not _SHA.fullmatch(trusted_manifest_sha256):
        raise ModelByteWindowError("a separately trusted manifest SHA-256 is required")
    if (type(execute) is not bool or type(seed) is not int
            or type(steps_per_cycle) is not int or not 1 <= steps_per_cycle <= 32):
        raise ModelByteWindowError("invalid bounded training configuration")
    if type(learning_rate) is not float or not 0 < learning_rate <= .01:
        raise ModelByteWindowError("invalid learning rate")
    validate_fixtures()
    try:
        run_root = plan_output(output_parent, run_name)
    except (OSError, ValueError) as exc:
        raise ModelByteWindowError("unsafe or existing output run") from exc
    root = Path(model_root).absolute()
    manifest, _manifest_file_digest = _read_json(Path(manifest_json))
    _verify_model(root, manifest, trusted_manifest_sha256)
    runner_sha = _digest_file(_RUNNER)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "execute": execute,
        "model_root": str(root),
        "run_root": str(run_root),
        "model_manifest_sha256": trusted_manifest_sha256,
        "runner_sha256": runner_sha,
        "origin_attested": False,
        "actual_loader_file_reads_attested": False,
        "graduation_authorized": False,
    }
    if not execute:
        return summary

    args = [
        sys.executable, "-m", "scripts.qualify_minilm_cpu",
        "--output-parent", str(Path(output_parent).resolve(strict=True)),
        "--run-name", run_name, "--model-id", str(root),
        "--seed", str(seed), "--steps-per-cycle", str(steps_per_cycle),
        "--learning-rate", str(learning_rate), "--execute",
    ]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    executor = _invoke or subprocess.run
    try:
        completed = executor(args, cwd=str(_REPO), env=env, text=True,
                             capture_output=True, timeout=900, check=False)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        raise ModelByteWindowError("isolated MiniLM pilot process failed") from exc
    if completed.returncode != 0:
        raise ModelByteWindowError("isolated MiniLM pilot subprocess failed")
    _require_output_root(run_root)
    report_path = run_root / "report.json"
    report, report_sha = _read_json(report_path)
    _check_report(report, model_root=root, runner_sha=runner_sha,
                  seed=seed, steps=steps_per_cycle, rate=learning_rate)
    if _digest_file(_RUNNER) != runner_sha:
        raise ModelByteWindowError("pilot source changed during execution")
    if _digest_file(report_path) != report_sha:
        raise ModelByteWindowError("pilot report mutated after validation")
    _verify_model(root, manifest, trusted_manifest_sha256)
    _require_output_root(run_root)

    receipt = {
        **summary,
        "execute": True,
        "pilot_report_sha256": report_sha,
        "pilot_report_schema": PILOT_SCHEMA,
        "preflight_manifest_verified": True,
        "postflight_manifest_verified": True,
        "qualification": "PUBLIC_DEVELOPMENT_BYTE_WINDOW_ONLY",
        "publisher_origin_attested": False,
        "actual_loader_file_reads_attested": False,
        "independent_final_exam": "NOT_RUN",
        "training_worker_identity_attested": False,
        "manifest_commitment_custody": "CALLER_SUPPLIED_NOT_VERIFIED",
    }
    with (run_root / "provenance_binding.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--trusted-manifest-sha256", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_bound_pilot(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"model-byte window failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
