"""Preserve two real *toy* CPU probe reports as one non-authorizing receipt.

This opt-in tool only runs when ``--execute`` is supplied. Its two producers
are the R25 weight-update and R22 full-state checkpoint *toy* probes. The
receipt records their original JSON reports, with source/run identifiers
provided by the caller. It is not a pretrained-model, examiner, consent,
provider-identity, antirollback, or host-execution attestation.

A receiving verifier must supply the receipt digest and every expectation
from independently controlled state. Recomputing the SHA of an attacker-
controlled document does NOT authenticate it. Neither a passing inspection
nor a signed-looking digest grants any effect or promotion authority.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from scripts.qualify_v2_checkpoint_cpu_r22 import STATUS as CHECKPOINT_STATUS
from scripts.qualify_v2_weight_updates_cpu_r25 import STATUS as WEIGHT_STATUS


SCHEMA = "labyrinth.toy-cpu-evidence-receipt.v1"
STATUS = "SOURCE_ONLY_NONAUTHORIZING"
MAX_RECEIPT_BYTES = 32 * 1024
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_WEIGHT_KEYS = frozenset({
    "status", "physical_cpu_toy_executed",
    "actual_trainable_tensor_updates_both_cycles", "frozen_base_weights_unchanged",
    "adamw_moments_advanced_both_cycles", "frozen_tensor_sha256",
    "trainable_tensor_sha256", "optimizer_moments_sha256", "optimizer_steps",
    "pretrained_model_loaded", "host_custody_verified",
    "neural_generalization_verified", "graduation_authorized",
    "persistent_host_artifacts",
})
_CHECKPOINT_KEYS = frozenset({
    "status", "physical_cpu_toy_executed", "pretrained_model_loaded",
    "maze_training_prompts_executed",
    "optimizer_fullstate_continuity_matches_uninterrupted_toy",
    "final_fresh_fullstate_restore_verified", "checkpoint_cycle_receipt_sha256",
    "toy_final_state_sha256", "synthetic_identity_pins",
    "independent_host_custody_verified", "independent_antirollback_verified",
    "neural_generalization_verified", "graduation_authorized",
    "persistent_host_artifacts",
})
_RECEIPT_KEYS = frozenset({
    "schema", "status", "source_head", "source_tree", "run_id",
    "weight_report", "checkpoint_report", "graduation_authorized",
    "deployment_authorized", "identity_consent_transfer_authorized",
    "host_custody_verified", "pretrained_model_learning_verified",
})


class EvidenceError(ValueError):
    """Non-authorizing toy evidence failed strict source consistency checks."""


def _require(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise EvidenceError(f"invalid toy evidence flag: {label}")


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise EvidenceError(f"invalid SHA-256: {label}")
    return value


def _digest_list(value: Any, count: int, label: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) != count:
        raise EvidenceError(f"invalid digest inventory: {label}")
    return tuple(_digest(item, label) for item in value)


def _pins(head: Any, tree: Any, run: Any) -> None:
    if type(head) is not str or _HEX40.fullmatch(head) is None:
        raise EvidenceError("source head must be an exact lowercase SHA-1")
    if type(tree) is not str or _HEX40.fullmatch(tree) is None:
        raise EvidenceError("source tree must be an exact lowercase SHA-1")
    if type(run) is not str or _RUN.fullmatch(run) is None:
        raise EvidenceError("invalid run identifier")


def _validate_weight(report: Any) -> None:
    if type(report) is not dict or set(report) != _WEIGHT_KEYS:
        raise EvidenceError("invalid weight report fields")
    if type(report["status"]) is not str or report["status"] != WEIGHT_STATUS:
        raise EvidenceError("invalid weight report status")
    for key in ("physical_cpu_toy_executed", "actual_trainable_tensor_updates_both_cycles",
                "frozen_base_weights_unchanged", "adamw_moments_advanced_both_cycles"):
        _require(report[key], True, key)
    for key in ("pretrained_model_loaded", "host_custody_verified",
                "neural_generalization_verified", "graduation_authorized",
                "persistent_host_artifacts"):
        _require(report[key], False, key)
    frozen = _digest_list(report["frozen_tensor_sha256"], 3, "frozen")
    trained = _digest_list(report["trainable_tensor_sha256"], 3, "trainable")
    moments = _digest_list(report["optimizer_moments_sha256"], 2, "moments")
    if frozen[0] != frozen[1] or frozen[1] != frozen[2]:
        raise EvidenceError("frozen model changed across toy cycles")
    if trained[0] == trained[1] or trained[1] == trained[2]:
        raise EvidenceError("trainable model did not change in both toy cycles")
    if moments[0] == moments[1]:
        raise EvidenceError("AdamW toy moments did not advance")
    if type(report["optimizer_steps"]) is not list or any(
        type(step) is not int for step in report["optimizer_steps"]
    ) or report["optimizer_steps"] != [4, 8]:
        raise EvidenceError("unexpected toy training step trajectory")


def _validate_checkpoint(report: Any) -> None:
    if type(report) is not dict or set(report) != _CHECKPOINT_KEYS:
        raise EvidenceError("invalid checkpoint report fields")
    if type(report["status"]) is not str or report["status"] != CHECKPOINT_STATUS:
        raise EvidenceError("invalid checkpoint report status")
    for key in ("physical_cpu_toy_executed", "maze_training_prompts_executed",
                "optimizer_fullstate_continuity_matches_uninterrupted_toy",
                "final_fresh_fullstate_restore_verified", "synthetic_identity_pins"):
        _require(report[key], True, key)
    for key in ("pretrained_model_loaded", "independent_host_custody_verified",
                "independent_antirollback_verified", "neural_generalization_verified",
                "graduation_authorized", "persistent_host_artifacts"):
        _require(report[key], False, key)
    checkpoints = _digest_list(report["checkpoint_cycle_receipt_sha256"], 2,
                               "checkpoint cycle")
    if checkpoints[0] == checkpoints[1]:
        raise EvidenceError("checkpoint did not advance across toy cycles")
    _digest(report["toy_final_state_sha256"], "final toy state")


def canonical_bytes(value: Any) -> bytes:
    try:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise EvidenceError("unserializable toy evidence") from exc
    if len(raw) > MAX_RECEIPT_BYTES:
        raise EvidenceError("oversized toy evidence")
    return raw


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate toy evidence key")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Any:
    if type(raw) is not bytes or len(raw) > MAX_RECEIPT_BYTES:
        raise EvidenceError("invalid or oversized evidence bytes")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              EvidenceError("nonfinite JSON constant")))
    except (ValueError, UnicodeError, RecursionError, TypeError) as exc:
        raise EvidenceError("invalid JSON toy evidence") from exc


def _validate_receipt(value: Any) -> None:
    if type(value) is not dict or set(value) != _RECEIPT_KEYS:
        raise EvidenceError("invalid toy receipt fields")
    if type(value["schema"]) is not str or value["schema"] != SCHEMA:
        raise EvidenceError("unsupported toy receipt schema")
    if type(value["status"]) is not str or value["status"] != STATUS:
        raise EvidenceError("toy receipt must remain non-authorizing")
    _pins(value["source_head"], value["source_tree"], value["run_id"])
    _validate_weight(value["weight_report"])
    _validate_checkpoint(value["checkpoint_report"])
    for key in ("graduation_authorized", "deployment_authorized",
                "identity_consent_transfer_authorized", "host_custody_verified",
                "pretrained_model_learning_verified"):
        _require(value[key], False, key)


def create_receipt(weight_report: dict[str, Any], checkpoint_report: dict[str, Any],
                   *, source_head: str, source_tree: str, run_id: str) -> bytes:
    value = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_head": source_head,
        "source_tree": source_tree,
        "run_id": run_id,
        "weight_report": weight_report,
        "checkpoint_report": checkpoint_report,
        "graduation_authorized": False,
        "deployment_authorized": False,
        "identity_consent_transfer_authorized": False,
        "host_custody_verified": False,
        "pretrained_model_learning_verified": False,
    }
    _validate_receipt(value)
    return canonical_bytes(value)


def inspect_receipt(raw: bytes, *, expected_receipt_sha256: str,
                    expected_source_head: str, expected_source_tree: str,
                    expected_run_id: str) -> dict[str, Any]:
    """Inspect with independently supplied immutable pins; grant no effects."""
    _digest(expected_receipt_sha256, "independently pinned receipt")
    _pins(expected_source_head, expected_source_tree, expected_run_id)
    if type(raw) is not bytes or sha256(raw).hexdigest() != expected_receipt_sha256:
        raise EvidenceError("receipt differs from independently pinned digest")
    value = _strict_json(raw)
    if canonical_bytes(value) != raw:
        raise EvidenceError("receipt not in exact canonical JSON encoding")
    _validate_receipt(value)
    if (value["source_head"] != expected_source_head
            or value["source_tree"] != expected_source_tree
            or value["run_id"] != expected_run_id):
        raise EvidenceError("receipt identity differs from external pins")
    return {
        "status": STATUS,
        "receipt_sha256": expected_receipt_sha256,
        "physical_toy_evidence_consistent": True,
        "graduation_authorized": False,
        "deployment_authorized": False,
        "identity_consent_transfer_authorized": False,
        "host_custody_verified": False,
        "pretrained_model_learning_verified": False,
    }


def _run_probe(module: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--execute"],
        capture_output=True, text=True, check=True, timeout=180,
    )
    value = _strict_json(completed.stdout.encode("utf-8"))
    if type(value) is not dict:
        raise EvidenceError("physical toy probe did not report an object")
    return value


def _check_destination(target: Path) -> None:
    # Check before running expensive physical probes, then again immediately
    # before publication. The hard link remains the final no-overwrite fence.
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise EvidenceError("absolute, unoccupied toy output path required")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise EvidenceError("existing non-symlink toy output directory required")
    for ancestor in target.parents:
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise EvidenceError("symlink or junction in toy output ancestry")


def _publish_new(target: Path, raw: bytes) -> None:
    # The destination must be explicit and unoccupied; hard-link publication is
    # atomic and cannot silently overwrite a prior independently pinned run.
    _check_destination(target)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".toy-evidence-",
                                         dir=target.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        if os.name == "posix":
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise EvidenceError("could not atomically publish new toy receipt") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    _pins(args.source_head, args.source_tree, args.run_id)
    if not args.execute:
        print(json.dumps({"status": "PREFLIGHT_ONLY_NO_EFFECT",
                          "physical_toy_evidence_consistent": False,
                          "graduation_authorized": False}, sort_keys=True))
        return 0
    _check_destination(args.output)
    weights = _run_probe("scripts.qualify_v2_weight_updates_cpu_r25")
    checkpoints = _run_probe("scripts.qualify_v2_checkpoint_cpu_r22")
    raw = create_receipt(weights, checkpoints, source_head=args.source_head,
                         source_tree=args.source_tree, run_id=args.run_id)
    _publish_new(args.output, raw)
    print(json.dumps({"status": STATUS, "receipt_sha256": sha256(raw).hexdigest(),
                      "physical_toy_evidence_consistent": True,
                      "graduation_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
