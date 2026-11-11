"""Offline *file-backed* HF-LoRA training/reload receipt integrity gate.

No ML imports, model execution, network access, signatures or claims of
independent provenance. PASS means bytes and self-reported records agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hf_pilot import (
    EVAL_SCHEMA, _adapter_manifest, _checkpoint_manifest, _validate_sharded_index,
)
from .hf_report_audit import audit_hf_report
from .runner import _MAX_JSON_BYTES, _reject_nonfinite_json, _unique_json_pairs

SCHEMA = "labyrinth.hf-persistence-audit.v1"


def _read_regular_bounded(path: Path) -> bytes:
    """Read only a regular nonsymlink source; cap input even across growth."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("receipt must be a regular nonsymlink file")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("receipt exceeds 8 MiB input limit")
    return raw


def audit_hf_persistence(run_dir: str | Path, reload_report: str | Path,
                         base_model_dir: str | Path) -> dict[str, Any]:
    """Audit independently supplied checkpoint bytes and paired JSON records.

    The process IDs, optimizer/tensor hashes and outputs are *self-reported*;
    even matching files can be constructed without any model execution.
    """
    failures: list[str] = []
    unknown: list[str] = []
    base_verified = False
    adapter_verified = False

    def fail(reason: str) -> None:
        failures.append(reason)

    def result() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": "FAIL" if failures else "INCOMPLETE" if unknown else "PASS",
            "meaning": "file_backed_receipt_consistency_only",
            "base_bytes_verified": base_verified,
            "adapter_bytes_verified": adapter_verified,
            "independent_execution_attested": False,
            "reasoning_transfer_demonstrated": False,
            "failures": failures,
            "unchecked": unknown,
            "limitations": [
                "A self-consistent pair of forged receipts and checkpoint bytes can PASS.",
                "PID differences do not independently attest process separation.",
                "Adapter tensor-value digests are compared as reported, not decoded by this offline gate.",
                "No blind structurally novel evaluation or improved reasoning is demonstrated.",
            ],
        }

    root = Path(run_dir)
    base = Path(base_model_dir)
    source = root / "pilot_report.json"
    try:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("run directory must be a nonsymlink directory")
        if not base.is_dir() or base.is_symlink():
            raise ValueError("base model directory must be a nonsymlink directory")
        raw = _read_regular_bounded(source)
        training = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_pairs,
                              parse_constant=_reject_nonfinite_json)
        reload_path = Path(reload_report)
        reload_bytes = _read_regular_bounded(reload_path)
        # Parse only the exact bytes that passed bounded nonsymlink validation.
        # Reopening the path here can follow a swapped symlink or different file.
        reload = json.loads(reload_bytes.decode("utf-8"),
                            object_pairs_hook=_unique_json_pairs,
                            parse_constant=_reject_nonfinite_json)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError,
            json.JSONDecodeError) as exc:
        fail(f"invalid input receipt or directory: {exc}")
        return result()

    if not isinstance(training, dict) or not isinstance(reload, dict):
        fail("expected two JSON objects")
        return result()
    try:
        internal = audit_hf_report(training)
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
        fail(f"training receipt internal audit raised: {exc}")
        return result()
    if internal["status"] == "FAIL":
        fail("training receipt failed its internal four-arm audit")
    elif internal["status"] != "PASS":
        unknown.append("training receipt source or response fields not fully auditable")

    if reload.get("schema") != EVAL_SCHEMA:
        fail("reload receipt schema differs from supported exact version")
    if reload.get("training_report_sha256") != hashlib.sha256(raw).hexdigest():
        fail("reload record is not bound to exact training-report bytes")
    training_pid = training.get("training_process_id")
    reload_pid = reload.get("process_id")
    if (type(training_pid) is not int or training_pid <= 0
            or type(reload_pid) is not int or reload_pid <= 0
            or training_pid == reload_pid or reload.get("process_id_differs") is not True):
        fail("reported training and reload process IDs must be distinct positive integers")
    if reload.get("claimed_fresh_process", "missing") is not None:
        fail("unsupported independent-process claim in reload report")
    if reload.get("matches_original_candidate_trials") is not True:
        fail("reload record does not claim same-process trial reproduction")
    candidate = training.get("evaluations", {}).get("candidate") if isinstance(training.get("evaluations"), dict) else None
    if not isinstance(candidate, dict) or reload.get("evaluation") != candidate:
        fail("reload evaluation differs from complete recorded trained evaluation")
    if (reload.get("reloaded_lora_tensor_value_sha256")
            != training.get("candidate_lora_tensor_value_sha256")):
        fail("trained versus reloaded LoRA tensor-value digest mismatch")
    if reload.get("adapter_sha256") != training.get("candidate_adapter_sha256"):
        fail("reload checkpoint manifest differs from trained checkpoint manifest")

    try:
        weights = sorted(base.glob("*.safetensors"))
        if not weights or any(not p.is_file() or p.is_symlink() for p in weights):
            raise ValueError("base weights must be regular local safetensors files")
        _validate_sharded_index(base, weights)
        checkpoint = _checkpoint_manifest(base, weights)
        if (checkpoint["weights"] != training.get("base_weights_sha256")
                or checkpoint["metadata"] != training.get("model_metadata_sha256")):
            fail("base model/tokenizer bytes differ from training receipt")
        else:
            base_verified = True
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        fail(f"cannot verify base checkpoint bytes: {exc}")

    try:
        adapter = _adapter_manifest(root / "candidate_adapter")
        if adapter != training.get("candidate_adapter_sha256"):
            fail("adapter checkpoint bytes differ from training receipt")
        else:
            adapter_verified = True
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        fail(f"cannot verify adapter checkpoint bytes: {exc}")
    return result()
