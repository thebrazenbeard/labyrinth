"""Source-only compatibility boundary for v3 LoRA and experimental v4 MeZO reports.

PR #62's resume bridge verifies *v3 AdamW LoRA* report and adapter bytes.
Draft PR #60 emits *v4 MeZO LoRA* reports but does not provide an integrated
v4 resume verifier here. A schema rename cannot constitute a migration.

This module classifies untrusted report *metadata*. Its sole positive result
permits calling the existing, stricter v3 byte/lineage preflight; it does not
verify model files, optimizer state, a source tree, independent custody, or
any physical learning result. In particular no supported report contains a
complete in-cycle restart bundle (weights, optimizer or direction state, RNG,
training cursor, and checkpoint fence).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


V3_SCHEMA = "labyrinth.hf-lora-pilot.v3"
V4_MEZO_SCHEMA = "labyrinth.hf-lora-pilot.v4"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CheckpointCompatibility:
    source_schema: str
    source_optimizer: str
    target_optimizer: str
    continuity: str
    status: str
    may_call_v3_preflight: bool
    optimizer_restart_supported: bool = False
    bytes_authenticated: bool = False
    graduation_authorized: bool = False


def _digest(value: Any, field: str) -> None:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _manifest(value: Any, field: str) -> None:
    if (type(value) is not dict or not value or len(value) > 128):
        raise ValueError(f"{field} must be a bounded nonempty digest map")
    for name, digest in value.items():
        if (type(name) is not str or not name or len(name) > 256
                or name.startswith("/") or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError(f"{field} contains an invalid file label")
        _digest(digest, f"{field}[{name}]")


def classify_checkpoint_report(
    report: dict[str, Any], *, continuity: str = "adapter_boundary",
    target_optimizer: str = "adamw",
) -> CheckpointCompatibility:
    """Classify an untrusted receipt without promoting it to verified lineage.

    ``adapter_boundary`` means a *completed-cycle* adapter may be considered
    for a new cycle only after the ordinary v3 ``preflight_resume`` verifies
    actual bytes, source pin, objective and cycle. ``mid_cycle`` requires a
    separate complete optimizer/RNG/cursor/fence checkpoint, unavailable in
    either report format. v4 MeZO remains blocked from this v3 bridge even
    when the next optimizer is also MeZO.
    """
    if continuity not in ("adapter_boundary", "mid_cycle") or type(continuity) is not str:
        raise ValueError("continuity must be adapter_boundary or mid_cycle")
    if target_optimizer not in ("adamw", "mezo") or type(target_optimizer) is not str:
        raise ValueError("target_optimizer must be adamw or mezo")
    if (type(report) is not dict or type(report.get("schema")) is not str
            or report.get("schema") not in (V3_SCHEMA, V4_MEZO_SCHEMA)):
        raise ValueError("unknown or malformed HF pilot report schema")
    schema = report["schema"]
    source = report.get("source")
    if type(source) is not dict:
        raise ValueError("missing source-tree metadata")
    _digest(source.get("source_tree_sha256"), "source_tree_sha256")

    config = report.get("config")
    if type(config) is not dict:
        raise ValueError("missing pilot configuration")
    for field in ("model_path", "output_dir"):
        value = config.get(field)
        if type(value) is not str or not value or len(value) > 4096:
            raise ValueError(f"missing or invalid config.{field}")
    for field in ("lora_rank", "lora_alpha"):
        value = config.get(field)
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError(f"missing or invalid config.{field}")
    if type(config.get("answer_only")) is not bool:
        raise ValueError("missing or invalid config.answer_only")

    if schema == V3_SCHEMA:
        # The v3 format has an implicit AdamW optimizer. A renamed v4 report
        # must not be imported by merely changing its schema string.
        if "optimizer" in config or "zo_epsilon" in config:
            raise ValueError("v3 AdamW report cannot contain MeZO optimizer metadata")
        optimizer = "adamw"
    else:
        if config.get("optimizer") != "mezo" or type(config.get("optimizer")) is not str:
            raise ValueError("v4 experimental report must declare MeZO optimizer")
        epsilon = config.get("zo_epsilon")
        if (type(epsilon) is not float or not math.isfinite(epsilon)
                or not 0 < epsilon <= 1):
            raise ValueError("v4 MeZO epsilon must be a bounded finite float")
        optimizer = "mezo"

    for field in ("base_weights_sha256", "model_metadata_sha256",
                  "candidate_adapter_sha256"):
        _manifest(report.get(field), field)
    for field in ("candidate_lora_parameter_sha256",
                  "candidate_lora_tensor_value_sha256",
                  "initial_frozen_parameter_sha256",
                  "initial_persistent_buffers_sha256"):
        _digest(report.get(field), field)
    if report["candidate_lora_parameter_sha256"] != report["candidate_lora_tensor_value_sha256"]:
        raise ValueError("saved LoRA value digest contradicts training digest")

    if continuity == "mid_cycle":
        status = "MID_CYCLE_STATE_UNAVAILABLE"
    elif optimizer != target_optimizer:
        status = "CROSS_OPTIMIZER_MIGRATION_REQUIRED"
    elif schema == V4_MEZO_SCHEMA:
        status = "V4_MEZO_NATIVE_PREFLIGHT_REQUIRED"
    else:
        status = "V3_ADAPTER_PREFLIGHT_REQUIRED"
    return CheckpointCompatibility(
        source_schema=schema,
        source_optimizer=optimizer,
        target_optimizer=target_optimizer,
        continuity=continuity,
        status=status,
        may_call_v3_preflight=(status == "V3_ADAPTER_PREFLIGHT_REQUIRED"),
    )


def require_v3_adapter_preflight(
    report: dict[str, Any], *, target_optimizer: str = "adamw",
) -> CheckpointCompatibility:
    """Reject unsupported imports; callers MUST still run real v3 preflight.

    Returning this plan is never an authorization to load a checkpoint on its
    own. In particular v4 MeZO needs a separate source-and-bytes-qualified
    native loader and cannot be mechanically migrated to v3.
    """
    result = classify_checkpoint_report(report, target_optimizer=target_optimizer)
    if not result.may_call_v3_preflight:
        if result.source_schema == V4_MEZO_SCHEMA:
            raise ValueError("v4 MeZO report requires native adapter requalification, not v3 renaming")
        raise ValueError("cross-optimizer migration cannot reuse v3 adapter preflight")
    return result
