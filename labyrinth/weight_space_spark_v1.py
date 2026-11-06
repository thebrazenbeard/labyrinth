"""Bounded source-only direct weight-space edit candidates for Labyrinth.

This is *not* gradient training or neural improvement evidence. An untrusted
model can suggest a numeric direction and columns, but only an external,
source-owned policy controls which exact physical tensors may be edited.
All bytes live in a new candidate genome; the parent is never edited.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any

from .campaign_full_state_v1 import (
    capture_campaign_state, save_campaign_state_once, verify_campaign_state,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_CLASS = "SOURCE_ONLY_DIRECT_WEIGHT_EDIT_NO_NEURAL_GAIN_PROOF"
_METHOD = "selected_columns_direction_subtraction_f32_v1"
_MAX_REASON = 1024


class WeightSpaceBoundaryError(ValueError):
    """A proposed direct edit is outside a source-owned physical boundary."""


@dataclass(frozen=True)
class WeightSpaceEditProposal:
    proposal_id: str
    parent_state_sha256: str
    public_train_source_sha256: str
    shard_name: str
    tensor_name: str
    selected_columns: tuple[int, ...]
    unit_direction: tuple[float, ...]
    gain: float
    rationale: str
    method: str = _METHOD


@dataclass(frozen=True)
class WeightSpaceEditPolicy:
    """Operator-owned bounds. This is not a cryptographic authorization."""

    allowed_shards: tuple[str, ...]
    allowed_tensor_names: tuple[str, ...]
    maximum_modified_fraction: float = 0.25
    maximum_relative_norm: float = 0.20
    maximum_file_bytes: int = 4 * 1024 * 1024


@dataclass(frozen=True)
class WeightSpaceEditReceipt:
    proposal_id: str
    proposal_sha256: str
    parent_state_sha256: str
    child_state_sha256: str
    child_manifest_path: Path
    changed_element_count: int
    tensor_element_count: int
    relative_edit_norm: float
    parent_unchanged: bool
    evidence_class: str = _CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _CLASS:
            raise WeightSpaceBoundaryError("direct-edit receipt cannot claim neural improvement")
        if self.parent_unchanged is not True:
            raise WeightSpaceBoundaryError("direct-edit receipt needs untouched parent")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise WeightSpaceBoundaryError("proposal contains non-canonical data") from exc


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise WeightSpaceBoundaryError("duplicate safetensors header field")
        result[key] = value
    return result


def _safe_name(value: object, label: str) -> str:
    if (type(value) is not str or not value or len(value) > 256
            or "\x00" in value or value.startswith(".")
            or Path(value).name != value or any(ch in value for ch in ("/", "\\", ":"))):
        raise WeightSpaceBoundaryError(f"invalid relative {label}")
    return value


def _validate(proposal: WeightSpaceEditProposal, policy: WeightSpaceEditPolicy) -> bytes:
    if type(proposal) is not WeightSpaceEditProposal or type(policy) is not WeightSpaceEditPolicy:
        raise WeightSpaceBoundaryError("exact proposal and operator policy required")
    if type(proposal.proposal_id) is not str or _ID.fullmatch(proposal.proposal_id) is None:
        raise WeightSpaceBoundaryError("invalid proposal ID")
    for label, value in (("parent state", proposal.parent_state_sha256),
                         ("public TRAIN source", proposal.public_train_source_sha256)):
        if type(value) is not str or _SHA.fullmatch(value) is None:
            raise WeightSpaceBoundaryError(f"invalid {label} hash")
    _safe_name(proposal.shard_name, "shard")
    _safe_name(proposal.tensor_name, "tensor")
    if (type(policy.allowed_shards) is not tuple
            or type(policy.allowed_tensor_names) is not tuple
            or not policy.allowed_shards or not policy.allowed_tensor_names
            or any(type(name) is not str or not name for name in
                   (*policy.allowed_shards, *policy.allowed_tensor_names))
            or proposal.shard_name not in policy.allowed_shards
            or proposal.tensor_name not in policy.allowed_tensor_names):
        raise WeightSpaceBoundaryError("tensor/shard not approved for source-only direct edit")
    if (proposal.method != _METHOD or type(proposal.rationale) is not str
            or not 1 <= len(proposal.rationale) <= _MAX_REASON
            or "\x00" in proposal.rationale):
        raise WeightSpaceBoundaryError("unsupported method or invalid rationale")
    if (type(proposal.selected_columns) is not tuple
            or not 1 <= len(proposal.selected_columns) <= 128
            or any(type(col) is not int or col < 0 for col in proposal.selected_columns)
            or len(set(proposal.selected_columns)) != len(proposal.selected_columns)
            or type(proposal.unit_direction) is not tuple
            or not 1 <= len(proposal.unit_direction) <= 8192
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in proposal.unit_direction)):
        raise WeightSpaceBoundaryError("invalid numeric direction or selected columns")
    norm = math.sqrt(sum(float(v) ** 2 for v in proposal.unit_direction))
    if not math.isfinite(norm) or not math.isclose(norm, 1, rel_tol=1e-5, abs_tol=1e-5):
        raise WeightSpaceBoundaryError("direction must be a finite unit vector")
    if (type(proposal.gain) not in (int, float) or not math.isfinite(proposal.gain)
            or not 0 < proposal.gain <= 2.0):
        raise WeightSpaceBoundaryError("finite positive bounded projection gain required")
    if (type(policy.maximum_file_bytes) is not int
            or not 128 <= policy.maximum_file_bytes <= 64 * 1024 * 1024
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
                   for v in (policy.maximum_modified_fraction, policy.maximum_relative_norm))):
        raise WeightSpaceBoundaryError("invalid operator physical edit budget")
    return _canonical({
        "schema": "labyrinth.weight-space-edit-proposal.v1",
        "proposal_id": proposal.proposal_id,
        "parent_state_sha256": proposal.parent_state_sha256,
        "public_train_source_sha256": proposal.public_train_source_sha256,
        "shard_name": proposal.shard_name, "tensor_name": proposal.tensor_name,
        "selected_columns": proposal.selected_columns,
        "unit_direction": proposal.unit_direction, "gain": proposal.gain,
        "rationale": proposal.rationale, "method": proposal.method,
    })


def _edited_tensor_file(data: bytes, proposal: WeightSpaceEditProposal,
                        policy: WeightSpaceEditPolicy) -> tuple[bytes, int, int, float]:
    """Only modify selected F32 matrix columns; retain every other file byte."""
    if not 16 <= len(data) <= policy.maximum_file_bytes:
        raise WeightSpaceBoundaryError("safetensors shard exceeds bounded pilot size")
    hsize = struct.unpack_from("<Q", data)[0]
    if not 2 <= hsize <= min(16 * 1024 * 1024, len(data) - 9):
        raise WeightSpaceBoundaryError("invalid safetensors header size")
    try:
        header = json.loads(data[8:8 + hsize], object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightSpaceBoundaryError("invalid safetensors header") from exc
    if type(header) is not dict or proposal.tensor_name not in header:
        raise WeightSpaceBoundaryError("target tensor absent from physical shard")
    spec = header[proposal.tensor_name]
    if (type(spec) is not dict or set(spec) != {"dtype", "shape", "data_offsets"}
            or spec["dtype"] != "F32"
            or type(spec["shape"]) is not list or len(spec["shape"]) != 2
            or any(type(v) is not int or v <= 0 for v in spec["shape"])
            or type(spec["data_offsets"]) is not list
            or len(spec["data_offsets"]) != 2):
        raise WeightSpaceBoundaryError("pilot supports an exact F32 rank-two tensor only")
    rows, width = spec["shape"]
    first, last = spec["data_offsets"]
    if (type(first) is not int or type(last) is not int
            or first < 0 or last - first != rows * width * 4
            or 8 + hsize + last > len(data)):
        raise WeightSpaceBoundaryError("tensor offsets do not match physical F32 matrix")
    if (len(proposal.unit_direction) != rows
            or max(proposal.selected_columns) >= width
            or len(proposal.selected_columns) / width > policy.maximum_modified_fraction):
        raise WeightSpaceBoundaryError("proposal does not fit approved matrix shape/budget")
    position = 8 + hsize + first
    matrix = list(struct.unpack_from(f"<{rows * width}f", data, position))
    if any(not math.isfinite(v) for v in matrix):
        raise WeightSpaceBoundaryError("input tensor contains non-finite weights")
    updated = matrix[:]
    for col in proposal.selected_columns:
        projection = sum(float(proposal.unit_direction[row]) * matrix[row * width + col]
                         for row in range(rows))
        for row in range(rows):
            index = row * width + col
            updated[index] = matrix[index] - proposal.gain * float(
                proposal.unit_direction[row]) * projection
    try:
        packed = struct.pack(f"<{rows * width}f", *updated)
    except (OverflowError, struct.error) as exc:
        raise WeightSpaceBoundaryError("edit overflowed F32 tensor representation") from exc
    actual = list(struct.unpack(f"<{rows * width}f", packed))
    if any(not math.isfinite(v) for v in actual):
        raise WeightSpaceBoundaryError("edit produced non-finite weights")
    changed = sum(a != b for a, b in zip(actual, matrix))
    if changed == 0:
        raise WeightSpaceBoundaryError("direct edit made no physical weight change")
    if changed / len(matrix) > policy.maximum_modified_fraction:
        raise WeightSpaceBoundaryError("direct edit changed too many F32 entries")
    old_norm = math.sqrt(sum(v * v for v in matrix))
    delta_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, matrix)))
    relative = delta_norm / max(old_norm, 1e-12)
    if not math.isfinite(relative) or relative > policy.maximum_relative_norm:
        raise WeightSpaceBoundaryError("direct edit norm exceeds operator budget")
    return data[:position] + packed + data[position + len(packed):], changed, len(matrix), relative


def create_weight_space_candidate(
    *, parent_manifest_path: Path, candidate_root: Path,
    proposal: WeightSpaceEditProposal, policy: WeightSpaceEditPolicy,
) -> WeightSpaceEditReceipt:
    """Copy verified W0/Wt to a new genome and apply a bounded gradient-free edit.

    The caller-supplied policy is a DEVELOPMENT approval, not a secure signer.
    Failed candidates are source-only and are never promoted automatically.
    """
    proposal_data = _validate(proposal, policy)
    if (not isinstance(parent_manifest_path, Path) or not parent_manifest_path.is_absolute()
            or not isinstance(candidate_root, Path) or not candidate_root.is_absolute()
            or candidate_root.exists() or candidate_root.is_symlink()
            or not candidate_root.parent.is_dir()
            or candidate_root.parent != candidate_root.parent.resolve(strict=True)):
        raise WeightSpaceBoundaryError("canonical existing parent and fresh candidate output required")
    verify_campaign_state(parent_manifest_path, fresh_process=True)
    parent = json.loads(parent_manifest_path.read_bytes(), object_pairs_hook=_unique_pairs)
    if (proposal.parent_state_sha256 != parent["state_sha256"]
            or parent["family"] not in ("hf_safetensors_full", "hf_safetensors_lora")):
        raise WeightSpaceBoundaryError("proposal not bound to exact supported Wt genome")
    resolved_output = candidate_root.resolve(strict=False)
    if resolved_output != candidate_root:
        raise WeightSpaceBoundaryError("candidate output cannot traverse symlink ancestors")
    for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
        item = parent[role]
        if item is not None:
            source = Path(item["root"])
            if resolved_output == source or source in resolved_output.parents:
                raise WeightSpaceBoundaryError("candidate root overlaps the parent genome")
    target_role = "base" if parent["mutation_kind"] == "full_weight_edit" else "adapter"
    if proposal.shard_name not in parent[target_role]["files"]:
        raise WeightSpaceBoundaryError("proposed shard not in parent manifest inventory")
    parent_shard = Path(parent[target_role]["root"]) / proposal.shard_name
    if parent_shard.is_symlink() or not parent_shard.is_file():
        raise WeightSpaceBoundaryError("symlink or missing tensor shard refused")
    if parent_shard.stat().st_size > policy.maximum_file_bytes:
        raise WeightSpaceBoundaryError("proposed tensor shard exceeds bounded memory budget")
    old = parent_shard.read_bytes()
    new, changed, total, relative = _edited_tensor_file(old, proposal, policy)
    # Validate the complete parent AGAIN after reading the edit source. No
    # direct edit writes to any parent or shared model file.
    verify_campaign_state(parent_manifest_path, fresh_process=True)
    # mkdir is an exclusive atomic claim; a rename of a directory could
    # otherwise replace a concurrently created empty user directory on POSIX.
    # After a failed copy, leave this newly owned, unqualified directory in
    # place for analysis. Never delete or overwrite a possible user artifact.
    try:
        candidate_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise WeightSpaceBoundaryError("candidate destination claimed concurrently") from exc
    for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
        member = parent[role]
        if member is not None:
            shutil.copytree(Path(member["root"]), candidate_root / role)
    shutil.copy2(Path(parent["data_cursor"]["path"]), candidate_root / "cursor.json")
    destination = candidate_root / target_role / proposal.shard_name
    with destination.open("wb") as stream:
        stream.write(new)
        stream.flush()
        os.fsync(stream.fileno())
    child = capture_campaign_state(
        base_dir=candidate_root / "base",
        adapter_dir=(candidate_root / "adapter"
                     if parent["mutation_kind"] == "lora_adapter" else None),
        optimizer_dir=candidate_root / "optimizer",
        scheduler_dir=candidate_root / "scheduler",
        rng_dir=candidate_root / "rng",
        data_cursor_path=candidate_root / "cursor.json",
        family=parent["family"], mutation_kind=parent["mutation_kind"],
        parent_manifest=parent,
    )
    if child["state_sha256"] == parent["state_sha256"]:
        raise WeightSpaceBoundaryError("physical successor equals Wt parent")
    child_path = candidate_root / "manifest.json"
    save_campaign_state_once(child, child_path)
    verify_campaign_state(child_path, fresh_process=True)
    verify_campaign_state(parent_manifest_path, fresh_process=True)
    return WeightSpaceEditReceipt(
        proposal.proposal_id,
        hashlib.sha256(b"labyrinth.weight-space-proposal.v1\0"
                       + proposal_data).hexdigest(),
        parent["state_sha256"], child["state_sha256"], child_path,
        changed, total, relative, True,
    )


__all__ = [
    "WeightSpaceBoundaryError", "WeightSpaceEditProposal",
    "WeightSpaceEditPolicy", "WeightSpaceEditReceipt",
    "create_weight_space_candidate",
]
