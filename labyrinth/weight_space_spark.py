"""Opt-in, SOURCE-ONLY sparse weight-edit *proposal*; never edits weights.

Inspired by Takehi et al. (2026-11-02), https://rikiyat.github.io/blog/weight-space-programming/
Experiment 8 logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math

Original implementation; no upstream code is included. Tensor digests and a
model digest are asserted metadata: this module neither reads checkpoint bytes
nor verifies old values against real tensors. A caller must independently pin
the authentic W0 manifest and model digest, and persist anti-replay state.
No model execution, training, patch application, eval/gold access, permission
grant, rollback, or independently observed improvement occurs here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence


SCHEMA = "labyrinth.weight-space-spark.proposal.v1"
MAX_EDIT_PPM = 10_000  # At most one percent of W0, even if caller asks for more.
MAX_EDIT_COUNT = 4096
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
_MAX_ABS = {
    "float16": 65504.0,
    "bfloat16": 3.3895313892515355e38,
    "float32": 3.4028234663852886e38,
    "float64": 1.7976931348623157e308,
}
_MANIFEST_KEYS = frozenset({"name", "shape", "dtype", "tensor_sha256"})
_EDIT_KEYS = frozenset({"tensor", "index", "old", "new"})
_PROPOSAL_KEYS = frozenset({
    "schema", "proposal_nonce_sha256", "source_model_sha256", "w0_manifest",
    "w0_manifest_sha256", "budget_ppm", "edits", "provenance",
    "rollback_receipt", "paired_control_receipt", "execution_authorized",
    "proposal_sha256",
})


def _require_hash(value: object, label: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("proposal is not canonical JSON") from exc


def _commit(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + _canonical(value)).hexdigest()


def _manifest_count(manifest: object) -> tuple[int, dict[str, tuple[tuple[int, ...], str]]]:
    if type(manifest) is not list or not manifest or len(manifest) > 20_000:
        raise ValueError("W0 must have a bounded, nonempty full tensor manifest")
    total = 0
    lookup: dict[str, tuple[tuple[int, ...], str]] = {}
    for entry in manifest:
        if type(entry) is not dict or frozenset(entry) != _MANIFEST_KEYS:
            raise ValueError("unexpected W0 tensor manifest fields")
        name = entry["name"]
        shape = entry["shape"]
        dtype = entry["dtype"]
        if type(name) is not str or not _NAME.fullmatch(name) or name in lookup:
            raise ValueError("invalid or duplicated W0 tensor name")
        if type(dtype) is not str or dtype not in _MAX_ABS:
            raise ValueError("W0 tensor must use a supported floating dtype")
        if (type(shape) is not list or not 1 <= len(shape) <= 8
                or any(type(axis) is not int or not 1 <= axis <= 2**31 for axis in shape)):
            raise ValueError("invalid W0 tensor shape")
        _require_hash(entry["tensor_sha256"], "W0 tensor")
        count = math.prod(shape)
        total += count
        if total > 2**63 - 1:
            raise ValueError("W0 tensor parameter count exceeds supported bound")
        lookup[name] = (tuple(shape), dtype)
    if list(lookup) != sorted(lookup):
        raise ValueError("full W0 manifest must be sorted by tensor name")
    return total, lookup


def w0_manifest_sha256(manifest: list[dict]) -> str:
    """Hash an explicitly sorted, full W0 tensor metadata manifest."""
    _manifest_count(manifest)
    return _commit(b"labyrinth-w0-tensor-manifest-v1", manifest)


def _expected_receipts(model_sha: str, manifest_sha: str) -> tuple[dict, dict]:
    return (
        {
            "kind": "restore_exact_w0", "target_model_sha256": model_sha,
            "target_manifest_sha256": manifest_sha, "status": "NOT_EXECUTED",
        },
        {
            "control": "unchanged_w0", "candidate": "sparse_edit_proposal",
            "control_model_sha256": model_sha, "status": "NOT_EVALUATED",
        },
    )


def _validate_fields(proposal: Mapping, *, model_sha256: str,
                     authoritative_manifest: list[dict],
                     seen_nonces: Collection[str], seen_proposals: Collection[str]) -> str:
    if type(proposal) is not dict or frozenset(proposal) != _PROPOSAL_KEYS:
        raise ValueError("weight-space proposal has missing or unexpected fields")
    if proposal["schema"] != SCHEMA or proposal["execution_authorized"] is not False:
        raise ValueError("proposal schema or non-execution boundary violated")
    _require_hash(model_sha256, "authoritative source model")
    _require_hash(proposal["source_model_sha256"], "source model")
    if proposal["source_model_sha256"] != model_sha256:
        raise ValueError("proposal is for the wrong W0 model")
    nonce = proposal["proposal_nonce_sha256"]
    _require_hash(nonce, "proposal nonce")
    _require_hash(proposal["proposal_sha256"], "proposal digest")
    if nonce in seen_nonces or proposal["proposal_sha256"] in seen_proposals:
        raise ValueError("replayed weight-space proposal or nonce")
    total, tensors = _manifest_count(authoritative_manifest)
    _manifest_count(proposal["w0_manifest"])
    if (proposal["w0_manifest"] != authoritative_manifest
            or proposal["w0_manifest_sha256"] != w0_manifest_sha256(authoritative_manifest)):
        raise ValueError("proposal W0 manifest does not match pinned tensor metadata")
    budget = proposal["budget_ppm"]
    if type(budget) is not int or not 1 <= budget <= MAX_EDIT_PPM:
        raise ValueError("sparse edit budget exceeds one percent or is invalid")
    edits = proposal["edits"]
    if type(edits) is not list or not 1 <= len(edits) <= MAX_EDIT_COUNT:
        raise ValueError("sparse proposal must contain a bounded number of edits")
    if len(edits) * 1_000_000 > total * budget:
        raise ValueError("sparse proposal exceeds its declared fraction of W0")
    coordinates: set[tuple[str, tuple[int, ...]]] = set()
    for edit in edits:
        if type(edit) is not dict or frozenset(edit) != _EDIT_KEYS:
            raise ValueError("unexpected sparse edit fields")
        tensor = edit["tensor"]
        if type(tensor) is not str or tensor not in tensors:
            raise ValueError("selected tensor absent from W0")
        shape, dtype = tensors[tensor]
        index = edit["index"]
        if (type(index) is not list or len(index) != len(shape)
                or any(type(axis) is not int or not 0 <= axis < limit
                       for axis, limit in zip(index, shape))):
            raise ValueError("tensor coordinate is out of bounds")
        coordinate = (tensor, tuple(index))
        if coordinate in coordinates:
            raise ValueError("duplicate tensor coordinate (colliding edit)")
        coordinates.add(coordinate)
        for field in ("old", "new"):
            number = edit[field]
            try:
                finite_and_bounded = (
                    type(number) in (int, float) and math.isfinite(number)
                    and abs(number) <= _MAX_ABS[dtype]
                )
            except OverflowError:
                finite_and_bounded = False
            if not finite_and_bounded:
                raise ValueError("sparse edit values must be finite and dtype-bounded")
        if edit["old"] == edit["new"]:
            raise ValueError("sparse edit may not be a no-op")
    provenance = proposal["provenance"]
    if (type(provenance) is not dict or frozenset(provenance) != {
        "method", "input_boundary", "proposer_receipt_sha256",
    } or provenance["method"] != "reasoned_sparse_weight_proposal"
            or provenance["input_boundary"] != "w0_metadata_only"):
        raise ValueError("unexpected provenance or TRAIN/heldout access claim")
    _require_hash(provenance["proposer_receipt_sha256"], "proposer provenance")
    rollback, control = _expected_receipts(model_sha256,
                                           proposal["w0_manifest_sha256"])
    if proposal["rollback_receipt"] != rollback or proposal["paired_control_receipt"] != control:
        raise ValueError("rollback or paired-control precommit does not match W0")
    content = {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    calculated = _commit(b"labyrinth-weight-space-spark-proposal-v1", content)
    if calculated != proposal["proposal_sha256"]:
        raise ValueError("weight-space proposal digest does not match its contents")
    return calculated


def verify_sparse_edit_proposal(
    proposal: Mapping, *, expected_source_model_sha256: str,
    expected_w0_manifest: list[dict],
    seen_proposal_nonces: Collection[str],
    seen_proposal_sha256: Collection[str],
) -> str:
    """Check a *proposal*, returning its digest; never approve/apply any edits.

    Both replay registries must be supplied by the caller. This function does
    not mutate them, persist its result, or establish independent authority.
    """
    return _validate_fields(
        proposal, model_sha256=expected_source_model_sha256,
        authoritative_manifest=expected_w0_manifest,
        seen_nonces=seen_proposal_nonces, seen_proposals=seen_proposal_sha256,
    )


def create_sparse_edit_proposal(
    *, source_model_sha256: str, w0_manifest: list[dict],
    proposal_nonce_sha256: str, proposer_receipt_sha256: str,
    edits: Sequence[dict], budget_ppm: int = MAX_EDIT_PPM,
) -> dict:
    """Construct a JSON-compatible, intentionally non-executable precommit."""
    _require_hash(source_model_sha256, "source model")
    manifest_sha = w0_manifest_sha256(w0_manifest)
    rollback, control = _expected_receipts(source_model_sha256, manifest_sha)
    # Round-trip copy prevents caller mutation of nested manifest or edits.
    artifact = {
        "schema": SCHEMA,
        "proposal_nonce_sha256": proposal_nonce_sha256,
        "source_model_sha256": source_model_sha256,
        "w0_manifest": json.loads(_canonical(w0_manifest)),
        "w0_manifest_sha256": manifest_sha,
        "budget_ppm": budget_ppm,
        "edits": json.loads(_canonical(list(edits))),
        "provenance": {
            "method": "reasoned_sparse_weight_proposal",
            "input_boundary": "w0_metadata_only",
            "proposer_receipt_sha256": proposer_receipt_sha256,
        },
        "rollback_receipt": rollback,
        "paired_control_receipt": control,
        "execution_authorized": False,
    }
    artifact["proposal_sha256"] = _commit(b"labyrinth-weight-space-spark-proposal-v1", artifact)
    verify_sparse_edit_proposal(
        artifact, expected_source_model_sha256=source_model_sha256,
        expected_w0_manifest=w0_manifest,
        seen_proposal_nonces=(), seen_proposal_sha256=(),
    )
    return artifact
