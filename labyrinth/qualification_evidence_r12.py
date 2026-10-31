"""Source-level public campaign and full-state byte readback, witness gated.

This is an opt-in four-arm *public-development* inspector, not a private
examiner or a credential service. Its three injected callbacks are trusted
host APIs by CONTRACT, not through an assertion made here:

* ``authenticate_witness`` verifies an independently witnessed, signed
  pre-outcome campaign commitment and recovery sequence;
* ``read_authoritative_sequence`` reads a non-rollbackable external head;
* ``reserve_witness_once`` atomically consumes the witness at that head.

No local SQLite, in-process signature flag, clock, or hash can implement those
authorities. A copied old host database + matching old bytes can still replay
unless the external provider correctly enforces those contracts. File-system
readback is susceptible to a writer changing files after inspection; only a
host-owned immutable artifact store and lease can eliminate that race.

The scorer is intentionally the R11 FOUR arm scorer. PR #62's opt-in FIVE
arm runner has different arm identities; do NOT reinterpret a five-arm report
as an equivalent four-arm study or as a qualification of those controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointReceipt, SCHEMA,
    _context_json, _manifest_digest, _safe_snapshot_dir, _strict_json,
)
from labyrinth.preregistered_minilm_campaign import (
    PublicCampaignScore, PublicMiniLMCampaign, ReportReceipt,
    score_public_minilm_campaign,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_MAX_TENSOR_BYTES = 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 8 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3FN": 1, "F8_E5M2": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}


@dataclass(frozen=True)
class WitnessRecord:
    """Signed EXTERNAL claims; records themselves are NOT authentication."""

    witness_id: str
    authority_id: str
    campaign_sha256: str
    origin_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    precommit_sequence: int
    first_outcome_sequence: int
    recovery_sequence: int


@dataclass(frozen=True)
class PublicRecoveryEvidence:
    status: str
    public_score: PublicCampaignScore
    witness_sequence: int
    origin_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    snapshot_bytes_checked: bool
    trainable_bytes_changed: bool
    limitations: tuple[str, ...]
    physical_model_verified: bool = False
    examiner_custody_verified: bool = False
    graduation_authorized: bool = False


def _identifier(value: Any) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _digest(value: Any) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors header key")
        result[key] = value
    return result


def _model_tensor_names(manifest: dict[str, Any]) -> dict[str, str]:
    """Extract trainable-name -> tensor-key from checkpoint model_tree."""
    tree = manifest.get("model_tree")
    names = manifest.get("trainable_names")
    if (type(tree) is not dict or set(tree) != {"type", "items"}
            or tree["type"] != "dict" or type(tree["items"]) is not list
            or type(names) is not list or not names
            or any(type(name) is not str or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("malformed checkpoint trainable inventory")
    selected: dict[str, str] = {}
    for item in tree["items"]:
        if (type(item) is not list or len(item) != 2
                or type(item[0]) is not dict
                or item[0].keys() != {"type", "value"}
                or item[0]["type"] != "scalar"
                or type(item[0]["value"]) is not str
                or type(item[1]) is not dict
                or item[1].keys() != {"type", "key"}
                or item[1]["type"] != "tensor"
                or type(item[1]["key"]) is not str
                or item[0]["value"] in selected):
            raise ValueError("malformed trainable model tensor tree")
        selected[item[0]["value"]] = item[1]["key"]
    if set(selected) != set(names) or len(set(selected.values())) != len(selected):
        raise ValueError("trainable model tensor inventory does not match manifest")
    return selected


def _tensor_byte_fingerprints(raw: bytes, model_keys: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Hash actual safetensors model byte ranges; no parameter restore claimed."""
    if not 8 < len(raw) <= _MAX_TENSOR_BYTES:
        raise ValueError("invalid bounded safetensors payload")
    size = int.from_bytes(raw[:8], "little")
    if not 2 <= size <= _MAX_HEADER_BYTES or 8 + size >= len(raw):
        raise ValueError("invalid safetensors header length")
    try:
        header = json.loads(raw[8:8 + size].decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("malformed safetensors header") from exc
    if type(header) is not dict:
        raise ValueError("malformed safetensors header mapping")
    payload = raw[8 + size:]
    selected: dict[str, str] = {}
    intervals = []
    for key, entry in header.items():
        if key == "__metadata__":
            if type(entry) is not dict:
                raise ValueError("malformed safetensors metadata")
            continue
        if (type(key) is not str or type(entry) is not dict
                or set(entry) != {"dtype", "shape", "data_offsets"}
                or type(entry["dtype"]) is not str
                or entry["dtype"] not in _DTYPE_BYTES
                or type(entry["shape"]) is not list
                or len(entry["shape"]) > 64
                or any(type(dim) is not int or dim < 0 for dim in entry["shape"])
                or type(entry["data_offsets"]) is not list
                or len(entry["data_offsets"]) != 2):
            raise ValueError("malformed safetensors tensor header")
        start, end = entry["data_offsets"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(payload)):
            raise ValueError("invalid safetensors tensor byte range")
        elements = 1
        for dim in entry["shape"]:
            elements *= dim
            if elements > _MAX_TENSOR_BYTES:
                raise ValueError("safetensors tensor element limit exceeded")
        if (end - start) != elements * _DTYPE_BYTES[entry["dtype"]]:
            raise ValueError("safetensors dtype/shape byte length mismatch")
        intervals.append((start, end))
        selected[key] = sha256(payload[start:end]).hexdigest()
    if (not intervals or
            sorted(intervals)[0][0] != 0 or
            sorted(intervals)[-1][1] != len(payload) or
            any(left[1] != right[0]
                for left, right in zip(sorted(intervals), sorted(intervals)[1:]))):
        raise ValueError("safetensors byte ranges overlap or leave gaps")
    if not set(model_keys.values()) <= set(selected):
        raise ValueError("missing trainable tensor bytes in checkpoint")
    return tuple(sorted((name, selected[key]) for name, key in model_keys.items()))


def _snapshot_fingerprints(
    directory: Path, receipt: FullCheckpointReceipt,
) -> tuple[tuple[str, str], ...]:
    if (type(receipt) is not FullCheckpointReceipt
            or type(receipt.context) is not FullCheckpointContext
            or receipt.graduation_authorized is not False
            or receipt.optimizer != "AdamW" or not _digest(receipt.sha256)):
        raise ValueError("nonprovisional full-state receipt required")
    receipt.context.validate()
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ValueError("absolute checkpoint artifact path required")
    root = _safe_snapshot_dir(directory, existing=True)
    manifest_path = root / "manifest.json"
    tensor_path = root / "state.safetensors"
    if (not 0 < manifest_path.stat().st_size <= _MAX_MANIFEST_BYTES
            or not 0 < tensor_path.stat().st_size <= _MAX_TENSOR_BYTES):
        raise ValueError("checkpoint byte budget invalid")
    manifest_bytes = manifest_path.read_bytes()
    if _manifest_digest(manifest_bytes) != receipt.sha256:
        raise ValueError("checkpoint manifest digest differs from receipt")
    manifest = _strict_json(manifest_bytes)
    if (manifest.get("schema") != SCHEMA
            or manifest.get("optimizer") != "AdamW"
            or manifest.get("context") != _context_json(receipt.context)
            or manifest.get("trainable_names") != list(receipt.trainable_parameters)
            or not _digest(manifest.get("tensor_sha256"))):
        raise ValueError("checkpoint context, optimizer or inventory mismatch")
    raw = tensor_path.read_bytes()
    if sha256(raw).hexdigest() != manifest["tensor_sha256"]:
        raise ValueError("checkpoint tensor digest mismatch")
    return _tensor_byte_fingerprints(raw, _model_tensor_names(manifest))


def _validate_record(
    witness: WitnessRecord, campaign: PublicMiniLMCampaign,
    origin: FullCheckpointReceipt, successor: FullCheckpointReceipt,
    authority: str,
) -> None:
    if (type(witness) is not WitnessRecord
            or not _identifier(witness.witness_id)
            or not _identifier(witness.authority_id)
            or not _identifier(authority)
            or witness.authority_id != authority):
        raise ValueError("invalid witness authority identity")
    if (not _digest(witness.campaign_sha256)
            or not _digest(witness.origin_checkpoint_sha256)
            or not _digest(witness.successor_checkpoint_sha256)
            or witness.campaign_sha256 != campaign.sha256
            or witness.origin_checkpoint_sha256 != origin.sha256
            or witness.successor_checkpoint_sha256 != successor.sha256):
        raise ValueError("witness does not bind exact campaign and checkpoint digests")
    sequences = (witness.precommit_sequence, witness.first_outcome_sequence,
                 witness.recovery_sequence)
    if (any(type(value) is not int or not 0 <= value < 2**63-1
            for value in sequences)
            or not sequences[0] < sequences[1] < sequences[2]):
        raise ValueError("pre-outcome witness sequence and recovery order required")
    if (origin.graduation_authorized or successor.graduation_authorized
            or origin.context.run_id != successor.context.run_id
            or origin.context.objective_id != successor.context.objective_id
            or origin.context.base_checkpoint_sha256 != successor.context.base_checkpoint_sha256
            or origin.context.exposures != successor.context.exposures
            or successor.context.prior_checkpoint_sha256 != origin.sha256
            or successor.context.fence < origin.context.fence
            or successor.context.cursor <= origin.context.cursor):
        raise ValueError("provisional successor full-state lineage required")


def inspect_witnessed_public_recovery(
    *, campaign: PublicMiniLMCampaign, reports: Iterable[ReportReceipt],
    origin_receipt: FullCheckpointReceipt, origin_dir: Path,
    successor_receipt: FullCheckpointReceipt, successor_dir: Path,
    witness: WitnessRecord, expected_authority_id: str,
    authenticate_witness: Callable[[WitnessRecord], bool] | None,
    read_authoritative_sequence: Callable[[str], int] | None,
    reserve_witness_once: Callable[[str, str, int], bool] | None,
) -> PublicRecoveryEvidence:
    """Read actual public report/checkpoint bytes, then consume external ticket.

    A PASS here establishes only coherent source-level *development* readback.
    A caller must ensure independent, immutable host witness verification;
    this function cannot prove callbacks were not supplied by the trainee.
    """
    if (type(campaign) is not PublicMiniLMCampaign
            or not callable(authenticate_witness)
            or not callable(read_authoritative_sequence)
            or not callable(reserve_witness_once)):
        raise ValueError("campaign and three trusted host witness callbacks required")
    if type(origin_receipt) is not FullCheckpointReceipt or type(successor_receipt) is not FullCheckpointReceipt:
        raise ValueError("full checkpoint receipts required")
    _validate_record(witness, campaign, origin_receipt, successor_receipt,
                     expected_authority_id)
    if authenticate_witness(witness) is not True:
        raise ValueError("independent host witness authentication failed")
    public = score_public_minilm_campaign(
        campaign, expected_campaign_sha256=witness.campaign_sha256,
        reports=reports,
    )
    if (public.status != "PROVISIONAL_PUBLIC_DEVELOPMENT"
            or public.graduation_authorized
            or public.physical_model_verified
            or public.evaluator_custody_verified):
        raise ValueError("public score cannot assert independent qualification")
    before = _snapshot_fingerprints(origin_dir, origin_receipt)
    after = _snapshot_fingerprints(successor_dir, successor_receipt)
    if before == after:
        raise ValueError("checkpoint trainable bytes unchanged across recovery")
    if {name for name, _ in before} != {name for name, _ in after}:
        raise ValueError("checkpoint trainable inventory changed")
    head = read_authoritative_sequence(expected_authority_id)
    if type(head) is not int or head != witness.recovery_sequence:
        raise ValueError("stale or rollback-prone authoritative witness head")
    if reserve_witness_once(witness.witness_id, witness.campaign_sha256, head) is not True:
        raise ValueError("witness replay or atomic reservation denied")
    return PublicRecoveryEvidence(
        status="WITNESSED_PUBLIC_DEVELOPMENT_ONLY",
        public_score=public, witness_sequence=head,
        origin_checkpoint_sha256=origin_receipt.sha256,
        successor_checkpoint_sha256=successor_receipt.sha256,
        snapshot_bytes_checked=True, trainable_bytes_changed=True,
        limitations=(
            "Trusted host witness origin, independent time and monotonicity cannot be proven in-process",
            "Static checkpoint bytes and public reports do not prove physical optimizer execution",
            "No independent private examiner, protected heldout, graduation or deployment",
            "Only legacy FOUR-arm public reports supported, not the distinct FIVE-arm PR62 pilot",
        ),
    )
