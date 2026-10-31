"""Optional signed readback of a *pending* v2 neural checkpoint capture.

The R19 journal binds immediate predecessors locally. This separate contract
permits a caller to verify a signature and double readback of a checkpoint
source claim before committing that pending cycle. It deliberately performs no
reservation or training effect and supplies no host-owned signing service.

The expected host, public key, authority head, and readback callback are all
caller inputs. A trainer that controls those inputs can forge an internally
consistent result; real host custody and monotonic anti-rollback are UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
from pathlib import Path
import re
from typing import Callable

from .maze_crash_resume_journal import (
    MazeJournalBinding, MazeJournalError, _inspect, _snapshot,
)
from .neural_resume_checkpoint import FullCheckpointContextV2, _context_json


SCHEMA = "labyrinth.pending-checkpoint-source-proof.v2"
DOMAIN = b"labyrinth.pending-checkpoint-source-proof.v2\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_SIG = re.compile(r"[a-f0-9]{128}\Z", re.ASCII)


class SignedV2SourceError(ValueError):
    """An opt-in v2 checkpoint source/signature claim was refused."""


@dataclass(frozen=True)
class SignedPendingCheckpointV2:
    provider_id: str
    host_id: str
    run_id: str
    checkpoint_run_id: str
    objective_id: str
    arm: str
    candidate_artifact_sha256: str
    plan_sha256: str
    scope_sha256: str
    source_sha256: str
    base_checkpoint_sha256: str
    initial_checkpoint_sha256: str
    immediate_predecessor_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_tensor_sha256: str
    cycle_index: int
    fence: int
    authority_epoch: int
    previous_authority_head_sha256: str
    signature_hex: str
    schema: str = SCHEMA

    def validate(self) -> None:
        if type(self) is not SignedPendingCheckpointV2 or self.schema != SCHEMA:
            raise SignedV2SourceError("wrong signed v2 source schema")
        for name in ("provider_id", "host_id", "run_id", "checkpoint_run_id",
                     "objective_id", "arm"):
            value = getattr(self, name)
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise SignedV2SourceError(f"invalid signed v2 {name}")
        for name in ("candidate_artifact_sha256", "plan_sha256", "scope_sha256",
                     "source_sha256", "base_checkpoint_sha256",
                     "initial_checkpoint_sha256", "immediate_predecessor_sha256",
                     "checkpoint_manifest_sha256", "checkpoint_tensor_sha256",
                     "previous_authority_head_sha256"):
            value = getattr(self, name)
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise SignedV2SourceError(f"invalid signed v2 {name}")
        if (type(self.cycle_index) is not int or not 0 <= self.cycle_index <= 127
                or type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1
                or type(self.authority_epoch) is not int
                or not 1 <= self.authority_epoch < 2**63
                or type(self.signature_hex) is not str
                or _SIG.fullmatch(self.signature_hex) is None):
            raise SignedV2SourceError("invalid cycle, fence, epoch, or signature")

    def signed_bytes(self) -> bytes:
        self.validate()
        record = {field: getattr(self, field)
                  for field in self.__dataclass_fields__ if field != "signature_hex"}
        return DOMAIN + (json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class PendingCheckpointV2SourceReceipt:
    checkpoint_manifest_sha256: str
    immediate_predecessor_sha256: str
    authority_epoch: int
    source_bytes_rehashed: bool = True
    signed_source_claim_verified: bool = True
    neural_derivation_verified: bool = False
    host_custody_verified: bool = False
    rollback_resistance_verified: bool = False
    graduation_authorized: bool = False
    status: str = "SIGNED_PENDING_V2_SOURCE_ONLY"


def _verify_signature(proof: SignedPendingCheckpointV2, public_key: bytes) -> None:
    if type(public_key) is not bytes or len(public_key) != 32:
        raise SignedV2SourceError("external pinned Ed25519 public key required")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(proof.signature_hex), proof.signed_bytes())
    except (ImportError, InvalidSignature, ValueError, TypeError) as exc:
        raise SignedV2SourceError("signed v2 source signature refused") from exc


def verify_pending_checkpoint_source_v2(
    *,
    journal_root: Path,
    checkpoint_path: Path,
    binding: MazeJournalBinding,
    context: FullCheckpointContextV2,
    proof: SignedPendingCheckpointV2,
    expected_provider_id: str,
    expected_host_id: str,
    expected_plan_sha256: str,
    expected_authority_epoch: int,
    expected_previous_authority_head_sha256: str,
    trusted_external_public_key: bytes,
    read_external_proof: Callable[[str, int], SignedPendingCheckpointV2],
) -> PendingCheckpointV2SourceReceipt:
    """Verify signed v2 source bytes while journal intent is still pending.

    Call after full checkpoint save, before journal commit. The separately
    trusted privileged service would have to sign and persist this claim, then
    expose its receipt for repeated reads. Without one, this remains an
    injection-tested source check and MUST NOT be used as a host permit.
    """
    if (type(binding) is not MazeJournalBinding
            or type(context) is not FullCheckpointContextV2
            or type(proof) is not SignedPendingCheckpointV2
            or not callable(read_external_proof)):
        raise SignedV2SourceError("exact typed v2 binding, context, proof and host port required")
    try:
        binding.validate()
        context.validate()
        proof.validate()
        _verify_signature(proof, trusted_external_public_key)
        status, pending_intent = _inspect(
            journal_root, binding, schema_version="v2")
        expected_context_json = _context_json(context)
        if (status.pending_cycle != context.cycle_index
                or status.committed_cycles != context.cycle_index
                or proof.cycle_index != context.cycle_index
                or proof.cycle_index >= binding.cycle_count
                or not status.last_checkpoint_sha256
                or pending_intent is None
                or pending_intent.get("context") != expected_context_json):
            raise SignedV2SourceError("missing pending v2 cycle intent")
        if (type(expected_authority_epoch) is not int
                or expected_authority_epoch != proof.authority_epoch
                or type(expected_previous_authority_head_sha256) is not str
                or _SHA.fullmatch(expected_previous_authority_head_sha256) is None):
            raise SignedV2SourceError("stale or invalid authority epoch/head")
        expected = {
            "provider_id": expected_provider_id,
            "host_id": expected_host_id,
            "run_id": binding.run_id,
            "checkpoint_run_id": binding.checkpoint_run_id,
            "objective_id": binding.objective_id,
            "arm": binding.arm,
            "candidate_artifact_sha256": binding.candidate_artifact_sha256,
            "plan_sha256": expected_plan_sha256,
            "scope_sha256": binding.scope_fingerprint,
            "source_sha256": binding.source_sha256,
            "base_checkpoint_sha256": binding.base_checkpoint_sha256,
            "initial_checkpoint_sha256": binding.prior_checkpoint_sha256,
            "immediate_predecessor_sha256": status.last_checkpoint_sha256,
            "cycle_index": context.cycle_index,
            "fence": binding.fence,
            "authority_epoch": expected_authority_epoch,
            "previous_authority_head_sha256": expected_previous_authority_head_sha256,
        }
        if any(getattr(proof, key) != value for key, value in expected.items()):
            raise SignedV2SourceError("signed v2 source differs from pinned journal/authority")
        if (context.run_id != binding.checkpoint_run_id
                or context.objective_id != binding.objective_id
                or context.base_checkpoint_sha256 != binding.base_checkpoint_sha256
                or context.prior_checkpoint_sha256 != binding.prior_checkpoint_sha256
                or context.fence != binding.fence
                or context.immediate_predecessor_sha256 != status.last_checkpoint_sha256):
            raise SignedV2SourceError("v2 context differs from pending journal predecessor")

        def external_matches() -> bool:
            observed = read_external_proof(proof.run_id, proof.cycle_index)
            if type(observed) is not SignedPendingCheckpointV2:
                return False
            observed.validate()
            _verify_signature(observed, trusted_external_public_key)
            return observed == proof

        if not external_matches():
            raise SignedV2SourceError("signed v2 external readback missing before snapshot")
        _, tensor_sha = _snapshot(
            journal_root, checkpoint_path, expected_context_json,
            proof.checkpoint_manifest_sha256, arm=binding.arm, schema_version="v2")
        if not hmac.compare_digest(tensor_sha, proof.checkpoint_tensor_sha256):
            raise SignedV2SourceError("signed v2 tensor differs from checkpoint bytes")
        if not external_matches():
            raise SignedV2SourceError("signed v2 external readback changed during snapshot")
        # The external callback is untrusted and may alter local state. Reject
        # changes to either the pending intent or checkpoint after its final
        # readback, rather than returning a receipt for stale source bytes.
        final_status, final_intent = _inspect(
            journal_root, binding, schema_version="v2")
        if final_status != status or final_intent != pending_intent:
            raise SignedV2SourceError("pending v2 journal state changed during readback")
        _, final_tensor_sha = _snapshot(
            journal_root, checkpoint_path, expected_context_json,
            proof.checkpoint_manifest_sha256, arm=binding.arm, schema_version="v2")
        if not hmac.compare_digest(final_tensor_sha, proof.checkpoint_tensor_sha256):
            raise SignedV2SourceError("signed v2 checkpoint changed during final readback")
    except SignedV2SourceError:
        raise
    except (MazeJournalError, ValueError, TypeError, OSError, AttributeError) as exc:
        raise SignedV2SourceError("pending v2 checkpoint source unavailable") from exc
    except Exception as exc:
        raise SignedV2SourceError("external signed v2 source provider failed") from exc
    return PendingCheckpointV2SourceReceipt(
        proof.checkpoint_manifest_sha256,
        proof.immediate_predecessor_sha256,
        proof.authority_epoch,
    )
