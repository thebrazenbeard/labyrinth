"""Opt-in v2 source proof for immediate-predecessor checkpoint custody.

The v1 full-state checkpoint manifest identifies the initial baseline in its
``prior_checkpoint_sha256`` field; the Maze journal separately records the
immediately preceding *cycle* checkpoint. Neither alone can establish that a
later tensor snapshot is a continuation of the previous cycle. This module
defines a distinct, externally signed v2 binding of those source identities.

This is not a v1 migration, not installed in the journal commit path, and not
evidence of neural-weight derivation. A genuine host must pin/sign at the
checkpoint capture boundary and provide non-rollbackable independent readback.
Synthetic tests with caller-provided keys and callbacks do NOT prove that.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
from pathlib import Path
import re
from typing import Callable

from .maze_crash_resume_journal import (
    MazeJournalBinding, MazeJournalError, _snapshot, inspect_maze_journal,
)
from .neural_resume_checkpoint import FullCheckpointContext, _context_json


SCHEMA = "labyrinth.cycle-predecessor-proof.v2"
_DOMAIN = b"labyrinth.cycle-predecessor-proof.v2\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_SIG = re.compile(r"[a-f0-9]{128}\Z", re.ASCII)


class PredecessorProofError(ValueError):
    """The source checkpoint or externally governed v2 binding was refused."""


@dataclass(frozen=True)
class SignedCyclePredecessorV2:
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

    def signed_bytes(self) -> bytes:
        fields = {name: getattr(self, name)
                  for name in self.__dataclass_fields__ if name != "signature_hex"}
        return _DOMAIN + (json.dumps(fields, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise PredecessorProofError("wrong predecessor source schema")
        for name in ("provider_id", "host_id", "run_id", "checkpoint_run_id",
                     "objective_id", "arm"):
            value = getattr(self, name)
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise PredecessorProofError(f"invalid predecessor {name}")
        for name in ("candidate_artifact_sha256", "plan_sha256", "scope_sha256",
                     "source_sha256", "base_checkpoint_sha256",
                     "initial_checkpoint_sha256", "immediate_predecessor_sha256",
                     "checkpoint_manifest_sha256", "checkpoint_tensor_sha256",
                     "previous_authority_head_sha256"):
            value = getattr(self, name)
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise PredecessorProofError(f"invalid predecessor {name}")
        if (type(self.cycle_index) is not int or not 1 <= self.cycle_index <= 127
                or type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1
                or type(self.authority_epoch) is not int
                or not 1 <= self.authority_epoch < 2**63
                or type(self.signature_hex) is not str
                or _SIG.fullmatch(self.signature_hex) is None):
            raise PredecessorProofError("invalid cycle, fence, authority epoch or signature")


@dataclass(frozen=True)
class PredecessorSourceEvidence:
    status: str
    checkpoint_manifest_sha256: str
    immediate_predecessor_sha256: str
    authority_epoch: int
    checkpoints_bytes_rehashed: bool = True
    neural_derivation_verified: bool = False
    external_host_custody_verified: bool = False
    rollback_resistance_verified: bool = False
    graduation_authorized: bool = False


def _verify_signature(proof: SignedCyclePredecessorV2, public_key: bytes) -> None:
    if type(public_key) is not bytes or len(public_key) != 32:
        raise PredecessorProofError("pinned external Ed25519 public key required")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(proof.signature_hex), proof.signed_bytes())
    except (ImportError, InvalidSignature, ValueError, TypeError) as exc:
        raise PredecessorProofError("external predecessor signature refused") from exc


def verify_cycle_predecessor_source_v2(
    *,
    journal_root: Path,
    checkpoint_path: Path,
    binding: MazeJournalBinding,
    context: FullCheckpointContext,
    proof: SignedCyclePredecessorV2,
    expected_provider_id: str,
    expected_host_id: str,
    expected_plan_sha256: str,
    expected_authority_epoch: int,
    expected_previous_authority_head_sha256: str,
    trusted_external_public_key: bytes,
    read_external_proof: Callable[[str, int], SignedCyclePredecessorV2],
) -> PredecessorSourceEvidence:
    """Check a signed immediate-predecessor claim against current source bytes.

    The caller must have separately obtained the host/proof identities and
    cryptographic pin outside the learner. This opt-in check does NOT modify
    the journal or upgrade its v1 `commit_maze_cycle` acceptance contract.
    """
    if (type(binding) is not MazeJournalBinding
            or type(context) is not FullCheckpointContext
            or type(proof) is not SignedCyclePredecessorV2
            or not callable(read_external_proof)):
        raise PredecessorProofError("typed journal, context, proof and host port required")
    try:
        binding.validate()
        context.validate()
        proof.validate()
        _verify_signature(proof, trusted_external_public_key)
        status = inspect_maze_journal(journal_root, binding)
        if (status.pending_cycle != proof.cycle_index
                or status.committed_cycles != proof.cycle_index
                or type(expected_authority_epoch) is not int
                or expected_authority_epoch != proof.authority_epoch
                or proof.cycle_index >= binding.cycle_count
                or type(expected_previous_authority_head_sha256) is not str
                or not _SHA.fullmatch(expected_previous_authority_head_sha256)):
            raise PredecessorProofError("uncommitted, wrong cycle or stale authority epoch")
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
            "fence": binding.fence,
            "previous_authority_head_sha256": expected_previous_authority_head_sha256,
        }
        if any(getattr(proof, key) != value for key, value in expected.items()):
            raise PredecessorProofError("v2 proof is not pinned to the journal predecessor")
        if (context.run_id != binding.checkpoint_run_id
                or context.objective_id != binding.objective_id
                or context.base_checkpoint_sha256 != binding.base_checkpoint_sha256
                or context.prior_checkpoint_sha256 != binding.prior_checkpoint_sha256
                or context.fence != binding.fence):
            raise PredecessorProofError("snapshot v1 context differs from pinned run")

        def external_matches() -> bool:
            observed = read_external_proof(proof.run_id, proof.cycle_index)
            if type(observed) is not SignedCyclePredecessorV2:
                return False
            observed.validate()
            _verify_signature(observed, trusted_external_public_key)
            return observed == proof

        if not external_matches():
            raise PredecessorProofError("external predecessor readback missing before snapshot")
        _, tensor_sha = _snapshot(
            journal_root, checkpoint_path, _context_json(context),
            proof.checkpoint_manifest_sha256, arm=binding.arm)
        if not hmac.compare_digest(tensor_sha, proof.checkpoint_tensor_sha256):
            raise PredecessorProofError("v2 tensor commitment differs from source bytes")
        if not external_matches():
            raise PredecessorProofError("external predecessor readback changed during snapshot")
    except PredecessorProofError:
        raise
    except (MazeJournalError, ValueError, TypeError, OSError, AttributeError) as exc:
        raise PredecessorProofError("v2 predecessor source or external readback unavailable") from exc
    except Exception as exc:
        raise PredecessorProofError("external v2 predecessor provider failed") from exc
    return PredecessorSourceEvidence(
        "CHECKPOINT_PREDECESSOR_SOURCE_PROOF_ONLY",
        proof.checkpoint_manifest_sha256,
        proof.immediate_predecessor_sha256,
        proof.authority_epoch,
    )
