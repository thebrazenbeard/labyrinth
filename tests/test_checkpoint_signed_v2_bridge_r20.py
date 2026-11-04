"""Adversarial signed pending-v2 source checks; synthetic, not host custody."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.checkpoint_signed_v2_bridge_r20 import (
    SignedPendingCheckpointV2, SignedV2SourceError,
    verify_pending_checkpoint_source_v2,
)
from labyrinth.maze_crash_resume_journal import (
    commit_maze_cycle, initialize_maze_journal, inspect_maze_journal,
    precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import _strict_json
from test_checkpoint_manifest_v2_r19 import versioned_context, versioned_fixture
from test_maze_crash_resume_journal import binding


@unittest.skipIf(os.name == "nt", "Windows fsync requires a qualified host provider")
class PendingV2SignedSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.parent = Path(temp.name) / "arm-correct"
        self.parent.mkdir()
        self.journal = self.parent / "journal"
        self.binding = binding()
        initialize_maze_journal(self.journal, self.binding, schema_version="v2")
        self.private = Ed25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.context, self.path, self.sha = self._pending(
            0, self.binding.prior_checkpoint_sha256, b"first")
        self.proof = self._proof(self.context, self.path, self.sha)

    def _pending(self, cycle: int, predecessor: str, salt: bytes):
        ctx = versioned_context(cycle, predecessor)
        path = self.parent / f"source-{cycle}-{salt.decode('ascii')}"
        digest = versioned_fixture(path, ctx, tensor_salt=salt)
        precommit_maze_cycle(self.journal, self.binding, cycle, ctx,
                             schema_version="v2")
        return ctx, path, digest

    def _proof(self, ctx, path, digest):
        tensor = _strict_json((path / "manifest.json").read_bytes())["tensor_sha256"]
        unsigned = SignedPendingCheckpointV2(
            provider_id="provider-1", host_id="host-1",
            run_id=self.binding.run_id,
            checkpoint_run_id=self.binding.checkpoint_run_id,
            objective_id=self.binding.objective_id, arm=self.binding.arm,
            candidate_artifact_sha256=self.binding.candidate_artifact_sha256,
            plan_sha256="f" * 64, scope_sha256=self.binding.scope_fingerprint,
            source_sha256=self.binding.source_sha256,
            base_checkpoint_sha256=self.binding.base_checkpoint_sha256,
            initial_checkpoint_sha256=self.binding.prior_checkpoint_sha256,
            immediate_predecessor_sha256=ctx.immediate_predecessor_sha256,
            checkpoint_manifest_sha256=digest, checkpoint_tensor_sha256=tensor,
            cycle_index=ctx.cycle_index, fence=self.binding.fence,
            authority_epoch=ctx.cycle_index + 1,
            previous_authority_head_sha256="1" * 64,
            signature_hex="0" * 128,
        )
        return replace(unsigned, signature_hex=self.private.sign(
            unsigned.signed_bytes()).hex())

    def _verify(self, **changes):
        args = dict(
            journal_root=self.journal, checkpoint_path=self.path,
            binding=self.binding, context=self.context, proof=self.proof,
            expected_provider_id="provider-1", expected_host_id="host-1",
            expected_plan_sha256="f" * 64,
            expected_authority_epoch=self.proof.authority_epoch,
            expected_previous_authority_head_sha256="1" * 64,
            trusted_external_public_key=self.public,
            read_external_proof=lambda *_: self.proof,
        )
        args.update(changes)
        return verify_pending_checkpoint_source_v2(**args)

    def test_pending_cycle_zero_and_one_signed_source_checks_only(self):
        before = inspect_maze_journal(self.journal, self.binding,
                                      schema_version="v2")
        receipt = self._verify()
        self.assertEqual(before.pending_cycle, 0)
        self.assertEqual(receipt.status, "SIGNED_PENDING_V2_SOURCE_ONLY")
        self.assertEqual(receipt.immediate_predecessor_sha256,
                         self.binding.prior_checkpoint_sha256)
        self.assertTrue(receipt.signed_source_claim_verified)
        self.assertFalse(receipt.neural_derivation_verified)
        self.assertFalse(receipt.host_custody_verified)
        self.assertFalse(receipt.rollback_resistance_verified)
        self.assertFalse(receipt.graduation_authorized)
        commit_maze_cycle(self.journal, self.binding, 0, self.context,
                          self.path, self.sha, schema_version="v2")
        self.context, self.path, self.sha = self._pending(1, self.sha, b"second")
        self.proof = self._proof(self.context, self.path, self.sha)
        receipt = self._verify()
        self.assertEqual(receipt.immediate_predecessor_sha256,
                         self.context.immediate_predecessor_sha256)
        self.assertEqual(receipt.authority_epoch, 2)

    def test_forged_signature_and_wrong_signer_are_rejected(self):
        with self.assertRaisesRegex(SignedV2SourceError, "signature"):
            self._verify(proof=replace(self.proof, signature_hex="0" * 128))
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        with self.assertRaisesRegex(SignedV2SourceError, "signature"):
            self._verify(trusted_external_public_key=other)

    def test_signed_wrong_predecessor_and_run_are_rejected(self):
        forged = replace(self.proof, immediate_predecessor_sha256="a" * 64,
                         signature_hex="0" * 128)
        forged = replace(forged, signature_hex=self.private.sign(
            forged.signed_bytes()).hex())
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(proof=forged, read_external_proof=lambda *_: forged)
        foreign = replace(self.proof, run_id="foreign-run", signature_hex="0" * 128)
        foreign = replace(foreign, signature_hex=self.private.sign(
            foreign.signed_bytes()).hex())
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(proof=foreign, read_external_proof=lambda *_: foreign)

    def test_wrong_plan_host_epoch_and_authority_head_are_rejected(self):
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(expected_plan_sha256="a" * 64)
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(expected_host_id="shadow-host")
        with self.assertRaisesRegex(SignedV2SourceError, "epoch/head"):
            self._verify(expected_authority_epoch=42)
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(expected_previous_authority_head_sha256="2" * 64)

    def test_changed_external_readback_or_checkpoint_bytes_are_rejected(self):
        seen = [0]

        def changing(_run, _cycle):
            seen[0] += 1
            return self.proof if seen[0] == 1 else replace(
                self.proof, authority_epoch=4)

        with self.assertRaises(SignedV2SourceError):
            self._verify(read_external_proof=changing)
        self.assertEqual(seen[0], 2)
        (self.path / "state.safetensors").write_bytes(b"tampered")
        with self.assertRaises(SignedV2SourceError):
            self._verify()

    def test_v1_downgrade_and_missing_pending_intent_cannot_pass(self):
        legacy = self.parent / "legacy-journal"
        initialize_maze_journal(legacy, self.binding)
        with self.assertRaises(SignedV2SourceError):
            self._verify(journal_root=legacy)
        commit_maze_cycle(self.journal, self.binding, 0, self.context,
                          self.path, self.sha, schema_version="v2")
        with self.assertRaisesRegex(SignedV2SourceError, "pending"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
