"""Independent hostile cases for the optional pending-v2 signed source bridge.

These synthetic fixtures exercise source readback timing and caller-controlled
authority substitution. They do NOT demonstrate a deployed hostile actor,
model-weight derivation, or a host-owned monotonic provider.
"""

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
    initialize_maze_journal, precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import _strict_json
from test_checkpoint_manifest_v2_r19 import versioned_context, versioned_fixture
from test_maze_crash_resume_journal import binding


@unittest.skipIf(os.name == "nt", "Windows fsync requires a qualified host provider")
class SignedBridgeHostileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "arm-correct"
        self.root.mkdir()
        self.journal = self.root / "journal"
        self.binding = binding()
        initialize_maze_journal(self.journal, self.binding, schema_version="v2")
        self.context = versioned_context(
            0, self.binding.prior_checkpoint_sha256)
        self.path = self.root / "cycle-zero"
        self.checkpoint_sha = versioned_fixture(
            self.path, self.context, tensor_salt=b"original")
        precommit_maze_cycle(self.journal, self.binding, 0, self.context,
                             schema_version="v2")
        self.signer = Ed25519PrivateKey.generate()
        self.public = self.signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        tensor_sha = _strict_json(
            (self.path / "manifest.json").read_bytes())["tensor_sha256"]
        candidate = SignedPendingCheckpointV2(
            provider_id="provider-1", host_id="host-1",
            run_id=self.binding.run_id,
            checkpoint_run_id=self.binding.checkpoint_run_id,
            objective_id=self.binding.objective_id, arm=self.binding.arm,
            candidate_artifact_sha256=self.binding.candidate_artifact_sha256,
            plan_sha256="f" * 64, scope_sha256=self.binding.scope_fingerprint,
            source_sha256=self.binding.source_sha256,
            base_checkpoint_sha256=self.binding.base_checkpoint_sha256,
            initial_checkpoint_sha256=self.binding.prior_checkpoint_sha256,
            immediate_predecessor_sha256=self.binding.prior_checkpoint_sha256,
            checkpoint_manifest_sha256=self.checkpoint_sha,
            checkpoint_tensor_sha256=tensor_sha,
            cycle_index=0, fence=self.binding.fence, authority_epoch=1,
            previous_authority_head_sha256="1" * 64,
            signature_hex="0" * 128,
        )
        self.proof = self._sign(candidate)

    def _sign(self, candidate: SignedPendingCheckpointV2) -> SignedPendingCheckpointV2:
        return replace(candidate, signature_hex=self.signer.sign(
            candidate.signed_bytes()).hex())

    def _verify(self, **changes):
        kwargs = dict(
            journal_root=self.journal, checkpoint_path=self.path,
            binding=self.binding, context=self.context, proof=self.proof,
            expected_provider_id="provider-1", expected_host_id="host-1",
            expected_plan_sha256="f" * 64, expected_authority_epoch=1,
            expected_previous_authority_head_sha256="1" * 64,
            trusted_external_public_key=self.public,
            read_external_proof=lambda *_: self.proof,
        )
        kwargs.update(changes)
        return verify_pending_checkpoint_source_v2(**kwargs)

    def test_mutation_during_second_authority_readback_must_invalidate_receipt(self):
        """RED: final callback can change bytes *after* the only tensor hash.

        The proof and both external return objects are byte-identical. The
        checkpoint itself is not. A source receipt should not cover bytes no
        longer matching the signed checkpoint commitment at return.
        """
        seen = [0]

        def changing_on_final_read(_run_id: str, _cycle: int):
            seen[0] += 1
            if seen[0] == 2:
                (self.path / "state.safetensors").write_bytes(b"posthash mutation")
            return self.proof

        with self.assertRaises(SignedV2SourceError):
            self._verify(read_external_proof=changing_on_final_read)
        self.assertEqual(seen[0], 2)

    def test_pending_intent_removed_during_final_readback_must_invalidate_receipt(self):
        """RED: after pending-state inspection, the final callback can erase it."""
        seen = [0]

        def removing_intent_on_final_read(_run_id: str, _cycle: int):
            seen[0] += 1
            if seen[0] == 2:
                (self.journal / "cycle-0000.intent.json").unlink()
            return self.proof

        with self.assertRaises(SignedV2SourceError):
            self._verify(read_external_proof=removing_intent_on_final_read)
        self.assertEqual(seen[0], 2)

    def test_malformed_or_throwing_external_readback_fails_closed(self):
        with self.assertRaises(SignedV2SourceError):
            self._verify(read_external_proof=lambda *_: object())

        def broken(_run_id: str, _cycle: int):
            raise RuntimeError("synthetic provider unavailable")

        with self.assertRaisesRegex(SignedV2SourceError, "provider failed"):
            self._verify(read_external_proof=broken)

    def test_symlinked_checkpoint_tensor_is_rejected(self):
        original = self.path / "state.safetensors"
        elsewhere = self.root / "outside-state.safetensors"
        elsewhere.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(elsewhere)
        with self.assertRaises(SignedV2SourceError):
            self._verify()

    def test_signer_with_wrong_arm_and_host_is_not_accepted(self):
        wrong_arm = self._sign(replace(
            self.proof, arm="inverted", signature_hex="0" * 128))
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(proof=wrong_arm,
                         read_external_proof=lambda *_: wrong_arm)
        wrong_host = self._sign(replace(
            self.proof, host_id="other-host", signature_hex="0" * 128))
        with self.assertRaisesRegex(SignedV2SourceError, "journal/authority"):
            self._verify(proof=wrong_host,
                         read_external_proof=lambda *_: wrong_host)

    def test_self_consistent_key_substitution_is_only_source_verification(self):
        """A caller-controlled replacement authority may be self-consistent.

        The receipt must not claim host custody or rollback resistance even
        though a substituted synthetic key can pass signature verification.
        """
        replacement = Ed25519PrivateKey.generate()
        public = replacement.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        forged = replace(self.proof, signature_hex="0" * 128)
        forged = replace(forged, signature_hex=replacement.sign(
            forged.signed_bytes()).hex())
        receipt = self._verify(
            trusted_external_public_key=public, proof=forged,
            read_external_proof=lambda *_: forged)
        self.assertTrue(receipt.signed_source_claim_verified)
        self.assertFalse(receipt.host_custody_verified)
        self.assertFalse(receipt.rollback_resistance_verified)
        self.assertFalse(receipt.neural_derivation_verified)
        self.assertFalse(receipt.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
