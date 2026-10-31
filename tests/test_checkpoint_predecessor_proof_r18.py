"""Synthetic v2 predecessor proof tests; no neural or host attestation."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.checkpoint_predecessor_proof_r18 import (
    PredecessorProofError, SignedCyclePredecessorV2,
    verify_cycle_predecessor_source_v2,
)
from labyrinth.maze_crash_resume_journal import (
    commit_maze_cycle, initialize_maze_journal, inspect_maze_journal,
    precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import _strict_json
from test_maze_crash_resume_journal import binding, context, fake_bytes_snapshot


@unittest.skipIf(os.name == "nt", "Windows fsync requires a qualified host provider")
class CheckpointPredecessorProofR18Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / "arm-correct"
        self.parent.mkdir()
        self.root = self.parent / "journal"
        self.binding = binding()
        initialize_maze_journal(self.root, self.binding)
        self.private = Ed25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        first_context = context(0, self.binding)
        first_path = self.parent / "snapshots" / "cycle-0"
        self.first_sha = fake_bytes_snapshot(first_path, first_context,
                                             manifest_changes={"rng_tree": {"type": "scalar", "value": "A0"}})
        precommit_maze_cycle(self.root, self.binding, 0, first_context)
        commit_maze_cycle(self.root, self.binding, 0, first_context,
                          first_path, self.first_sha)
        self.second_context = context(1, self.binding)
        self.second_path = self.parent / "snapshots" / "cycle-1"
        self.second_sha = fake_bytes_snapshot(self.second_path, self.second_context)
        precommit_maze_cycle(self.root, self.binding, 1, self.second_context)
        self.proof = self._signed_proof(self.first_sha, self.second_sha,
                                        self.second_path)

    def _signed_proof(self, prior: str, digest: str,
                      snapshot_path: Path) -> SignedCyclePredecessorV2:
        manifest = _strict_json((snapshot_path / "manifest.json").read_bytes())
        base = SignedCyclePredecessorV2(
            provider_id="provider-1", host_id="host-1",
            run_id=self.binding.run_id,
            checkpoint_run_id=self.binding.checkpoint_run_id,
            objective_id=self.binding.objective_id,
            arm=self.binding.arm,
            candidate_artifact_sha256=self.binding.candidate_artifact_sha256,
            plan_sha256="f" * 64,
            scope_sha256=self.binding.scope_fingerprint,
            source_sha256=self.binding.source_sha256,
            base_checkpoint_sha256=self.binding.base_checkpoint_sha256,
            initial_checkpoint_sha256=self.binding.prior_checkpoint_sha256,
            immediate_predecessor_sha256=prior,
            checkpoint_manifest_sha256=digest,
            checkpoint_tensor_sha256=manifest["tensor_sha256"],
            cycle_index=1, fence=self.binding.fence,
            authority_epoch=2, previous_authority_head_sha256="1" * 64,
            signature_hex="0" * 128,
        )
        return replace(base, signature_hex=self.private.sign(
            base.signed_bytes()).hex())

    def _verify(self, **changes):
        args = dict(
            journal_root=self.root, checkpoint_path=self.second_path,
            binding=self.binding, context=self.second_context,
            proof=self.proof, expected_provider_id="provider-1",
            expected_host_id="host-1", expected_plan_sha256="f" * 64,
            expected_authority_epoch=2,
            expected_previous_authority_head_sha256="1" * 64,
            trusted_external_public_key=self.public,
            read_external_proof=lambda _run, _cycle: self.proof,
        )
        args.update(changes)
        return verify_cycle_predecessor_source_v2(**args)

    def test_valid_source_binding_requires_v2_predecessor_and_never_graduates(self):
        before = inspect_maze_journal(self.root, self.binding)
        result = self._verify()
        self.assertEqual(result.status, "CHECKPOINT_PREDECESSOR_SOURCE_PROOF_ONLY")
        self.assertEqual(result.immediate_predecessor_sha256, self.first_sha)
        self.assertTrue(result.checkpoints_bytes_rehashed)
        self.assertFalse(result.neural_derivation_verified)
        self.assertFalse(result.external_host_custody_verified)
        self.assertFalse(result.rollback_resistance_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertEqual(before, inspect_maze_journal(self.root, self.binding))

    def test_foreign_fork_signed_predecessor_must_not_extend_current_fork(self):
        branch = Path(self.temp.name) / "arm-fork-B"
        branch.mkdir()
        fork_root = branch / "journal"
        initialize_maze_journal(fork_root, self.binding)
        first_context = context(0, self.binding)
        fork_first_path = branch / "snapshots" / "cycle-0"
        fork_first_sha = fake_bytes_snapshot(
            fork_first_path, first_context,
            manifest_changes={"rng_tree": {"type": "scalar", "value": "B0"}},
        )
        self.assertNotEqual(self.first_sha, fork_first_sha)
        precommit_maze_cycle(fork_root, self.binding, 0, first_context)
        commit_maze_cycle(fork_root, self.binding, 0, first_context,
                          fork_first_path, fork_first_sha)
        fork_second_path = branch / "snapshots" / "cycle-1"
        fork_second_sha = fake_bytes_snapshot(fork_second_path, self.second_context)
        shutil.rmtree(self.second_path)
        shutil.copytree(fork_second_path, self.second_path)
        foreign_proof = self._signed_proof(fork_first_sha, fork_second_sha,
                                           fork_second_path)
        with self.assertRaisesRegex(PredecessorProofError, "journal predecessor"):
            self._verify(proof=foreign_proof,
                         read_external_proof=lambda *_: foreign_proof)

    def test_signed_rebinding_and_wrong_host_epoch_rejected(self):
        alternate = replace(self.proof, host_id="foreign-host", signature_hex="0" * 128)
        alternate = replace(alternate, signature_hex=self.private.sign(
            alternate.signed_bytes()).hex())
        with self.assertRaisesRegex(PredecessorProofError, "pinned"):
            self._verify(proof=alternate,
                         read_external_proof=lambda *_: alternate)
        with self.assertRaisesRegex(PredecessorProofError, "epoch"):
            self._verify(expected_authority_epoch=3)
        with self.assertRaisesRegex(PredecessorProofError, "signature"):
            self._verify(proof=replace(self.proof, signature_hex="0" * 128))

    def test_checkpoint_tensor_changed_after_precommit_is_rejected(self):
        (self.second_path / "state.safetensors").write_bytes(b"altered")
        with self.assertRaises(PredecessorProofError):
            self._verify()

    def test_optimizer_manifest_changed_after_signing_is_rejected(self):
        manifest_path = self.second_path / "manifest.json"
        manifest = _strict_json(manifest_path.read_bytes())
        manifest["optimizer_groups"] = [{"forged": "new optimizer"}]
        manifest_path.write_bytes((json.dumps(
            manifest, sort_keys=True, separators=(",", ":")) + "\n").encode())
        with self.assertRaises(PredecessorProofError):
            self._verify()

    def test_plan_and_external_authority_head_cannot_be_rebound(self):
        with self.assertRaisesRegex(PredecessorProofError, "journal predecessor"):
            self._verify(expected_plan_sha256="c" * 64)
        with self.assertRaisesRegex(PredecessorProofError, "journal predecessor"):
            self._verify(expected_previous_authority_head_sha256="2" * 64)

    def test_external_host_proof_readback_changes_after_snapshot_is_denied(self):
        seen = [0]

        def changing(_run, _cycle):
            seen[0] += 1
            return self.proof if seen[0] == 1 else replace(
                self.proof, authority_epoch=3)

        with self.assertRaises(PredecessorProofError):
            self._verify(read_external_proof=changing)
        self.assertEqual(seen[0], 2)

    def test_old_baseline_is_not_the_immediate_predecessor(self):
        fake = self._signed_proof(self.binding.prior_checkpoint_sha256,
                                  self.second_sha, self.second_path)
        with self.assertRaisesRegex(PredecessorProofError, "journal predecessor"):
            self._verify(proof=fake, read_external_proof=lambda *_: fake)


if __name__ == "__main__":
    unittest.main()
