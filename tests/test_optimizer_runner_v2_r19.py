"""R19 source-only MiniLM runner v2 immediate-predecessor integration.

Pure tests run without the optional neural CPU cache. The conditional AdamW
probe uses a toy torch model, not pretrained MiniLM or independent host CAS.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from labyrinth.minilm_optimizer_state_resume import (
    load_maze_optimizer_snapshot, save_maze_optimizer_snapshot,
)
from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointContextV2, FullCheckpointError,
    SCHEMA, SCHEMA_V2, _context_json, _json_bytes, _manifest_digest,
)
from labyrinth.maze_crash_resume_journal import (
    MazeJournalBinding, MazeJournalError, MazeJournalUnknown,
    commit_maze_cycle, initialize_maze_journal, inspect_maze_journal,
    precommit_maze_cycle,
)
from scripts.qualify_maze_conditioned_minilm_cpu import (
    OptimizerContinuityRequest, main,
)
from tests.test_minilm_optimizer_state_resume import plan
from tests.test_maze_crash_resume_journal import fake_bytes_snapshot


def source_only_v2_snapshot(path: Path, ctx: FullCheckpointContextV2,
                            salt: bytes) -> str:
    """Synthetic digest-valid manifest, NOT physical pretrained/AdamW proof."""
    fake_bytes_snapshot(path, ctx)
    tensor = b"runner-v2-byte-fixture-" + salt
    (path / "state.safetensors").write_bytes(tensor)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["schema"] = SCHEMA_V2
    manifest["context"] = _context_json(ctx)
    manifest["tensor_sha256"] = sha256(tensor).hexdigest()
    raw = _json_bytes(manifest)
    (path / "manifest.json").write_bytes(raw)
    return _manifest_digest(raw, schema=SCHEMA_V2)


class OptimizerRunnerV2SourceTests(unittest.TestCase):
    def test_legacy_optin_is_explicitly_unqualified_and_has_no_v2_candidate(self):
        request = OptimizerContinuityRequest("operator", "a" * 64, 7)
        request.validate()
        self.assertFalse(request.source_journal_v2)
        self.assertIsNone(request.candidate_artifact_sha256)
        self.assertIs(type(plan().context(4)), FullCheckpointContext)

    def test_v2_optin_requires_exact_candidate_pin_and_boolean_mode(self):
        valid = OptimizerContinuityRequest(
            "operator", "a" * 64, 7, True, "c" * 64)
        valid.validate()
        for changed in (
            replace(valid, source_journal_v2=False),
            replace(valid, source_journal_v2=1),
            replace(valid, candidate_artifact_sha256=None),
            replace(valid, candidate_artifact_sha256="C" * 64),
            replace(valid, candidate_artifact_sha256="../escape"),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                changed.validate()

    def test_v2_context_binds_actual_prior_without_reinterpreting_baseline(self):
        current = plan()
        previous_receipt = "d" * 64
        first = current.context(
            4, cycle_index=0, immediate_predecessor_sha256="b" * 64)
        second = current.context(
            8, cycle_index=1,
            immediate_predecessor_sha256=previous_receipt)
        self.assertIs(type(first), FullCheckpointContextV2)
        self.assertIs(type(second), FullCheckpointContextV2)
        self.assertEqual(first.prior_checkpoint_sha256, "b" * 64)
        self.assertEqual(first.immediate_predecessor_sha256, "b" * 64)
        self.assertEqual(second.prior_checkpoint_sha256, "b" * 64)
        self.assertEqual(second.immediate_predecessor_sha256, previous_receipt)
        self.assertEqual(second.cycle_index, 1)
        self.assertNotEqual(second, current.context(8))
        for options in (
            dict(cycle_index=0),
            dict(immediate_predecessor_sha256="b" * 64),
            dict(cycle_index=0, immediate_predecessor_sha256="d" * 64),
            dict(cycle_index=1, immediate_predecessor_sha256="d" * 64),
            dict(cycle_index=True, immediate_predecessor_sha256="b" * 64),
            dict(cycle_index=0, immediate_predecessor_sha256="z" * 64),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                current.context(4, **options)

    def test_cli_v2_is_optin_and_reports_source_only_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            common = ["--output-parent", temp, "--run-name", "candidate"]
            legacy = ["--full-optimizer-cycle-continuity",
                      "--optimizer-run-id", "operator",
                      "--optimizer-prior-sha256", "a" * 64,
                      "--optimizer-fence", "7"]
            for extra in (
                ["--optimizer-v2-journal"],
                legacy + ["--optimizer-v2-journal"],
                legacy + ["--optimizer-v2-candidate-sha256", "c" * 64],
            ):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main(common + extra)
                    self.assertEqual(caught.exception.code, 2)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(common + legacy + [
                    "--optimizer-v2-journal",
                    "--optimizer-v2-candidate-sha256", "c" * 64,
                ]), 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["full_optimizer_continuity"],
                             "V2_IMMEDIATE_PREDECESSOR_SOURCE_JOURNAL_ONLY")
            self.assertFalse(report["execute"])
            self.assertFalse(report["graduation_authorized"])
            self.assertFalse((Path(temp) / "candidate").exists())

    @unittest.skipIf(os.name == "nt", "Windows requires qualified journal fsync")
    def test_real_runner_context_cycle1_rejects_foreign_fork_on_source_bytes(self):
        # These contexts are produced by the *runner's* plan, not a generic
        # neural checkpoint fixture. The journal must bind exactly its live
        # source-derived run/plan/exposure/arm identities.
        current = plan()
        with tempfile.TemporaryDirectory() as temp:
            arm_root = Path(temp) / "correct"
            arm_root.mkdir()
            journal_root = arm_root / "source-journal-v2"
            binding = MazeJournalBinding(
                run_id=current.logical_run_id,
                objective_id=current.scope.objective_id,
                arm=current.arm,
                candidate_artifact_sha256="c" * 64,
                scope_fingerprint=current.scope.fingerprint,
                base_checkpoint_sha256=current.base_checkpoint_sha256,
                prior_checkpoint_sha256=current.prior_checkpoint_sha256,
                checkpoint_run_id=current.checkpoint_run_id,
                source_sha256=current.plan_sha256,
                fence=current.fence,
                cycle_count=2,
            )
            initialize_maze_journal(journal_root, binding, schema_version="v2")
            first = current.context(
                4, cycle_index=0,
                immediate_predecessor_sha256=current.prior_checkpoint_sha256)
            path0 = arm_root / "a0"
            sha_a0 = source_only_v2_snapshot(path0, first, b"a0")
            precommit_maze_cycle(journal_root, binding, 0, first,
                                 schema_version="v2")
            commit_maze_cycle(journal_root, binding, 0, first, path0, sha_a0,
                              schema_version="v2")
            local = current.context(
                8, cycle_index=1, immediate_predecessor_sha256=sha_a0)
            self.assertEqual(local.prior_checkpoint_sha256,
                             current.prior_checkpoint_sha256)
            self.assertEqual(local.immediate_predecessor_sha256, sha_a0)

            # Fork B independently saves a byte-different checkpoint with
            # otherwise identical run, original baseline and schedule.
            path_b0 = arm_root / "b0"
            sha_b0 = source_only_v2_snapshot(path_b0, first, b"b0")
            self.assertNotEqual(sha_b0, sha_a0)
            foreign = current.context(
                8, cycle_index=1, immediate_predecessor_sha256=sha_b0)
            with self.assertRaisesRegex(MazeJournalError, "immediate predecessor"):
                precommit_maze_cycle(journal_root, binding, 1, foreign,
                                     schema_version="v2")
            precommit_maze_cycle(journal_root, binding, 1, local,
                                 schema_version="v2")
            path_b1 = arm_root / "b1"
            sha_b1 = source_only_v2_snapshot(path_b1, foreign, b"b1")
            with self.assertRaisesRegex(MazeJournalError, "context"):
                commit_maze_cycle(journal_root, binding, 1, local, path_b1,
                                  sha_b1, schema_version="v2")
            status = inspect_maze_journal(journal_root, binding,
                                          schema_version="v2")
            self.assertEqual((status.committed_cycles, status.pending_cycle),
                             (1, 1))
            with self.assertRaises(MazeJournalUnknown):
                precommit_maze_cycle(journal_root, binding, 1, local,
                                     schema_version="v2")


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "optional torch+safetensors physical CPU dependencies unavailable",
)
class OptimizerRunnerV2ToyAdamWTests(unittest.TestCase):
    @staticmethod
    def _make():
        from tests.test_minilm_optimizer_state_resume import PhysicalMazeAdamWResumeTests
        return PhysicalMazeAdamWResumeTests._make()

    @staticmethod
    def _step(encoder, head, optimizer):
        from tests.test_minilm_optimizer_state_resume import PhysicalMazeAdamWResumeTests
        PhysicalMazeAdamWResumeTests._step(encoder, head, optimizer)

    @staticmethod
    def _binding(current):
        return MazeJournalBinding(
            run_id=current.logical_run_id,
            objective_id=current.scope.objective_id,
            arm=current.arm,
            candidate_artifact_sha256="c" * 64,
            scope_fingerprint=current.scope.fingerprint,
            base_checkpoint_sha256=current.base_checkpoint_sha256,
            prior_checkpoint_sha256=current.prior_checkpoint_sha256,
            checkpoint_run_id=current.checkpoint_run_id,
            source_sha256=current.plan_sha256,
            fence=current.fence,
            cycle_count=2,
        )

    def test_v2_journal_binds_cycle0_receipt_and_restores_continuous_adamw(self):
        import torch

        current = plan()
        encoder, head, optimizer = self._make()
        with tempfile.TemporaryDirectory() as temp:
            arm_root = Path(temp) / "correct"
            arm_root.mkdir()
            journal_root = arm_root / "optimizer-source-journal-v2"
            binding = self._binding(current)
            initialize_maze_journal(journal_root, binding, schema_version="v2")
            context0 = current.context(
                4, cycle_index=0, immediate_predecessor_sha256="b" * 64)
            precommit_maze_cycle(journal_root, binding, 0, context0,
                                 schema_version="v2")
            for _ in range(4):
                self._step(encoder, head, optimizer)
            checkpoint0 = arm_root / "cycle-0" / "optimizer-state"
            checkpoint0.parent.mkdir()
            receipt0 = save_maze_optimizer_snapshot(
                checkpoint0, encoder=encoder, head=head, optimizer=optimizer,
                plan=current, completed_steps=4, cycle_index=0,
                immediate_predecessor_sha256="b" * 64)
            self.assertEqual(json.loads((checkpoint0 / "manifest.json").read_text())["schema"],
                             SCHEMA_V2)
            commit_maze_cycle(journal_root, binding, 0, context0,
                              checkpoint0, receipt0.sha256, schema_version="v2")
            context1 = current.context(
                8, cycle_index=1,
                immediate_predecessor_sha256=receipt0.sha256)
            precommit_maze_cycle(journal_root, binding, 1, context1,
                                 schema_version="v2")
            for _ in range(4):
                self._step(encoder, head, optimizer)
            expected_params = [p.detach().clone() for module in (encoder, head)
                               for p in module.parameters() if p.requires_grad]

            # A fresh module and optimizer resume from the cycle0 v2 receipt.
            new_encoder, new_head, new_optimizer = self._make()
            load_maze_optimizer_snapshot(
                checkpoint0, encoder=new_encoder, head=new_head,
                optimizer=new_optimizer, plan=current, completed_steps=4,
                independently_pinned_checkpoint_sha256=receipt0.sha256,
                cycle_index=0, immediate_predecessor_sha256="b" * 64)
            for _ in range(4):
                self._step(new_encoder, new_head, new_optimizer)
            actual_params = [p.detach() for module in (new_encoder, new_head)
                             for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(torch.equal(a, b)
                                for a, b in zip(expected_params, actual_params)))
            checkpoint1 = arm_root / "cycle-1" / "optimizer-state"
            checkpoint1.parent.mkdir()
            receipt1 = save_maze_optimizer_snapshot(
                checkpoint1, encoder=new_encoder, head=new_head,
                optimizer=new_optimizer, plan=current, completed_steps=8,
                cycle_index=1, immediate_predecessor_sha256=receipt0.sha256)
            commit_maze_cycle(journal_root, binding, 1, context1,
                              checkpoint1, receipt1.sha256, schema_version="v2")
            final = inspect_maze_journal(journal_root, binding, schema_version="v2")
            self.assertEqual(final.last_checkpoint_sha256, receipt1.sha256)
            self.assertEqual(final.committed_cycles, 2)
            self.assertTrue(final.safe_to_begin_next_cycle)
            self.assertNotEqual(receipt1.sha256, receipt0.sha256)

    def test_v2_foreign_fork_is_rejected_and_pending_is_not_replayed(self):
        current = plan()
        encoder, head, optimizer = self._make()
        with tempfile.TemporaryDirectory() as temp:
            arm_root = Path(temp) / "correct"
            arm_root.mkdir()
            root = arm_root / "optimizer-source-journal-v2"
            binding = self._binding(current)
            initialize_maze_journal(root, binding, schema_version="v2")
            context0 = current.context(
                4, cycle_index=0, immediate_predecessor_sha256="b" * 64)
            precommit_maze_cycle(root, binding, 0, context0, schema_version="v2")
            for _ in range(4):
                self._step(encoder, head, optimizer)
            checkpoint0 = arm_root / "cycle-0" / "optimizer-state"
            checkpoint0.parent.mkdir()
            receipt0 = save_maze_optimizer_snapshot(
                checkpoint0, encoder=encoder, head=head, optimizer=optimizer,
                plan=current, completed_steps=4, cycle_index=0,
                immediate_predecessor_sha256="b" * 64)
            commit_maze_cycle(root, binding, 0, context0, checkpoint0,
                              receipt0.sha256, schema_version="v2")

            foreign = "d" * 64
            foreign_context = current.context(
                8, cycle_index=1, immediate_predecessor_sha256=foreign)
            with self.assertRaises(MazeJournalError):
                precommit_maze_cycle(root, binding, 1, foreign_context,
                                     schema_version="v2")
            self.assertTrue(inspect_maze_journal(root, binding,
                                                schema_version="v2").safe_to_begin_next_cycle)
            proper = current.context(
                8, cycle_index=1, immediate_predecessor_sha256=receipt0.sha256)
            precommit_maze_cycle(root, binding, 1, proper, schema_version="v2")
            with self.assertRaises(MazeJournalUnknown):
                precommit_maze_cycle(root, binding, 1, proper, schema_version="v2")
            status = inspect_maze_journal(root, binding, schema_version="v2")
            self.assertEqual(status.pending_cycle, 1)
            self.assertFalse(status.safe_to_begin_next_cycle)


if __name__ == "__main__":
    unittest.main()
