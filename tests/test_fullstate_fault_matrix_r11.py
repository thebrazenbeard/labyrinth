"""Source-only recovery fault matrix over a real temporary SQLite journal.

Model serialization and neural fingerprints are replaced at the physical
boundary. These tests prove journal/runner ordering, never Torch recovery or
independent host attestation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.fullstate_successor import (
    DevelopmentStepReceipt, FullStateSuccessorError,
    run_fullstate_development_cycle,
)
from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError, FullStateRunBinding,
    claim_development_run_resume, initialize_development_event_journal,
    inspect_development_event,
)
from labyrinth.neural_resume_checkpoint import FullCheckpointReceipt


class _CPUModel:
    def parameters(self):
        return (SimpleNamespace(device=SimpleNamespace(type="cpu")),)

    def named_buffers(self):
        return ()


class FullStateFaultMatrixTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "journal.sqlite"
        self.key = b"r11-journal-test-key-32-bytes-plus"
        initialize_development_event_journal(self.db, state_key=self.key)
        self.event = "event-r11"
        self.binding = FullStateRunBinding(
            run_id="run-r11", objective_id="program-reasoning",
            base_checkpoint_sha256="a" * 64,
            source_tree_sha256="b" * 64,
            maze_commitment_sha256="c" * 64,
            prior_checkpoint_sha256="d" * 64,
            initial_checkpoint_sha256="e" * 64,
            initial_cursor=0,
            initial_exposures=(("case-0", "f" * 64),),
        )
        self.initial = self.root / "initial"
        self.successor = self.root / "successor"

    def state(self):
        return inspect_development_event(self.db, self.event, state_key=self.key)

    @staticmethod
    def _load(_path, *, expected_sha256, expected_context, **_kwargs):
        return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

    @staticmethod
    def _save(path, *, context, **_kwargs):
        # Make an actual orphan directory: the mock is a serialization boundary.
        path.mkdir()
        (path / "fake-state-bytes").write_bytes(b"source-test-only")
        return FullCheckpointReceipt("9" * 64, context, ("weight",))

    @staticmethod
    def _progress(_model, _optimizer, planned, cursor):
        return DevelopmentStepReceipt(tuple(planned), len(planned),
                                      cursor, cursor + len(planned))

    @staticmethod
    def _fingerprints():
        return [("a" * 64, (0,), "f" * 64),
                ("b" * 64, (1,), "f" * 64),
                ("b" * 64, (1,), "f" * 64)]

    def invoke(self, *, attest=None, training=None, fingerprints=None,
               save=None, destination=None):
        with (patch("labyrinth.fullstate_successor.load_full_checkpoint",
                    side_effect=self._load),
              patch("labyrinth.fullstate_successor.save_full_checkpoint",
                    side_effect=save or self._save),
              patch("labyrinth.fullstate_successor._neural_fingerprint",
                    side_effect=fingerprints or self._fingerprints())):
            return run_fullstate_development_cycle(
                binding=self.binding, event_id=self.event,
                journal_db=self.db, state_key=self.key,
                initial_checkpoint_dir=self.initial,
                successor_checkpoint_dir=destination or self.successor,
                model_factory=_CPUModel, optimizer_factory=lambda _model: object(),
                training_step=training or self._progress, step_count=1,
                verify_signed_event=lambda *_: True,
                attest_successor_bytes=attest or (lambda *_: True),
            )

    def test_published_bytes_without_attestation_remain_orphan_and_uncommitted(self):
        with self.assertRaisesRegex(FullStateSuccessorError, "attestation"):
            self.invoke(attest=lambda *_: False)
        self.assertTrue((self.successor / "fake-state-bytes").is_file())
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)
        self.assertEqual(self.state()["run_state"], "ACTIVE")

    def test_readback_neural_mismatch_never_advances_journal(self):
        values = [("a" * 64, (0,), "f" * 64),
                  ("b" * 64, (1,), "f" * 64),
                  ("c" * 64, (1,), "f" * 64)]
        with self.assertRaisesRegex(FullStateSuccessorError, "readback.*differs"):
            self.invoke(fingerprints=values)
        self.assertTrue(self.successor.exists())
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)

    def test_changed_parameters_without_adamw_step_advance_cannot_publish(self):
        values = [("a" * 64, (0,), "f" * 64),
                  ("b" * 64, (0,), "f" * 64)]
        with self.assertRaisesRegex(FullStateSuccessorError, "AdamW optimizer step"):
            self.invoke(fingerprints=values)
        self.assertFalse(self.successor.exists())
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)

    def test_stale_fence_appearing_after_attestation_blocks_commit(self):
        def race(_path, _receipt):
            fence = claim_development_run_resume(
                self.db, self.event, state_key=self.key,
                run_id=self.binding.run_id,
                run_commitment_sha256=self.binding.commitment_sha256(),
                checkpoint_sha256=self.binding.initial_checkpoint_sha256,
                expected_fence=0,
            )
            self.assertEqual(fence, 1)
            return True

        with self.assertRaisesRegex(FullStateSuccessorError, "fence mismatch"):
            self.invoke(attest=race)
        self.assertEqual(self.state()["fence"], 1)
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)

    def test_fence_claim_between_inspection_and_atomic_cas_blocks_commit(self):
        real_inspect = inspect_development_event

        def race_after_inspection(*args, **kwargs):
            snapshot = real_inspect(*args, **kwargs)
            next_fence = claim_development_run_resume(
                self.db, self.event, state_key=self.key,
                run_id=self.binding.run_id,
                run_commitment_sha256=self.binding.commitment_sha256(),
                checkpoint_sha256=self.binding.initial_checkpoint_sha256,
                expected_fence=0,
            )
            self.assertEqual(next_fence, 1)
            return snapshot

        with patch("labyrinth.fullstate_successor.inspect_development_event",
                   side_effect=race_after_inspection):
            with self.assertRaisesRegex(DevelopmentJournalError, "fence mismatch"):
                self.invoke()
        self.assertTrue(self.successor.exists())
        self.assertEqual(self.state()["fence"], 1)
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)

    def test_failure_during_final_journal_commit_keeps_published_orphan(self):
        with patch("labyrinth.fullstate_successor.commit_development_run_checkpoint",
                   side_effect=DevelopmentJournalError("storage unavailable")):
            with self.assertRaisesRegex(DevelopmentJournalError, "storage unavailable"):
                self.invoke()
        self.assertTrue((self.successor / "fake-state-bytes").is_file())
        self.assertEqual(self.state()["checkpoint_sha256"], "e" * 64)

    def test_double_reservation_cannot_run_training_twice(self):
        first = self.invoke()
        self.assertEqual(first.checkpoint_sha256, "9" * 64)
        called = []

        def second_train(*args):
            called.append(args)
            return self._progress(*args)

        with self.assertRaisesRegex(DevelopmentJournalError, "already consumed"):
            self.invoke(training=second_train,
                        destination=self.root / "second-successor")
        self.assertEqual(called, [])
        self.assertEqual(self.state()["checkpoint_sha256"], "9" * 64)

    def test_relative_or_ancestor_alias_journal_path_rejected_before_training(self):
        original_db = self.db
        self.db = Path("relative-journal.sqlite")
        with self.assertRaisesRegex(FullStateSuccessorError, "absolute"):
            self.invoke()
        self.db = original_db
        alias = self.root / "alias"
        nested = self.root / "journal-home"
        nested.mkdir()
        (nested / "child").mkdir()
        try:
            alias.symlink_to(nested, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this platform")
        self.db = alias / "child" / "journal.sqlite"
        with self.assertRaisesRegex(ValueError, "symlink|junction"):
            initialize_development_event_journal(self.db, state_key=self.key)
        self.assertFalse(self.successor.exists())


if __name__ == "__main__":
    unittest.main()
