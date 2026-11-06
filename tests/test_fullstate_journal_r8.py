"""Full-state host journal preflight and fencing contracts, not host attestations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from dataclasses import replace

from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError, FullStateRunBinding,
    claim_development_run_resume, commit_development_run_checkpoint,
    inspect_development_event, reserve_full_state_development_run,
    initialize_development_event_journal,
    reserve_development_event,
)
from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointReceipt,
)


class JournalCanonicalPathTests(TestCase):
    KEY = b"fullstate-journal-custodian-secret-32!!"

    def test_rejects_symlink_in_any_ancestor_not_only_immediate_parent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "first"
            (source / "sub").mkdir(parents=True)
            alias = root / "alias"
            try:
                alias.symlink_to(source, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks unavailable")
            # Lexical parent alias/sub is not itself a symlink; only its
            # ancestor is. Accepting it lets a caller retarget one path to
            # fresh databases and repeatedly consume the same event ID.
            unsafe = alias / "sub" / "journal.sqlite"
            with self.assertRaises(ValueError):
                initialize_development_event_journal(unsafe, state_key=self.KEY)

    def test_rejects_noncanonical_dotdot_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            unsafe = root / "sub" / ".." / "journal.sqlite"
            with self.assertRaises(ValueError):
                initialize_development_event_journal(unsafe, state_key=self.KEY)

    def test_canonical_regular_path_still_works(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.sqlite"
            initialize_development_event_journal(path, state_key=self.KEY)
            self.assertTrue(reserve_development_event(path, "event", state_key=self.KEY))
            self.assertFalse(reserve_development_event(path, "event", state_key=self.KEY))


class FullStateJournalBindingTests(TestCase):
    KEY = b"separate-host-fullstate-journal-key-32!"

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "journal.sqlite"
        initialize_development_event_journal(self.path, state_key=self.KEY)
        self.binding = FullStateRunBinding(
            run_id="run-1", objective_id="learn-arithmetic",
            base_checkpoint_sha256="a" * 64, source_tree_sha256="b" * 64,
            maze_commitment_sha256="c" * 64,
            prior_checkpoint_sha256="d" * 64,
            initial_checkpoint_sha256="e" * 64,
            initial_cursor=1,
            initial_exposures=(("task-1", "f" * 64),),
        )
        self.receipt = FullCheckpointReceipt(
            sha256="e" * 64,
            context=FullCheckpointContext(
                run_id="run-1", objective_id="learn-arithmetic",
                base_checkpoint_sha256="a" * 64,
                prior_checkpoint_sha256="d" * 64,
                fence=0, cursor=1,
                exposures=(("task-1", "f" * 64),),
            ),
            trainable_parameters=("adapter.weight",),
        )

    def reserve(self, event="event-1", *, binding=None, receipt=None):
        return reserve_full_state_development_run(
            self.path, event, state_key=self.KEY,
            binding=self.binding if binding is None else binding,
            initial_receipt=self.receipt if receipt is None else receipt,
        )

    def test_full_state_commitment_is_domain_separated_and_semantically_bound(self):
        original = self.binding.commitment_sha256()
        self.assertEqual(len(original), 64)
        self.assertNotEqual(original, self.binding.initial_checkpoint_sha256)
        for field, alternative in (
            ("objective_id", "learn-empathy"),
            ("source_tree_sha256", "0" * 64),
            ("maze_commitment_sha256", "1" * 64),
            ("prior_checkpoint_sha256", "2" * 64),
            ("initial_cursor", 0),
            ("initial_exposures", (("task-2", "f" * 64),)),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    original, replace(self.binding, **{field: alternative}).commitment_sha256(),
                )

    def test_receipt_mismatch_does_not_consume_event(self):
        for incompatible in (
            replace(self.receipt, sha256="9" * 64),
            replace(self.receipt, context=replace(self.receipt.context, fence=1)),
            replace(self.receipt, context=replace(self.receipt.context, cursor=0)),
            replace(self.receipt, context=replace(self.receipt.context, exposures=())),
        ):
            with self.subTest(incompatible=incompatible), self.assertRaises(ValueError):
                self.reserve(receipt=incompatible)
        self.assertFalse(inspect_development_event(
            self.path, "event-1", state_key=self.KEY,
        )["consumed"])
        self.assertEqual(self.reserve(), 0)

    def test_cas_crash_window_and_stale_fence_are_fail_closed(self):
        binding = self.binding
        commitment = binding.commitment_sha256()
        self.assertEqual(self.reserve(), 0)
        # Simulate a complete candidate checkpoint written to a distinct path,
        # then a crash BEFORE journal CAS. The persisted authority remains the
        # initial checkpoint. Orphan bytes are not authoritative.
        uncommitted = "7" * 64
        state = inspect_development_event(self.path, "event-1", state_key=self.KEY)
        self.assertEqual(state["checkpoint_sha256"], binding.initial_checkpoint_sha256)
        fence = claim_development_run_resume(
            self.path, "event-1", state_key=self.KEY,
            run_id=binding.run_id, run_commitment_sha256=commitment,
            checkpoint_sha256=binding.initial_checkpoint_sha256, expected_fence=0,
        )
        self.assertEqual(fence, 1)
        # Paused old worker can't publish even with correct candidate bytes.
        with self.assertRaisesRegex(DevelopmentJournalError, "mismatch"):
            commit_development_run_checkpoint(
                self.path, "event-1", state_key=self.KEY,
                run_id=binding.run_id, run_commitment_sha256=commitment,
                checkpoint_sha256=binding.initial_checkpoint_sha256,
                expected_fence=0, next_checkpoint_sha256=uncommitted,
            )
        # Current owner may commit only when the host has verified the exact
        # new checkpoint bytes; journal itself cannot inspect neural tensors.
        commit_development_run_checkpoint(
            self.path, "event-1", state_key=self.KEY,
            run_id=binding.run_id, run_commitment_sha256=commitment,
            checkpoint_sha256=binding.initial_checkpoint_sha256,
            expected_fence=1, next_checkpoint_sha256=uncommitted,
        )
        self.assertEqual(inspect_development_event(
            self.path, "event-1", state_key=self.KEY,
        )["checkpoint_sha256"], uncommitted)

    def test_changed_source_or_maze_commitment_cannot_claim_old_run(self):
        self.reserve()
        for forged in (
            replace(self.binding, source_tree_sha256="0" * 64),
            replace(self.binding, maze_commitment_sha256="1" * 64),
        ):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                DevelopmentJournalError, "mismatch",
            ):
                claim_development_run_resume(
                    self.path, "event-1", state_key=self.KEY,
                    run_id=forged.run_id,
                    run_commitment_sha256=forged.commitment_sha256(),
                    checkpoint_sha256=forged.initial_checkpoint_sha256,
                    expected_fence=0,
                )
