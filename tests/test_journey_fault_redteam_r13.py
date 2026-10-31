"""Adversarial R13 checks of the signed Journey/full-state handoff.

This uses a real temporary SQLite journal and synthetic host/neural callbacks.
It does not exercise pretrained weights, host key custody, or protected exams.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import test_journey_execution_r12 as _fixture_module

from labyrinth.fullstate_successor import (
    FullStateSuccessorError,
    FullStateSuccessorReceipt,
)
from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError,
    inspect_development_event,
)
from labyrinth.journey_execution_r12 import (
    SignedJourneyExecutionError,
    execute_signed_fullstate_journey,
)


class JourneyFaultRedteamTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = _fixture_module.SignedFullStateJourneyTests(
            "test_real_journal_executes_signed_next_maze_with_byte_readback")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.subject = fixture

    def _journal(self) -> dict:
        s = self.subject
        return inspect_development_event(s.db, "dev-event-1", state_key=s.state_key)

    def test_roster_mismatch_preflight_preserves_event_for_corrected_retry(self):
        """R14 preflight prevents the prior R13 event-burn availability defect."""
        s = self.subject
        reversed_roster = replace(
            s.binding, initial_exposures=tuple(reversed(s.exposures)))
        with self.assertRaisesRegex(SignedJourneyExecutionError, "exposure roster"):
            execute_signed_fullstate_journey(
                **s.params(fullstate_binding=reversed_roster))
        self.assertEqual(s.seen, set())
        self.assertEqual(s.model_created, 0)
        self.assertFalse(self._journal()["consumed"])
        # Correcting the local exposure roster must still permit one valid
        # external reservation, under the same synthetic checkpoint mocks.
        mocks = s.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            result = execute_signed_fullstate_journey(**s.params())
        self.assertEqual(s.seen, {"dev-event-1"})
        self.assertFalse(result.graduation_authorized)
        self.assertTrue(self._journal()["consumed"])

    def test_host_binding_denial_consumes_external_event_but_no_worker(self):
        s = self.subject
        with self.assertRaisesRegex(SignedJourneyExecutionError, "host binding"):
            execute_signed_fullstate_journey(
                **s.params(verify_host_binding=lambda _plan, _binding: False))
        self.assertEqual(s.seen, {"dev-event-1"})
        self.assertEqual(s.model_created, 0)
        self.assertFalse(self._journal()["consumed"])

    def test_host_event_revocation_before_neural_load_fails_without_replay(self):
        s = self.subject
        with self.assertRaisesRegex(FullStateSuccessorError, "not verified"):
            execute_signed_fullstate_journey(
                **s.params(verify_signed_event=lambda _event, _binding: False))
        self.assertEqual(s.seen, {"dev-event-1"})
        self.assertEqual(s.model_created, 0)
        self.assertFalse(self._journal()["consumed"])
        with self.assertRaisesRegex(ValueError, "consumed event"):
            execute_signed_fullstate_journey(**s.params())

    def test_model_factory_crash_after_host_reservation_retains_no_commit(self):
        s = self.subject

        def crashed_model():
            s.model_created += 1
            raise RuntimeError("synthetic worker crashed before checkpoint load")

        with self.assertRaisesRegex(RuntimeError, "synthetic worker crashed"):
            execute_signed_fullstate_journey(
                **s.params(model_factory=crashed_model))
        self.assertEqual(s.seen, {"dev-event-1"})
        self.assertEqual(s.model_created, 1)
        self.assertFalse(self._journal()["consumed"])
        self.assertFalse(s.successor.exists())

    def test_attestation_refusal_preserves_uncommitted_checkpoint_orphan(self):
        s = self.subject
        # Synthetic save writes a distinguishable orphan; it does not serialize
        # actual optimizer/model state. The SQLite journal itself is real.
        existing = s.checkpoint_mocks()

        def save_orphan(path: Path, *, context, **_):
            path.mkdir()
            (path / "synthetic-orphan-marker").write_text("uncommitted")
            from labyrinth.neural_resume_checkpoint import FullCheckpointReceipt
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        with existing[0], existing[1], patch(
                "labyrinth.fullstate_successor.save_full_checkpoint",
                side_effect=save_orphan):
            with self.assertRaisesRegex(FullStateSuccessorError,
                                        "attestation missing"):
                execute_signed_fullstate_journey(**s.params(
                    attest_successor_bytes=lambda _path, _receipt: False))
        self.assertTrue((s.successor / "synthetic-orphan-marker").is_file())
        record = self._journal()
        self.assertNotEqual(record["checkpoint_sha256"], "9" * 64)
        self.assertEqual(s.seen, {"dev-event-1"})
        with self.assertRaisesRegex(ValueError, "consumed event"):
            execute_signed_fullstate_journey(**s.params(
                successor_checkpoint_dir=s.db.parent / "retry-next"))

    def test_symlinked_journal_ancestor_fails_before_host_reservation(self):
        s = self.subject
        alias = s.db.parent / "alias-parent"
        try:
            alias.symlink_to(s.db.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("sandbox does not permit temporary symlinks")
        with self.assertRaisesRegex(SignedJourneyExecutionError, "preflight"):
            execute_signed_fullstate_journey(
                **s.params(journal_db=alias / s.db.name))
        self.assertFalse(s.seen)
        self.assertEqual(s.model_created, 0)

    def test_two_claimed_hosts_cannot_repeat_training_via_same_sqlite_journal(self):
        """Host callback alone is not a replay defense; SQLite denies second run."""
        s = self.subject
        trained: list[tuple] = []

        def train(_model, _optimizer, planned, cursor):
            from labyrinth.fullstate_successor import DevelopmentStepReceipt
            trained.append((planned, cursor))
            return DevelopmentStepReceipt(
                planned, len(planned), cursor, cursor + len(planned))

        mocks = s.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            execute_signed_fullstate_journey(
                **s.params(training_step=train))
        self.assertEqual(len(trained), 1)
        record = self._journal()
        self.assertEqual(record["checkpoint_sha256"], "9" * 64)

        # Simulate a second, uncoordinated fake host that claims the same
        # authenticated signed event. Real external host custody is NOT modeled.
        s.seen.clear()
        next_root = s.db.parent / "sibling-host-successor"
        mocks = s.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            with self.assertRaisesRegex(DevelopmentJournalError,
                                        "already consumed"):
                execute_signed_fullstate_journey(**s.params(
                    successor_checkpoint_dir=next_root,
                    training_step=train))
        self.assertEqual(len(trained), 1)
        self.assertFalse(next_root.exists())
        self.assertEqual(self._journal()["checkpoint_sha256"], "9" * 64)

    def test_stale_host_fence_refused_before_model_worker(self):
        s = self.subject
        def stale(proposal: bytes):
            return replace(s.reserve(proposal), fence=6)
        with self.assertRaises(ValueError):
            execute_signed_fullstate_journey(
                **s.params(host_reserve_once=stale))
        self.assertEqual(s.model_created, 0)
        self.assertFalse(self._journal()["consumed"])

    def test_spoofed_successor_run_or_future_fence_not_accepted(self):
        s = self.subject
        for forged in (
            FullStateSuccessorReceipt("another-run", "9" * 64, 0, len(s.exposures)),
            FullStateSuccessorReceipt(s.binding.run_id, "9" * 64, 987, len(s.exposures)),
        ):
            with self.subTest(forged=forged):
                s.seen.clear()  # reset fake host reservation for isolated attempt
                with patch("labyrinth.journey_execution_r12.run_fullstate_development_cycle",
                           return_value=forged):
                    with self.assertRaisesRegex(SignedJourneyExecutionError,
                                                "receipt does not match"):
                        execute_signed_fullstate_journey(**s.params())
        self.assertEqual(s.model_created, 0)
        self.assertFalse(self._journal()["consumed"])


if __name__ == "__main__":
    unittest.main()
