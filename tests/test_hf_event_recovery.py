"""Host journal recovery-fence contracts, not physical optimizer recovery."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase

from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError, claim_development_run_resume,
    commit_development_run_checkpoint, complete_development_run,
    initialize_development_event_journal, inspect_development_event,
    reconcile_development_event, reserve_development_event, reserve_development_run,
)


class RecoveryFenceTests(TestCase):
    KEY = b"separate-host-journal-custodian-key-32!!"
    RUN = "same-logical-run"
    COMMITMENT = "a" * 64
    INITIAL = "b" * 64

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "host-events.sqlite"
        initialize_development_event_journal(self.db, state_key=self.KEY)

    def reserve(self, event="event-fail-1"):
        return reserve_development_run(
            self.db, event, state_key=self.KEY, run_id=self.RUN,
            run_commitment_sha256=self.COMMITMENT,
            initial_checkpoint_sha256=self.INITIAL,
        )

    def resume(self, *, run=None, commitment=None, checkpoint=None, fence=0):
        return claim_development_run_resume(
            self.db, "event-fail-1", state_key=self.KEY,
            run_id=self.RUN if run is None else run,
            run_commitment_sha256=self.COMMITMENT if commitment is None else commitment,
            checkpoint_sha256=self.INITIAL if checkpoint is None else checkpoint,
            expected_fence=fence,
        )

    def test_run_reservation_is_consumed_and_not_reissued_even_for_same_run(self):
        self.assertEqual(self.reserve(), 0)
        state = inspect_development_event(self.db, "event-fail-1", state_key=self.KEY)
        self.assertTrue(state["consumed"])
        self.assertEqual(state["fence"], 0)
        self.assertEqual(state["run_state"], "ACTIVE")
        self.assertFalse(reserve_development_event(self.db, "event-fail-1", state_key=self.KEY))
        with self.assertRaisesRegex(DevelopmentJournalError, "already consumed"):
            self.reserve()
        self.assertNotIn("graduated", state)

    def test_new_run_or_other_checkpoint_cannot_recover_same_signed_event(self):
        self.reserve()
        for kwargs in ({"run": "impostor"}, {"commitment": "f" * 64},
                       {"checkpoint": "f" * 64}, {"fence": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                DevelopmentJournalError, "mismatch"
            ):
                self.resume(**kwargs)
        self.assertEqual(self.resume(), 1)
        with self.assertRaisesRegex(DevelopmentJournalError, "mismatch"):
            self.resume()

    def test_checkpoint_resume_fences_stale_worker_and_completion_is_terminal(self):
        self.reserve()
        new_fence = self.resume()
        with self.assertRaisesRegex(DevelopmentJournalError, "mismatch"):
            commit_development_run_checkpoint(
                self.db, "event-fail-1", state_key=self.KEY, run_id=self.RUN,
                run_commitment_sha256=self.COMMITMENT,
                checkpoint_sha256=self.INITIAL, expected_fence=0,
                next_checkpoint_sha256="c" * 64,
            )
        commit_development_run_checkpoint(
            self.db, "event-fail-1", state_key=self.KEY, run_id=self.RUN,
            run_commitment_sha256=self.COMMITMENT,
            checkpoint_sha256=self.INITIAL, expected_fence=new_fence,
            next_checkpoint_sha256="c" * 64,
        )
        with self.assertRaisesRegex(DevelopmentJournalError, "mismatch"):
            self.resume(fence=1)
        self.assertEqual(self.resume(checkpoint="c" * 64, fence=1), 2)
        complete_development_run(
            self.db, "event-fail-1", state_key=self.KEY, run_id=self.RUN,
            run_commitment_sha256=self.COMMITMENT,
            checkpoint_sha256="c" * 64, expected_fence=2,
            result_sha256="d" * 64,
        )
        state = inspect_development_event(self.db, "event-fail-1", state_key=self.KEY)
        self.assertEqual(state["run_state"], "COMPLETED")
        self.assertEqual(state["result_sha256"], "d" * 64)
        reconcile_development_event(
            self.db, "event-fail-1", state_key=self.KEY,
            observation="REPORTED_RESULT", artifact_sha256="d" * 64,
            review_ref="operator-review-after-result",
        )
        observed = inspect_development_event(self.db, "event-fail-1", state_key=self.KEY)
        self.assertEqual(observed["run_state"], "COMPLETED")
        self.assertEqual(observed["fence"], 2)
        with self.assertRaisesRegex(DevelopmentJournalError, "completed"):
            self.resume(checkpoint="c" * 64, fence=2)

    def test_concurrent_same_run_resume_claims_only_one_fence(self):
        self.reserve()
        outcomes, failures = [], []

        def worker():
            try:
                outcomes.append(self.resume())
            except DevelopmentJournalError as exc:
                failures.append(str(exc))

        threads = [Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes, [1])
        self.assertEqual(len(failures), 5)
        self.assertTrue(all("mismatch" in message for message in failures))

    def test_legacy_event_cannot_be_silently_upgraded_and_tamper_fails(self):
        self.assertTrue(reserve_development_event(self.db, "event-fail-1", state_key=self.KEY))
        with self.assertRaisesRegex(DevelopmentJournalError, "legacy-only"):
            self.resume()
        with self.assertRaisesRegex(DevelopmentJournalError, "already consumed"):
            self.reserve()
        with sqlite3.connect(self.db) as connection:
            connection.execute("UPDATE journal SET events_json = ? WHERE id=1", ('{"events":{}}',))
        with self.assertRaisesRegex(DevelopmentJournalError, "seal"):
            self.resume()

    def test_malformed_run_inputs_fail_without_reserving_event(self):
        for bad in ("", "../run", "r" * 129):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reserve_development_run(
                    self.db, "event-fail-1", state_key=self.KEY, run_id=bad,
                    run_commitment_sha256=self.COMMITMENT,
                    initial_checkpoint_sha256=self.INITIAL,
                )
        self.assertFalse(inspect_development_event(
            self.db, "event-fail-1", state_key=self.KEY,
        )["consumed"])

    def test_operator_partial_observation_preserves_run_identity_and_recovery(self):
        self.reserve()
        reconcile_development_event(
            self.db, "event-fail-1", state_key=self.KEY,
            observation="PARTIAL_ARTIFACT", artifact_sha256="e" * 64,
            review_ref="operator-review-1",
        )
        state = inspect_development_event(self.db, "event-fail-1", state_key=self.KEY)
        self.assertEqual(state["run_id"], self.RUN)
        self.assertEqual(state["checkpoint_sha256"], self.INITIAL)
        self.assertEqual(state["observation"], "PARTIAL_ARTIFACT")
        self.assertEqual(self.resume(), 1)
        self.assertFalse(reserve_development_event(self.db, "event-fail-1", state_key=self.KEY))
