"""Durable, bounded local event journal; simulated crashes, not host attestation."""

from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase, mock

from labyrinth.hf_development_event_journal import (
    DevelopmentJournalError, initialize_development_event_journal,
    inspect_development_event, reconcile_development_event,
    reserve_development_event,
)
from labyrinth.hf_development_transition import (
    SCHEMA as TRANSITION_SCHEMA, _DOMAIN as TRANSITION_DOMAIN,
    run_exam_bound_successor,
)
from labyrinth.hf_pilot import PilotConfig
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.training_scope import TrainingScopeSpec


class DevelopmentEventJournalTests(TestCase):
    KEY = b"host-owned-journal-secret-32-bytes!!"
    EVALUATOR_KEY = b"distinct-development-evaluator-secret-32!!"

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "development-events.sqlite"
        initialize_development_event_journal(self.path, state_key=self.KEY)

    def reserve(self, event_id: str) -> bool:
        return reserve_development_event(self.path, event_id, state_key=self.KEY)

    def inspect(self, event_id: str) -> dict:
        return inspect_development_event(self.path, event_id, state_key=self.KEY)

    def test_event_reservation_survives_new_connection_and_replay(self):
        self.assertFalse(self.inspect("event-one")["consumed"])
        self.assertIs(self.reserve("event-one"), True)
        self.assertIs(self.inspect("event-one")["consumed"], True)
        self.assertIs(self.reserve("event-one"), False)
        self.assertTrue(self.reserve("event-two"))
        self.assertFalse(self.reserve("event-two"))
        with self.assertRaises(DevelopmentJournalError):
            initialize_development_event_journal(self.path, state_key=self.KEY)

    def test_observation_never_releases_event_or_asserts_graduation(self):
        self.reserve("event-one")
        reconcile_development_event(
            self.path, "event-one", state_key=self.KEY,
            observation="REPORTED_RESULT", artifact_sha256="a" * 64,
            review_ref="operator-review-1",
        )
        result = self.inspect("event-one")
        self.assertTrue(result["consumed"])
        self.assertTrue(result["operator_observation_only"])
        self.assertEqual(result["observation"], "REPORTED_RESULT")
        self.assertEqual(result["review_ref"], "operator-review-1")
        self.assertFalse(self.reserve("event-one"))
        with self.assertRaisesRegex(DevelopmentJournalError, "already reconciled"):
            reconcile_development_event(
                self.path, "event-one", state_key=self.KEY,
                observation="NOT_STARTED", artifact_sha256="b" * 64,
                review_ref="operator-review-2",
            )

    def test_no_reconciliation_of_unreserved_or_malformed_events(self):
        with self.assertRaisesRegex(DevelopmentJournalError, "unreserved"):
            reconcile_development_event(
                self.path, "not-reserved", state_key=self.KEY,
                observation="UNKNOWN", artifact_sha256="0" * 64,
                review_ref="operator-1",
            )
        for bad in ("../escape", "", "x" * 129, 12):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.reserve(bad)
        self.reserve("good-event")
        for bad in ("PASS", "GRADUATED", "RELEASED", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reconcile_development_event(
                    self.path, "good-event", state_key=self.KEY,
                    observation=bad, artifact_sha256="0" * 64,
                    review_ref="operator-1",
                )
        self.assertIsNone(self.inspect("good-event")["observation"])

    def test_wrong_key_or_state_modification_fails_closed(self):
        self.reserve("event-one")
        with self.assertRaisesRegex(DevelopmentJournalError, "seal"):
            inspect_development_event(self.path, "event-one", state_key=b"x" * 32)
        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute("UPDATE journal SET events_json=? WHERE id=1",
                           ('{"events":{}}',))
        with self.assertRaisesRegex(DevelopmentJournalError, "seal"):
            self.reserve("event-two")

    def test_transaction_abandoned_before_commit_is_not_consumed(self):
        # Simulated interruption BEFORE the reservation callback can return True.
        # SQLite rolls back the abandoned transaction and training must not launch.
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE journal SET version=version+1 WHERE id=1")
            db.rollback()
        self.assertFalse(self.inspect("event-one")["consumed"])
        self.assertTrue(self.reserve("event-one"))

    def test_concurrent_duplicate_attempts_admit_at_most_one(self):
        outcomes = []
        failures = []
        def worker():
            try:
                outcomes.append(self.reserve("single-event"))
            except BaseException as exc:
                failures.append(exc)
        threads = [Thread(target=worker) for _ in range(8)]
        for worker_thread in threads:
            worker_thread.start()
        for worker_thread in threads:
            worker_thread.join(timeout=10)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), 7)

    def test_signed_transition_post_reservation_exception_is_consumed(self):
        # This mocks the model trainer raising after a committed host reservation.
        # No actual neural process, crash, or external evaluator was exercised.
        scope = TrainingScopeSpec("learn-arithmetic", "reasoning", ("arithmetic",), 1, 1, 3, 17)
        context = ResumeContext(scope.fingerprint, 1, "a" * 64, "b" * 64, "authority-one")
        receipt = {
            "schema": TRANSITION_SCHEMA, "event_id": "failed-development-1",
            "exam_id": "development-exam-1", "exam_kind": "development", "exam_status": "FAIL",
            "objective_id": scope.objective_id, "scope_fingerprint": scope.fingerprint,
            "from_cycle": 0, "to_cycle": 1,
            "predecessor_report_sha256": context.predecessor_report_sha256,
            "predecessor_lora_value_sha256": context.predecessor_lora_value_sha256,
            "resume_authority_ref": context.authority_ref, "weak_skills": ["arithmetic"],
            "exam_case_commitment_sha256": "c" * 64,
        }
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(self.EVALUATOR_KEY, TRANSITION_DOMAIN + raw,
                             hashlib.sha256).hexdigest()
        config = PilotConfig(model_path="unused", output_dir="unused")
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         side_effect=RuntimeError("simulated post-reservation interruption")) as train):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_exam_bound_successor(
                    config, scope, "unused", context, raw, signature,
                    independent_development_key=self.EVALUATOR_KEY,
                    reserve_development_event=self.reserve,
                )
            self.assertTrue(self.inspect("failed-development-1")["consumed"])
            with self.assertRaisesRegex(ValueError, "not newly reserved"):
                run_exam_bound_successor(
                    config, scope, "unused", context, raw, signature,
                    independent_development_key=self.EVALUATOR_KEY,
                    reserve_development_event=self.reserve,
                )
            self.assertEqual(train.call_count, 1)

    def test_preinitialized_directory_and_full_rollback_limitation(self):
        with self.assertRaises(ValueError):
            reserve_development_event(Path("relative.db"), "event", state_key=self.KEY)
        with self.assertRaises(ValueError):
            reserve_development_event(self.path, "event", state_key=b"short")
        alternate = self.directory / "symlink.sqlite"
        try:
            alternate.symlink_to(self.path)
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaisesRegex(ValueError, "type"):
                reserve_development_event(alternate, "event", state_key=self.KEY)
