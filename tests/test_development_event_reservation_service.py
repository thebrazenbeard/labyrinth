"""Real SQLite event-consumption tests; these do NOT establish host custody."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest import mock
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from labyrinth.development_event_reservation_service import (
    ReservationConflict, SQLiteDevelopmentEventReservationSink,
)
from tests import test_signed_exam_lineage as lineage_fixtures


class DevelopmentEventReservationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = lineage_fixtures.SignedExamLineageTests(
            "test_exact_scored_exam_candidate_transition_and_nonce_verified")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.binding = fixture.bind()
        self.verify_options = fixture.options.copy()
        self.verify_options.pop("independent_binding_key")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "consumption.sqlite"
        self.sink = self.open_sink(self.db)
        self.run_id = fixture.options["candidate_run_id"]
        self.sink.register_run(
            self.run_id,
            expected_candidate_artifact_sha256=fixture.options[
                "candidate_artifact_sha256"],
            expected_scope_fingerprint=fixture.options["scope"].fingerprint,
            expected_to_cycle=fixture.options["context"].cycle_index,
        )

    def open_sink(self, path: Path, *, sink_id: str = "event-sink-1",
                  signer_id: str = lineage_fixtures.SIGNER,
                  binding_key: bytes = lineage_fixtures.BINDING_KEY,
                  receipt_key: bytes = b"receipt-owned-by-test-sink-20261031!!"):
        return SQLiteDevelopmentEventReservationSink(
            path, sink_id=sink_id, expected_binding_signer_id=signer_id,
            independent_binding_key=binding_key, receipt_key=receipt_key,
        )

    def reserve(self, *, sink=None, binding=None, epoch=1, fence=0, **overrides):
        return (sink or self.sink).reserve_authenticated_binding(
            binding or self.binding, epoch=epoch, fence=fence,
            **{**self.verify_options, **overrides},
        )

    def test_signed_exam_reserved_once_with_exact_scored_identity(self):
        receipt = self.reserve()
        self.assertEqual(receipt.event_id, "dev-event-1")
        self.assertEqual(receipt.exam_id, "dev-exam-0")
        self.assertEqual(receipt.run_id, self.run_id)
        self.assertEqual(receipt.candidate_artifact_sha256,
                         self.fixture.options["candidate_artifact_sha256"])
        self.assertEqual(receipt.scope_fingerprint, self.fixture.options["scope"].fingerprint)
        self.assertEqual(receipt.to_cycle, 1)
        self.assertFalse(receipt.graduation_authorized)
        self.assertFalse(receipt.model_execution_authenticated)
        self.assertEqual(self.sink.reconcile("dev-event-1"), receipt)
        self.assertNotIn("receipt-owned-by-test-sink", repr(receipt))
        with self.assertRaises(ReservationConflict):
            self.reserve()

    def test_tampered_signature_candidate_nonce_or_signer_is_never_consumed(self):
        attempts = (
            (replace(self.binding, signature_hex="f" * 64), {}),
            (self.binding, {"candidate_artifact_sha256": "f" * 64}),
            (self.binding, {"expected_nonce_sha256": "f" * 64}),
            (self.binding, {"binding_signer_id": "wrong-signer"}),
        )
        for binding, overrides in attempts:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.reserve(binding=binding, **overrides)
            self.assertIsNone(self.sink.reconcile("dev-event-1"))

    def test_clone_state_callback_replay_rejected_and_wrong_event_not_consumed(self):
        callback = self.sink.callback_for_binding(
            self.binding, epoch=1, fence=0, **self.verify_options)
        self.assertFalse(callback("unrelated-event"))
        self.assertIsNone(self.sink.reconcile("dev-event-1"))
        self.assertTrue(callback("dev-event-1"))
        self.assertFalse(callback("dev-event-1"))

    def test_two_real_sqlite_handles_race_only_one_reservation(self):
        other = self.open_sink(self.db)
        barrier = threading.Barrier(2)

        def attempt(sink):
            barrier.wait(timeout=3)
            try:
                return self.reserve(sink=sink).event_id
            except ReservationConflict:
                return "DENIED"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, (self.sink, other)))
        self.assertCountEqual(results, ("dev-event-1", "DENIED"))
        self.assertIsNotNone(self.sink.reconcile("dev-event-1"))

    def test_reissued_event_id_for_same_exam_cycle_does_not_reserve_twice(self):
        # The independent signer might reissue a new event ID for the same
        # examined learner/cycle. Isolate the SQL uniqueness invariant after
        # verification, without forging another actual scorer signature.
        self.reserve()
        original = self.sink._authenticated_body(self.binding, self.verify_options)
        rewritten = {**original, "event_id": "reissued-event-2"}
        with mock.patch.object(self.sink, "_authenticated_body", return_value=rewritten):
            with self.assertRaises(ReservationConflict):
                self.reserve()
        self.assertIsNone(self.sink.reconcile("reissued-event-2"))

    def test_sql_tampering_of_reservation_receipt_is_detected(self):
        self.reserve()
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("""UPDATE development_event_reservations
                SET candidate_sha=? WHERE event_id=?""", ("f" * 64, "dev-event-1"))
        with self.assertRaises(ReservationConflict):
            self.sink.reconcile("dev-event-1")

    def test_in_process_receipt_key_swap_refused_before_db_effect(self):
        self.sink._receipt_key = b"foreign-receipt-key-20261031-abcdef"
        with self.assertRaises(ReservationConflict):
            self.reserve()
        with self.assertRaises(ReservationConflict):
            self.sink.reconcile("dev-event-1")
        self.sink = self.open_sink(self.db)
        self.assertIsNone(self.sink.reconcile("dev-event-1"))

    def test_stale_fence_and_malformed_transition_do_not_consume_event(self):
        self.sink.advance_fence(self.run_id, expected_epoch=1, expected_fence=0,
                                new_epoch=2, new_fence=1,
                                new_expected_candidate_artifact_sha256="f" * 64,
                                new_expected_scope_fingerprint=self.fixture.options[
                                    "scope"].fingerprint,
                                new_expected_to_cycle=2)
        with self.assertRaises(ReservationConflict):
            self.reserve(epoch=1, fence=0)
        self.assertIsNone(self.sink.reconcile("dev-event-1"))
        # An old fully authenticated exam cannot be made current merely by
        # giving it the newer epoch/fence: the pinned candidate/cycle moved.
        with self.assertRaises(ReservationConflict):
            self.reserve(epoch=2, fence=1)
        self.assertIsNone(self.sink.reconcile("dev-event-1"))
        with self.assertRaises(ReservationConflict):
            self.sink.advance_fence(
                self.run_id, expected_epoch=1, expected_fence=0,
                new_epoch=2, new_fence=1,
                new_expected_candidate_artifact_sha256="e" * 64,
                new_expected_scope_fingerprint=self.fixture.options[
                    "scope"].fingerprint,
                new_expected_to_cycle=2)

    def test_wrong_pinned_candidate_blocks_even_authentic_four_key_exam(self):
        db = Path(self.temp.name) / "wrong-candidate.sqlite"
        sink = self.open_sink(db)
        sink.register_run(
            self.run_id, expected_candidate_artifact_sha256="f" * 64,
            expected_scope_fingerprint=self.fixture.options["scope"].fingerprint,
            expected_to_cycle=self.fixture.options["context"].cycle_index,
        )
        with self.assertRaises(ReservationConflict):
            self.reserve(sink=sink)
        self.assertIsNone(sink.reconcile("dev-event-1"))

    def test_wrong_database_sink_or_key_cannot_take_over_run(self):
        for changes in (
            {"sink_id": "foreign-sink"},
            {"binding_key": b"foreign-binding-key-20261031-abcdef"},
            {"receipt_key": b"foreign-receipt-key-20261031-abcdef"},
            {"signer_id": "foreign-signer"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ReservationConflict):
                self.open_sink(self.db, **changes)
        self.assertIsNone(self.sink.reconcile("dev-event-1"))
        self.reserve()
        self.assertIsNotNone(self.sink.reconcile("dev-event-1"))

    def test_fault_after_insert_rolls_back_atomically_and_allows_retry(self):
        def fault():
            raise RuntimeError("controlled before-commit failure")

        self.sink._before_commit = fault
        with self.assertRaisesRegex(RuntimeError, "before-commit"):
            self.reserve()
        self.assertIsNone(self.sink.reconcile("dev-event-1"))
        self.sink._before_commit = lambda: None
        self.reserve()

    def test_lost_ack_after_commit_reconciles_and_replay_stays_denied(self):
        self.sink._after_commit = lambda: (_ for _ in ()).throw(
            RuntimeError("transport lost acknowledgment"))
        with self.assertRaisesRegex(RuntimeError, "lost acknowledgment"):
            self.reserve()
        receipt = self.sink.reconcile("dev-event-1")
        self.assertIsNotNone(receipt)
        self.sink._after_commit = lambda: None
        with self.assertRaises(ReservationConflict):
            self.reserve()

    def test_valid_older_db_snapshot_can_replay_without_external_anchor(self):
        snapshot = Path(self.temp.name) / "prior-valid.sqlite"
        with sqlite3.connect(str(self.db)) as source:
            with sqlite3.connect(str(snapshot)) as target:
                source.backup(target)
        self.reserve()
        self.assertIsNotNone(self.sink.reconcile("dev-event-1"))
        restored = self.open_sink(snapshot)
        self.assertIsNone(restored.reconcile("dev-event-1"))
        # Deliberate FAIL evidence: independent monotonic anchor is absent.
        self.assertEqual(self.reserve(sink=restored).event_id, "dev-event-1")

    def test_opt_in_measured_successor_reaches_real_reservation_callback(self):
        from labyrinth.hf_pilot import PilotConfig
        from tests import test_measured_taskmaster_transition as taskmaster_fixtures

        fixture = taskmaster_fixtures.MeasuredTaskmasterTransitionTests(
            "test_same_signed_score_yields_weakness_tactic_and_retained_learner")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        options = fixture.common()
        config = PilotConfig(model_path="/not-loaded", output_dir="/not-written",
                             seed=33, train_count=4, eval_count=4,
                             tiers=(1,), steps=4)

        def downstream(*args, **kwargs):
            # Mock ONLY the final physical trainer; the real source preflight,
            # signed lineage and real SQLite callback all execute.
            return {"reserved": kwargs["reserve_development_event"]("dev-event-1")}

        with mock.patch("labyrinth.hf_development_transition.run_exam_bound_successor",
                        side_effect=downstream):
            prepared, result = self.sink.run_bound_measured_successor(
                config, Path("/nonexistent-predecessor"), epoch=1, fence=0,
                **options)
        self.assertTrue(result["reserved"])
        self.assertFalse(prepared.host_event_consumption_verified)
        self.assertFalse(prepared.model_execution_authenticated)
        self.assertIsNotNone(self.sink.reconcile("dev-event-1"))
        callback = self.sink.callback_for_binding(
            self.binding, epoch=1, fence=0, **self.verify_options)
        self.assertFalse(callback("dev-event-1"))

    def test_safe_call_site_rejects_wrong_journey_before_event_reservation(self):
        from labyrinth.hf_pilot import PilotConfig
        from tests import test_measured_taskmaster_transition as taskmaster_fixtures

        fixture = taskmaster_fixtures.MeasuredTaskmasterTransitionTests(
            "test_same_signed_score_yields_weakness_tactic_and_retained_learner")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        options = fixture.common()
        options["state"] = replace(options["state"], checkpoint_sha256="f" * 64)
        config = PilotConfig(model_path="/not-loaded", output_dir="/not-written",
                             seed=33, train_count=4, eval_count=4,
                             tiers=(1,), steps=4)
        with self.assertRaisesRegex(ValueError, "Journey differs"):
            self.sink.run_bound_measured_successor(
                config, Path("/nonexistent"), epoch=1, fence=0, **options)
        self.assertIsNone(self.sink.reconcile("dev-event-1"))


if __name__ == "__main__":
    unittest.main()
