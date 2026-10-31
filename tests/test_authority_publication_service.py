"""Real SQLite transaction tests for the uninstalled sink service candidate."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from labyrinth.atomic_artifact_publication import (
    ArtifactPublicationError, PublicationDenied, PublicationIntent,
    PublicationOutcomeUnknown, SinkConflict, publish_with_sink,
)
from labyrinth.authority_publication_service import (
    SQLiteAtomicPublicationSink, verify_sink_receipt,
)


class SQLiteSinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "private-sink.db"
        self.key = b"sink-only-test-key-not-worker-owned!!"
        self.sink = SQLiteAtomicPublicationSink(
            self.db, sink_id="sink-alpha", receipt_key=self.key,
        )
        self.sink.register_run("run-one", current_sha256="0" * 64)

    def _intent(self, operation: str, staged: str, payload: bytes,
                *, epoch: int = 1, fence: int = 0,
                run: str = "run-one", predecessor: str = "0" * 64):
        return PublicationIntent(
            "sink-alpha", operation, run, "checkpoint", staged,
            epoch, fence, predecessor, sha256(payload).hexdigest(),
        )

    def _verify(self, payload: bytes, signature: str, sink_id: str) -> bool:
        return verify_sink_receipt(
            payload, signature, sink_id,
            pinned_sink_id="sink-alpha", pinned_receipt_key=self.key,
        )

    def _publish(self, intent: PublicationIntent, *, sink=None):
        return publish_with_sink(
            self.sink if sink is None else sink, intent,
            expected_sink_id="sink-alpha",
            verify_receipt_signature=self._verify,
        )

    def test_durable_commit_reopen_and_readback(self):
        body = b"model-weights-v1"
        self.assertEqual(self.sink.stage(
            run_id="run-one", staged_object_id="stage-one", payload=body,
        ), sha256(body).hexdigest())
        intent = self._intent("op-one", "stage-one", body)
        receipt = self._publish(intent)
        self.assertEqual(receipt.intent, intent)
        reopened = SQLiteAtomicPublicationSink(
            self.db, sink_id="sink-alpha", receipt_key=self.key,
        )
        current = reopened.read_current("run-one")
        self.assertIsNotNone(current)
        self.assertEqual(current.payload, body)
        self.assertEqual(current.receipt, receipt)
        self.assertEqual(reopened.reconcile_receipt("op-one"), receipt)

    def test_same_db_rejects_foreign_sink_id_and_key_before_mutation(self):
        body = b"original-authorized-weights"
        self.sink.stage(run_id="run-one", staged_object_id="stage-one", payload=body)
        original = self._publish(self._intent("op-original", "stage-one", body))
        for sink_id, key in (
            ("sink-beta", b"another-independent-test-sink-key!!"),
            ("sink-alpha", b"different-key-for-the-same-sink!!!!"),
        ):
            with self.subTest(sink_id=sink_id, key_changed=key != self.key):
                with self.assertRaises((SinkConflict, ArtifactPublicationError)):
                    SQLiteAtomicPublicationSink(
                        self.db, sink_id=sink_id, receipt_key=key,
                    )
        # Cross-tenant callers never damaged the original authority or MAC.
        self.assertEqual(self.sink.read_current("run-one").receipt, original)
        self.assertEqual(self.sink.read_current("run-one").payload, body)
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM publications").fetchone()[0], 1)

    def test_unbound_legacy_authority_database_refuses_implicit_ownership(self):
        """An identity-less populated DB cannot be claimed with a new key."""
        legacy = Path(self.tmp.name) / "unbound-legacy.db"
        with sqlite3.connect(str(legacy)) as conn:
            conn.execute("CREATE TABLE run_authority (run_id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO run_authority VALUES ('old-run')")
        with self.assertRaises((SinkConflict, ArtifactPublicationError)):
            SQLiteAtomicPublicationSink(
                legacy, sink_id="sink-new", receipt_key=self.key,
            )

    def test_existing_handle_refuses_changed_db_binding_before_new_write(self):
        """A process must not keep using a DB after its sink identity changes."""
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute("UPDATE sink_identity SET sink_id=? WHERE singleton=1",
                         ("different-sink",))
        with self.assertRaises(SinkConflict):
            self.sink.stage(run_id="run-one", staged_object_id="must-not-stage",
                            payload=b"not-authorized")
        with self.assertRaises(SinkConflict):
            self.sink.read_current("run-one")
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM staged").fetchone()[0], 0)

    def test_stale_worker_denied_after_fence_advance_no_visible_blob(self):
        old = b"old-worker-weights"
        new = b"new-worker-weights"
        self.sink.stage(run_id="run-one", staged_object_id="old-stage", payload=old)
        self.sink.stage(run_id="run-one", staged_object_id="new-stage", payload=new)
        stale = self._intent("op-old", "old-stage", old)
        self.sink.advance_fence("run-one", expected_epoch=1, expected_fence=0,
                                new_epoch=2, new_fence=1)
        with self.assertRaises(PublicationDenied):
            self._publish(stale)
        self.assertIsNone(self.sink.read_current("run-one"))
        self.assertIsNone(self.sink.reconcile_receipt("op-old"))
        fresh = self._intent("op-new", "new-stage", new, epoch=2, fence=1)
        self._publish(fresh)
        self.assertEqual(self.sink.read_current("run-one").payload, new)

    def test_two_real_sqlite_connections_one_publication(self):
        first, second = b"candidate-one", b"candidate-two"
        self.sink.stage(run_id="run-one", staged_object_id="s-one", payload=first)
        self.sink.stage(run_id="run-one", staged_object_id="s-two", payload=second)
        intents = (
            self._intent("op-one", "s-one", first),
            self._intent("op-two", "s-two", second),
        )
        barrier = Barrier(2)

        def attempt(intent):
            # Each competing caller owns a distinct actual DB connection.
            sink = SQLiteAtomicPublicationSink(
                self.db, sink_id="sink-alpha", receipt_key=self.key,
            )
            barrier.wait(timeout=5)
            try:
                return ("published", self._publish(intent, sink=sink))
            except PublicationDenied:
                return ("denied", None)

        with ThreadPoolExecutor(max_workers=2) as workers:
            outcomes = list(workers.map(attempt, intents))
        self.assertEqual(sorted(status for status, _ in outcomes),
                         ["denied", "published"])
        winner = next(item[1] for item in outcomes if item[0] == "published")
        self.assertEqual(self.sink.read_current("run-one").receipt, winner)
        self.assertEqual(self.sink.read_current("run-one").payload,
                         first if winner.intent.operation_id == "op-one" else second)

    def test_same_operation_cannot_publish_twice(self):
        body = b"one-shot"
        self.sink.stage(run_id="run-one", staged_object_id="stage-one", payload=body)
        intent = self._intent("op-once", "stage-one", body)
        first = self._publish(intent)
        with self.assertRaises(PublicationDenied):
            self._publish(intent)
        self.assertEqual(self.sink.reconcile_receipt("op-once"), first)
        self.assertEqual(self.sink.read_current("run-one").payload, body)

    def test_cross_run_stage_and_wrong_digest_never_visible(self):
        body = b"cross-run"
        self.sink.stage(run_id="run-one", staged_object_id="stage-one", payload=body)
        self.sink.register_run("run-two")
        cross = self._intent("op-cross", "stage-one", body, run="run-two")
        with self.assertRaises(PublicationDenied):
            self._publish(cross)
        self.assertIsNone(self.sink.read_current("run-two"))
        mismatch = PublicationIntent(
            "sink-alpha", "op-wrong-hash", "run-one", "checkpoint",
            "stage-one", 1, 0, "0" * 64, "f" * 64,
        )
        with self.assertRaises(PublicationDenied):
            self._publish(mismatch)
        self.assertIsNone(self.sink.read_current("run-one"))

    def test_timeout_after_durable_commit_is_unknown_and_reconcilable(self):
        body = b"crash-after-commit"
        self.sink.stage(run_id="run-one", staged_object_id="stage-one", payload=body)
        intent = self._intent("op-ambiguous", "stage-one", body)
        real_sink = self.sink

        class FaultAfterCommit:
            def commit_if_current(self, call):
                real_sink.commit_if_current(call)
                raise TimeoutError("response lost after commit")

        with self.assertRaises(PublicationOutcomeUnknown):
            self._publish(intent, sink=FaultAfterCommit())
        acknowledged = self.sink.reconcile_receipt("op-ambiguous")
        self.assertIsNotNone(acknowledged)
        self.assertEqual(acknowledged.intent, intent)
        self.assertEqual(self.sink.read_current("run-one").payload, body)

    def test_sqlite_fault_after_insert_before_head_update_is_atomic_rollback(self):
        body = b"must-never-be-half-visible"
        self.sink.stage(run_id="run-one", staged_object_id="stage-fault", payload=body)
        intent = self._intent("op-fault", "stage-fault", body)
        # SQLite itself aborts the second write *inside* the transaction,
        # after the publication INSERT and before authority-head UPDATE.
        with sqlite3.connect(str(self.db)) as conn:
            conn.executescript("""
                CREATE TRIGGER fault_head BEFORE UPDATE ON run_authority
                BEGIN SELECT RAISE(ABORT, 'injected-head-write-fault'); END;
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.sink.commit_if_current(intent)
        self.assertIsNone(self.sink.read_current("run-one"))
        self.assertIsNone(self.sink.reconcile_receipt("op-fault"))
        with sqlite3.connect(str(self.db)) as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM publications").fetchone()[0], 0)

    def test_forged_receipt_rejected_on_readback(self):
        body = b"authenticated"
        self.sink.stage(run_id="run-one", staged_object_id="stage-one", payload=body)
        self._publish(self._intent("op-auth", "stage-one", body))
        with sqlite3.connect(str(self.db)) as conn:
            conn.execute(
                "UPDATE publications SET signature_hex=? WHERE operation_id=?",
                ("0" * 64, "op-auth"),
            )
        with self.assertRaises(ArtifactPublicationError):
            self.sink.read_current("run-one")
        with self.assertRaises(ArtifactPublicationError):
            self.sink.reconcile_receipt("op-auth")

    def test_database_rollback_remains_possible_without_external_anchor(self):
        """Expected falsifier: local signed DB can be rolled back by its admin."""
        first = b"accepted-v1"
        self.sink.stage(run_id="run-one", staged_object_id="s-one", payload=first)
        self._publish(self._intent("op-one", "s-one", first))
        snapshot = Path(self.tmp.name) / "old-valid.db"
        with sqlite3.connect(str(self.db)) as source:
            with sqlite3.connect(str(snapshot)) as backup:
                source.backup(backup)
        second = b"accepted-v2"
        self.sink.stage(run_id="run-one", staged_object_id="s-two", payload=second)
        self.sink.advance_fence("run-one", expected_epoch=1, expected_fence=0,
                                new_epoch=2, new_fence=1)
        self._publish(self._intent(
            "op-two", "s-two", second, epoch=2, fence=1,
            predecessor=sha256(first).hexdigest(),
        ))
        self.assertEqual(self.sink.read_current("run-one").payload, second)
        # The older MAC-valid DB snapshot still reads as authoritative if
        # independently administered minimum epoch is missing.
        rolled_back = SQLiteAtomicPublicationSink(
            snapshot, sink_id="sink-alpha", receipt_key=self.key,
        )
        self.assertEqual(rolled_back.read_current("run-one").payload, first)
        self.assertIsNone(rolled_back.reconcile_receipt("op-two"))

    def test_invalid_transition_denied_without_mutation(self):
        for next_epoch, next_fence in (
            (2.0, 1), (2, 1.0), (True, 1), (2, True),
            (2**63 - 1, 1), (2, 2**63 - 1),
        ):
            with self.subTest(next_epoch=next_epoch, next_fence=next_fence):
                with self.assertRaises(SinkConflict):
                    self.sink.advance_fence("run-one", expected_epoch=1,
                                            expected_fence=0,
                                            new_epoch=next_epoch,
                                            new_fence=next_fence)
        with self.assertRaises(SinkConflict):
            self.sink.advance_fence("run-one", expected_epoch=1,
                                    expected_fence=0, new_epoch=2, new_fence=0)
        with self.assertRaises(SinkConflict):
            self.sink.advance_fence("run-one", expected_epoch=9,
                                    expected_fence=9, new_epoch=10, new_fence=10)
        self.assertIsNone(self.sink.read_current("run-one"))
        body = b"still-authorized"
        self.sink.stage(run_id="run-one", staged_object_id="s-valid", payload=body)
        self._publish(self._intent("op-valid", "s-valid", body))
        self.assertEqual(self.sink.read_current("run-one").payload, body)


if __name__ == "__main__":
    unittest.main()
