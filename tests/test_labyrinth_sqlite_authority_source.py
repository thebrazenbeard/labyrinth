"""Storage/CAS contract tests; no host-ownership or graduation claim."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict,
    ExternalAuthorityError,
    ExternalAuthorityOutcomeUnknown,
    ExternalMonotonicReservationClient,
    ExternalReservationRequest,
)
from experiments.labyrinth_sqlite_authority.sqlite_reservation_port import SQLiteReservationPort


def request(**overrides) -> ExternalReservationRequest:
    values = dict(
        provider_id="host-operator", run_id="run-1",
        event_id="dev-event-1", exam_id="dev-exam-1",
        candidate_artifact_sha256="a" * 64,
        scope_fingerprint="b" * 64,
        exam_lineage_sha256="c" * 64,
        prior_checkpoint_sha256="d" * 64,
        to_cycle=2, expected_epoch=3, expected_fence=11,
        expected_commit_index=0, expected_head_sha256="0" * 64,
        operation_nonce="nonce-1",
    )
    values.update(overrides)
    return ExternalReservationRequest(**values)


class SQLiteReservationPortTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "authority.sqlite3"
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.intent = request()
        self.port = SQLiteReservationPort(
            db_path=self.database,
            provider_id=self.intent.provider_id,
            sign_bytes=self.private_key.sign,
            pinned_ed25519_public_key=self.public_key,
        )
        self.port.operator_bootstrap_run(self.intent.expected_head())
        self.port.operator_stage_intent(self.intent)

    def client(self, port=None, intent=None):
        return ExternalMonotonicReservationClient(
            port=self.port if port is None else port,
            provider_id=self.intent.provider_id,
            pinned_request=self.intent if intent is None else intent,
            pinned_ed25519_public_key=self.public_key,
        )

    def test_real_ed25519_client_single_reservation_and_readback(self):
        receipt = self.client().reserve(self.intent)
        self.assertEqual(receipt.commit_index, 1)
        self.assertEqual(self.port.read_head("run-1"), receipt.post_head())
        self.assertEqual(self.port.read_receipt("run-1", "dev-event-1"), receipt)
        self.assertFalse(receipt.graduation_authorized)
        self.assertFalse(receipt.model_execution_authenticated)

    def test_duplicate_event_replay_denied_before_signing(self):
        self.client().reserve(self.intent)
        with self.assertRaises(ExternalAuthorityConflict):
            self.client().reserve(self.intent)
        self.assertEqual(self.port.read_head("run-1").commit_index, 1)

    def test_unstaged_fabricated_candidate_rejected_even_with_valid_head(self):
        bogus = replace(self.intent, candidate_artifact_sha256="e" * 64)
        with self.assertRaises(ExternalAuthorityConflict):
            self.port.reserve_once(bogus)
        self.assertEqual(self.port.read_head("run-1"), self.intent.expected_head())

    def test_operator_cannot_reset_existing_run_or_stage_duplicate_exam(self):
        with self.assertRaises(ExternalAuthorityConflict):
            self.port.operator_bootstrap_run(self.intent.expected_head())
        next_event = replace(self.intent, event_id="other-event", operation_nonce="other-nonce")
        with self.assertRaises(ExternalAuthorityConflict):
            self.port.operator_stage_intent(next_event)
        next_nonce = replace(self.intent, event_id="another-event", exam_id="other-exam")
        with self.assertRaises(ExternalAuthorityConflict):
            self.port.operator_stage_intent(next_nonce)

    def test_pre_admitted_competing_events_cannot_share_same_head(self):
        second = replace(self.intent, event_id="other-event", exam_id="other-exam",
                         operation_nonce="other-nonce")
        self.port.operator_stage_intent(second)
        self.client().reserve(self.intent)
        with self.assertRaises(ExternalAuthorityConflict):
            self.port.reserve_once(second)
        self.assertIsNone(self.port.read_receipt("run-1", "other-event"))

    def test_forged_string_subclasses_denied_before_sql_comparison(self):
        class GhostStr(str):
            def __eq__(self, _other):
                return True

            def __ne__(self, _other):
                return False

        bogus = replace(self.intent, provider_id=GhostStr("other-host"))
        with self.assertRaises(ExternalAuthorityError):
            self.port.operator_stage_intent(bogus)
        with self.assertRaises(ExternalAuthorityError):
            self.port.reserve_once(bogus)

    def test_atomicity_when_signer_fails(self):
        def broken_signer(_payload):
            raise IOError("signing device offline")

        bad = SQLiteReservationPort(
            db_path=self.database, provider_id="host-operator",
            sign_bytes=broken_signer,
            pinned_ed25519_public_key=self.public_key,
        )
        with self.assertRaises(OSError):
            bad.reserve_once(self.intent)
        self.assertEqual(bad.read_head("run-1"), self.intent.expected_head())
        self.assertIsNone(bad.read_receipt("run-1", "dev-event-1"))
        # A healthy signer may still commit because nothing was consumed.
        self.assertEqual(self.client().reserve(self.intent).commit_index, 1)

    def test_invalid_but_64_byte_signature_does_not_consume_event(self):
        for signer in (lambda _payload: b"\x00" * 64,
                       Ed25519PrivateKey.generate().sign):
            bad = SQLiteReservationPort(
                db_path=self.database, provider_id="host-operator",
                sign_bytes=signer, pinned_ed25519_public_key=self.public_key,
            )
            with self.assertRaises(ExternalAuthorityError):
                bad.reserve_once(self.intent)
            self.assertEqual(bad.read_head("run-1"), self.intent.expected_head())
            self.assertIsNone(bad.read_receipt("run-1", "dev-event-1"))
        self.assertEqual(self.client().reserve(self.intent).commit_index, 1)

    def test_database_file_and_ancestor_symlinks_rejected(self):
        direct = Path(self.temporary.name) / "redirect.sqlite3"
        parent = Path(self.temporary.name) / "redirect-dir"
        try:
            direct.symlink_to(self.database)
            parent.symlink_to(Path(self.temporary.name), target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        for redirected in (direct, parent / "authority.sqlite3"):
            with self.subTest(redirected=str(redirected)):
                with self.assertRaises(ExternalAuthorityError):
                    SQLiteReservationPort(
                        db_path=redirected, provider_id="host-operator",
                        sign_bytes=self.private_key.sign,
                        pinned_ed25519_public_key=self.public_key,
                    )

    def test_concurrent_process_style_writers_one_wins(self):
        barrier = Barrier(3)
        results: list[str] = []

        def writer():
            other = SQLiteReservationPort(
                db_path=self.database, provider_id="host-operator",
                sign_bytes=self.private_key.sign,
                pinned_ed25519_public_key=self.public_key,
            )
            barrier.wait()
            try:
                other.reserve_once(self.intent)
                results.append("COMMIT")
            except ExternalAuthorityConflict:
                results.append("CONFLICT")

        threads = [Thread(target=writer), Thread(target=writer)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(results, ["COMMIT", "CONFLICT"])
        self.assertEqual(self.port.read_head("run-1").commit_index, 1)

    def test_lost_ack_outcome_unknown_and_reconciliation(self):
        class LostAck:
            def read_head(_self, run_id):
                return self.port.read_head(run_id)

            def reserve_once(_self, pinned):
                self.port.reserve_once(pinned)
                raise ConnectionError("transport ACK lost after commit")

        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.client(port=LostAck()).reserve(self.intent)
        self.assertEqual(self.port.read_head("run-1").commit_index, 1)
        self.assertIsNotNone(self.port.read_receipt("run-1", "dev-event-1"))
        with self.assertRaises(ExternalAuthorityConflict):
            self.client().reserve(self.intent)

    def test_restart_reloads_committed_head_and_receipt(self):
        receipt = self.client().reserve(self.intent)
        restarted = SQLiteReservationPort(
            db_path=self.database, provider_id="host-operator",
            sign_bytes=self.private_key.sign,
            pinned_ed25519_public_key=self.public_key,
        )
        self.assertEqual(restarted.read_head("run-1"), receipt.post_head())
        self.assertEqual(restarted.read_receipt("run-1", "dev-event-1"), receipt)

    def test_cloned_snapshot_and_key_double_authorizes_falsifier(self):
        """Known FAIL of host ownership if storage and signer are cloneable."""
        with self.port._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        clone1 = Path(self.temporary.name) / "copy1.sqlite3"
        clone2 = Path(self.temporary.name) / "copy2.sqlite3"
        shutil.copyfile(self.database, clone1)
        shutil.copyfile(self.database, clone2)
        hosts = [
            SQLiteReservationPort(db_path=path, provider_id="host-operator",
                                  sign_bytes=self.private_key.sign,
                                  pinned_ed25519_public_key=self.public_key)
            for path in (clone1, clone2)
        ]
        receipts = [host.reserve_once(self.intent) for host in hosts]
        self.assertEqual([r.commit_index for r in receipts], [1, 1])
        self.assertEqual([h.read_head("run-1").commit_index for h in hosts], [1, 1])
        # This must NOT be relabeled rollback-resistant or physical Q8 PASS.


if __name__ == "__main__":
    unittest.main()
