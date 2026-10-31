"""SOURCE-ONLY service-core tests; these do not authenticate a real host."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.authority_service_core_r20 import (
    AuthorityCoreConflict, AuthorityCoreError, HostProvisionedJourney,
    SourceOnlyJourneyAuthorityCore,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, build_journey_external_request,
)
from labyrinth.external_monotonic_authority_contract import _verify_ed25519


def binding(run: str, event: str, exam: str) -> JourneyExternalAuthorityBinding:
    return JourneyExternalAuthorityBinding(
        provider_id="provider-a", host_id="host-a", run_id=run,
        event_id=event, exam_id=exam, scope_sha256="1" * 64,
        objective_contract_sha256="2" * 64, exam_lineage_sha256="3" * 64,
        journey_sha256="4" * 64, signed_binding_sha256="5" * 64,
        learner_checkpoint_sha256="6" * 64, prior_training_roster_sha256="7" * 64,
        next_training_allocation_sha256="8" * 64, host_policy_sha256="9" * 64,
        source_tree_sha256="a" * 64, fullstate_binding_sha256="b" * 64,
        consumed_event_ledger_root_sha256="c" * 64,
        migration_witness_sha256="d" * 64, previous_cycle=0, next_cycle=1,
    )


def pin(b: JourneyExternalAuthorityBinding) -> HostProvisionedJourney:
    return HostProvisionedJourney(
        binding=b, authenticated_principal_id="trainer-a",
        operator_pinned_ledger_root_sha256="c" * 64,
        operator_pinned_migration_witness_sha256="d" * 64,
        initial_authority_epoch=1, initial_fence=7,
    )


class SourceOnlyAuthorityCoreR20Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "host-authority.sqlite3"
        self.private = Ed25519PrivateKey.generate()
        self.key = self.private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw,
        )
        self.a = binding("run-a", "event-a", "exam-a")
        self.b = binding("run-b", "event-b", "exam-b")

    def service(self, *bindings: JourneyExternalAuthorityBinding):
        return SourceOnlyJourneyAuthorityCore(
            database_path=self.db, provider_id="provider-a",
            service_signing_key=self.key,
            provisioned_journeys=tuple(pin(b) for b in bindings),
        )

    def request(self, svc, b, nonce="operation-a"):
        return build_journey_external_request(
            b, svc.read_head(b.run_id), operation_nonce=nonce)

    def test_full_binding_is_validated_before_atomic_signed_commit(self):
        svc = self.service(self.a)
        req = self.request(svc, self.a)
        receipt = svc.reserve_journey(
            self.a, req, authenticated_principal_id="trainer-a")
        self.assertEqual(receipt.request_sha256, req.digest())
        self.assertEqual(receipt.commit_index, 1)
        self.assertFalse(receipt.graduation_authorized)
        self.assertFalse(receipt.model_execution_authenticated)
        _verify_ed25519(self.public, receipt.signed_payload(),
                        bytes.fromhex(receipt.signature_hex))
        self.assertEqual(svc.read_head("run-a"), receipt.post_head())
        self.assertEqual(svc.read_receipt("run-a", "event-a", "operation-a"), receipt)

    def test_full_preimage_policy_principal_and_migration_pins_reject_substitution(self):
        svc = self.service(self.a)
        req = self.request(svc, self.a)
        with self.assertRaises(AuthorityCoreConflict):
            svc.reserve_journey(replace(self.a, host_policy_sha256="f" * 64), req,
                                authenticated_principal_id="trainer-a")
        with self.assertRaises(AuthorityCoreConflict):
            svc.reserve_journey(self.a, req,
                                authenticated_principal_id="attacker")
        with self.assertRaises(AuthorityCoreError):
            HostProvisionedJourney(
                self.a, "trainer-a", "e" * 64, "d" * 64, 1, 7).validate()
        with self.assertRaises(AuthorityCoreError):
            HostProvisionedJourney(
                self.a, "trainer-a", "c" * 64, "e" * 64, 1, 7).validate()
        self.assertEqual(svc.read_head("run-a").commit_index, 0)

    def test_global_cross_run_event_and_exam_uniqueness(self):
        shared_event = replace(self.b, event_id="event-a")
        svc = self.service(self.a, shared_event)
        svc.reserve_journey(self.a, self.request(svc, self.a),
                            authenticated_principal_id="trainer-a")
        with self.assertRaises(AuthorityCoreConflict):
            svc.reserve_journey(shared_event, self.request(svc, shared_event, "operation-b"),
                                authenticated_principal_id="trainer-a")
        self.assertEqual(svc.read_head("run-b").commit_index, 0)
        # A second independent database exercises the exam uniqueness path.
        self.db = Path(self.temp.name) / "exam-authority.sqlite3"
        shared_exam = replace(self.b, exam_id="exam-a")
        svc = self.service(self.a, shared_exam)
        svc.reserve_journey(self.a, self.request(svc, self.a),
                            authenticated_principal_id="trainer-a")
        with self.assertRaises(AuthorityCoreConflict):
            svc.reserve_journey(shared_exam, self.request(svc, shared_exam, "operation-b"),
                                authenticated_principal_id="trainer-a")

    def test_lost_ack_reconciliation_is_idempotent_and_restart_persistent(self):
        svc = self.service(self.a)
        req = self.request(svc, self.a)
        first = svc.reserve_journey(self.a, req, authenticated_principal_id="trainer-a")
        reopened = self.service(self.a)
        self.assertEqual(reopened.read_receipt("run-a", "event-a", "operation-a"), first)
        self.assertEqual(reopened.reserve_journey(
            self.a, req, authenticated_principal_id="trainer-a"), first)
        self.assertEqual(reopened.read_head("run-a").commit_index, 1)
        with self.assertRaises(AuthorityCoreConflict):
            reopened.reserve_journey(
                self.a, replace(req, operation_nonce="operation-new"),
                authenticated_principal_id="trainer-a")

    def test_service_restart_rejects_changed_signing_identity_and_host_pins(self):
        self.service(self.a)
        with self.assertRaises(AuthorityCoreConflict):
            SourceOnlyJourneyAuthorityCore(
                database_path=self.db, provider_id="provider-a",
                service_signing_key=Ed25519PrivateKey.generate().private_bytes(
                    serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                    serialization.NoEncryption()),
                provisioned_journeys=(pin(self.a),))
        with self.assertRaises(AuthorityCoreConflict):
            self.service(replace(self.a, host_policy_sha256="f" * 64))

    def test_concurrent_duplicate_requests_consume_once(self):
        svc = self.service(self.a)
        req = self.request(svc, self.a)
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(
                lambda _: svc.reserve_journey(
                    self.a, req, authenticated_principal_id="trainer-a"),
                range(8),
            ))
        self.assertTrue(all(r == receipts[0] for r in receipts))
        self.assertEqual(svc.read_head("run-a").commit_index, 1)

    def test_concurrent_cross_run_collision_has_one_global_winner(self):
        shared_event = replace(self.b, event_id="event-a")
        svc = self.service(self.a, shared_event)
        requests = (self.request(svc, self.a),
                    self.request(svc, shared_event, "operation-b"))
        def attempt(index):
            try:
                return svc.reserve_journey(
                    (self.a, shared_event)[index], requests[index],
                    authenticated_principal_id="trainer-a")
            except AuthorityCoreConflict:
                return "CONFLICT"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, (0, 1)))
        self.assertEqual(sum(result == "CONFLICT" for result in results), 1)
        self.assertEqual(sum(result != "CONFLICT" for result in results), 1)
        self.assertEqual(svc.read_head("run-a").commit_index +
                         svc.read_head("run-b").commit_index, 1)

    def test_corrupted_receipt_cannot_be_used_for_lost_ack_reconciliation(self):
        svc = self.service(self.a)
        req = self.request(svc, self.a)
        svc.reserve_journey(self.a, req, authenticated_principal_id="trainer-a")
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE reservations SET receipt_json=?", ('{"forged":true}',))
        with self.assertRaises(AuthorityCoreError):
            svc.read_receipt("run-a", "event-a", "operation-a")
        with self.assertRaises(AuthorityCoreError):
            svc.reserve_journey(self.a, req, authenticated_principal_id="trainer-a")


if __name__ == "__main__":
    unittest.main()
