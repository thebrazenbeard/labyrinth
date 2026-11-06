"""Contract-only R17 checks; no independent external host/provider is present."""

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityHead, ExternalReservationRequest,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyAuthorityBindingError, JourneyExternalAuthorityBinding,
    build_journey_external_request, verify_journey_external_request,
)


class JourneyExternalAuthorityContractR17Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = JourneyExternalAuthorityBinding(
            provider_id="provider-a", host_id="host-a", run_id="run-a",
            event_id="event-a", exam_id="exam-a",
            scope_sha256="1" * 64, objective_contract_sha256="2" * 64,
            exam_lineage_sha256="3" * 64, journey_sha256="4" * 64,
            signed_binding_sha256="5" * 64,
            learner_checkpoint_sha256="6" * 64,
            prior_training_roster_sha256="7" * 64,
            next_training_allocation_sha256="8" * 64,
            host_policy_sha256="9" * 64, source_tree_sha256="a" * 64,
            fullstate_binding_sha256="b" * 64,
            consumed_event_ledger_root_sha256="c" * 64,
            migration_witness_sha256="d" * 64,
            previous_cycle=4, next_cycle=5,
        )
        self.head = ExternalAuthorityHead("provider-a", "run-a", 4, 7, 9, "e" * 64)

    def _request(self):
        return build_journey_external_request(
            self.binding, self.head, operation_nonce="nonce-a")

    def test_canonical_journey_bundle_maps_to_native_ed25519_request(self):
        req = self._request()
        self.assertIs(type(req), ExternalReservationRequest)
        self.assertEqual(req.candidate_artifact_sha256, self.binding.digest())
        self.assertEqual(req.scope_fingerprint, self.binding.scope_sha256)
        self.assertEqual(req.exam_lineage_sha256, self.binding.exam_lineage_sha256)
        self.assertEqual(req.prior_checkpoint_sha256,
                         self.binding.learner_checkpoint_sha256)
        self.assertEqual(req.expected_head(), self.head)
        self.assertEqual(req.to_cycle, 5)
        self.assertIsNone(verify_journey_external_request(
            self.binding, self.head, req, operation_nonce="nonce-a"))
        self.assertNotIn(b"graduation", self.binding.payload())

    def test_every_security_sensitive_binding_change_changes_candidate_digest(self):
        original = self._request()
        for field in (
            "provider_id", "host_id", "run_id", "event_id", "exam_id",
            "scope_sha256", "objective_contract_sha256", "exam_lineage_sha256",
            "journey_sha256", "signed_binding_sha256",
            "learner_checkpoint_sha256", "prior_training_roster_sha256",
            "next_training_allocation_sha256", "host_policy_sha256",
            "source_tree_sha256", "fullstate_binding_sha256",
            "consumed_event_ledger_root_sha256", "migration_witness_sha256",
        ):
            with self.subTest(field=field):
                current = getattr(self.binding, field)
                changed = "alternate" if field.endswith("_id") else "f" * 64
                self.assertNotEqual(replace(self.binding, **{field: changed}).digest(),
                                    original.candidate_artifact_sha256)

    def test_valid_native_shape_rejects_cross_run_exam_or_host_rebinding(self):
        req = self._request()
        variants = (
            replace(req, event_id="event-b"),
            replace(req, exam_id="exam-b"),
            replace(req, run_id="run-b"),
            replace(req, candidate_artifact_sha256="f" * 64),
            replace(req, expected_head_sha256="f" * 64),
            replace(req, operation_nonce="nonce-b"),
        )
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaises(JourneyAuthorityBindingError):
                    verify_journey_external_request(
                        self.binding, self.head, changed, operation_nonce="nonce-a")
        with self.assertRaises(JourneyAuthorityBindingError):
            build_journey_external_request(
                self.binding, replace(self.head, provider_id="provider-b"),
                operation_nonce="nonce-a")

    def test_strict_epoch_cycle_and_migration_commitments(self):
        for changed in (
            replace(self.binding, next_cycle=7),
            replace(self.binding, previous_cycle=True),
            replace(self.binding, consumed_event_ledger_root_sha256=""),
            replace(self.binding, migration_witness_sha256="F" * 64),
            replace(self.binding, purpose="GRADUATION"),
            replace(self.binding, schema="labyrinth.journey-native-authority-binding.v0"),
        ):
            with self.subTest(changed=changed), self.assertRaises(JourneyAuthorityBindingError):
                changed.validate()
        with self.assertRaises(JourneyAuthorityBindingError):
            build_journey_external_request(
                self.binding, replace(self.head, epoch=True),
                operation_nonce="nonce-a")

    def test_native_journey_digest_uses_separate_protocol_domain(self):
        payload = self.binding.payload()
        self.assertTrue(payload.endswith(b"\n"))
        self.assertNotEqual(self.binding.digest(), sha256(payload).hexdigest())
        self.assertNotEqual(self.binding.digest(), self._request().digest())
        changed_cycle = replace(self.binding, previous_cycle=5, next_cycle=6)
        self.assertNotEqual(changed_cycle.digest(), self.binding.digest())
        with self.assertRaises(JourneyAuthorityBindingError):
            verify_journey_external_request(
                changed_cycle, self.head, self._request(),
                operation_nonce="nonce-a")


if __name__ == "__main__":
    unittest.main()
