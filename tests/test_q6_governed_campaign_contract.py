"""Hostile source-contract tests; mock provider is NEVER a host attestation."""

from __future__ import annotations

from dataclasses import replace
import re
import unittest

from labyrinth.q6_governed_campaign_contract import (
    Q6AuthorityHead, Q6CampaignAuthorityError, Q6CampaignOutcomeUnknown,
    Q6CampaignReservationClient, Q6CampaignReservationIntent,
    Q6CampaignReservationReceipt, build_q6_campaign_manifest,
    execute_only_after_campaign_reservation, next_q6_head_digest,
)
from labyrinth.q6_stratified_toy_study_r93 import construct_q6_cohorts

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey = None


class UntrustedTestPort:
    """Synthetic signing port, deliberately NOT host-owned or rollback safe."""

    def __init__(self, intent, key):
        self.head = intent.expected_head()
        self.receipt = None
        self.intent = intent
        self.signing_key = key
        self.reserve_calls = 0
        self.readback_calls = 0
        self.raise_before_commit = False
        self.raise_after_commit = False
        self.mutate_transport_intent = False
        self.corrupt_signature = False
        self.corrupt_readback = False

    def read_head(self, run_id):
        if run_id != self.intent.run_id:
            raise ValueError("wrong run")
        return self.head

    def reserve_campaign(self, intent):
        self.reserve_calls += 1
        if self.raise_before_commit:
            raise ConnectionError("provider unavailable")
        if self.mutate_transport_intent:
            object.__setattr__(intent, "manifest_sha256", "f" * 64)
        record = Q6CampaignReservationReceipt(
            provider_id=self.intent.provider_id,
            run_id=self.intent.run_id,
            campaign_id=self.intent.campaign_id,
            reservation_id="res-test-1",
            request_sha256=self.intent.sha256,
            manifest_sha256=self.intent.manifest_sha256,
            epoch=self.intent.expected_epoch,
            fence=self.intent.expected_fence,
            commit_index=self.intent.expected_commit_index + 1,
            previous_head_sha256=self.intent.expected_head_sha256,
            new_head_sha256=next_q6_head_digest(self.intent, "res-test-1"),
            status="CAMPAIGN_RESERVED_SOURCE_ONLY", signature_hex="",
        )
        record = replace(record, signature_hex=self.signing_key.sign(record.signed_payload()).hex())
        self.receipt = record
        self.head = record.post_head()
        if self.raise_after_commit:
            raise ConnectionError("ACK lost after external commit")
        if self.corrupt_signature:
            return replace(record, signature_hex="0" * 128)
        return record

    def read_reservation(self, campaign_id, operation_nonce):
        self.readback_calls += 1
        if (campaign_id != self.intent.campaign_id
                or operation_nonce != self.intent.operation_nonce):
            raise ValueError("wrong lookup")
        if self.corrupt_readback:
            return replace(self.receipt, reservation_id="res-substituted")
        return self.receipt


class Q6GeneratorProvenanceTests(unittest.TestCase):
    """Mandatory without optional Ed25519 dependencies or host transport."""

    def setUp(self):
        self.cohorts = construct_q6_cohorts(seed=7, per_profile=2)

    def test_wrong_declared_seed_cannot_produce_a_manifest(self):
        with self.assertRaisesRegex(Q6CampaignAuthorityError, "generator/seed/order"):
            build_q6_campaign_manifest(
                self.cohorts, seed=8, per_profile=2,
                epochs_stage_a=3, epochs_stage_b=2)

    def test_forged_case_and_order_are_not_source_provenance(self):
        forged = replace(
            self.cohorts,
            train_a=(replace(self.cohorts.train_a[0], id="source-alias"),)
            + self.cohorts.train_a[1:],
        )
        reordered = replace(self.cohorts, train_a=tuple(reversed(self.cohorts.train_a)))
        for untrusted in (forged, reordered):
            with self.assertRaisesRegex(Q6CampaignAuthorityError, "generator/seed/order"):
                build_q6_campaign_manifest(
                    untrusted, seed=7, per_profile=2,
                    epochs_stage_a=3, epochs_stage_b=2)

    def test_direct_manifest_constructor_is_subject_to_regeneration(self):
        manifest = build_q6_campaign_manifest(
            self.cohorts, seed=7, per_profile=2,
            epochs_stage_a=3, epochs_stage_b=2)
        with self.assertRaises(Q6CampaignAuthorityError):
            replace(manifest, seed=8).payload()
        with self.assertRaises(Q6CampaignAuthorityError):
            replace(manifest, cases=tuple(reversed(manifest.cases))).payload()


@unittest.skipUnless(Ed25519PrivateKey, "cryptography optional Ed25519 verifier unavailable")
class Q6GovernedSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.cohorts = construct_q6_cohorts(seed=7, per_profile=2)
        self.manifest = build_q6_campaign_manifest(
            self.cohorts, seed=7, per_profile=2,
            epochs_stage_a=3, epochs_stage_b=2,
        )
        self.intent = Q6CampaignReservationIntent(
            provider_id="operator-host", run_id="q6-run-1",
            campaign_id="q6-campaign-1", manifest_sha256=self.manifest.sha256,
            candidate_checkpoint_sha256="a" * 64,
            operation_nonce="q6-op-1", expected_epoch=1, expected_fence=0,
            expected_commit_index=3, expected_head_sha256="b" * 64,
        )
        self.key = Ed25519PrivateKey.generate()
        self.port = UntrustedTestPort(self.intent, self.key)
        self.client = Q6CampaignReservationClient(
            port=self.port, pinned_intent=self.intent,
            pinned_manifest=self.manifest,
            pinned_ed25519_public_key=self.key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw),
        )
        self.callback_calls = 0

    def _callback(self):
        # Called only after independently verified synthetic readback in tests.
        self.assertEqual(self.port.readback_calls, 1)
        self.callback_calls += 1
        return "toy-score"

    def _run(self, client=None, *, intent=None, manifest=None):
        return execute_only_after_campaign_reservation(
            client=self.client if client is None else client,
            intent=self.intent if intent is None else intent,
            manifest=self.manifest if manifest is None else manifest,
            operation=self._callback,
        )

    def test_public_gold_free_manifest_fixes_arms_phases_and_visit_counts(self):
        self.assertEqual(len(self.manifest.cases), 32)
        self.assertEqual(self.manifest.source_generator_schema,
                         "labyrinth.q6-stratified-toy-study-r93.v1")
        self.assertEqual(len(self.manifest.planned_exposures), 32 * 3 * 2)
        self.assertNotIn(b"gold", self.manifest.payload())
        self.assertNotIn(b"answer", self.manifest.payload())
        train = [slot for slot in self.manifest.planned_exposures
                 if slot.partition == "train_a"]
        self.assertEqual({slot.planned_visits for slot in train}, {3})
        heldout = [slot for slot in self.manifest.planned_exposures
                   if slot.partition == "heldout"]
        self.assertEqual({slot.phase for slot in heldout}, {"initial", "after_a", "after_b"})
        self.assertEqual({slot.planned_visits for slot in heldout}, {1})

    def test_verified_source_reservation_is_still_unqualified(self):
        result = self._run()
        self.assertEqual(result.status, "SOURCE_ONLY_UNQUALIFIED")
        self.assertTrue(result.callback_invoked)
        self.assertEqual(result.result, "toy-score")
        self.assertEqual(self.callback_calls, 1)
        self.assertEqual(self.port.reserve_calls, 1)
        self.assertEqual(self.port.readback_calls, 1)
        self.assertEqual(len(result.source_reservation_sha256), 64)

    def test_missing_external_authority_cannot_call_model(self):
        result = execute_only_after_campaign_reservation(
            client=None, intent=self.intent, manifest=self.manifest,
            operation=self._callback)
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertEqual(self.port.reserve_calls, 0)
        self.assertEqual(self.callback_calls, 0)

    def test_precommit_unavailable_is_unqualified_with_zero_callback(self):
        self.port.raise_before_commit = True
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertEqual(self.port.reserve_calls, 1)
        self.assertEqual(self.callback_calls, 0)

    def test_postcommit_lost_ack_is_not_blindly_retried(self):
        self.port.raise_after_commit = True
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertIn("reconcile", result.failure)
        self.assertEqual(self.port.reserve_calls, 1)
        self.assertIsNotNone(self.port.receipt)  # synthetic durable event exists
        self.assertEqual(self.callback_calls, 0)

    def test_invalid_signature_and_corrupt_readback_never_reach_model(self):
        for flag in ("corrupt_signature", "corrupt_readback"):
            with self.subTest(flag=flag):
                self.setUp()
                setattr(self.port, flag, True)
                result = self._run()
                self.assertEqual(result.status, "UNQUALIFIED")
                self.assertFalse(result.callback_invoked)
                self.assertEqual(self.callback_calls, 0)

    def test_stale_monotonic_head_refuses_before_provider_mutation(self):
        self.port.head = replace(self.port.head, commit_index=4)
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertEqual(self.port.reserve_calls, 0)
        self.assertEqual(self.callback_calls, 0)

    def test_readback_loss_after_commit_refuses_callback(self):
        def fail_readback(campaign_id, nonce):
            raise TimeoutError("postcommit ACK/readback unavailable")

        self.port.read_reservation = fail_readback
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertIsNotNone(self.port.receipt)
        self.assertEqual(self.callback_calls, 0)

    def test_replaying_stale_reservation_after_head_advance_refuses_callback(self):
        self.assertTrue(self._run().callback_invoked)
        self.assertEqual(self.callback_calls, 1)
        replay = self._run()
        self.assertEqual(replay.status, "UNQUALIFIED")
        self.assertFalse(replay.callback_invoked)
        self.assertEqual(self.port.reserve_calls, 1)
        self.assertEqual(self.callback_calls, 1)

    def test_host_pinned_campaign_rejects_substitution_before_provider_call(self):
        altered = replace(self.intent, candidate_checkpoint_sha256="c" * 64)
        result = self._run(intent=altered)
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertEqual(self.port.reserve_calls, 0)
        self.assertEqual(self.callback_calls, 0)

    def test_frozen_object_caller_mutation_cannot_change_pinned_intent(self):
        object.__setattr__(self.intent, "candidate_checkpoint_sha256", "d" * 64)
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertEqual(self.port.reserve_calls, 0)
        self.assertEqual(self.callback_calls, 0)

    def test_mutating_port_transport_argument_does_not_redirect_verified_intent(self):
        self.port.mutate_transport_intent = True
        result = self._run()
        self.assertEqual(result.status, "SOURCE_ONLY_UNQUALIFIED")
        self.assertEqual(self.intent.manifest_sha256, self.manifest.sha256)
        self.assertEqual(self.callback_calls, 1)

    def test_invalid_quota_and_duplicate_manifest_semantics_fail_before_reservation(self):
        with self.assertRaises(ValueError):
            build_q6_campaign_manifest(
                self.cohorts, seed=7, per_profile=True,
                epochs_stage_a=3, epochs_stage_b=2)
        with self.assertRaises(ValueError):
            build_q6_campaign_manifest(
                self.cohorts, seed=7, per_profile=2,
                epochs_stage_a=0, epochs_stage_b=2)
        # No external port is called by purely local invalid manifest building.
        self.assertEqual(self.port.reserve_calls, 0)

    def test_declared_seed_must_generate_exact_supplied_cohort(self):
        with self.assertRaisesRegex(Q6CampaignAuthorityError, "generator/seed/order"):
            build_q6_campaign_manifest(
                self.cohorts, seed=8, per_profile=2,
                epochs_stage_a=3, epochs_stage_b=2)
        self.assertEqual(self.port.reserve_calls, 0)

    def test_forged_case_id_and_reordered_cohort_fail_even_when_source_valid(self):
        forged_case = replace(self.cohorts.train_a[0], id="q6-forged-case-id")
        forged_id_cohort = replace(
            self.cohorts, train_a=(forged_case,) + self.cohorts.train_a[1:])
        reordered = replace(
            self.cohorts, train_a=tuple(reversed(self.cohorts.train_a)))
        for bad_cohort in (forged_id_cohort, reordered):
            with self.subTest(cohort=bad_cohort.train_a[0].id):
                with self.assertRaisesRegex(Q6CampaignAuthorityError, "generator/seed/order"):
                    build_q6_campaign_manifest(
                        bad_cohort, seed=7, per_profile=2,
                        epochs_stage_a=3, epochs_stage_b=2)
        self.assertEqual(self.port.reserve_calls, 0)

    def test_direct_manifest_construction_cannot_bypass_generator_provenance(self):
        # A caller could bypass the factory and maintain internal slot/ID
        # consistency. Validation must regenerate the real seed-7 cohort.
        original_id = self.manifest.cases[0].case_id
        forged_id = "q6-forged-case-id"
        cases = (replace(self.manifest.cases[0], case_id=forged_id)
                 ,) + self.manifest.cases[1:]
        slots = tuple(
            replace(slot, case_id=forged_id)
            if slot.case_id == original_id else slot
            for slot in self.manifest.planned_exposures
        )
        forged = replace(self.manifest, cases=cases, planned_exposures=slots)
        with self.assertRaisesRegex(Q6CampaignAuthorityError, "generator/seed/order"):
            forged.payload()
        with self.assertRaises(Q6CampaignAuthorityError):
            Q6CampaignReservationClient(
                port=self.port, pinned_intent=self.intent,
                pinned_manifest=forged,
                pinned_ed25519_public_key=self.key.public_key().public_bytes(
                    Encoding.Raw, PublicFormat.Raw),
            )
        self.assertEqual(self.port.reserve_calls, 0)

    def test_direct_manifest_seed_or_generator_version_substitution_fails(self):
        for forged in (
            replace(self.manifest, seed=8),
            replace(self.manifest, source_generator_schema="unverified-generator.v9"),
        ):
            with self.assertRaises(Q6CampaignAuthorityError):
                forged.payload()
        self.assertEqual(self.port.reserve_calls, 0)

    def test_forged_manifest_missing_evaluation_slot_fails_closed(self):
        forged = replace(self.manifest,
                         planned_exposures=self.manifest.planned_exposures[:-1])
        with self.assertRaises(Q6CampaignAuthorityError):
            forged.payload()
        result = self._run(manifest=forged)
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertEqual(self.port.reserve_calls, 0)

    def test_semantic_alias_between_train_and_heldout_cannot_be_manifested(self):
        training = self.cohorts.train_a[0]
        match = re.fullmatch(r"Calculate (\d+) \+ (\d+)\.\n"
                             r"Respond with exactly one line: ANSWER: <integer>\.",
                             training.prompt)
        self.assertIsNotNone(match)
        alias = replace(self.cohorts.heldout[0],
                        prompt=f"Calculate {match[2]} + {match[1]}.\n"
                               "Respond with exactly one line: ANSWER: <integer>.",
                        answer=training.answer)
        forged_cohorts = replace(self.cohorts,
                                 heldout=(alias,) + self.cohorts.heldout[1:])
        with self.assertRaises(ValueError):
            build_q6_campaign_manifest(
                forged_cohorts, seed=7, per_profile=2,
                epochs_stage_a=3, epochs_stage_b=2)
        self.assertEqual(self.port.reserve_calls, 0)

    def test_source_reservation_explicitly_cannot_claim_exposure_or_graduation(self):
        original = self.port.reserve_campaign

        def falsely_authoritative(intent):
            signed = original(intent)
            return replace(signed, individual_exposures_reserved=True)

        self.port.reserve_campaign = falsely_authoritative
        result = self._run()
        self.assertEqual(result.status, "UNQUALIFIED")
        self.assertFalse(result.callback_invoked)
        self.assertEqual(self.callback_calls, 0)

    def test_direct_reserve_raises_typed_outcome_unknown_on_lost_ack(self):
        self.port.raise_after_commit = True
        with self.assertRaises(Q6CampaignOutcomeUnknown):
            self.client.reserve(self.intent, self.manifest)

    def test_wrong_manifest_commitment_rejected_at_client_creation(self):
        changed = replace(self.intent, manifest_sha256="9" * 64)
        with self.assertRaises(Q6CampaignAuthorityError):
            Q6CampaignReservationClient(
                port=self.port, pinned_intent=changed,
                pinned_manifest=self.manifest,
                pinned_ed25519_public_key=self.key.public_key().public_bytes(
                    Encoding.Raw, PublicFormat.Raw),
            )


if __name__ == "__main__":
    unittest.main()
