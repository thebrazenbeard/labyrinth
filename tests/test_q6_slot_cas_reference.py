"""Hostile tests for reference Q6 per-case CAS; never a host custody proof."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import sys
import threading
import unittest

from labyrinth.q6_governed_campaign_contract import (
    Q6CampaignReservationClient, Q6CampaignReservationIntent,
    Q6CampaignReservationReceipt, build_q6_campaign_manifest,
    next_q6_head_digest,
)
from labyrinth.q6_slot_cas_reference import (
    Q6SlotAddress, Q6SlotCasError, Q6SlotCasReference,
    Q6SlotLedgerHead, Q6SlotOutcomeUnknown,
)
from labyrinth.q6_stratified_toy_study_r93 import construct_q6_cohorts

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:  # Pinned neural CPU workflow installs cryptography.
    Ed25519PrivateKey = None


class _SyntheticCampaignPort:
    """In-process signature fake: NOT owner-owned, durable or rollback safe."""

    def __init__(self, intent, key):
        self.intent, self.key = intent, key
        self.head = intent.expected_head()
        self.record = None
        self.reservations = 0

    def read_head(self, run_id):
        if run_id != self.intent.run_id:
            raise ValueError("wrong run")
        return self.head

    def reserve_campaign(self, intent):
        self.reservations += 1
        receipt = Q6CampaignReservationReceipt(
            provider_id=self.intent.provider_id,
            run_id=self.intent.run_id,
            campaign_id=self.intent.campaign_id,
            reservation_id="q6-slot-reservation",
            request_sha256=self.intent.sha256,
            manifest_sha256=self.intent.manifest_sha256,
            epoch=self.intent.expected_epoch,
            fence=self.intent.expected_fence,
            commit_index=self.intent.expected_commit_index + 1,
            previous_head_sha256=self.intent.expected_head_sha256,
            new_head_sha256=next_q6_head_digest(self.intent, "q6-slot-reservation"),
            status="CAMPAIGN_RESERVED_SOURCE_ONLY",
            signature_hex="",
        )
        receipt = replace(receipt, signature_hex=self.key.sign(receipt.signed_payload()).hex())
        self.record, self.head = receipt, receipt.post_head()
        return receipt

    def read_reservation(self, campaign_id, nonce):
        if (campaign_id != self.intent.campaign_id
                or nonce != self.intent.operation_nonce):
            raise ValueError("wrong receipt")
        return self.record


@unittest.skipUnless(Ed25519PrivateKey, "cryptography verifier unavailable")
class Q6PerExposureCasSourceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_q6_campaign_manifest(
            construct_q6_cohorts(seed=7, per_profile=2), seed=7,
            per_profile=2, epochs_stage_a=2, epochs_stage_b=3,
        )
        self.intent = Q6CampaignReservationIntent(
            provider_id="owner-test-provider", run_id="slot-run-1",
            campaign_id="slot-campaign-1", manifest_sha256=self.manifest.sha256,
            candidate_checkpoint_sha256="a" * 64, operation_nonce="slot-once-1",
            expected_epoch=4, expected_fence=9, expected_commit_index=11,
            expected_head_sha256="b" * 64,
        )
        key = Ed25519PrivateKey.generate()
        self.port = _SyntheticCampaignPort(self.intent, key)
        self.client = Q6CampaignReservationClient(
            port=self.port, pinned_intent=self.intent,
            pinned_manifest=self.manifest,
            pinned_ed25519_public_key=key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw),
        )
        self.hook_events = []
        self.ledger = Q6SlotCasReference(
            client=self.client, intent=self.intent, manifest=self.manifest,
        )
        self.case = self.manifest.cases[0]
        self.slot = self.ledger.address(self.case.partition, self.case.case_id,
                                        "trained", "after_a", 0)

    def _begin(self, slot=None, **kwargs):
        target = self.slot if slot is None else slot
        return self.ledger.begin_exposure(
            target, expected_head=kwargs.pop("expected_head", self.ledger.read_head()),
            **kwargs)

    def _complete(self, slot=None, **kwargs):
        target = self.slot if slot is None else slot
        return self.ledger.complete_exposure(
            target, expected_head=kwargs.pop("expected_head", self.ledger.read_head()),
            expected_slot_version=kwargs.pop("expected_slot_version", 1),
            completion_nonce=kwargs.pop("completion_nonce", "completion-1"),
            result_sha256=kwargs.pop("result_sha256", "c" * 64),
            execution_sha256=kwargs.pop("execution_sha256", "d" * 64),
            checkpoint_sha256=kwargs.pop("checkpoint_sha256", "e" * 64),
            **kwargs)

    def test_init_requires_signed_readback_campaign(self):
        self.assertEqual(self.port.reservations, 1)
        self.assertEqual(self.ledger.read_head().commit_index, 12)
        self.assertTrue(self.ledger.source_only_unqualified)

    def test_signed_manifest_snapshot_survives_postvalidation_constructor_race(self):
        # The signed payload has already been accepted by the client before
        # hostile mutation. A frozen dataclass alone is not a trusted snapshot.
        case = self.manifest.cases[0]
        exposure = self.manifest.planned_exposures[0]
        original_prompt = case.prompt_sha256
        original_visits = exposure.planned_visits
        original_campaign_sha = self.intent.manifest_sha256
        key = Ed25519PrivateKey.generate()
        fresh_port = _SyntheticCampaignPort(self.intent, key)
        fresh_client = Q6CampaignReservationClient(
            port=fresh_port, pinned_intent=self.intent,
            pinned_manifest=self.manifest,
            pinned_ed25519_public_key=key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw),
        )
        source_lines, first_line = inspect.getsourcelines(Q6SlotCasReference.__init__)
        target_line = next(first_line + i for i, source in enumerate(source_lines)
                           if "self._manifest_sha = source_manifest_sha" in source)
        mutation_started = threading.Event()
        mutation_done = threading.Event()
        fired = []

        def hostile_mutation():
            if mutation_started.wait(timeout=4):
                object.__setattr__(case, "prompt_sha256", "f" * 64)
                object.__setattr__(exposure, "planned_visits", 99)
                mutation_done.set()

        def after_signed_readback(frame, event, unused):
            if (event == "line" and frame.f_code is Q6SlotCasReference.__init__.__code__
                    and frame.f_lineno == target_line):
                fired.append(True)
                mutation_started.set()
                if not mutation_done.wait(timeout=4):
                    raise AssertionError("hostile mutation worker did not run")
            return after_signed_readback

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(hostile_mutation)
            try:
                sys.settrace(after_signed_readback)
                recovered = Q6SlotCasReference(
                    client=fresh_client, intent=self.intent, manifest=self.manifest,
                )
            finally:
                sys.settrace(None)
                pending.result(timeout=4)
                object.__setattr__(case, "prompt_sha256", original_prompt)
                object.__setattr__(exposure, "planned_visits", original_visits)

        self.assertTrue(fired, "mutation must happen after signed readback")
        self.assertEqual(recovered.campaign_sha256, original_campaign_sha)
        self.assertEqual(recovered.address(
            case.partition, case.case_id, "trained", "after_a", 0,
        ).prompt_sha256, original_prompt)
        with self.assertRaises(Q6SlotCasError):
            recovered.address(exposure.partition, exposure.case_id,
                              exposure.arm, exposure.phase, original_visits)

    def test_no_gold_answers_or_private_prompt_bytes_in_addresses(self):
        self.assertEqual(len(self.slot.slot_id), 64)
        self.assertNotIn("answer", repr(self.slot).lower())
        self.assertNotIn("calculate", repr(self.slot).lower())
        self.assertEqual(self.slot.prompt_sha256, self.case.prompt_sha256)

    def test_all_preregistered_train_visits_have_distinct_identities(self):
        addresses = [self.ledger.address(self.case.partition, self.case.case_id,
                                         "trained", "after_a", visit)
                     for visit in range(2)]
        self.assertNotEqual(addresses[0].slot_id, addresses[1].slot_id)

    def test_fixed_eval_panel_phases_are_distinct_authorized_slots(self):
        heldout = next(c for c in self.manifest.cases if c.partition == "heldout")
        slots = [self.ledger.address("heldout", heldout.case_id, "trained", phase, 0)
                 for phase in ("initial", "after_a", "after_b")]
        self.assertEqual(len({s.slot_id for s in slots}), 3)

    def test_invalid_partition_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("q6_private_exam", self.case.case_id,
                                "trained", "after_a", 0)

    def test_invalid_case_identity_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("train_a", "forged-case", "trained", "after_a", 0)

    def test_invalid_arm_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("train_a", self.case.case_id, "winner", "after_a", 0)

    def test_wrong_phase_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("train_a", self.case.case_id, "trained", "initial", 0)

    def test_out_of_bounds_visit_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("train_a", self.case.case_id, "trained", "after_a", 2)

    def test_boolean_visit_is_refused(self):
        with self.assertRaises(Q6SlotCasError):
            self.ledger.address("train_a", self.case.case_id, "trained", "after_a", True)

    def test_forged_slot_id_is_rejected_before_transition(self):
        with self.assertRaises(Q6SlotCasError):
            self._begin(replace(self.slot, slot_id="f" * 64))
        self.assertEqual(self.ledger.reconcile(self.slot).state, "RESERVED")

    def test_forged_prompt_digest_is_rejected(self):
        with self.assertRaises(Q6SlotCasError):
            self._begin(replace(self.slot, prompt_sha256="f" * 64))

    def test_forged_semantic_digest_is_rejected(self):
        with self.assertRaises(Q6SlotCasError):
            self._begin(replace(self.slot, arithmetic_semantic_sha256="f" * 64))

    def test_racing_caller_mutation_after_validation_cannot_forge_begin(self):
        original = self.ledger._slot
        supplied = replace(self.slot)
        def mutate_after_check(candidate):
            valid = original(candidate)
            object.__setattr__(candidate, "slot_id", "f" * 64)
            return valid
        self.ledger._slot = mutate_after_check
        try:
            receipt = self._begin(supplied)
        finally:
            self.ledger._slot = original
        self.assertEqual(receipt.slot_id, self.slot.slot_id)
        self.assertEqual(self.ledger.reconcile(self.slot).state, "REVEALED_OR_CONSUMED")
        self.assertNotIn("f" * 64, self.ledger._begun)

    def test_racing_caller_mutation_after_validation_cannot_forge_complete(self):
        self._begin()
        original = self.ledger._slot
        supplied = replace(self.slot)
        def mutate_after_check(candidate):
            valid = original(candidate)
            object.__setattr__(candidate, "slot_id", "f" * 64)
            return valid
        self.ledger._slot = mutate_after_check
        try:
            receipt = self._complete(supplied)
        finally:
            self.ledger._slot = original
        self.assertEqual(receipt.slot_id, self.slot.slot_id)
        self.assertEqual(self.ledger.reconcile(self.slot).state, "COMPLETED")

    def test_racing_caller_mutation_after_validation_cannot_fake_reconcile(self):
        self._begin()
        original = self.ledger._slot
        supplied = replace(self.slot)
        def mutate_after_check(candidate):
            valid = original(candidate)
            object.__setattr__(candidate, "slot_id", "f" * 64)
            return valid
        self.ledger._slot = mutate_after_check
        try:
            status = self.ledger.reconcile(supplied)
        finally:
            self.ledger._slot = original
        self.assertEqual(status.slot_id, self.slot.slot_id)
        self.assertEqual(status.state, "REVEALED_OR_CONSUMED")

    def test_callback_runs_only_after_consumption(self):
        observed = []
        def called(slot):
            observed.append(self.ledger.reconcile(slot).state)
            return "PUBLIC_PROMPT_ONLY"
        receipt, result = self.ledger.begin_then_call(
            self.slot, expected_head=self.ledger.read_head(), operation=called)
        self.assertEqual(observed, ["REVEALED_OR_CONSUMED"])
        self.assertEqual(result, "PUBLIC_PROMPT_ONLY")
        self.assertEqual(receipt.event, "BEGIN")

    def test_after_commit_caller_mutation_cannot_reveal_unconsumed_valid_slot(self):
        # Hostile post-BEGIN hook swaps the caller-owned frozen dataclass A
        # into a distinct *valid* preregistered B. The callback must receive
        # A's owned address, never the unconsumed B prompt.
        original = replace(self.slot)
        supplied = replace(self.slot)
        different = self.ledger.address(self.case.partition, self.case.case_id,
                                        "trained", "after_a", 1)
        observed = []

        def mutate_caller(event):
            self.assertEqual(event, "BEGIN")
            for field in ("partition", "case_id", "arm", "phase", "visit",
                          "slot_id", "prompt_sha256", "arithmetic_semantic_sha256"):
                object.__setattr__(supplied, field, getattr(different, field))

        self.ledger._after_commit_hook = mutate_caller

        def expose(slot):
            observed.append((slot.slot_id, self.ledger.reconcile(slot).state))
            return "PUBLIC_PROMPT_ONLY"

        receipt, result = self.ledger.begin_then_call(
            supplied, expected_head=self.ledger.read_head(), operation=expose)
        self.assertEqual(receipt.slot_id, original.slot_id)
        self.assertEqual(observed, [(original.slot_id, "REVEALED_OR_CONSUMED")])
        self.assertEqual(result, "PUBLIC_PROMPT_ONLY")
        self.assertEqual(supplied.slot_id, different.slot_id)
        self.assertEqual(self.ledger.reconcile(different).state, "RESERVED")
        self.assertEqual(self.ledger.reconcile(original).state, "REVEALED_OR_CONSUMED")

    def test_callback_failure_still_consumes_slot(self):
        def fail(_):
            raise RuntimeError("inference crashed")
        with self.assertRaisesRegex(RuntimeError, "crashed"):
            self.ledger.begin_then_call(
                self.slot, expected_head=self.ledger.read_head(), operation=fail)
        self.assertEqual(self.ledger.reconcile(self.slot).state, "REVEALED_OR_CONSUMED")

    def test_callback_not_run_if_head_is_stale(self):
        calls = []
        old_head = self.ledger.read_head()
        self._begin()
        with self.assertRaises(Q6SlotCasError):
            self.ledger.begin_then_call(
                self.ledger.address("train_a", self.case.case_id, "trained", "after_a", 1),
                expected_head=old_head, operation=lambda x: calls.append(x))
        self.assertFalse(calls)

    def test_second_begin_never_reveals_same_slot(self):
        self._begin()
        with self.assertRaisesRegex(Q6SlotCasError, "consumed"):
            self._begin()

    def test_stale_slot_version_cannot_reveal(self):
        with self.assertRaises(Q6SlotCasError):
            self._begin(expected_slot_version=1)
        self.assertEqual(self.ledger.reconcile(self.slot).slot_version, 0)

    def test_stale_head_cannot_reveal_new_slot(self):
        old = self.ledger.read_head()
        self._begin()
        another = self.ledger.address("train_a", self.case.case_id,
                                      "trained", "after_a", 1)
        with self.assertRaisesRegex(Q6SlotCasError, "stale"):
            self._begin(another, expected_head=old)

    def test_forged_epoch_or_fence_is_rejected(self):
        head = self.ledger.read_head()
        for changed in (replace(head, epoch=5), replace(head, fence=10)):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(Q6SlotCasError, "stale"):
                    self._begin(expected_head=changed)

    def test_postbegin_lost_ack_requires_read_only_reconciliation(self):
        self.ledger._after_commit_hook = lambda event: (_ for _ in ()).throw(
            ConnectionError("ACK lost")) if event == "BEGIN" else None
        with self.assertRaises(Q6SlotOutcomeUnknown):
            self._begin()
        self.assertEqual(self.ledger.reconcile(self.slot).state, "REVEALED_OR_CONSUMED")
        with self.assertRaises(Q6SlotCasError):
            self._begin()

    def test_readonly_reconcile_is_stable(self):
        old = self.ledger.read_head()
        a = self.ledger.reconcile(self.slot)
        b = self.ledger.reconcile(self.slot)
        self.assertEqual(a, b)
        self.assertEqual(self.ledger.read_head(), old)

    def test_forged_returned_receipt_cannot_mutate_ledger(self):
        receipt = self._begin()
        object.__setattr__(receipt, "state", "RESERVED")
        object.__setattr__(receipt, "slot_id", "f" * 64)
        self.assertEqual(self.ledger.reconcile(self.slot).state, "REVEALED_OR_CONSUMED")
        self.assertEqual(self.ledger.reconcile(self.slot).begin_receipt.slot_id,
                         self.slot.slot_id)

    def test_completion_before_begin_is_denied(self):
        with self.assertRaises(Q6SlotCasError):
            self._complete()

    def test_valid_complete_advances_from_revealed(self):
        self._begin()
        receipt = self._complete()
        self.assertEqual(receipt.event, "COMPLETE")
        self.assertEqual(receipt.slot_version, 2)
        self.assertEqual(self.ledger.reconcile(self.slot).state, "COMPLETED")

    def test_result_digest_required(self):
        self._begin()
        with self.assertRaises(Q6SlotCasError):
            self._complete(result_sha256="not-a-digest")

    def test_execution_and_checkpoint_digests_required(self):
        self._begin()
        for key in ("execution_sha256", "checkpoint_sha256"):
            with self.subTest(key=key):
                with self.assertRaises(Q6SlotCasError):
                    self._complete(**{key: "Z" * 64})

    def test_nonce_required(self):
        self._begin()
        for nonce in ("", "\n", "non ascii café", "x" * 129):
            with self.subTest(nonce=nonce):
                with self.assertRaises(Q6SlotCasError):
                    self._complete(completion_nonce=nonce)

    def test_exact_idempotent_completion_does_not_advance_head(self):
        self._begin()
        first = self._complete()
        head = self.ledger.read_head()
        second = self._complete(expected_head=replace(head, commit_index=0))
        self.assertEqual(first, second)
        self.assertEqual(self.ledger.read_head(), head)

    def test_divergent_second_completion_is_denied(self):
        self._begin()
        self._complete()
        with self.assertRaisesRegex(Q6SlotCasError, "pinned"):
            self._complete(result_sha256="f" * 64)

    def test_wrong_slot_version_completion_is_denied(self):
        self._begin()
        with self.assertRaises(Q6SlotCasError):
            self._complete(expected_slot_version=0)

    def test_stale_global_head_before_completion_is_denied(self):
        prior = self.ledger.read_head()
        self._begin()
        with self.assertRaisesRegex(Q6SlotCasError, "stale"):
            self._complete(expected_head=prior)

    def test_postcompletion_lost_ack_never_reinferences(self):
        self._begin()
        self.ledger._after_commit_hook = lambda event: (_ for _ in ()).throw(
            ConnectionError("completion ACK lost")) if event == "COMPLETE" else None
        with self.assertRaises(Q6SlotOutcomeUnknown):
            self._complete()
        self.assertEqual(self.ledger.reconcile(self.slot).state, "COMPLETED")
        with self.assertRaises(Q6SlotCasError):
            self._begin()
        again = self._complete(expected_head=replace(self.ledger.read_head(), commit_index=0))
        self.assertEqual(again, self.ledger.reconcile(self.slot).complete_receipt)

    def test_mutating_returned_complete_receipt_does_not_change_pin(self):
        self._begin()
        receipt = self._complete()
        object.__setattr__(receipt, "result_sha256", "f" * 64)
        with self.assertRaises(Q6SlotCasError):
            self._complete(result_sha256="f" * 64)
        self.assertEqual(self.ledger.reconcile(self.slot).complete_receipt.result_sha256,
                         "c" * 64)

    def test_same_slot_concurrent_begin_exactly_one_wins(self):
        start = threading.Event()
        head = self.ledger.read_head()
        def run(_):
            start.wait()
            try:
                return self.ledger.begin_exposure(self.slot, expected_head=head)
            except Q6SlotCasError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(run, i) for i in range(2)]
            start.set()
            results = [future.result() for future in pending]
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(self.ledger.reconcile(self.slot).slot_version, 1)

    def test_concurrent_distinct_slots_compete_on_global_head(self):
        other = self.ledger.address("train_a", self.case.case_id,
                                    "trained", "after_a", 1)
        start = threading.Event()
        head = self.ledger.read_head()
        def run(slot):
            start.wait()
            try:
                return self.ledger.begin_exposure(slot, expected_head=head)
            except Q6SlotCasError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(run, slot) for slot in (self.slot, other)]
            start.set()
            results = [future.result() for future in pending]
        self.assertEqual(sum(r is not None for r in results), 1)

    def test_no_missing_external_high_water_can_prove_rollback_safety(self):
        with self.assertRaises(Q6SlotOutcomeUnknown):
            self.ledger.assert_external_head_matches(None)

    def test_external_high_water_difference_detects_local_rollback(self):
        stale = self.ledger.read_head()
        self._begin()
        current = self.ledger.read_head()
        with self.assertRaises(Q6SlotOutcomeUnknown):
            self.ledger.assert_external_head_matches(stale)
        self.ledger.assert_external_head_matches(current)

    def test_manual_inmemory_snapshot_restoration_remains_insecure_without_external_witness(self):
        # This test deliberately exposes the limitation of a process-local
        # reference. An owner MUST store an independently anchored high-water.
        original_head = self.ledger.read_head()
        self._begin()
        external_head = self.ledger.read_head()
        self.ledger._begun.clear()
        self.ledger._commit_index = original_head.commit_index
        self.ledger._head_sha = original_head.head_sha256
        with self.assertRaises(Q6SlotOutcomeUnknown):
            self.ledger.assert_external_head_matches(external_head)

    def test_false_only_authority_fields(self):
        begin = self._begin()
        self.assertFalse(begin.model_execution_authenticated)
        self.assertFalse(begin.individual_exposures_host_verified)
        self.assertFalse(begin.graduation_authorized)
        self.assertTrue(begin.source_only_unqualified)
        self.assertFalse(hasattr(self.ledger, "private_gold"))


if __name__ == "__main__":
    unittest.main()
