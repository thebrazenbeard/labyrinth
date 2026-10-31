"""Source-only host-reserved continuation of a signed development FAIL."""

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from test_weakness_training_bridge_r9 import WeaknessTrainingBridgeTests

from labyrinth.measured_weakness_feedback import training_roster_commitment
from labyrinth.signed_journey_loop import (
    HostLoopReservation, prepare_signed_journey_loop, journey_state_commitment,
)


HOST_KEY = b"independent-host-test-key-123456"


class SignedJourneyLoopTests(unittest.TestCase):
    def setUp(self):
        self.fixture = WeaknessTrainingBridgeTests(
            "test_signed_program_weakness_allocates_deterministic_answerless_train_rooms")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.used = set()

    def reserve(self, payload: bytes, *, fence: int = 7) -> HostLoopReservation:
        body = json.loads(payload)
        digest = sha256(payload).hexdigest()
        if digest in self.used:
            raise ValueError("host event already reserved")
        self.used.add(digest)
        unsigned = {
            "schema": "labyrinth.host-loop-reservation.v1",
            "event_id": body["event_id"],
            "proposal_sha256": digest,
            "fence": fence,
            "host_id": "examiner-owned-host",
        }
        canonical = (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
        signature = hmac.new(HOST_KEY, canonical, sha256).hexdigest()
        return HostLoopReservation(**unsigned, signature_hex=signature)

    @staticmethod
    def verify(reservation: HostLoopReservation, signed: bytes) -> bool:
        return hmac.compare_digest(
            hmac.new(HOST_KEY, signed, sha256).hexdigest(),
            reservation.signature_hex,
        )

    def prepare(self, **overrides):
        fixture = self.fixture
        args = dict(
            contract=fixture.contract,
            journey=fixture.journey,
            binding=fixture.binding,
            training_exposures=fixture.training,
            expected_training_roster_sha256=fixture.roster,
            expected_objective_contract_sha256=fixture.contract.fingerprint,
            expected_journey_sha256=journey_state_commitment(fixture.journey),
            expected_learner_checkpoint_sha256=fixture.journey.checkpoint_sha256,
            expected_fence=7,
            host_reserve_once=self.reserve,
            verify_host_reservation=self.verify,
            verification_inputs=fixture.verification,
        )
        args.update(overrides)
        return prepare_signed_journey_loop(**args)

    def test_host_reserved_signed_failure_returns_only_new_train_views(self):
        result = self.prepare()
        self.assertEqual(result.learner_checkpoint_sha256,
                         self.fixture.journey.checkpoint_sha256)
        self.assertEqual(result.allocation.maze, self.fixture.journey.maze)
        self.assertEqual(result.previous_cycle, 0)
        self.assertEqual(result.next_cycle, 1)
        self.assertEqual(result.host_fence, 7)
        self.assertTrue(result.allocation.views)
        self.assertTrue(all(view.challenge_id.startswith("train-")
                            and not hasattr(view, "answer")
                            for view in result.allocation.views))
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.host_independence_verified)

    def test_no_host_reservation_or_verifier_must_fail_closed(self):
        for arg in ("host_reserve_once", "verify_host_reservation"):
            with self.subTest(arg=arg), self.assertRaises(ValueError):
                self.prepare(**{arg: None})
        with self.assertRaises(ValueError):
            self.prepare(verify_host_reservation=lambda *_: False)

    def test_host_one_shot_rejects_repeat_and_fence_change(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already reserved"):
            self.prepare()
        with self.assertRaises(ValueError):
            self.prepare(host_reserve_once=lambda payload: self.reserve(payload, fence=8))

    def test_changed_learner_and_journey_identity_are_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare(expected_learner_checkpoint_sha256="f" * 64)
        with self.assertRaises(ValueError):
            self.prepare(expected_journey_sha256="0" * 64)
        changed = replace(self.fixture.journey, checkpoint_sha256="f" * 64)
        with self.assertRaises(ValueError):
            self.prepare(journey=changed,
                         expected_journey_sha256=journey_state_commitment(changed),
                         expected_learner_checkpoint_sha256="f" * 64)

    def test_forged_receipt_and_unsupported_objectives_fail(self):
        def forged(payload):
            return replace(self.reserve(payload), signature_hex="0" * 64)
        with self.assertRaises(ValueError):
            self.prepare(host_reserve_once=forged)
        contract = replace(self.fixture.contract, targets=(replace(
            self.fixture.contract.targets[0], rubric_id="unsupported"),))
        with self.assertRaises(ValueError):
            self.prepare(contract=contract,
                         expected_objective_contract_sha256=contract.fingerprint)

    def test_training_identity_prompt_pairs_and_binding_cannot_be_reassigned(self):
        training = self.fixture.training
        swapped = (replace(training[0], prompt_sha256=training[1].prompt_sha256),
                   replace(training[1], prompt_sha256=training[0].prompt_sha256))
        with self.assertRaises(ValueError):
            self.prepare(training_exposures=swapped,
                         expected_training_roster_sha256=training_roster_commitment(swapped))
        with self.assertRaises(ValueError):
            self.prepare(binding=replace(self.fixture.binding, signature_hex="0" * 64))

    def test_host_receipt_cannot_relabel_event_or_proposal(self):
        for forged in (
            lambda payload: replace(self.reserve(payload), event_id="other-event"),
            lambda payload: replace(self.reserve(payload), proposal_sha256="f" * 64),
        ):
            with self.subTest(forged=forged), self.assertRaises(ValueError):
                self.used.clear()
                self.prepare(host_reserve_once=forged)

    def test_public_result_cannot_claim_independent_authority(self):
        result = self.prepare()
        for field in ("graduation_authorized", "model_execution_authenticated",
                      "host_independence_verified"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(result, **{field: True})


if __name__ == "__main__":
    unittest.main()
