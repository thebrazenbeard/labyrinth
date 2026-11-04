"""Provider/host Journey boundary tests; fake remote anchor is NOT host custody."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from test_host_authority_reservation_r12 import BINDING, FakeAnchor, PROVIDER, SECRETS

from labyrinth.host_authority_reservation_r12 import (
    HostAuthorityError, HostProviderReservationService, ProviderRequest,
    sign_provider_request,
)
from labyrinth.hf_development_event_journal import FullStateRunBinding
from labyrinth.journey_execution_r12 import maze_commitment_sha256
from labyrinth.provider_journey_adapter_r13 import (
    ProviderJourneyAdapter, ProviderJourneyPins, training_allocation_sha256,
)
from labyrinth.signed_journey_loop import journey_state_commitment, prepare_signed_journey_loop
from labyrinth.weakness_training_bridge import allocate_signed_program_training


def signer(binding, event_id, payload, expected_epoch):
    return sign_provider_request(
        binding, provider_key=PROVIDER, event_id=event_id,
        proposal_bytes=payload, expected_epoch=expected_epoch,
    )


class ProviderJourneyAdapterTests(unittest.TestCase):
    def setUp(self):
        from test_weakness_training_bridge_r9 import WeaknessTrainingBridgeTests
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = WeaknessTrainingBridgeTests(
            "test_signed_program_weakness_allocates_deterministic_answerless_train_rooms")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.anchor = FakeAnchor()
        self.host = HostProviderReservationService(
            Path(self.temp.name) / "authority.sqlite", BINDING,
            external=self.anchor, **SECRETS,
        )
        self.host.initialize()
        f = self.fixture
        self.allocation = allocate_signed_program_training(
            contract=f.contract, journey=f.journey, binding=f.binding,
            training_exposures=f.training,
            expected_training_roster_sha256=f.roster,
            expected_objective_contract_sha256=f.contract.fingerprint,
            **f.verification,
        )
        self.run = FullStateRunBinding(
            run_id="run-pinned", objective_id=f.journey.scope.objective_id,
            base_checkpoint_sha256=f.journey.baseline_checkpoint_sha256,
            source_tree_sha256="9" * 64,
            maze_commitment_sha256=maze_commitment_sha256(f.journey.maze),
            prior_checkpoint_sha256=f.journey.checkpoint_sha256,
            initial_checkpoint_sha256="8" * 64,
            initial_cursor=0,
            initial_exposures=tuple(
                (view.challenge_id, view.prompt_sha256)
                for view in self.allocation.views),
        )
        self.pins = ProviderJourneyPins(
            authority_binding_sha256=BINDING.sha256,
            event_id=self.allocation.event_id,
            exam_id=self.allocation.exam_id,
            scope_sha256=f.journey.scope.fingerprint,
            contract_sha256=f.contract.fingerprint,
            journey_sha256=journey_state_commitment(f.journey),
            learner_checkpoint_sha256=f.journey.checkpoint_sha256,
            signed_binding_sha256=sha256(f.binding.binding_bytes).hexdigest(),
            training_roster_sha256=self.allocation.training_roster_sha256,
            expected_train_allocation_sha256=training_allocation_sha256(
                [{"room_id": room.room_id, "skill": room.skill,
                  "case_seed": room.case_seed} for room in self.allocation.maze.rooms],
                [{"challenge_id": view.challenge_id,
                  "prompt_sha256": view.prompt_sha256, "skill": view.skill}
                 for view in self.allocation.views],
            ),
            previous_cycle=f.journey.attempts[-1].cycle_index,
            next_cycle=f.journey.cycle_index,
            expected_host_fence=1,
            weak_skills=self.allocation.weak_skills,
            expected_run_id=self.run.run_id,
            expected_objective_id=self.run.objective_id,
            expected_source_tree_sha256=self.run.source_tree_sha256,
            fullstate_binding_sha256=self.run.commitment_sha256(),
        )
        self.adapter = ProviderJourneyAdapter(self.host, BINDING, self.pins, signer)

    def prepare(self, **overrides):
        f = self.fixture
        params = dict(
            contract=f.contract, journey=f.journey, binding=f.binding,
            training_exposures=f.training,
            expected_training_roster_sha256=f.roster,
            expected_objective_contract_sha256=f.contract.fingerprint,
            expected_journey_sha256=self.pins.journey_sha256,
            expected_learner_checkpoint_sha256=f.journey.checkpoint_sha256,
            expected_fence=1,
            host_reserve_once=self.adapter.host_reserve_once,
            verify_host_reservation=self.adapter.verify_host_reservation,
            verification_inputs=f.verification,
        )
        params.update(overrides)
        return prepare_signed_journey_loop(**params)

    def test_real_signed_journey_calls_provider_then_remote_cas_and_host_verifier(self):
        plan = self.prepare()
        self.assertEqual(self.anchor.current.epoch, 1)
        self.assertEqual(plan.reservation.event_id, self.allocation.event_id)
        self.assertEqual(plan.reservation.fence, 1)
        self.assertTrue(self.adapter.verify_host_binding(plan, self.run))
        with self.assertRaises(HostAuthorityError):
            self.prepare()
        self.assertEqual(self.anchor.current.epoch, 1)

    def test_provider_request_forgery_or_boolean_callback_fails_without_effect(self):
        forged = lambda binding, event_id, payload, epoch: replace(
            signer(binding, event_id, payload, epoch), signature_hex="0" * 64)
        for callback in (forged, lambda *_: True,
                         lambda *_: ProviderRequest("foreign", "foreign", "0"*64,
                                                    "other", "0"*64, 0, "0"*64)):
            with self.subTest(callback=callback):
                adapter = ProviderJourneyAdapter(self.host, BINDING, self.pins, callback)
                with self.assertRaises(HostAuthorityError):
                    self.prepare(host_reserve_once=adapter.host_reserve_once)
                self.assertEqual(self.anchor.current.epoch, 0)

    def test_identity_and_policy_changes_fail_before_reservation(self):
        for name, replacement in (
            ("scope", replace(self.pins, scope_sha256="0" * 64)),
            ("journey", replace(self.pins, journey_sha256="0" * 64)),
            ("learner", replace(self.pins, learner_checkpoint_sha256="0" * 64)),
            ("contract", replace(self.pins, contract_sha256="0" * 64)),
            ("event", replace(self.pins, event_id="other-event")),
            ("run", replace(self.pins, fullstate_binding_sha256="0" * 64)),
        ):
            with self.subTest(name=name):
                adapter = ProviderJourneyAdapter(self.host, BINDING, replacement, signer)
                if name == "run":
                    plan = self.prepare()
                    self.assertFalse(adapter.verify_host_binding(plan, self.run))
                    break
                with self.assertRaises(HostAuthorityError):
                    self.prepare(host_reserve_once=adapter.host_reserve_once)
                self.assertEqual(self.anchor.current.epoch, 0)

    def test_cross_host_and_policy_receipts_are_rejected(self):
        plan = self.prepare()
        changed = replace(BINDING, policy_sha256="2" * 64)
        with self.assertRaises(HostAuthorityError):
            ProviderJourneyAdapter(self.host, changed, self.pins, signer)
        self.assertFalse(self.adapter.verify_host_reservation(
            replace(plan.reservation, host_id="alien-host"),
            plan.reservation.signed_bytes))
        self.assertFalse(self.adapter.verify_host_binding(
            plan, replace(self.run, source_tree_sha256="1" * 64)))

    def test_host_rejects_unpinned_skill_even_when_signed_by_provider(self):
        # Build a genuine proposal with the genuine signed Journey using an
        # intercepting callback, then alter its declared skill and re-sign it.
        captured = []

        def intercept(payload):
            captured.append(payload)
            raise RuntimeError("stop before reservation")

        with self.assertRaises(RuntimeError):
            self.prepare(host_reserve_once=intercept)
        raw = json.loads(captured[0])
        raw["weak_skills"] = ["unsupported"]
        raw["rooms"][0]["skill"] = "unsupported"
        raw["answerless_train_pairs"][0]["skill"] = "unsupported"
        changed = (json.dumps(raw, sort_keys=True, separators=(",", ":"),
                              allow_nan=False, ensure_ascii=True) + "\n").encode()
        with self.assertRaises(HostAuthorityError):
            self.adapter.host_reserve_once(changed)
        self.assertEqual(self.anchor.current.epoch, 0)

    def test_forged_host_receipt_or_fullstate_run_binds_fail_closed(self):
        plan = self.prepare()
        self.assertFalse(self.adapter.verify_host_reservation(
            replace(plan.reservation, signature_hex="0" * 64),
            plan.reservation.signed_bytes))
        self.assertFalse(self.adapter.verify_host_binding(
            plan, replace(self.run, run_id="other-run")))
        self.assertFalse(self.adapter.verify_host_binding(
            plan, replace(self.run, objective_id="other-objective")))

    def test_committed_receipt_requires_fresh_remote_authority_readback(self):
        plan = self.prepare()
        self.assertTrue(self.adapter.verify_host_reservation(
            plan.reservation, plan.reservation.signed_bytes))
        self.anchor.read_forge = True
        self.assertFalse(self.adapter.verify_host_reservation(
            plan.reservation, plan.reservation.signed_bytes))
        self.assertFalse(self.adapter.verify_host_binding(plan, self.run))


if __name__ == "__main__":
    unittest.main()
