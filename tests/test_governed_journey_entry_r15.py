"""Source-only governed caller tests; keys, model and external epoch are synthetic."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import unittest

from test_host_authority_reservation_r12 import BINDING, FakeAnchor, PROVIDER, SECRETS
from test_journey_execution_r12 import SignedFullStateJourneyTests

from labyrinth.hf_development_event_journal import inspect_development_event
from labyrinth.host_authority_reservation_r12 import (
    HostAuthorityError, HostProviderReservationService, sign_provider_request,
)
from labyrinth.provider_journey_adapter_r13 import (
    ProviderJourneyAdapter, ProviderJourneyPins, training_allocation_sha256,
)
from labyrinth.signed_journey_loop import journey_state_commitment
from labyrinth.governed_journey_entry_r15 import (
    GovernedJourneyEntryError, execute_provider_bound_fullstate_journey,
)


def _sign(binding, event_id, payload, expected_epoch):
    return sign_provider_request(
        binding, provider_key=PROVIDER, event_id=event_id,
        proposal_bytes=payload, expected_epoch=expected_epoch,
    )


class GovernedJourneyEntryR15(unittest.TestCase):
    def setUp(self):
        source = SignedFullStateJourneyTests(
            "test_real_journal_executes_signed_next_maze_with_byte_readback")
        source.setUp()
        self.addCleanup(source.doCleanups)
        self.source = source
        fixture = source.fixture
        allocation = fixture.allocate()
        self.anchor = FakeAnchor()
        self.host = HostProviderReservationService(
            Path(source.temp.name) / "provider-authority.sqlite",
            BINDING, external=self.anchor, **SECRETS,
        )
        self.host.initialize()
        self.pins = ProviderJourneyPins(
            authority_binding_sha256=BINDING.sha256,
            event_id=allocation.event_id,
            exam_id=allocation.exam_id,
            scope_sha256=fixture.journey.scope.fingerprint,
            contract_sha256=fixture.contract.fingerprint,
            journey_sha256=journey_state_commitment(fixture.journey),
            learner_checkpoint_sha256=fixture.journey.checkpoint_sha256,
            signed_binding_sha256=sha256(fixture.binding.binding_bytes).hexdigest(),
            training_roster_sha256=allocation.training_roster_sha256,
            expected_train_allocation_sha256=training_allocation_sha256(
                [{"room_id": room.room_id, "skill": room.skill,
                  "case_seed": room.case_seed} for room in allocation.maze.rooms],
                [{"challenge_id": view.challenge_id,
                  "prompt_sha256": view.prompt_sha256, "skill": view.skill}
                 for view in allocation.views],
            ),
            previous_cycle=fixture.journey.attempts[-1].cycle_index,
            next_cycle=fixture.journey.cycle_index,
            expected_host_fence=1,
            weak_skills=allocation.weak_skills,
            expected_run_id=source.binding.run_id,
            expected_objective_id=source.binding.objective_id,
            expected_source_tree_sha256=source.binding.source_tree_sha256,
            fullstate_binding_sha256=source.binding.commitment_sha256(),
        )
        self.adapter = ProviderJourneyAdapter(self.host, BINDING, self.pins, _sign)

    def call_args(self, **changes):
        args = self.source.params(expected_fence=1)
        for name in ("host_reserve_once", "verify_host_reservation",
                     "verify_host_binding"):
            del args[name]
        args.update(changes)
        return args

    def test_typed_provider_entry_completes_one_public_source_only_journey(self):
        s = self.source
        mocks = s.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            result = execute_provider_bound_fullstate_journey(
                provider_adapter=self.adapter, **self.call_args())
        self.assertEqual(self.anchor.current.epoch, 1)
        self.assertEqual(s.model_created, 2)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.host_independence_verified)
        self.assertTrue(inspect_development_event(
            s.db, "dev-event-1", state_key=s.state_key)["consumed"])

    def test_missing_provider_or_pin_mismatch_refuses_without_model_or_event(self):
        for adapter, changed in (
            (None, {}),
            (self.adapter, {"expected_source_tree_sha256": "2" * 64}),
            (self.adapter, {"expected_fence": 2}),
            (self.adapter, {"expected_training_roster_sha256": "1" * 64}),
        ):
            with self.subTest(adapter=adapter is None, changed=changed):
                with self.assertRaises(GovernedJourneyEntryError):
                    execute_provider_bound_fullstate_journey(
                        provider_adapter=adapter, **self.call_args(**changed))
                self.assertEqual(self.anchor.current.epoch, 0)
                self.assertEqual(self.source.model_created, 0)

    def test_host_authority_callbacks_cannot_be_injected_or_overridden(self):
        with self.assertRaises(TypeError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=self.adapter,
                host_reserve_once=lambda *_: True, **self.call_args())
        self.assertEqual(self.anchor.current.epoch, 0)

    def test_remote_committed_reply_forgery_never_loads_neural_model(self):
        self.anchor.forge_reply = True
        with self.assertRaises(HostAuthorityError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=self.adapter, **self.call_args())
        self.assertEqual(self.anchor.current.epoch, 1)
        self.assertEqual(self.source.model_created, 0)

    def test_second_provider_call_cannot_replay_consumed_event(self):
        s = self.source
        mocks = s.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            execute_provider_bound_fullstate_journey(
                provider_adapter=self.adapter, **self.call_args())
        with self.assertRaisesRegex(HostAuthorityError, "replay"):
            execute_provider_bound_fullstate_journey(
                provider_adapter=self.adapter, **self.call_args())
        self.assertEqual(self.anchor.current.epoch, 1)


if __name__ == "__main__":
    unittest.main()
