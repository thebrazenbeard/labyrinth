"""Adversarial synthetic checks of the opt-in provider-bound Journey boundary.

These tests use the existing fake external epoch, signer and model hooks. They
cannot authenticate real host custody or actual pretrained-model execution.
"""

from __future__ import annotations

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import test_governed_journey_entry_r15 as fixture_module

from labyrinth.governed_journey_entry_r15 import (
    GovernedJourneyEntryError, execute_provider_bound_fullstate_journey,
)
from labyrinth.host_authority_reservation_r12 import HostAuthorityError
from labyrinth.journey_execution_r12 import SignedJourneyExecutionError


class GovernedJourneyRedteamR17(TestCase):
    def setUp(self) -> None:
        f = fixture_module.GovernedJourneyEntryR15(
            "test_typed_provider_entry_completes_one_public_source_only_journey")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f

    def test_mutated_adapter_provider_binding_denies_before_reservation(self) -> None:
        f = self.f
        from test_host_authority_reservation_r12 import BINDING
        f.adapter.binding = replace(BINDING, policy_sha256="2" * 64)
        with self.assertRaises(GovernedJourneyEntryError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=f.adapter, **f.call_args())
        self.assertEqual(f.anchor.current.epoch, 0)
        self.assertEqual(f.source.model_created, 0)

    def test_mutated_next_train_allocation_pin_denies_before_host_cas(self) -> None:
        f = self.f
        f.adapter.pins = replace(f.pins, expected_train_allocation_sha256="0" * 64)
        with self.assertRaises(HostAuthorityError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=f.adapter, **f.call_args())
        self.assertEqual(f.anchor.current.epoch, 0)
        self.assertEqual(f.source.model_created, 0)

    def test_wrong_fullstate_run_id_denies_without_epoch(self) -> None:
        f = self.f
        shadow = replace(f.source.binding, run_id="shadow-run")
        with self.assertRaises(GovernedJourneyEntryError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=f.adapter,
                **f.call_args(fullstate_binding=shadow))
        self.assertEqual(f.anchor.current.epoch, 0)
        self.assertEqual(f.source.model_created, 0)

    def test_host_binding_denial_consumes_once_without_model_or_replay(self) -> None:
        f = self.f
        with patch.object(f.adapter, "verify_host_binding", return_value=False):
            with self.assertRaisesRegex(SignedJourneyExecutionError,
                                        "host binding authority"):
                execute_provider_bound_fullstate_journey(
                    provider_adapter=f.adapter, **f.call_args())
        self.assertEqual(f.anchor.current.epoch, 1)
        self.assertEqual(f.source.model_created, 0)
        with self.assertRaises(HostAuthorityError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=f.adapter, **f.call_args())
        self.assertEqual(f.anchor.current.epoch, 1)

    def test_host_entry_rejects_proxy_not_exact_provider_adapter(self) -> None:
        f = self.f

        class Proxy:
            def __getattr__(self, name):
                return getattr(f.adapter, name)

        with self.assertRaises(GovernedJourneyEntryError):
            execute_provider_bound_fullstate_journey(
                provider_adapter=Proxy(), **f.call_args())
        self.assertEqual(f.anchor.current.epoch, 0)

    def test_adapter_pins_rebinding_after_init_cannot_authorize_shadow_run(self) -> None:
        """Host/operator pins must not be silently replaced after admission.

        This deliberately challenges the in-process mutable wrapper boundary;
        independent host-managed immutable pins are not provided by this test.
        """
        f = self.f
        shadow = replace(f.source.binding, run_id="shadow-run")
        f.adapter.pins = replace(
            f.pins, expected_run_id=shadow.run_id,
            fullstate_binding_sha256=shadow.commitment_sha256(),
        )
        mocks = f.source.checkpoint_mocks()
        try:
            with mocks[0], mocks[1], mocks[2]:
                execute_provider_bound_fullstate_journey(
                    provider_adapter=f.adapter,
                    **f.call_args(fullstate_binding=shadow))
        except (GovernedJourneyEntryError, HostAuthorityError,
                SignedJourneyExecutionError, ValueError):
            pass
        self.assertEqual(f.anchor.current.epoch, 0,
                         "mutable adapter pins spent a host epoch for a new run")
        self.assertEqual(f.source.model_created, 0,
                         "mutable adapter pins reached the neural worker")
