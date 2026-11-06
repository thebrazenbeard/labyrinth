"""Regression gates for local signed-Journey checks before one-shot reservation.

The R12 test host stores consumed event IDs separately from the full-state
SQLite journal. A pristine SQLite journal alone does not mean the host event
can still be used.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

import test_journey_execution_r12 as r12

from labyrinth.hf_development_event_journal import inspect_development_event
from labyrinth.journey_execution_r12 import (
    SignedJourneyExecutionError, execute_signed_fullstate_journey,
)
from labyrinth.signed_journey_loop import (
    preview_signed_journey_loop, reserve_signed_journey_loop,
)


class JourneyPreflightR14Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = r12.SignedFullStateJourneyTests(
            "test_real_journal_executes_signed_next_maze_with_byte_readback")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_wrong_fullstate_train_exposure_does_not_spend_host_event(self):
        fixture = self.fixture
        bad = replace(fixture.binding,
                      initial_exposures=tuple(reversed(fixture.exposures)))
        self.assertNotEqual(bad.initial_exposures, fixture.exposures)

        with self.assertRaisesRegex(SignedJourneyExecutionError, "exposure"):
            execute_signed_fullstate_journey(**fixture.params(fullstate_binding=bad))

        self.assertFalse(fixture.seen, "local mismatch spent external one-shot event")
        self.assertEqual(fixture.model_created, 0)
        self.assertFalse(inspect_development_event(
            fixture.db, "dev-event-1", state_key=fixture.state_key)["consumed"])

    def test_successful_execution_reserves_once_after_local_preflight(self):
        fixture = self.fixture
        reservation_count = 0

        def count_reserve(proposal):
            nonlocal reservation_count
            reservation_count += 1
            return fixture.reserve(proposal)

        original = r12.execute_signed_fullstate_journey
        mocks = fixture.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            result = original(**fixture.params(host_reserve_once=count_reserve))
        self.assertEqual(result.plan.allocation.event_id, "dev-event-1")
        self.assertEqual(reservation_count, 1)
        self.assertEqual(fixture.seen, {"dev-event-1"})

    def test_corrected_local_roster_can_retry_original_unspent_event(self):
        fixture = self.fixture
        bad = replace(fixture.binding,
                      initial_exposures=tuple(reversed(fixture.exposures)))
        with self.assertRaisesRegex(SignedJourneyExecutionError, "exposure"):
            execute_signed_fullstate_journey(**fixture.params(fullstate_binding=bad))
        self.assertFalse(fixture.seen)
        mocks = fixture.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            result = execute_signed_fullstate_journey(**fixture.params())
        self.assertEqual(result.successor.next_cursor, len(fixture.exposures))
        self.assertEqual(fixture.seen, {"dev-event-1"})

    def test_modified_unsigned_preview_is_rejected_without_host_side_effect(self):
        fixture = self.fixture
        args = fixture.params()
        preview_keys = (
            "contract", "journey", "binding", "training_exposures",
            "expected_training_roster_sha256", "expected_objective_contract_sha256",
            "expected_journey_sha256", "expected_learner_checkpoint_sha256",
            "expected_fence", "verification_inputs",
        )
        preview = preview_signed_journey_loop(**{
            key: args[key] for key in preview_keys
        })
        self.assertFalse(fixture.seen)
        # Even deliberate bypass of the frozen dataclass constructor must be
        # caught at the irreversible reserve boundary.
        object.__setattr__(preview, "proposal_bytes", preview.proposal_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "commitment"):
            reserve_signed_journey_loop(
                preview,
                host_reserve_once=fixture.reserve,
                verify_host_reservation=fixture.verify_reservation,
            )
        self.assertFalse(fixture.seen)

    def test_host_rejection_remains_fail_closed_after_signed_reservation(self):
        """R12's reservation-bearing host verifier cannot run before reserve.

        Its externally held host identity proof is not available to this
        source-only caller without an additional, independently trusted host
        preflight protocol. R14 must preserve the final host-verifier gate.
        """
        fixture = self.fixture
        with self.assertRaisesRegex(SignedJourneyExecutionError, "authority"):
            execute_signed_fullstate_journey(**fixture.params(
                verify_host_binding=lambda plan, fullstate: False))
        self.assertEqual(fixture.seen, {"dev-event-1"})
        self.assertEqual(fixture.model_created, 0)


if __name__ == "__main__":
    unittest.main()
