"""Exercises the opt-in signed Journey -> full-state training runtime seam.

Torch serialization and private signer custody are deliberately mocked; the
SQLite run journal, signed-development verifier and Journey allocation are real.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_weakness_training_bridge_r9 import WeaknessTrainingBridgeTests

from labyrinth.hf_development_event_journal import (
    FullStateRunBinding, initialize_development_event_journal,
    inspect_development_event,
)
from labyrinth.fullstate_successor import DevelopmentStepReceipt, FullStateSuccessorError
from labyrinth.neural_resume_checkpoint import FullCheckpointReceipt
from labyrinth.signed_journey_loop import (
    HostLoopReservation, journey_state_commitment,
)
from labyrinth.journey_execution_r12 import (
    SignedJourneyExecutionError, execute_signed_fullstate_journey,
    maze_commitment_sha256,
)


_KEY = b"test signer for a non-production host lease"


class SignedFullStateJourneyTests(unittest.TestCase):
    def setUp(self):
        fixture = WeaknessTrainingBridgeTests(
            "test_signed_program_weakness_allocates_deterministic_answerless_train_rooms")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.db = root / "journal.sqlite3"
        self.state_key = b"separate fullstate journal signer 32+ bytes"
        initialize_development_event_journal(self.db, state_key=self.state_key)
        self.initial = root / "initial"
        self.initial.mkdir()
        self.successor = root / "next"
        self.seen = set()
        self.event_checks = []
        self.model_created = 0
        allocation = fixture.allocate()
        self.exposures = tuple((view.challenge_id, view.prompt_sha256)
                               for view in allocation.views)
        self.binding = FullStateRunBinding(
            run_id="fullstate-run-1",
            objective_id=fixture.scope.objective_id,
            base_checkpoint_sha256=fixture.journey.baseline_checkpoint_sha256,
            source_tree_sha256="f" * 64,
            maze_commitment_sha256=maze_commitment_sha256(fixture.journey.maze),
            prior_checkpoint_sha256=fixture.journey.checkpoint_sha256,
            initial_checkpoint_sha256="e" * 64,
            initial_cursor=0,
            initial_exposures=self.exposures,
        )

    def reserve(self, proposal: bytes):
        body = json.loads(proposal)
        self.assertEqual(body["event_id"], "dev-event-1")
        if body["event_id"] in self.seen:
            raise ValueError("persistently consumed event")
        self.seen.add(body["event_id"])
        unsigned = dict(
            schema="labyrinth.host-loop-reservation.v1",
            event_id=body["event_id"],
            proposal_sha256=sha256(proposal).hexdigest(),
            fence=7,
            host_id="test-host-operator",
        )
        canonical = (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return HostLoopReservation(**unsigned,
                                   signature_hex=hmac.new(_KEY, canonical, sha256).hexdigest())

    @staticmethod
    def verify_reservation(reservation, signed):
        return hmac.compare_digest(reservation.signature_hex,
                                   hmac.new(_KEY, signed, sha256).hexdigest())

    def model_factory(self):
        self.model_created += 1

        class CPUModel:
            def parameters(self):
                return (SimpleNamespace(device=SimpleNamespace(type="cpu")),)

            def named_buffers(self):
                return ()

        return CPUModel()

    def params(self, **changes):
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
            verify_host_reservation=self.verify_reservation,
            verification_inputs=fixture.verification,
            fullstate_binding=self.binding,
            expected_source_tree_sha256="f" * 64,
            verify_host_binding=lambda plan, binding: True,
            journal_db=self.db,
            journal_state_key=self.state_key,
            initial_checkpoint_dir=self.initial,
            successor_checkpoint_dir=self.successor,
            model_factory=self.model_factory,
            optimizer_factory=lambda _model: object(),
            training_step=lambda _model, _optimizer, planned, cursor:
                DevelopmentStepReceipt(planned, len(planned), cursor, cursor + len(planned)),
            verify_signed_event=self.verify_event,
            attest_successor_bytes=lambda path, receipt: True,
        )
        args.update(changes)
        return args

    def verify_event(self, event_id, binding):
        self.event_checks.append((event_id, binding))
        return True

    def checkpoint_mocks(self, neural=None):
        fingerprints = neural if neural is not None else [
            ("a" * 64, (0,), "d" * 64),
            ("b" * 64, (len(self.exposures),), "d" * 64),
            ("b" * 64, (len(self.exposures),), "d" * 64),
        ]

        def load(_path, *, expected_sha256, expected_context, **_):
            return FullCheckpointReceipt(expected_sha256, expected_context, ("weight",))

        def save(_path, *, context, **_):
            return FullCheckpointReceipt("9" * 64, context, ("weight",))

        return (
            patch("labyrinth.fullstate_successor._neural_fingerprint",
                  side_effect=fingerprints),
            patch("labyrinth.fullstate_successor.load_full_checkpoint", side_effect=load),
            patch("labyrinth.fullstate_successor.save_full_checkpoint", side_effect=save),
        )

    def test_real_journal_executes_signed_next_maze_with_byte_readback(self):
        order = []

        def train(model, optimizer, planned, cursor):
            order.append((planned, cursor))
            return DevelopmentStepReceipt(planned, len(planned), cursor, cursor + len(planned))

        mocks = self.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            output = execute_signed_fullstate_journey(
                **self.params(training_step=train))
        self.assertEqual(order, [(self.exposures, 0)])
        self.assertEqual(self.event_checks, [("dev-event-1", self.binding)])
        self.assertGreaterEqual(self.model_created, 2)
        self.assertEqual(output.plan.allocation.event_id, "dev-event-1")
        self.assertEqual(output.successor.next_cursor, len(self.exposures))
        self.assertEqual(output.successor.checkpoint_sha256, "9" * 64)
        self.assertFalse(output.graduation_authorized)
        self.assertFalse(output.model_execution_authenticated)
        self.assertFalse(output.host_independence_verified)
        record = inspect_development_event(self.db, "dev-event-1", state_key=self.state_key)
        self.assertEqual(record["checkpoint_sha256"], "9" * 64)

    def test_binding_changed_learner_or_maze_rejected_before_event_reservation(self):
        for changed in (
            replace(self.binding, prior_checkpoint_sha256="0" * 64),
            replace(self.binding, maze_commitment_sha256="0" * 64),
            replace(self.binding, source_tree_sha256="0" * 64),
            replace(self.binding, objective_id="wrong-objective"),
        ):
            with self.subTest(changed=changed), self.assertRaises(SignedJourneyExecutionError):
                execute_signed_fullstate_journey(**self.params(fullstate_binding=changed))
        self.assertFalse(self.seen)
        self.assertEqual(self.model_created, 0)

    def test_mismatched_train_prompts_do_not_reach_model(self):
        bad = replace(self.binding, initial_exposures=tuple(reversed(self.exposures)))
        with self.assertRaisesRegex(SignedJourneyExecutionError, "exposure"):
            execute_signed_fullstate_journey(**self.params(fullstate_binding=bad))
        self.assertEqual(self.model_created, 0)
        self.assertFalse(inspect_development_event(self.db, "dev-event-1",
                                                  state_key=self.state_key)["consumed"])

    def test_missing_external_host_authority_or_attestation_is_denied(self):
        for missing in ("host_reserve_once", "verify_host_reservation",
                        "verify_host_binding", "verify_signed_event",
                        "attest_successor_bytes"):
            with self.subTest(missing=missing), self.assertRaises(SignedJourneyExecutionError):
                execute_signed_fullstate_journey(**self.params(**{missing: None}))
        self.assertFalse(self.seen)
        self.assertEqual(self.model_created, 0)

    def test_missing_host_journal_or_bad_checkpoint_destination_never_consumes_event(self):
        for changes in (
            dict(journal_db=self.db.parent / "not-initialized.sqlite3"),
            dict(initial_checkpoint_dir=self.db.parent / "missing-initial"),
            dict(successor_checkpoint_dir=self.db.parent / "missing" / "next"),
            dict(journal_state_key=b"too-short"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                    SignedJourneyExecutionError, "preflight"):
                execute_signed_fullstate_journey(**self.params(**changes))
        self.assertFalse(self.seen)
        self.assertEqual(self.model_created, 0)

    def test_forged_signature_and_wrong_host_binding_denied(self):
        def forged(payload):
            return replace(self.reserve(payload), signature_hex="0" * 64)

        with self.assertRaisesRegex(ValueError, "signature"):
            execute_signed_fullstate_journey(**self.params(host_reserve_once=forged))
        self.seen.clear()
        with self.assertRaisesRegex(SignedJourneyExecutionError, "authority"):
            execute_signed_fullstate_journey(**self.params(verify_host_binding=lambda *_: False))
        self.assertEqual(self.model_created, 0)

    def test_boolean_sink_or_foreign_reservation_receipt_is_not_host_authority(self):
        # PR #62 at 709b3e0 exposes a distinct authenticated-binding sink API
        # with no mapping to signed Journey's proposal/host signature contract.
        # Its True result or an alien receipt must not be silently adapted.
        for foreign in (
            lambda _proposal: True,
            lambda _proposal: SimpleNamespace(event_id="dev-event-1", fence=7),
        ):
            with self.subTest(foreign=foreign), self.assertRaisesRegex(
                    ValueError, "typed one-shot reservation"):
                execute_signed_fullstate_journey(**self.params(host_reserve_once=foreign))
        self.assertEqual(self.model_created, 0)
        self.assertFalse(inspect_development_event(self.db, "dev-event-1",
                                                  state_key=self.state_key)["consumed"])

    def test_double_reservation_cannot_repeat_the_neural_cycle(self):
        mocks = self.checkpoint_mocks()
        with mocks[0], mocks[1], mocks[2]:
            execute_signed_fullstate_journey(**self.params())
        created = self.model_created
        with self.assertRaisesRegex(ValueError, "consumed event"):
            execute_signed_fullstate_journey(**self.params(
                successor_checkpoint_dir=self.successor.parent / "another"))
        self.assertEqual(self.model_created, created)

    def test_pure_metadata_advance_cannot_publish_unmodified_neural_bytes(self):
        mocks = self.checkpoint_mocks(neural=[
            ("a" * 64, (0,), "d" * 64),
            ("a" * 64, (len(self.exposures),), "d" * 64),
        ])
        with mocks[0], mocks[1], mocks[2]:
            with self.assertRaisesRegex(FullStateSuccessorError, "neural parameter"):
                execute_signed_fullstate_journey(**self.params())
        record = inspect_development_event(self.db, "dev-event-1", state_key=self.state_key)
        self.assertEqual(record["checkpoint_sha256"], "e" * 64)

    def test_host_event_checker_must_affirm_exact_signed_event_and_run(self):
        def wrong(event_id, binding):
            self.event_checks.append((event_id, binding))
            return False

        with self.assertRaises(FullStateSuccessorError):
            execute_signed_fullstate_journey(**self.params(verify_signed_event=wrong))
        self.assertEqual(self.model_created, 0)
        self.assertEqual(self.event_checks, [("dev-event-1", self.binding)])


if __name__ == "__main__":
    unittest.main()
