"""Source-only, independently pinned five-arm batch custody tests.

The signed fixture uses a real local scorer and four-key lineage verifier,
but fixture answers are synthetic. These tests never execute a neural model
and cannot establish a host-owned signing key, reservation, or graduation.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from labyrinth.measured_neural_batch_binding import (
    BoundMeasuredNeuralBatches, propose_signed_neural_batch_schedule,
    verify_precommitted_neural_batch_schedule,
)
from scripts import qualify_maze_conditioned_minilm_cpu as probe
from tests import test_qualify_maze_measured_tactics_cpu as source_fixtures


class BoundNeuralMeasuredBatchesTests(unittest.TestCase):
    def setUp(self):
        self.signed = source_fixtures.SignedMeasuredTacticBatchesTests(
            "test_real_signed_two_polarity_score_changes_actual_train_visits")
        self.signed.setUp()
        self.addCleanup(self.signed.doCleanups)
        self.prev = self.signed.fixture.context.predecessor_lora_value_sha256

    def proposal(self, **changed):
        signed = self.signed
        kwargs = dict(
            declared_scope=signed.scope, training_cycles=signed.cycles,
            steps_per_cycle=16, binding=signed.binding,
            training_exposures=signed.training,
            expected_training_roster_sha256=signed.training_sha,
            predecessor_checkpoint_sha256=self.prev, fence=7,
            **signed.inputs,
        )
        kwargs.update(changed)
        return propose_signed_neural_batch_schedule(**kwargs)

    def pinned(self, schedule: BoundMeasuredNeuralBatches, **changed):
        kwargs = dict(
            training_cycles=self.signed.cycles,
            expected_schedule_sha256=schedule.schedule_sha256,
            expected_run_id=self.signed.inputs["candidate_run_id"],
            expected_candidate_artifact_sha256=self.signed.fixture.artifact_hash,
            expected_predecessor_checkpoint_sha256=self.prev,
            expected_scope_fingerprint=self.signed.scope.fingerprint,
            expected_fence=7,
        )
        kwargs.update(changed)
        return verify_precommitted_neural_batch_schedule(schedule, **kwargs)

    def test_actual_four_key_signed_development_produces_bound_five_arm_schedule(self):
        proposal = self.proposal()
        self.assertIs(self.pinned(proposal), proposal)
        self.assertEqual(proposal.weak_skill, "negative_polarity")
        self.assertEqual(proposal.tactic, "transfer")
        self.assertEqual(proposal.arms, probe.PREREG_ARMS)
        self.assertEqual(proposal.source_event_id, "dev-event-1")
        self.assertEqual(len(proposal.batches), 16)
        self.assertEqual(len(proposal.schedule_sha256), 64)
        self.assertNotEqual(proposal.batches[0],
                            probe.select_training_batch(self.signed.cycles[1], 0))
        roster = {row.exposure_id for row in self.signed.cycles[1].examples}
        self.assertEqual({row.exposure_id for batch in proposal.batches
                          for row in batch}, roster)
        self.assertFalse(proposal.host_signature_attested)
        self.assertFalse(proposal.model_execution_authenticated)
        self.assertFalse(proposal.graduation_authorized)

    def test_every_arm_uses_identical_committed_batch_ids_not_identical_labels(self):
        proposal = self.pinned(self.proposal())
        expected = tuple(tuple(row.exposure_id for row in batch)
                         for batch in proposal.batches)
        randomized = probe.preregistered_randomized_labels(
            proposal.cycle, self.signed.scope.seed)
        labels = {}
        for arm in probe.PREREG_ARMS:
            observed = tuple(tuple(row.exposure_id for row in
                                   proposal.batch_for_arm(arm, step))
                             for step in range(proposal.steps_per_cycle))
            self.assertEqual(observed, expected)
            labels[arm] = tuple(probe.training_batch_labels(
                proposal.batch_for_arm(arm, step), arm,
                randomized if arm == "randomized_labels" else None)
                for step in range(proposal.steps_per_cycle))
        self.assertEqual(labels["correct"], labels["zero_update"])
        self.assertEqual(labels["correct"], labels["zero_lr"])
        self.assertNotEqual(labels["correct"], labels["inverted"])
        self.assertNotEqual(labels["correct"], labels["randomized_labels"])
        for invalid_arm, step in (("unknown", 0), ("correct", -1),
                                  ("correct", 16), ("correct", True)):
            with self.subTest(arm=invalid_arm, step=step), self.assertRaises(ValueError):
                proposal.batch_for_arm(invalid_arm, step)

    def test_precommit_rejects_wrong_candidate_checkpoint_fence_and_scope(self):
        proposal = self.proposal()
        failures = (
            {"expected_schedule_sha256": "f" * 64},
            {"expected_run_id": "foreign-run"},
            {"expected_candidate_artifact_sha256": "a" * 64},
            {"expected_predecessor_checkpoint_sha256": "f" * 64},
            {"expected_scope_fingerprint": "f" * 64},
            {"expected_fence": 8},
            {"expected_fence": True},
            {"training_cycles": probe.build_training_cycles(self.signed.scope)},
        )
        for change in failures:
            with self.subTest(change=list(change)), self.assertRaises((TypeError, ValueError)):
                self.pinned(proposal, **change)

    def test_signed_input_tampering_and_operator_weak_hint_rejected(self):
        failures = (
            {"scored_signature_hex": "f" * 64},
            {"candidate_lineage_signature_hex": "f" * 64},
            {"independent_binding_key": b"z" * 32},
            {"candidate_run_id": "different-run"},
            {"candidate_artifact_sha256": "f" * 64},
            {"predecessor_checkpoint_sha256": "a" * 64},
            {"expected_training_roster_sha256": "f" * 64},
            {"training_exposures": self.signed.training[:-1]},
            {"training_cycles": probe.build_training_cycles(self.signed.scope)},
            {"weak_skills": ("negative_polarity",)},
            {"operator_tactic": "transfer"},
            {"fence": True},
        )
        for change in failures:
            with self.subTest(change=list(change)), self.assertRaises((TypeError, ValueError)):
                self.proposal(**change)

    def test_recommitted_tamper_cannot_evade_independent_precommit(self):
        proposal = self.proposal()
        mutation = replace(proposal, batches=proposal.batches[::-1])
        with self.assertRaisesRegex(ValueError, "commitment"):
            self.pinned(mutation)
        # The schedule hash itself is not an authority credential: a caller
        # can compute a new hash but cannot match an independently pinned old
        # digest. No test claims the independent pin exists on a live host.
        with self.assertRaisesRegex(ValueError, "commitment"):
            self.pinned(replace(proposal, schedule_sha256="f" * 64))
        forged = replace(proposal, host_signature_attested=True)
        with self.assertRaisesRegex(ValueError, "authority"):
            self.pinned(forged)

    def test_incomplete_or_forged_train_batches_blocked_before_neural_load(self):
        proposal = self.proposal()
        row = proposal.batches[0][0]
        variants = (
            replace(proposal, batches=proposal.batches[:-1]),
            replace(proposal, batches=((row,) * 4,) + proposal.batches[1:]),
            replace(proposal, batches=((replace(row, prompt="injected"),)
                                      + proposal.batches[0][1:],)
                    + proposal.batches[1:]),
            replace(proposal, arms=probe.ARMS),
            replace(proposal, graduation_authorized=True),
        )
        for candidate in variants:
            with self.subTest(kind=repr(candidate)[:80]), self.assertRaises(ValueError):
                self.pinned(candidate)


if __name__ == "__main__":
    unittest.main()
