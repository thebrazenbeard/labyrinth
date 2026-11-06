"""Signed Minotaur tactic changes actual neural-loop selection (source-only).

Every development aggregate is re-signed through the real four-key lineage
fixtures. No pretrained model, independent examiner or host effect runs here.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from labyrinth.measured_neural_batch_binding import (
    propose_signed_neural_batch_schedule,
    verify_precommitted_neural_batch_schedule,
)
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from scripts import qualify_maze_conditioned_minilm_cpu as runner
from tests import test_measured_neural_batch_binding as measured_fixture
from tests import test_blind_transition_bridge as bridge_fixture


_SCORER_DOMAIN = b"labyrinth.blind-development-result.v1\0"


class SignedTacticNeuralSelectionTests(unittest.TestCase):
    def setUp(self):
        fixture = measured_fixture.BoundNeuralMeasuredBatchesTests(
            "test_actual_four_key_signed_development_produces_bound_five_arm_schedule"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture

    def _signed_plan(self, *, heldout_correct: int, transfer_correct: int):
        f = self.fixture
        inputs = dict(f.signed.inputs)
        receipt = json.loads(inputs["scored_receipt_bytes"])
        for group in receipt["groups"]:
            if group["skill"] == "negative_polarity":
                group["correct"] = (heldout_correct if group["split"] == "heldout"
                                    else transfer_correct)
        raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"))
               + "\n").encode("ascii")
        signature = hmac.new(bridge_fixture.EVALUATOR_KEY,
                             _SCORER_DOMAIN + raw, sha256).hexdigest()
        inputs["transition"] = replace(
            inputs["transition"], scored_receipt_sha256=sha256(raw).hexdigest())
        inputs["scored_receipt_bytes"] = raw
        inputs["scored_signature_hex"] = signature
        binding = sign_exam_lineage_binding(**inputs)
        proposal = propose_signed_neural_batch_schedule(
            declared_scope=f.signed.scope,
            training_cycles=f.signed.cycles,
            steps_per_cycle=16,
            binding=binding,
            training_exposures=f.signed.training,
            expected_training_roster_sha256=f.signed.training_sha,
            predecessor_checkpoint_sha256=f.prev,
            fence=7,
            **inputs,
        )
        return verify_precommitted_neural_batch_schedule(
            proposal,
            training_cycles=f.signed.cycles,
            expected_schedule_sha256=proposal.schedule_sha256,
            expected_run_id=inputs["candidate_run_id"],
            expected_candidate_artifact_sha256=inputs["candidate_artifact_sha256"],
            expected_predecessor_checkpoint_sha256=f.prev,
            expected_scope_fingerprint=f.signed.scope.fingerprint,
            expected_fence=7,
        )

    def test_signed_development_split_weakness_changes_neural_tactic_and_batches(self):
        # Exact signed heldout/transfer counts, not operator-chosen tactics.
        # Each plan selects *the selector used inside the actual execute loop*.
        plans = {
            "practice": self._signed_plan(heldout_correct=0, transfer_correct=2),
            "transfer": self._signed_plan(heldout_correct=2, transfer_correct=0),
            "contrast": self._signed_plan(heldout_correct=0, transfer_correct=0),
        }
        self.assertEqual({name: plan.tactic for name, plan in plans.items()},
                         {name: name for name in plans})
        visits = {}
        for name, plan in plans.items():
            by_arm = {
                arm: tuple(tuple(row.exposure_id for row in
                                 runner.select_executed_training_batch(
                                     plan.cycle, step, arm, plan))
                           for step in range(16))
                for arm in runner.PREREG_ARMS
            }
            self.assertEqual(len(set(by_arm.values())), 1)
            self.assertEqual(len({item for batch in by_arm["correct"]
                                  for item in batch}), len(plan.cycle.examples))
            visits[name] = by_arm["correct"]
            self.assertFalse(plan.model_execution_authenticated)
            self.assertFalse(plan.graduation_authorized)
        self.assertEqual(len(set(visits.values())), 3)

    def test_signed_tactic_cannot_be_reassigned_without_new_precommit(self):
        plan = self._signed_plan(heldout_correct=2, transfer_correct=0)
        self.assertEqual(plan.tactic, "transfer")
        with self.assertRaises(ValueError):
            verify_precommitted_neural_batch_schedule(
                replace(plan, tactic="practice"),
                training_cycles=self.fixture.signed.cycles,
                expected_schedule_sha256=plan.schedule_sha256,
                expected_run_id=self.fixture.signed.inputs["candidate_run_id"],
                expected_candidate_artifact_sha256=(
                    self.fixture.signed.inputs["candidate_artifact_sha256"]),
                expected_predecessor_checkpoint_sha256=self.fixture.prev,
                expected_scope_fingerprint=self.fixture.signed.scope.fingerprint,
                expected_fence=7,
            )


if __name__ == "__main__":
    unittest.main()
