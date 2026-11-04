"""R25 public/synthetic Labyrinth→Minotaur handoff tests; no real exam or key."""

from dataclasses import replace
import unittest

from test_graduation_gate import KEY, POLICY, evaluated_receipt, independently_signed
from labyrinth.graduation_gate import GateState, _policy_fingerprint
from labyrinth.trainingday_graduation_bridge import (
    AdapterTrainingReceipt, CurriculumAdmission, LearningIntent,
    assess_graduation_handoff,
)


INTENT = LearningIntent(POLICY.objective_id, "training_effect", "9" * 64)
CURRICULUM = CurriculumAdmission(
    POLICY.objective_id, INTENT.intent_sha256, "1" * 64, "2" * 64,
    "6" * 64, "5" * 64, "4" * 64,
    POLICY.expected_evaluator_sha256, POLICY.expected_conditions_sha256,
    "train", 1024, True, False,
)
TRAINING = AdapterTrainingReceipt(
    POLICY.objective_id, CURRICULUM.manifest_sha256, CURRICULUM.admitted_corpus_sha256,
    POLICY.base_sha256, "4" * 64, POLICY.trainee_sha256,
    "5" * 64, "6" * 64, 32, 531.0, 120.0, "VALID_TIME_BUDGET",
)


def assess(*, intent=INTENT, curriculum=CURRICULUM, training=TRAINING,
           policy=POLICY, state=None, receipt=None, signature=None, key=None):
    return assess_graduation_handoff(
        intent, curriculum, training, policy, GateState() if state is None else state,
        exam_receipt_bytes=receipt, exam_signature_hex=signature,
        independent_evaluator_key=key,
    )


class TrainingDayGraduationBridgeTests(unittest.TestCase):
    def test_valid_source_lineage_is_only_ready_for_an_independent_exam(self):
        result = assess()
        self.assertEqual(result.status, "EXAM_REQUIRED")
        self.assertEqual(result.corridor_action, "REQUEST_INDEPENDENT_MINOTAUR_EXAM")
        self.assertEqual(result.minotaur_gate_disposition, "NOT_CALLED")
        self.assertFalse(result.promotion_allowed)
        self.assertFalse(result.qualified_neural_improvement)
        self.assertFalse(result.independently_verified_host_receipts)

    def test_corpus_only_cannot_qualify_without_neural_adapter_receipt(self):
        result = assess(training=None)
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("NO_NEURAL_ADAPTER_RECEIPT", result.reasons)

    def test_historical_r18_invalid_budget_cannot_be_recast_as_admitted(self):
        for receipt in (
            replace(TRAINING, time_integrity_status="TIME_BUDGET_OVERRUN_INVALID"),
            replace(TRAINING, optimizer_elapsed_seconds=19813.562),
            replace(TRAINING, optimizer_elapsed_seconds=float("nan")),
            replace(TRAINING, optimizer_elapsed_seconds=float("inf")),
            replace(TRAINING, optimizer_elapsed_seconds=10**1000),
            replace(TRAINING, optimizer_elapsed_seconds=0.0),
            replace(TRAINING, optimizer_elapsed_seconds=5e-324),
            replace(TRAINING, optimizer_elapsed_seconds=0.01),
        ):
            with self.subTest(receipt=receipt):
                result = assess(training=receipt)
                self.assertEqual(result.status, "BLOCKED")
                self.assertFalse(result.promotion_allowed)

    def test_zero_subnormal_or_implausibly_short_step_time_is_not_exam_ready(self):
        for elapsed in (0, 5e-324, 0.01):
            with self.subTest(elapsed=elapsed):
                result = assess(training=replace(TRAINING, optimizer_elapsed_seconds=elapsed))
                self.assertEqual(result.status, "BLOCKED")
                self.assertEqual(result.reasons, ("INVALID_TRAINING_RECEIPT",))
                self.assertEqual(result.minotaur_gate_disposition, "NOT_CALLED")
        self.assertEqual(assess().status, "EXAM_REQUIRED")

    def test_leaked_or_unlicensed_curriculum_never_reaches_minotaur_gate(self):
        for admission in (
            replace(CURRICULUM, eval_overlap_detected=True),
            replace(CURRICULUM, rights_approved=False),
            replace(CURRICULUM, split="eval"),
            replace(CURRICULUM, admitted_case_count=0),
        ):
            with self.subTest(admission=admission):
                result = assess(curriculum=admission)
                self.assertEqual(result.status, "BLOCKED")
                self.assertEqual(result.minotaur_gate_disposition, "NOT_CALLED")

    def test_exact_artifact_and_subject_bindings_are_required(self):
        for changed in (
            replace(TRAINING, manifest_sha256="a" * 64),
            replace(TRAINING, admitted_corpus_sha256="f" * 64),
            replace(TRAINING, trained_adapter_sha256="b" * 64),
            replace(TRAINING, base_sha256="e" * 64),
            replace(TRAINING, initial_adapter_sha256=POLICY.trainee_sha256),
            replace(TRAINING, tokenizer_sha256="7" * 64),
            replace(TRAINING, source_sha256="8" * 64),
            replace(TRAINING, completed_steps=0),
            replace(TRAINING, optimizer_budget_seconds=True),
        ):
            with self.subTest(receipt=changed):
                self.assertEqual(assess(training=changed).status, "BLOCKED")
        self.assertEqual(assess(intent=replace(INTENT, objective_id="another-objective")).status,
                         "BLOCKED")
        self.assertEqual(assess(curriculum=replace(CURRICULUM, evaluator_sha256="d" * 64)).status,
                         "BLOCKED")
        self.assertEqual(assess(curriculum=replace(CURRICULUM, intent_sha256="d" * 64)).status,
                         "BLOCKED")

    def test_forged_pass_receipt_does_not_bypass_existing_minotaur_gate(self):
        forged = b'{"status":"PASS","accuracy":1}'
        result = assess(receipt=forged, signature="f" * 64, key=KEY)
        self.assertEqual(result.status, "GATE_BLOCKED")
        self.assertEqual(result.minotaur_gate_disposition, "BLOCKED")
        self.assertEqual(result.corridor_action, "REMAIN_IN_LABYRINTH")
        self.assertFalse(result.promotion_allowed)

    def test_synthetic_signed_exam_delegates_without_deployment_permission(self):
        raw, sig = independently_signed(evaluated_receipt())
        result = assess(receipt=raw, signature=sig, key=KEY)
        self.assertEqual(result.status, "SIGNED_GATE_REPORTED_GRADUATED")
        self.assertEqual(result.minotaur_gate_disposition, "GRADUATED")
        self.assertEqual(result.corridor_action, "REQUIRES_INDEPENDENT_CUSTODY_REVIEW")
        self.assertFalse(result.promotion_allowed)
        self.assertFalse(result.qualified_neural_improvement)

    def test_prior_trusted_graduation_does_not_claim_new_signature_check(self):
        prior = GateState(next_attempt=2, graduated=True,
                          policy_fingerprint=_policy_fingerprint(POLICY))
        result = assess(state=prior, receipt=b'{"forged":true}',
                        signature="0" * 64, key=KEY)
        self.assertEqual(result.status, "PRIOR_TRUSTED_GATE_GRADUATION_REPORTED")
        self.assertEqual(result.minotaur_gate_disposition, "GRADUATED")
        self.assertEqual(result.reasons, ("prior graduation, no new exam verified",))
        self.assertFalse(result.promotion_allowed)
        self.assertFalse(result.qualified_neural_improvement)
        unrelated = replace(prior, policy_fingerprint="f" * 64)
        self.assertEqual(assess(state=unrelated, receipt=b'{}',
                                signature="0" * 64, key=KEY).status, "GATE_BLOCKED")

    def test_public_exam_or_trainer_signature_claim_is_insufficient(self):
        raw, sig = independently_signed(evaluated_receipt())
        self.assertEqual(assess(receipt=raw, signature=sig).status, "EXAM_REQUIRED")
        self.assertEqual(assess(receipt=raw, signature="0" * 64, key=KEY).status,
                         "GATE_BLOCKED")
        self.assertEqual(assess(receipt=raw, signature=sig, key=b"trainer-owned").status,
                         "GATE_BLOCKED")


if __name__ == "__main__":
    unittest.main()
