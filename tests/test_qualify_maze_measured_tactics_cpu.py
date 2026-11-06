"""Source-verified two-polarity scheduling; no host or model execution claim."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.blind_evaluation_custody import (
    PrivateExamCase, initialize_assessment, score_assessment,
)
from labyrinth.blind_transition_bridge import PublicAssessmentCase
from labyrinth.hf_development_transition import (
    DevelopmentCaseRef, development_case_commitment,
)
from labyrinth.measured_weakness_feedback import (
    TrainingExposureRef, training_roster_commitment,
)
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from scripts import qualify_maze_conditioned_minilm_cpu as probe
from tests import test_blind_transition_bridge as bridge_fixtures
from tests import test_signed_exam_lineage as lineage_fixtures


class SignedMeasuredTacticBatchesTests(unittest.TestCase):
    def setUp(self):
        fixture = bridge_fixtures.BlindTransitionBridgeTests(
            "test_authenticated_score_plus_lineage_produces_verified_development_fail")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.scope = probe.default_scope()
        fixture.scope = self.scope
        fixture.context = replace(
            fixture.context, scope_fingerprint=self.scope.fingerprint)
        cases = (
            PrivateExamCase("ph1", "positive_polarity", "heldout",
                            "The collaborator found the reply encouraging.",
                            "positive-heldout", "positive"),
            PrivateExamCase("ph2", "positive_polarity", "heldout",
                            "Everyone appreciated the courteous assistance.",
                            "positive-heldout", "positive"),
            PrivateExamCase("pt1", "positive_polarity", "transfer",
                            "The review made her day brighter.",
                            "positive-transfer", "positive"),
            PrivateExamCase("pt2", "positive_polarity", "transfer",
                            "They welcomed his kind response warmly.",
                            "positive-transfer", "positive"),
            PrivateExamCase("nh1", "negative_polarity", "heldout",
                            "The outcome made the customer regret asking.",
                            "negative-heldout", "negative"),
            PrivateExamCase("nh2", "negative_polarity", "heldout",
                            "She rejected the unacceptable treatment.",
                            "negative-heldout", "negative"),
            PrivateExamCase("nt1", "negative_polarity", "transfer",
                            "His disappointment was difficult to conceal.",
                            "negative-transfer", "negative"),
            PrivateExamCase("nt2", "negative_polarity", "transfer",
                            "The complaint went unresolved all week.",
                            "negative-transfer", "negative"),
        )
        public = tuple(PublicAssessmentCase(
            case.case_id, case.skill, case.split,
            sha256(case.prompt.encode()).hexdigest(), case.structure_key,
        ) for case in cases)
        dev = tuple(DevelopmentCaseRef(
            case.case_id, case.skill, case.prompt_sha256, case.structure_key,
        ) for case in public)
        fixture.public_precommit = initialize_assessment(
            fixture.path / "polarity-assessment.sqlite", cases,
            run_id="candidate-run-0", objective_id=self.scope.objective_id,
            checkpoint_sha256=fixture.artifact_hash,
            evaluator_key=bridge_fixtures.EVALUATOR_KEY,
            state_key=bridge_fixtures.STATE_KEY,
        )
        responses = {case.case_id: case.gold_answer for case in cases}
        for case_id in ("nh1", "nt1", "nt2"):
            responses[case_id] = "positive"
        fixture.scored = score_assessment(
            fixture.path / "polarity-assessment.sqlite", cases, responses,
            artifact_root=fixture.path, artifact_name="candidate.safetensors",
            evaluator_key=bridge_fixtures.EVALUATOR_KEY,
            state_key=bridge_fixtures.STATE_KEY,
        )
        fixture.attestation.update({
            "objective_id": self.scope.objective_id,
            "scope_fingerprint": self.scope.fingerprint,
            "public_exam_sha256": fixture.public_precommit,
        })
        transition = fixture.bridge(
            public_cases=public,
            development_roster_precommitment_sha256=development_case_commitment(dev),
        )
        raw, sig = fixture.lineage()
        self.inputs = dict(
            scope=self.scope, context=fixture.context, transition=transition,
            scored_receipt_bytes=fixture.scored.receipt_bytes,
            scored_signature_hex=fixture.scored.signature_hex,
            candidate_lineage_bytes=raw,
            candidate_lineage_signature_hex=sig,
            candidate_run_id="candidate-run-0",
            candidate_artifact_sha256=fixture.artifact_hash,
            expected_evaluator_roster_sha256=fixture.public_precommit,
            expected_development_roster_sha256=development_case_commitment(dev),
            failure_threshold_percent=75,
            binding_signer_id=lineage_fixtures.SIGNER,
            expected_nonce_sha256=lineage_fixtures.NONCE,
            independent_evaluator_key=bridge_fixtures.EVALUATOR_KEY,
            independent_lineage_key=bridge_fixtures.LINEAGE_KEY,
            independent_transition_key=bridge_fixtures.TRANSITION_KEY,
            independent_binding_key=lineage_fixtures.BINDING_KEY,
        )
        self.binding = sign_exam_lineage_binding(**self.inputs)
        self.cycles = probe.build_training_cycles(
            self.scope, weak_cycle_one=("negative_polarity",))
        self.training = tuple(TrainingExposureRef(
            row.exposure_id, sha256(row.prompt.encode()).hexdigest(),
        ) for row in self.cycles[0].examples)
        self.training_sha = training_roster_commitment(self.training)

    def verified(self, **changes):
        args = {**self.inputs,
                "declared_scope": self.scope,
                "training_cycles": self.cycles,
                "steps_per_cycle": 16,
                "binding": self.binding,
                "training_exposures": self.training,
                "expected_training_roster_sha256": self.training_sha}
        args.update(changes)
        return probe.plan_signed_measured_sentiment_batches(**args)

    def test_real_signed_two_polarity_score_changes_actual_train_visits(self):
        planned = self.verified()
        self.assertEqual(planned.weak_skill, "negative_polarity")
        self.assertEqual(planned.tactic, "transfer")
        self.assertEqual(planned.cycle, self.cycles[1])
        self.assertEqual(planned.unique_training_exposures, 12)
        self.assertEqual(planned.class_counts_per_batch[0], (3, 1))
        self.assertNotEqual(
            planned.batches[0], probe.select_training_batch(self.cycles[1], 0))
        self.assertEqual(
            {row.exposure_id for batch in planned.batches for row in batch},
            {row.exposure_id for row in self.cycles[1].examples},
        )
        self.assertFalse(planned.source_signature_host_attested)
        self.assertFalse(planned.model_execution_authenticated)
        self.assertFalse(planned.graduation_authorized)

    def test_tampered_signatures_unpinned_train_and_wrong_scope_fail_closed(self):
        for mutation in (
            {"binding": replace(self.binding, signature_hex="f" * 64)},
            {"scored_signature_hex": "f" * 64},
            {"independent_evaluator_key": b"z" * 32},
            {"expected_training_roster_sha256": "f" * 64},
            {"training_exposures": self.training[:-1]},
            {"scope": replace(self.scope, seed=123456)},
            {"training_cycles": probe.build_training_cycles(self.scope)},
        ):
            with self.subTest(mutation=list(mutation)), self.assertRaises(
                    (TypeError, ValueError, RuntimeError)):
                self.verified(**mutation)

    def test_planned_five_arm_train_exposures_match_exactly(self):
        # Source-level invariant only: no physical optimizer actually ran.
        planned = self.verified()
        randomized = probe.preregistered_randomized_labels(
            planned.cycle, self.scope.seed)
        expected_visits = tuple(tuple(row.exposure_id for row in batch)
                                for batch in planned.batches)
        for arm in probe.configured_arms(True):
            visited = []
            for batch in planned.batches:
                visits = tuple(row.exposure_id for row in batch)
                visited.append(visits)
                actual_labels = probe.training_batch_labels(
                    batch, arm, randomized if arm == "randomized_labels" else None)
                self.assertEqual(len(actual_labels), len(visits))
            self.assertEqual(tuple(visited), expected_visits, arm)

    def test_synthetic_dataclass_or_operator_hint_is_not_a_verifier(self):
        from tests.test_measured_sentiment_corridors import synthetic_feedback

        for fake in (
            {"binding": synthetic_feedback(self.scope)},
            {"weak_skills": ("negative_polarity",)},
            {"operator_tactic": "transfer"},
        ):
            with self.subTest(fake=list(fake)), self.assertRaises(
                    (TypeError, ValueError)):
                self.verified(**fake)


if __name__ == "__main__":
    unittest.main()
