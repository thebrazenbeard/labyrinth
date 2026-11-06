"""Signed development weakness can drive TRAIN-only objective materialization."""

from dataclasses import replace
from hashlib import sha256
import hmac
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.blind_evaluation_custody import PrivateExamCase, initialize_assessment, score_assessment
from labyrinth.blind_transition_bridge import (
    CANDIDATE_LINEAGE_DOMAIN, CANDIDATE_LINEAGE_SCHEMA, PublicAssessmentCase,
    authorize_signed_regeneration,
)
from labyrinth.hf_development_transition import DevelopmentCaseRef, development_case_commitment
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.journey import JourneyAttempt, JourneyState, TRAINING
from labyrinth.measured_weakness_feedback import TrainingExposureRef, training_roster_commitment
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.weakness_training_bridge import allocate_signed_program_training


EVALUATOR = b"e" * 32
STATE = b"s" * 32
LINEAGE = b"l" * 32
TRANSITION = b"t" * 32
BINDING = b"x" * 32


class WeaknessTrainingBridgeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        artifact = root / "candidate.safetensors"
        artifact.write_bytes(b"synthetic model artifact for source contract tests")
        model_sha = sha256(artifact.read_bytes()).hexdigest()
        self.scope = TrainingScopeSpec(
            "program-weakness", "reasoning", ("program_reasoning",), 3, 1, 3, 17,
        )
        self.contract = ObjectiveContract(
            scope=self.scope,
            intention="Practice four-step register-dependent programs.",
            targets=(SkillTarget(
                "program_reasoning", ("four_step_cross_register_dependency_path",),
                ("Computes final register A after dependencies.",),
                "program_structure_v1", 75, 10,
            ),),
            excluded_scopes=("Private final examination examples.",),
            forbidden_claims=("Development feedback proves general mastery.",),
        )
        cases = (
            PrivateExamCase("held-1", "program_reasoning", "heldout", "Register exercise A", "held-pattern", "4"),
            PrivateExamCase("held-2", "program_reasoning", "heldout", "Register exercise B", "held-pattern", "8"),
            PrivateExamCase("trans-1", "program_reasoning", "transfer", "Register story C", "trans-pattern", "12"),
            PrivateExamCase("trans-2", "program_reasoning", "transfer", "Register story D", "trans-pattern", "16"),
        )
        self.context = ResumeContext(self.scope.fingerprint, 1, "a" * 64, "b" * 64,
                                     "operator-cycle-one")
        precommit = initialize_assessment(
            root / "exam.sqlite", cases, run_id="candidate-run-0",
            objective_id=self.scope.objective_id, checkpoint_sha256=model_sha,
            evaluator_key=EVALUATOR, state_key=STATE,
        )
        answers = {case.case_id: case.gold_answer for case in cases}
        answers["trans-2"] = "0"
        scored = score_assessment(
            root / "exam.sqlite", cases, answers, artifact_root=root,
            artifact_name="candidate.safetensors", evaluator_key=EVALUATOR, state_key=STATE,
        )
        public = tuple(PublicAssessmentCase(
            case.case_id, case.skill, case.split,
            sha256(case.prompt.encode()).hexdigest(), case.structure_key,
        ) for case in cases)
        public_dev = tuple(DevelopmentCaseRef(
            case.case_id, case.skill, case.prompt_sha256, case.structure_key,
        ) for case in public)
        lineage = {
            "schema": CANDIDATE_LINEAGE_SCHEMA,
            "run_id": "candidate-run-0", "exam_id": "dev-exam-0",
            "objective_id": self.scope.objective_id,
            "scope_fingerprint": self.scope.fingerprint, "cycle_index": 0,
            "checkpoint_artifact_sha256": model_sha,
            "predecessor_report_sha256": self.context.predecessor_report_sha256,
            "predecessor_lora_value_sha256": self.context.predecessor_lora_value_sha256,
            "public_exam_sha256": precommit,
        }
        lineage_bytes = (json.dumps(lineage, sort_keys=True, separators=(",", ":")) + "\n").encode()
        lineage_sig = hmac.new(LINEAGE, CANDIDATE_LINEAGE_DOMAIN + lineage_bytes,
                               sha256).hexdigest()
        transition = authorize_signed_regeneration(
            scope=self.scope, context=self.context,
            scored_receipt_bytes=scored.receipt_bytes,
            scored_signature_hex=scored.signature_hex,
            candidate_run_id="candidate-run-0", candidate_artifact_sha256=model_sha,
            public_cases=public, evaluator_precommitment_sha256=precommit,
            development_roster_precommitment_sha256=development_case_commitment(public_dev),
            candidate_lineage_bytes=lineage_bytes,
            candidate_lineage_signature_hex=lineage_sig,
            exam_id="dev-exam-0", event_id="dev-event-1",
            failure_threshold_percent=75,
            independent_evaluator_key=EVALUATOR, independent_lineage_key=LINEAGE,
            independent_transition_key=TRANSITION,
        )
        self.verification = {
            "scope": self.scope, "context": self.context, "transition": transition,
            "scored_receipt_bytes": scored.receipt_bytes,
            "scored_signature_hex": scored.signature_hex,
            "candidate_lineage_bytes": lineage_bytes,
            "candidate_lineage_signature_hex": lineage_sig,
            "candidate_run_id": "candidate-run-0",
            "candidate_artifact_sha256": model_sha,
            "expected_evaluator_roster_sha256": precommit,
            "expected_development_roster_sha256": development_case_commitment(public_dev),
            "failure_threshold_percent": 75,
            "binding_signer_id": "independent-lineage-operator",
            "expected_nonce_sha256": sha256(b"precommitted one-shot development binding").hexdigest(),
            "independent_evaluator_key": EVALUATOR,
            "independent_lineage_key": LINEAGE,
            "independent_transition_key": TRANSITION,
            "independent_binding_key": BINDING,
        }
        self.binding = sign_exam_lineage_binding(**self.verification)
        self.training = (TrainingExposureRef("train-old-0", "c" * 64),
                         TrainingExposureRef("train-old-1", "d" * 64))
        self.roster = training_roster_commitment(self.training)
        dev_ids = tuple(case.case_id for case in public)
        dev_prompts = tuple(case.prompt_sha256 for case in public)
        self.journey = JourneyState(
            scope=self.scope, baseline_checkpoint_sha256="e" * 64,
            checkpoint_sha256=self.context.predecessor_lora_value_sha256,
            maze=generate_maze_plan(self.scope, 1, weak_skills=("program_reasoning",)),
            phase=TRAINING,
            attempts=(JourneyAttempt(
                cycle_index=0, exam_id="dev-exam-0",
                checkpoint_before_sha256="e" * 64,
                checkpoint_after_sha256=self.context.predecessor_lora_value_sha256,
                training_case_ids=tuple(x.case_id for x in self.training),
                training_prompt_sha256=tuple(x.prompt_sha256 for x in self.training),
                exam_case_ids=dev_ids, exam_prompt_sha256=dev_prompts,
                exam_status="FAIL", weak_skills=("program_reasoning",),
            ),),
            used_training_case_ids=tuple(x.case_id for x in self.training),
            used_training_prompt_sha256=tuple(x.prompt_sha256 for x in self.training),
            sealed_exam_ids=("dev-exam-0",), exam_event_statuses=("FAIL",),
            sealed_exam_case_ids=dev_ids,
            sealed_exam_prompt_sha256=dev_prompts,
        )

    def allocate(self, **changes):
        args = dict(
            contract=self.contract, journey=self.journey, binding=self.binding,
            training_exposures=self.training, expected_training_roster_sha256=self.roster,
            expected_objective_contract_sha256=self.contract.fingerprint,
            **self.verification,
        )
        args.update(changes)
        return allocate_signed_program_training(**args)

    def test_signed_program_weakness_allocates_deterministic_answerless_train_rooms(self):
        first = self.allocate()
        second = self.allocate()
        self.assertEqual(first, second)
        self.assertEqual(first.maze, self.journey.maze)
        self.assertEqual(first.weak_skills, ("program_reasoning",))
        self.assertTrue(first.views)
        self.assertTrue(all(view.challenge_id.startswith("train-") for view in first.views))
        self.assertEqual(len({view.challenge_id for view in first.views}), len(first.views))
        self.assertEqual(len({view.prompt_sha256 for view in first.views}), len(first.views))
        self.assertFalse(first.graduation_authorized)
        self.assertFalse(first.model_execution_authenticated)
        self.assertTrue(all(not hasattr(view, "answer") for view in first.views))

    def test_signed_context_and_journey_partition_cannot_be_swapped(self):
        for changes in (
            {"binding": replace(self.binding, signature_hex="0" * 64)},
            {"expected_training_roster_sha256": "0" * 64},
            {"journey": replace(self.journey, checkpoint_sha256="f" * 64)},
            {"journey": replace(self.journey, sealed_exam_case_ids=("different",))},
            {"journey": replace(self.journey, used_training_case_ids=("different",))},
            {"journey": replace(self.journey, phase="AWAITING_EXAM")},
            {"contract": replace(self.contract, intention="Practice a different target.")},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.allocate(**changes)

    def test_signed_case_prompt_pairs_not_just_separate_sets(self):
        swapped_training = (
            TrainingExposureRef(self.training[0].case_id, self.training[1].prompt_sha256),
            TrainingExposureRef(self.training[1].case_id, self.training[0].prompt_sha256),
        )
        with self.assertRaisesRegex(ValueError, "roster differs"):
            self.allocate(
                training_exposures=swapped_training,
                expected_training_roster_sha256=training_roster_commitment(swapped_training),
            )
        prior = self.journey.attempts[-1]
        swapped_exam = replace(prior,
            exam_prompt_sha256=tuple(reversed(prior.exam_prompt_sha256)))
        swapped_journey = replace(self.journey,
            attempts=(swapped_exam,),
            sealed_exam_prompt_sha256=tuple(reversed(self.journey.sealed_exam_prompt_sha256)),
        )
        with self.assertRaises(ValueError):
            self.allocate(journey=swapped_journey)

    def test_unsupported_empathy_and_unapproved_rubric_fail_closed(self):
        empathy = replace(self.scope, objective_id="empathy-example", domain="empathy",
                          skills=("cognitive_empathy",))
        target = replace(self.contract.targets[0], skill="cognitive_empathy")
        for contract in (
            replace(self.contract, scope=empathy, targets=(target,)),
            replace(self.contract, targets=(replace(
                self.contract.targets[0], rubric_id="unapproved_program_rubric"),)),
        ):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                self.allocate(contract=contract)

    def test_allocation_receipt_cannot_claim_exam_or_execution_authority(self):
        allocation = self.allocate()
        for field in ("graduation_authorized", "model_execution_authenticated",
                      "evaluator_host_independence_verified"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(allocation, **{field: True})


if __name__ == "__main__":
    unittest.main()
