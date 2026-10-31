"""Real one-shot signed development assessments feeding the provisional Minotaur."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.blind_evaluation_custody import (
    PrivateExamCase, initialize_assessment, score_assessment,
)
from labyrinth.blind_transition_bridge import PublicAssessmentCase
from labyrinth.journey import (
    AWAITING_EXAM, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    TrainingCompletionReceipt, begin_journey, complete_maze,
)
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.signed_minotaur_pair_r25 import (
    SignedMinotaurPairPrecommit, apply_signed_minotaur_pair,
)
from labyrinth.training_scope import TrainingScopeSpec
import labyrinth.signed_minotaur_pair_r25 as signed_pair_module


def digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return sha256(value).hexdigest()


class SignedMinotaurPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        scope = TrainingScopeSpec(
            objective_id="arithmetic-mastery", domain="reasoning", skills=("arithmetic",),
            specificity=2, scale=1, max_cycles=2, seed=27,
        )
        self.contract = ObjectiveContract(
            scope=scope, intention="Learn to solve unfamiliar arithmetic tasks",
            targets=(SkillTarget(
                skill="arithmetic", focus_tags=("arithmetic",),
                observable_behaviors=("Generalize to unseen arithmetic cases",),
                rubric_id="paired_binary_arithmetic_v1", minimum_score_percent=75,
                minimum_gain_percent=10,
            ),),
            excluded_scopes=(), forbidden_claims=("No graduation from a development exam",),
        )
        self.baseline = root / "baseline.safetensors"
        self.candidate = root / "candidate.safetensors"
        self.baseline.write_bytes(b"untrained-baseline-fixture")
        self.candidate.write_bytes(b"changed-trained-fixture")
        # The Journey checkpoint identity and evaluator artifact identity must
        # be exactly the same at this source-level seam.
        initial = begin_journey(scope, checkpoint_sha256=digest(self.baseline.read_bytes()))
        self.state = complete_maze(initial, TrainingCompletionReceipt(
            objective_id=scope.objective_id, cycle_index=0,
            checkpoint_before_sha256=initial.checkpoint_sha256,
            checkpoint_after_sha256=digest(self.candidate.read_bytes()),
            room_case_ids=tuple((room.room_id, f"train-r{i}", digest(f"train-r{i}"))
                                for i, room in enumerate(initial.maze.rooms)),
        ))
        self.assertEqual(self.state.phase, AWAITING_EXAM)
        self.cases = tuple(
            PrivateExamCase(f"exam-{split}-{i}", "arithmetic", split,
                            f"Independent {split} arithmetic task {i}?",
                            f"{split}-structure-{i}", "right")
            for split in ("heldout", "transfer") for i in range(4)
        )
        self.public_cases = tuple(PublicAssessmentCase(
            case.case_id, case.skill, case.split, digest(case.prompt), case.structure_key,
        ) for case in self.cases)
        self.evaluator_key = b"c" * 32
        self.state_key = b"s" * 32
        self.candidate_run_id = "candidate-run"
        self.baseline_run_id = "baseline-run"
        self.candidate_evidence = self.score(
            root / "candidate.db", self.candidate_run_id, self.candidate,
            {case.case_id: "right" for case in self.cases},
        )
        self.baseline_evidence = self.score(
            root / "baseline.db", self.baseline_run_id, self.baseline,
            {case.case_id: "wrong" for case in self.cases},
        )
        self.exam = MinotaurExamReceipt(
            exam_id="minotaur-development-0", objective_id=scope.objective_id,
            cycle_index=0, checkpoint_sha256=self.state.checkpoint_sha256,
            baseline_sha256=self.state.baseline_checkpoint_sha256,
            evaluator_id="test-only-independent-evaluator",
            cases=tuple(ExamCaseResult(
                case_id=case.case_id, skill=case.skill, split=case.split,
                score=1.0, baseline_score=0.0,
                prompt_sha256=digest(case.prompt),
            ) for case in self.cases),
        )
        self.precommit = SignedMinotaurPairPrecommit(
            objective_id=scope.objective_id,
            scope_fingerprint=scope.fingerprint,
            objective_contract_sha256=self.contract.fingerprint,
            exam_id=self.exam.exam_id,
            cycle_index=0,
            candidate_run_id=self.candidate_run_id,
            baseline_run_id=self.baseline_run_id,
            candidate_checkpoint_sha256=self.state.checkpoint_sha256,
            baseline_checkpoint_sha256=self.state.baseline_checkpoint_sha256,
            public_exam_sha256=self.candidate_evidence[1],
        )
        self.assertEqual(self.candidate_evidence[1], self.baseline_evidence[1])

    def score(self, db_path, run_id, artifact, answers):
        roster_sha = initialize_assessment(
            db_path, self.cases, run_id=run_id,
            objective_id=self.contract.scope.objective_id,
            checkpoint_sha256=digest(artifact.read_bytes()),
            evaluator_key=self.evaluator_key, state_key=self.state_key,
        )
        signed = score_assessment(
            db_path, self.cases, answers, artifact_root=artifact.parent,
            artifact_name=artifact.name,
            evaluator_key=self.evaluator_key, state_key=self.state_key,
        )
        return signed, roster_sha

    def apply(self, **changes):
        args = dict(
            contract=self.contract, journey=self.state, exam=self.exam,
            public_cases=self.public_cases,
            candidate_assessment=self.candidate_evidence[0],
            baseline_assessment=self.baseline_evidence[0],
            evaluator_key=self.evaluator_key, trusted_precommit=self.precommit,
        )
        args.update(changes)
        return apply_signed_minotaur_pair(**args)

    def test_real_scored_candidate_and_baseline_reach_provisional_dev_pass(self):
        transition = self.apply()
        self.assertEqual(transition.next_journey.phase,
                         DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
        self.assertEqual(transition.next_journey.attempts[-1].exam_status, "PASS")
        self.assertEqual(transition.qualification, "NOT_ESTABLISHED")
        self.assertFalse(transition.graduation_authorized)
        self.assertFalse(transition.model_execution_authenticated)
        self.assertFalse(transition.host_custody_verified)

    def test_missing_or_forged_signed_assessments_refused(self):
        for changed in (
            {"candidate_assessment": replace(self.candidate_evidence[0], signature_hex="0" * 64)},
            {"baseline_assessment": replace(self.baseline_evidence[0], signature_hex="0" * 64)},
            {"candidate_assessment": self.baseline_evidence[0]},
            {"baseline_assessment": self.candidate_evidence[0]},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.apply(**changed)

    def test_precommit_identity_and_structure_binding_refused(self):
        for precommit in (
            replace(self.precommit, candidate_run_id="different-run"),
            replace(self.precommit, public_exam_sha256="0" * 64),
            replace(self.precommit, objective_contract_sha256="0" * 64),
            replace(self.precommit, candidate_checkpoint_sha256="0" * 64),
            replace(self.precommit, exam_id="other-exam"),
        ):
            with self.subTest(precommit=precommit), self.assertRaises(ValueError):
                self.apply(trusted_precommit=precommit)
        with self.assertRaises(ValueError):
            replace(self.precommit, baseline_run_id=self.candidate_run_id)
        changed = replace(self.public_cases[0], structure_key="forged-structure")
        with self.assertRaises(ValueError):
            self.apply(public_cases=(changed,) + self.public_cases[1:])

    def test_fabricated_per_case_scores_cannot_rebind_signed_aggregate(self):
        forged = replace(self.exam, cases=(replace(self.exam.cases[0], score=0.0),)
                         + self.exam.cases[1:])
        with self.assertRaises(ValueError):
            self.apply(exam=forged)
        forged = replace(self.exam, cases=(replace(self.exam.cases[0], baseline_score=1.0),)
                         + self.exam.cases[1:])
        with self.assertRaises(ValueError):
            self.apply(exam=forged)
        forged = replace(self.exam, cases=(replace(self.exam.cases[0], score=0.5),)
                         + self.exam.cases[1:])
        with self.assertRaises(ValueError):
            self.apply(exam=forged)

    def test_no_training_case_reuse_or_private_final_crossing(self):
        forbidden = replace(self.exam, cases=(replace(
            self.exam.cases[0], case_id=self.state.used_training_case_ids[0]),)
            + self.exam.cases[1:])
        with self.assertRaises(ValueError):
            self.apply(exam=forbidden)
        with self.assertRaises(ValueError):
            self.apply(exam=replace(self.exam, exam_kind="private_final"))

    def test_mutated_frozen_receipt_cannot_claim_graduation(self):
        forged = replace(self.candidate_evidence[0], graduation_authorized=True)
        with self.assertRaises(ValueError):
            self.apply(candidate_assessment=forged)

    def test_signed_development_fail_regenerates_maze_and_preserves_checkpoint(self):
        from labyrinth.journey import TRAINING

        # A predeclared low policy target does not dilute the Minotaur's fixed
        # stronger 75% threshold. Its independent signed score still FAILs.
        target = replace(self.contract.targets[0], minimum_score_percent=0,
                         minimum_gain_percent=0)
        contract = replace(self.contract, targets=(target,))
        precommit = replace(self.precommit, objective_contract_sha256=contract.fingerprint)
        right_ids = {case.case_id for i, case in enumerate(self.cases) if i % 4 < 2}
        answers = {case.case_id: "right" if case.case_id in right_ids else "wrong"
                   for case in self.cases}
        signed, _roster = self.score(
            Path(self.tmp.name) / "candidate-fail.db", self.candidate_run_id,
            self.candidate, answers,
        )
        exam = replace(self.exam, cases=tuple(replace(
            case, score=1.0 if case.case_id in right_ids else 0.0,
        ) for case in self.exam.cases))
        transition = self.apply(contract=contract, trusted_precommit=precommit,
                                candidate_assessment=signed, exam=exam)
        self.assertEqual(transition.next_journey.phase, TRAINING)
        self.assertEqual(transition.next_journey.cycle_index, 1)
        self.assertEqual(transition.next_journey.attempts[-1].exam_status, "FAIL")
        self.assertEqual(transition.next_journey.checkpoint_sha256,
                         self.state.checkpoint_sha256)
        self.assertNotEqual(transition.next_journey.maze, self.state.maze)

    def test_stricter_declared_target_cannot_be_silently_lowered(self):
        target = replace(self.contract.targets[0], minimum_score_percent=100)
        contract = replace(self.contract, targets=(target,))
        precommit = replace(self.precommit, objective_contract_sha256=contract.fingerprint)
        wrong_ids = {case.case_id for i, case in enumerate(self.cases) if i % 4 == 0}
        answers = {case.case_id: "wrong" if case.case_id in wrong_ids else "right"
                   for case in self.cases}
        signed, _roster = self.score(
            Path(self.tmp.name) / "candidate-strict.db", self.candidate_run_id,
            self.candidate, answers,
        )
        exam = replace(self.exam, cases=tuple(replace(
            case, score=0.0 if case.case_id in wrong_ids else 1.0,
        ) for case in self.exam.cases))
        with self.assertRaisesRegex(ValueError, "declared observable development target"):
            self.apply(contract=contract, trusted_precommit=precommit,
                       candidate_assessment=signed, exam=exam)

    def test_precommit_mutation_between_signed_reads_cannot_swap_baseline(self):
        """A callback cannot turn a foreign signed baseline into our entrant."""
        foreign = Path(self.tmp.name) / "foreign-baseline.safetensors"
        foreign.write_bytes(b"a-different-entrant-model-fixture")
        foreign_run = "foreign-baseline-run"
        foreign_signed, roster = self.score(
            Path(self.tmp.name) / "foreign-baseline.db", foreign_run, foreign,
            {case.case_id: "wrong" for case in self.cases},
        )
        self.assertEqual(roster, self.precommit.public_exam_sha256)
        original = signed_pair_module._checked_signed_assessment
        mutated = False

        def rebind_after_candidate(*args, **kwargs):
            nonlocal mutated
            value = original(*args, **kwargs)
            if not mutated:
                mutated = True
                object.__setattr__(self.precommit, "baseline_run_id", foreign_run)
                object.__setattr__(self.precommit, "baseline_checkpoint_sha256",
                                   digest(foreign.read_bytes()))
            return value

        with patch.object(signed_pair_module, "_checked_signed_assessment",
                          side_effect=rebind_after_candidate):
            with self.assertRaisesRegex(ValueError, "mutated|signed development assessment"):
                self.apply(baseline_assessment=foreign_signed)
        self.assertTrue(mutated)
        self.assertEqual(self.state.phase, AWAITING_EXAM)

    def test_objective_policy_mutation_during_signature_verification_cannot_downgrade_target(self):
        """A callback cannot turn signed strict 100% into trainer-owned 75%."""
        strict = replace(self.contract, targets=(replace(
            self.contract.targets[0], minimum_score_percent=100,
        ),))
        strict_precommit = replace(
            self.precommit, objective_contract_sha256=strict.fingerprint,
        )
        wrong_ids = {case.case_id for i, case in enumerate(self.cases) if i % 4 == 0}
        signed, _ = self.score(
            Path(self.tmp.name) / "policy-downgrade.db", self.candidate_run_id,
            self.candidate, {case.case_id: (
                "wrong" if case.case_id in wrong_ids else "right"
            ) for case in self.cases},
        )
        exam = replace(self.exam, cases=tuple(replace(
            case, score=0.0 if case.case_id in wrong_ids else 1.0,
        ) for case in self.exam.cases))
        original = signed_pair_module.verify_assessment
        mutated = False

        def policy_downgrade(*args, **kwargs):
            nonlocal mutated
            value = original(*args, **kwargs)
            if not mutated:
                mutated = True
                object.__setattr__(strict, "targets", self.contract.targets)
            return value

        with patch.object(signed_pair_module, "verify_assessment", side_effect=policy_downgrade):
            with self.assertRaisesRegex(ValueError, "mutated|declared observable development target"):
                self.apply(contract=strict, trusted_precommit=strict_precommit,
                           candidate_assessment=signed, exam=exam)
        self.assertTrue(mutated)
        self.assertEqual(self.state.phase, AWAITING_EXAM)


if __name__ == "__main__":
    unittest.main()
