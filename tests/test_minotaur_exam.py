"""Development exam tests; no test here confers formal graduation."""

import hashlib
import unittest

from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.minotaur_exam import (
    ExamCaseResult,
    MinotaurExamReceipt,
    evaluate_minotaur,
)


def make_scope():
    return TrainingScopeSpec(
        objective_id="empathy.v1", domain="empathy",
        skills=("cognitive_empathy", "compassionate_empathy"),
        specificity=3, scale=2, max_cycles=3, seed=42,
    )


def make_cases(*, weak_skill=None, weak_heldout_skill=None):
    cases = []
    for skill in make_scope().skills:
        for split in ("heldout", "transfer"):
            for index in range(4):
                text = f"{skill}:{split}:{index}"
                candidate = 0.5 if ((weak_skill == skill and split == "transfer")
                                    or (weak_heldout_skill == skill and split == "heldout")) else 1.0
                cases.append(ExamCaseResult(
                    case_id=f"{skill}-{split}-{index}", skill=skill, split=split,
                    score=candidate, baseline_score=0.25,
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ))
    return tuple(cases)


def make_receipt(*, cases=None, baseline="b" * 64, current="a" * 64, **kwargs):
    return MinotaurExamReceipt(
        exam_id=kwargs.pop("exam_id", "exam1"), objective_id=kwargs.pop("objective_id", "empathy.v1"),
        cycle_index=kwargs.pop("cycle_index", 0), checkpoint_sha256=current,
        baseline_sha256=baseline, evaluator_id=kwargs.pop("evaluator_id", "external.dev"),
        cases=make_cases() if cases is None else cases,
        **kwargs,
    )


class MinotaurExamTests(unittest.TestCase):
    def test_adequate_new_checkpoint_only_internal_pass(self):
        decision = evaluate_minotaur(make_scope(), make_receipt())
        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.weak_skills, ())
        self.assertIn("independent attestation required", decision.reason)
        self.assertEqual(dict(decision.mean_accuracy_by_skill)["cognitive_empathy"], 1)

    def test_targeted_skill_failure_for_next_maze(self):
        decision = evaluate_minotaur(
            make_scope(), make_receipt(cases=make_cases(weak_skill="compassionate_empathy"))
        )
        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.weak_skills, ("compassionate_empathy",))

    def test_weak_heldout_cannot_be_hidden_by_perfect_transfer(self):
        # Eight cases for this skill average 0.75, but heldout is only 0.50.
        # Development mastery must pass BOTH independently covered splits.
        decision = evaluate_minotaur(
            make_scope(), make_receipt(cases=make_cases(weak_heldout_skill="cognitive_empathy"))
        )
        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.weak_skills, ("cognitive_empathy",))
        self.assertEqual(dict(decision.mean_accuracy_by_skill)["cognitive_empathy"], 0.75)

    def test_incomplete_transfer_coverage_never_passes(self):
        cases = tuple(case for case in make_cases()
                      if not (case.skill == "cognitive_empathy" and case.split == "transfer"))
        decision = evaluate_minotaur(make_scope(), make_receipt(cases=cases))
        self.assertEqual(decision.status, "INCONCLUSIVE")
        self.assertEqual(decision.weak_skills, ("cognitive_empathy",))

    def test_unchanged_checkpoint_is_failure_even_perfect_scores(self):
        decision = evaluate_minotaur(make_scope(), make_receipt(current="b" * 64))
        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(set(decision.weak_skills), set(make_scope().skills))

    def test_exam_is_not_private_graduation(self):
        with self.assertRaises(ValueError):
            make_receipt(exam_kind="private_final")
        with self.assertRaises(TypeError):
            make_receipt(claimed_graduated=True)

    def test_train_or_control_splits_forbidden(self):
        for split in ("train", "control", "placebo", "final", "eval"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                ExamCaseResult("c", "cognitive_empathy", split, 1.0, 0.0, "a" * 64)

    def test_missing_or_extra_skills_forbidden(self):
        foreign = ExamCaseResult("foreign", "unrelated", "transfer", 1.0, 0.0, "c" * 64)
        with self.assertRaises(ValueError):
            evaluate_minotaur(make_scope(), make_receipt(cases=make_cases() + (foreign,)))
        with self.assertRaises(ValueError):
            evaluate_minotaur(make_scope(), make_receipt(objective_id="other.v1"))

    def test_nonfinite_or_boolean_scores_rejected(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                ExamCaseResult("c", "cognitive_empathy", "transfer", score, 0, "a" * 64)

    def test_duplicate_case_or_prompt_fails_closed(self):
        original = make_cases()
        first = original[0]
        with self.assertRaises(ValueError):
            make_receipt(cases=original + (first,))
        same_prompt = ExamCaseResult(
            "new-case-id", first.skill, first.split,
            first.score, first.baseline_score, first.prompt_sha256,
        )
        with self.assertRaises(ValueError):
            make_receipt(cases=original + (same_prompt,))

    def test_mutated_frozen_case_is_revalidated_before_decision(self):
        receipt = make_receipt()
        object.__setattr__(receipt.cases[0], "score", float("nan"))
        with self.assertRaises(ValueError):
            evaluate_minotaur(make_scope(), receipt)

    def test_cycle_must_fit_predeclared_limit(self):
        with self.assertRaises(ValueError):
            evaluate_minotaur(make_scope(), make_receipt(cycle_index=make_scope().max_cycles))

    def test_exam_size_is_bounded(self):
        with self.assertRaises(ValueError):
            make_receipt(cases=make_cases() * 130)


if __name__ == "__main__":
    unittest.main()
