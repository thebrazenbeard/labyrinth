"""Hostile source tests for evaluator-custodied experimental qualification.

All model responses are synthetic. No test here attests neural execution.
"""

from dataclasses import replace
import hashlib
import hmac
import json
import unittest

from labyrinth.independent_holdout import (
    ArmRun, ExamCase, InferenceClaim, TrainingExposure, case_commitment, private_gold_seal,
    evaluate_holdout,
)

EXAMINER_KEY = b"synthetic-test-key-not-an-actual-secret-2026"


def _fixture():
    cases = tuple(
        ExamCase(
            case_id=f"exam-{i:03d}",
            split="heldout" if i < 16 else "transfer" if i < 32 else "retention",
            skill="arithmetic", prompt=f"Private independently authored question {i}",
            answer=str(i + 5),
            structure_key=f"new-structure-{i}" if 16 <= i < 32 else "old-skill",
        )
        for i in range(40)
    )
    visits = ("train-a", "train-b") * 16
    exposure = TrainingExposure(
        case_ids=("train-a", "train-b"),
        prompt_sha256=(hashlib.sha256(b"train prompt a").hexdigest(),
                       hashlib.sha256(b"train prompt b").hexdigest()),
        structure_keys=("training-structure", "training-structure"),
    )
    def responses(correct):
        return tuple((case.case_id, f"ANSWER: {case.answer if correct else -1}")
                     for case in cases)
    common = dict(base_sha256="a" * 64, initial_adapter_sha256="b" * 64)
    arms = (
        ArmRun("baseline", final_adapter_sha256="b" * 64, optimizer_updates=0,
               compute_steps=0, visited_case_ids=(), responses=responses(False), **common),
        ArmRun("zero_update", final_adapter_sha256="b" * 64, optimizer_updates=0,
               compute_steps=32, visited_case_ids=visits, responses=responses(False), **common),
        ArmRun("correct_labels", final_adapter_sha256="c" * 64, optimizer_updates=32,
               compute_steps=32, visited_case_ids=visits, responses=responses(True), **common),
        ArmRun("corrupted_labels", final_adapter_sha256="d" * 64, optimizer_updates=32,
               compute_steps=32, visited_case_ids=visits, responses=responses(False), **common),
    )
    before = {case.case_id: f"ANSWER: {case.answer}" for case in cases
              if case.split == "retention"}
    return cases, exposure, arms, before


def _evaluate(cases=None, exposure=None, arms=None, before=None, commitment=None, gold_seal=None):
    base_cases, base_exposure, base_arms, base_before = _fixture()
    cases = base_cases if cases is None else cases
    return evaluate_holdout(
        cases, case_commitment(base_cases) if commitment is None else commitment,
        base_exposure if exposure is None else exposure,
        base_arms if arms is None else arms,
        base_before if before is None else before,
        examiner_gold_seal=(private_gold_seal(cases, EXAMINER_KEY)
                            if gold_seal is None else gold_seal),
        examiner_key=EXAMINER_KEY,
    )


class IndependentHoldoutTests(unittest.TestCase):
    def test_synthetic_positive_is_only_provisional_not_graduation(self):
        result = _evaluate()
        self.assertEqual(result.status, "PROVISIONAL")
        self.assertFalse(result.graduation_authorized)
        self.assertEqual(result.candidate_accuracy, 1.0)
        self.assertEqual(result.retention_drop, 0.0)
        self.assertTrue(all(p <= .05 for p in result.paired_p_values.values()))

    def test_private_final_split_cannot_impersonate_heldout_in_direct_evaluator(self):
        class ForgedHeldout(str):
            def __eq__(self, other):
                return other == "heldout"

            def __hash__(self):
                return hash("heldout")

        cases, exposure, arms, before = _fixture()
        for case in cases:
            if case.split == "heldout":
                object.__setattr__(case, "split", ForgedHeldout("private_final"))
        # Build the two commitments from raw bytes, without invoking the
        # prevalidating helpers; the target must be evaluate_holdout itself.
        public = [(case.case_id, str(case.split), case.skill,
                   hashlib.sha256(case.prompt.encode()).hexdigest(), case.structure_key)
                  for case in sorted(cases, key=lambda case: case.case_id)]
        commitment = hashlib.sha256(
            b"labyrinth.evaluator-public-manifest.v1\0" +
            json.dumps(public, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        private = [(case.case_id, case.answer)
                   for case in sorted(cases, key=lambda case: case.case_id)]
        gold_seal = hmac.new(
            EXAMINER_KEY,
            b"labyrinth.evaluator-gold.v1\0" +
            json.dumps(private, separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "split"):
            evaluate_holdout(cases, commitment, exposure, arms, before,
                             examiner_gold_seal=gold_seal,
                             examiner_key=EXAMINER_KEY)

    def test_private_final_arm_cannot_impersonate_baseline(self):
        class ForgedBaseline(str):
            def __eq__(self, other):
                return other == "baseline"

            def __hash__(self):
                return hash("baseline")

        cases, exposure, arms, before = _fixture()
        object.__setattr__(arms[0], "name", ForgedBaseline("private_final"))
        with self.assertRaisesRegex(ValueError, "arm"):
            _evaluate(cases=cases, exposure=exposure, arms=arms, before=before)

    def test_private_final_inference_arm_cannot_impersonate_baseline(self):
        class ForgedBaseline(str):
            def __eq__(self, other):
                return other == "baseline"

            def __hash__(self):
                return hash("baseline")

        claim = InferenceClaim(ForgedBaseline("private_final"), "pid-1",
                               "a" * 64, "b" * 64, "c" * 64)
        with self.assertRaisesRegex(ValueError, "arm"):
            claim.validate()

    def test_public_prompt_manifest_mutation_is_rejected(self):
        cases, exposure, arms, before = _fixture()
        edited = (replace(cases[0], prompt="modified after precommit"),) + cases[1:]
        with self.assertRaisesRegex(ValueError, "precommit"):
            _evaluate(cases=edited)

    def test_gold_swap_after_private_seal_rejected_without_exposing_gold_digest(self):
        cases, _, _, _ = _fixture()
        swapped = (replace(cases[0], answer="9999"),) + cases[1:]
        self.assertEqual(case_commitment(cases), case_commitment(swapped))
        with self.assertRaisesRegex(ValueError, "oracle gold"):
            _evaluate(cases=swapped, gold_seal=private_gold_seal(cases, EXAMINER_KEY))

    def test_training_case_and_prompt_overlap_are_rejected(self):
        cases, exposure, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "leakage"):
            _evaluate(exposure=replace(exposure, case_ids=(cases[0].case_id, "train-b")))
        overlapping_prompt = hashlib.sha256(cases[0].prompt.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "leakage"):
            _evaluate(exposure=replace(exposure, prompt_sha256=(overlapping_prompt,
                                                                 exposure.prompt_sha256[1])))

    def test_transfer_structure_cannot_have_appeared_in_train(self):
        cases, exposure, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "leakage"):
            _evaluate(exposure=replace(exposure, structure_keys=("training-structure",
                                                                cases[20].structure_key)))

    def test_missing_structural_exposure_cannot_make_transfer_provisional(self):
        cases, exposure, _, _ = _fixture()
        # This was a bypass: deleting the entire training structure manifest
        # yielded PROVISIONAL even though a declared transfer structure might
        # have occurred in TRAIN. Incomplete metadata must fail closed.
        self.assertEqual(_evaluate().status, "PROVISIONAL")
        with self.assertRaisesRegex(ValueError, "incomplete.*training exposure"):
            _evaluate(exposure=replace(exposure, structure_keys=()))
        with self.assertRaisesRegex(ValueError, "incomplete.*training exposure"):
            _evaluate(exposure=replace(exposure, structure_keys=("training-structure",)))
        with self.assertRaisesRegex(ValueError, "incomplete.*training exposure"):
            _evaluate(exposure=replace(exposure, prompt_sha256=()))

    def test_missing_or_duplicate_prediction_is_rejected(self):
        cases, _, arms, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "predictions"):
            _evaluate(arms=tuple(replace(arm, responses=arm.responses[:-1])
                                 if arm.name == "correct_labels" else arm for arm in arms))
        with self.assertRaisesRegex(ValueError, "predictions"):
            _evaluate(arms=tuple(replace(arm, responses=arm.responses[:-1] +
                                      (arm.responses[0],))
                                 if arm.name == "correct_labels" else arm for arm in arms))

    def test_unmatched_training_visits_or_stale_weights_rejected(self):
        _, _, arms, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "matched"):
            _evaluate(arms=tuple(replace(arm, visited_case_ids=arm.visited_case_ids[::-1])
                                 if arm.name == "corrupted_labels" else arm for arm in arms))
        with self.assertRaisesRegex(ValueError, "updated"):
            _evaluate(arms=tuple(replace(arm, final_adapter_sha256="b" * 64)
                                 if arm.name == "correct_labels" else arm for arm in arms))

    def test_transfer_failure_not_hidden_by_heldout_success(self):
        cases, _, arms, _ = _fixture()
        transfer_ids = {case.case_id for case in cases if case.split == "transfer"}
        impaired = tuple(replace(arm, responses=tuple(
            (id_, "ANSWER: -1" if id_ in transfer_ids else response)
            for id_, response in arm.responses)) if arm.name == "correct_labels" else arm
            for arm in arms)
        result = _evaluate(arms=impaired)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("transfer", " ".join(result.reasons))

    def test_weak_declared_skill_not_hidden_by_other_skills(self):
        cases, _, arms, _ = _fixture()
        second_ids = {cases[i].case_id for i in (12, 13, 14, 15, 28, 29, 30, 31)}
        cases = tuple(replace(case, skill="secondary") if case.case_id in second_ids
                      else case for case in cases)
        impaired = tuple(replace(arm, responses=tuple(
            (id_, "ANSWER: -1" if id_ in {cases[i].case_id for i in (28, 29, 30, 31)}
             else response) for id_, response in arm.responses))
            if arm.name == "correct_labels" else arm for arm in arms)
        result = _evaluate(cases=cases, commitment=case_commitment(cases), arms=impaired)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("secondary", " ".join(result.reasons))

    def test_forgetting_is_observed_against_previous_cycle(self):
        cases, _, arms, _ = _fixture()
        retention_ids = {case.case_id for case in cases if case.split == "retention"}
        impaired = tuple(replace(arm, responses=tuple(
            (id_, "ANSWER: -1" if id_ in retention_ids else response)
            for id_, response in arm.responses)) if arm.name == "correct_labels" else arm
            for arm in arms)
        result = _evaluate(arms=impaired)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.retention_drop, 1.0)

    def test_one_skill_retention_loss_is_not_hidden_by_good_average(self):
        cases, _, arms, before = _fixture()
        extra = tuple(replace(cases[32 + i], case_id=f"exam-extra-{i}",
                              prompt=f"Independent prior-skill retention {i}",
                              answer=str(100 + i), structure_key="other-old-skill")
                      for i in range(8))
        cases = cases + extra
        cases = tuple(replace(case, skill="previous-secondary")
                      if case.case_id in {cases[i].case_id for i in (36, 37, 38, 39)}
                      else case for case in cases)
        old_answers = {case.case_id: f"ANSWER: {case.answer}" for case in extra}
        before = {**before, **old_answers}
        extra_responses = tuple((case.case_id, f"ANSWER: {case.answer}") for case in extra)
        impaired = tuple(replace(arm, responses=arm.responses + extra_responses)
                         for arm in arms)
        # Only one of four previous-secondary cases is lost: 1/16 overall
        # forgetting is under 10%, but one skill fell by 25%.
        lost_id = cases[36].case_id
        impaired = tuple(replace(arm, responses=tuple(
            (id_, "ANSWER: -1" if id_ == lost_id else response)
            for id_, response in arm.responses)) if arm.name == "correct_labels" else arm
            for arm in impaired)
        result = _evaluate(cases=cases, commitment=case_commitment(cases),
                           arms=impaired, before=before)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("previous-secondary", " ".join(result.reasons))

    def test_missing_retention_before_does_not_pass(self):
        result = _evaluate(before={})
        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertFalse(result.graduation_authorized)

    def test_duplicate_exam_cases_and_unmatched_arm_base_are_rejected(self):
        cases, _, arms, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _evaluate(cases=cases[:1] + cases[:1] + cases[2:],
                      commitment=case_commitment(cases[:1] + cases[:1] + cases[2:]))
        with self.assertRaisesRegex(ValueError, "base"):
            _evaluate(arms=tuple(replace(arm, base_sha256="f" * 64)
                                 if arm.name == "corrupted_labels" else arm for arm in arms))
        with self.assertRaisesRegex(ValueError, "distinct"):
            _evaluate(arms=tuple(replace(arm, final_adapter_sha256="c" * 64)
                                 if arm.name == "corrupted_labels" else arm for arm in arms))

    def test_malformed_unhashable_exposure_and_noncanonical_oracle_fail_closed(self):
        cases, exposure, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "training case"):
            _evaluate(exposure=replace(exposure, case_ids=(["unhashable"],)))
        with self.assertRaisesRegex(ValueError, "oracle"):
            case_commitment((replace(cases[0], answer="-0"),))


if __name__ == "__main__":
    unittest.main()
