"""Contracts for the independent, public V8 arithmetic examiner."""

import unittest


class MinotaurV8ExaminerTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v8_examiner as examiner
        return examiner

    def _plan(self):
        return self._module().build_exam_plan(seed=19, holdout_fraction=0.10)

    def test_plan_uses_unique_unordered_groups_and_withholds_53(self):
        plan = self._plan()
        self.assertEqual(plan.modulus, 97)
        self.assertEqual(plan.unseen_modulus, 101)
        self.assertTrue(all(a <= b for a, b in plan.training_members))
        self.assertTrue(all(a != 53 and b != 53 and (a * b) % 97 != 53
                            for a, b in plan.training_members))
        all_eval = [pair for pairs in plan.evaluation_groups.values() for pair in pairs]
        self.assertEqual(len(all_eval), len(set(all_eval)))
        self.assertTrue(all(a <= b for a, b in all_eval))
        self.assertTrue(all((a * b) % 97 == 53
                            for a, b in plan.evaluation_groups["output_53"]))
        self.assertTrue(all(a == 53 or b == 53
                            for a, b in plan.evaluation_groups["input_53"]))
        self.assertTrue(all(0 <= a <= b < 101
                            for a, b in plan.evaluation_groups["modulus_101"]))

    def test_scores_one_value_per_unordered_pair_and_reports_controls(self):
        examiner = self._module()
        plan = self._plan()
        answers = {pair: (pair[0] * pair[1]) % (101 if pair in plan.evaluation_groups["modulus_101"] else 97)
                   for pairs in plan.evaluation_groups.values() for pair in pairs}
        submission = examiner.make_prediction_submission(plan, answers)
        report = examiner.score_submission(plan, plan.training_members, submission)
        self.assertEqual(report["verdict"], "SCORED")
        self.assertEqual(report["sample_group_accuracy"], 1.0)
        self.assertEqual(set(report["controls"]),
                         {"zero", "copy_x", "train_majority", "frozen_random"})
        for name, group in report["groups"].items():
            self.assertEqual(group["correct"], group["count"])
            self.assertEqual(group["accuracy"], 1.0)
            self.assertEqual(group["count"], len(plan.evaluation_groups[name]))
            for control in report["controls"].values():
                self.assertEqual(control[name]["count"], group["count"])
        self.assertEqual(report["row_policy"], "one canonical unordered pair is one sample")
        self.assertEqual(report["prediction_sha256"], submission.prediction_sha256)

    def test_rejects_reversed_training_leakage_before_scoring(self):
        examiner = self._module()
        plan = self._plan()
        pair = plan.evaluation_groups["independent_holdout"][0]
        answers = examiner.constant_predictions(plan, 0)
        submission = examiner.make_prediction_submission(plan, answers)
        with self.assertRaisesRegex(ValueError, "overlaps evaluation"):
            examiner.score_submission(plan, (*plan.training_members, (pair[1], pair[0])), submission)

    def test_rejects_malformed_labels_and_oracle_bearing_prediction_rows(self):
        examiner = self._module()
        plan = self._plan()
        answers = examiner.constant_predictions(plan, 0)
        bad_label = dict(answers)
        first = next(iter(bad_label))
        bad_label[first] = "0"
        with self.assertRaisesRegex(ValueError, "integer residue"):
            examiner.make_prediction_submission(plan, bad_label)
        cheating_rows = [{"x": 1, "y": 2, "prediction": 2, "product": 2}]
        with self.assertRaisesRegex(ValueError, "mapping"):
            examiner.make_prediction_submission(plan, cheating_rows)

    def test_rejects_wrong_custody_and_tampered_prediction_digest(self):
        examiner = self._module()
        plan = self._plan()
        submission = examiner.make_prediction_submission(plan, examiner.constant_predictions(plan, 0))
        wrong_custody = examiner.PredictionSubmission(
            predictions=submission.predictions, plan_sha256="0" * 64,
            prediction_sha256=submission.prediction_sha256,
        )
        with self.assertRaisesRegex(ValueError, "custody"):
            examiner.score_submission(plan, plan.training_members, wrong_custody)
        bad_digest = examiner.PredictionSubmission(
            predictions=submission.predictions, plan_sha256=plan.custody_sha256,
            prediction_sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "digest"):
            examiner.score_submission(plan, plan.training_members, bad_digest)


if __name__ == "__main__":
    unittest.main()
