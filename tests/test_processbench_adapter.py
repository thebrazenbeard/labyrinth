"""Offline adapter conformance using invented fixtures, NOT official scores."""

import json
import unittest

from labyrinth.processbench_adapter import (
    evaluate_isomorphic_pairs, evaluate_processbench, load_processbench_jsonl,
)


def fixture(*, label, name, correct, steps=None):
    return {
        "id": name,
        "generator": "fixture-model",
        "problem": "Add two integers: 2 + 3. (synthetic fixture)",
        "steps": steps or ["2 + 3 = 5", "Therefore the answer is 5"],
        "final_answer_correct": correct,
        "label": label,
    }


def encode(*cases):
    return ("\n".join(json.dumps(case) for case in cases) + "\n").encode("utf-8")


class ProcessBenchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.raw = encode(
            fixture(label=-1, name="fixture-good", correct=True),
            fixture(label=0, name="fixture-first", correct=False,
                    steps=["2 + 3 = 6", "Therefore the answer is 6"]),
            fixture(label=1, name="fixture-second", correct=False,
                    steps=["2 + 3 = 5", "Therefore the answer is 4"]),
            fixture(label=-1, name="fixture-good-other", correct=True,
                    steps=["Another reasoning step"]),
        )
        self.cases = load_processbench_jsonl(self.raw, split="gsm8k")

    def test_label_translation_and_balanced_metrics(self):
        self.assertEqual([case.first_error_step for case in self.cases],
                         [None, 1, 2, None])
        results = ["ALL_CORRECT", "ERROR_STEP:1", "ERROR_STEP:2", "ALL_CORRECT"]
        index = iter(results)
        report = evaluate_processbench(self.cases, lambda _q, _steps: next(index))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["error_accuracy"], 1.0)
        self.assertEqual(report["correct_accuracy"], 1.0)
        self.assertEqual(report["harmonic_f1"], 1.0)
        self.assertEqual(report["false_approval_rate"], 0.0)
        self.assertEqual(report["evidence_status"], "UNVERIFIED_CALLER_SUPPLIED_DATA")
        self.assertEqual(report["reasoning_qualification"], "NOT_ESTABLISHED")
        self.assertNotIn("predictions", report)
        self.assertNotIn("cases", report)

    def test_one_class_benchmark_does_not_claim_zero_error_rate(self):
        good = evaluate_processbench(self.cases[:1], lambda _q, _steps: "ALL_CORRECT")
        wrong = evaluate_processbench(self.cases[1:2], lambda _q, _steps: "ALL_CORRECT")
        self.assertIsNone(good["false_approval_rate"])
        self.assertIsNone(wrong["false_rejection_rate"])
        self.assertEqual(good["error_cases"], 0)
        self.assertEqual(wrong["correct_cases"], 0)

    def test_abstain_and_accept_all_negative_controls(self):
        allow = evaluate_processbench(self.cases, lambda _q, _steps: "ALL_CORRECT")
        self.assertEqual(allow["accuracy"], 0.5)
        self.assertEqual(allow["correct_accuracy"], 1.0)
        self.assertEqual(allow["error_accuracy"], 0.0)
        self.assertEqual(allow["harmonic_f1"], 0.0)
        self.assertEqual(allow["false_approval_rate"], 1.0)
        abstain = evaluate_processbench(self.cases,
                                        lambda _q, _steps: "ERROR_STEP:1\nALL_CORRECT")
        self.assertEqual(abstain["coverage"], 0.0)
        self.assertEqual(abstain["accuracy"], 0.0)
        self.assertEqual(abstain["abstentions"], 4)

    def test_reviewer_receives_no_labels_or_answer_correctness(self):
        inputs = []

        def reviewer(question, steps):
            inputs.append((question, steps))
            return "ALL_CORRECT"

        evaluate_processbench(self.cases, reviewer)
        self.assertEqual(len(inputs), 4)
        self.assertTrue(all(type(q) is str and type(s) is tuple for q, s in inputs))
        self.assertEqual(inputs[0][1], self.cases[0].steps)

    def test_rejects_duplicate_fields_bad_labels_and_duplicate_ids(self):
        invalid = [
            fixture(label=True, name="bad", correct=True),
            fixture(label=2, name="bad", correct=True),
            fixture(label=-2, name="bad", correct=True),
            fixture(label=-1, name="bad", correct=1),
            fixture(label=-1, name="bad", correct=True, steps=[""]),
        ]
        for case in invalid:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    load_processbench_jsonl(encode(case), split="math")
        with self.assertRaises(ValueError):
            load_processbench_jsonl(self.raw, split="unknown")
        with self.assertRaises(ValueError):
            load_processbench_jsonl(encode(fixture(label=-1, name="same", correct=True),
                                           fixture(label=-1, name="same", correct=True)),
                                    split="gsm8k")
        duplicated = b'{"id":"a","id":"b","generator":"x","problem":"p","steps":["s"],"final_answer_correct":true,"label":-1}\n'
        with self.assertRaises(ValueError):
            load_processbench_jsonl(duplicated, split="gsm8k")

    def test_payload_limits_and_unsupported_shapes(self):
        with self.assertRaises((TypeError, ValueError)):
            load_processbench_jsonl("not bytes", split="gsm8k")
        with self.assertRaises(ValueError):
            load_processbench_jsonl(b"x" * (4 * 1024 * 1024 + 1), split="gsm8k")
        with self.assertRaises(ValueError):
            load_processbench_jsonl(b'{}\n', split="gsm8k")
        with self.assertRaises(ValueError):
            load_processbench_jsonl(b"", split="gsm8k")
        with self.assertRaises(ValueError):
            evaluate_processbench([self.cases[0], self.cases[0]],
                                  lambda _q, _steps: "ALL_CORRECT")

    def test_semantics_preserving_pair_metric_does_not_confuse_consistency_with_truth(self):
        # Both fixture variants have same intended arithmetic semantics; their
        # claimed equivalence is NOT independently established by the loader.
        cases = load_processbench_jsonl(encode(
            fixture(label=0, name="variant-a", correct=False,
                    steps=["2 + 3 = 6", "answer 6"]),
            fixture(label=0, name="variant-b", correct=False,
                    steps=["Three added to two gives six", "answer six"]),
        ), split="math")
        same_wrong = evaluate_isomorphic_pairs([cases], lambda _q, _steps: "ALL_CORRECT")
        self.assertEqual(same_wrong["verdict_consistency"], 1.0)
        self.assertEqual(same_wrong["both_correct_rate"], 0.0)
        self.assertEqual(same_wrong["equivalence_proof"], "NOT_VERIFIED")

        outputs = iter(["ERROR_STEP:1", "ALL_CORRECT"])
        broken = evaluate_isomorphic_pairs([cases], lambda _q, _steps: next(outputs))
        self.assertEqual(broken["verdict_consistency"], 0.0)
        self.assertEqual(broken["one_sided_failure_rate"], 1.0)
        with self.assertRaises(ValueError):
            evaluate_isomorphic_pairs([(cases[0], cases[0])],
                                      lambda _q, _steps: "ALL_CORRECT")


if __name__ == "__main__":
    unittest.main()
