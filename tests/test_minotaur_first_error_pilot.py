"""Deterministic, dependency-free contracts for the opt-in first-error pilot."""

import unittest

from labyrinth.minotaur_first_error_pilot import (
    FirstErrorConfig, make_cases, validate_splits, training_pairs,
    prefix_examples, score_first_errors, encode_pair,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        if text == " Yes":
            return [11]
        if text == " No":
            return [12]
        return [ord(c) + 20 for c in text]


class FirstErrorTests(unittest.TestCase):
    def test_oracle_checked_traces_and_split_structures(self):
        train = make_cases(1357, "train", 16, "train")
        dev = make_cases(2468, "firsterror_dev", 8, "train")
        transfer = make_cases(3579, "firsterror_transfer", 8, "matched_length_ood")
        length = make_cases(4680, "firsterror_length", 8, "structural_holdout")
        self.assertEqual(train, make_cases(1357, "train", 16, "train"))
        validate_splits(train, dev, transfer, length)
        self.assertFalse({c.structure for c in train} & {c.structure for c in transfer})
        self.assertFalse({c.structure for c in train} & {c.structure for c in length})
        self.assertEqual({c.first_error for c in train if c.variant == "clean"}, {None})
        self.assertTrue(all(c.first_error is not None for c in train if c.variant == "error"))
        self.assertTrue(all(c.first_error == c.injected_step for c in train if c.variant == "error"))

    def test_no_answers_or_labels_and_negative_mix(self):
        cases = make_cases(12, "train", 12, "train")
        examples = prefix_examples(cases)
        self.assertTrue(any(e.is_correct for e in examples))
        self.assertTrue(any(not e.is_correct for e in examples))
        self.assertTrue(all("ANSWER:" not in e.prompt and "ERROR_STEP:" not in e.prompt
                            and "Verdict: Yes" not in e.prompt for e in examples))
        pairs = training_pairs(cases, 8, 777)
        self.assertEqual(len({p[0].case_id for p in pairs}), 8)
        self.assertTrue(all({a.is_correct, b.is_correct} == {True, False} for a, b in pairs))
        self.assertTrue(all(a.case_prompt_sha256 == b.case_prompt_sha256 for a, b in pairs))
        self.assertEqual({labels[-1] for item in encode_pair(Tokenizer(), pairs[0], 2048, False)
                          for _, labels in [item]}, {11, 12})
        self.assertEqual({labels[-1] for item in encode_pair(Tokenizer(), pairs[0], 2048, True)
                          for _, labels in [item]}, {11, 12})

    def test_first_error_metrics_count_clean_and_wrong(self):
        cases = make_cases(8, "firsterror_dev", 3, "train")
        expected = [c.first_error for c in cases]
        perfect = score_first_errors(cases, expected)
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["error_accuracy"], 1.0)
        self.assertEqual(perfect["correct_accuracy"], 1.0)
        approves = score_first_errors(cases, [None] * len(cases))
        self.assertEqual(approves["false_approval_rate"], 1.0)
        with self.assertRaises(ValueError):
            score_first_errors(cases, [0] * len(cases))

    def test_config_is_bounded(self):
        with self.assertRaises(ValueError):
            FirstErrorConfig("x", "y", steps=65).validate_fields()
        with self.assertRaises(ValueError):
            FirstErrorConfig("x", "y", train_count=2, steps=3).validate_fields()


if __name__ == "__main__":
    unittest.main()
