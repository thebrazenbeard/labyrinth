"""Contract tests for the opt-in pretrained Minotaur correctness-critic pilot."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.minotaur_critic_pilot import (
    CriticConfig, critic_examples, training_schedule, score_binary,
    encode_supervised_verdict, validate_dataset_separation, reload_critic, SCHEMA,
    verify_reload_contract, prepare_verdict_pair, score_selective_margins,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        if text == " Yes":
            return [11]
        if text == " No":
            return [12]
        return [ord(char) % 200 + 20 for char in text]


class CriticPilotTests(unittest.TestCase):
    def test_train_dev_and_transfer_are_deterministic_and_disjoint(self):
        train = critic_examples(1357, "train", 24, (1, 2))
        dev = critic_examples(2468, "critic_dev", 8, (1, 2))
        transfer = critic_examples(3579, "critic_transfer", 8, (4,))
        self.assertEqual(train, critic_examples(1357, "train", 24, (1, 2)))
        self.assertEqual([len(x) for x in (train, dev, transfer)], [96, 32, 16])
        validate_dataset_separation(train, dev, transfer)
        self.assertEqual({example.kind for example in transfer}, {"register_program"})
        for corpus in (train, dev, transfer):
            self.assertEqual(sum(x.is_correct for x in corpus), len(corpus) // 2)
            self.assertEqual(len({x.example_id for x in corpus}), len(corpus))
            for first, second in zip(corpus[::2], corpus[1::2]):
                self.assertEqual(first.case_id, second.case_id)
                self.assertNotEqual(first.trainee_answer, second.trainee_answer)
                self.assertEqual({first.is_correct, second.is_correct}, {False, True})
                self.assertEqual(first.case_prompt_sha256, second.case_prompt_sha256)
                self.assertNotIn("VERDICT: Yes", first.prompt)
                self.assertNotIn("VERDICT: No", first.prompt)

    def test_train_schedule_balanced_and_no_dev_reuse(self):
        train = critic_examples(1357, "train", 24, (1, 2))
        selected = training_schedule(train, steps=24, seed=777)
        self.assertEqual(selected, training_schedule(train, steps=24, seed=777))
        self.assertEqual(len(selected), 24)
        self.assertTrue(all(len(pair) == 2 for pair in selected))
        self.assertTrue(all({x.is_correct for x in pair} == {True, False} for pair in selected))
        self.assertTrue(all(len({x.case_id for x in pair}) == 1 for pair in selected))
        self.assertEqual(sum(x.is_correct for pair in selected for x in pair), 24)
        self.assertEqual(len({x.example_id for pair in selected for x in pair}), 48)
        self.assertEqual(len({x.case_id for pair in selected for x in pair}), 24)
        with self.assertRaises(ValueError):
            training_schedule(train[:2] + train[:2], steps=2, seed=777)

    def test_target_mask_supervises_only_one_verdict_token(self):
        correct = critic_examples(4, "train", 1, (1,))[0]
        incorrect = critic_examples(4, "train", 1, (1,))[1]
        tokenizer = FakeTokenizer()
        for example, expected_token in ((correct, 11), (incorrect, 12)):
            ids, labels = encode_supervised_verdict(tokenizer, example, max_tokens=512)
            self.assertEqual(len(ids), len(labels))
            self.assertEqual(labels[-1], expected_token)
            self.assertEqual(labels[:-1], [-100] * (len(labels) - 1))
        with self.assertRaises(ValueError):
            encode_supervised_verdict(tokenizer, correct, max_tokens=10)

    def test_paired_supervision_uses_both_labels_on_same_problem_every_update(self):
        correct, incorrect = critic_examples(4, "train", 1, (1,))
        tokenizer = FakeTokenizer()
        prepared = prepare_verdict_pair(tokenizer, (correct, incorrect),
                                        max_tokens=512, corrupt=False)
        self.assertEqual(len(prepared), 2)
        self.assertEqual({labels[-1] for _, labels in prepared}, {11, 12})
        corrupted = prepare_verdict_pair(tokenizer, (correct, incorrect),
                                         max_tokens=512, corrupt=True)
        self.assertEqual([labels[-1] for _, labels in corrupted], [12, 11])
        with self.assertRaises(ValueError):
            prepare_verdict_pair(tokenizer, (correct, correct),
                                 max_tokens=512, corrupt=False)
        other = critic_examples(5, "train", 1, (1,))[1]
        with self.assertRaises(ValueError):
            prepare_verdict_pair(tokenizer, (correct, other),
                                 max_tokens=512, corrupt=False)

    def test_false_approval_and_rejection_metrics_are_separate(self):
        examples = critic_examples(3, "critic_dev", 2, (1,))
        predictions = [True, True, False, False]
        result = score_binary(examples, predictions)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["false_approval_rate"], 0.5)
        self.assertEqual(result["false_rejection_rate"], 0.5)
        with self.assertRaises(ValueError):
            score_binary(examples, predictions[:-1])

    def test_selective_verdict_threshold_reports_abstention_and_false_approvals(self):
        examples = critic_examples(3, "critic_dev", 2, (1,))
        result = score_selective_margins(examples, [3.0, 0.1, -0.1, -3.0], threshold=0.5)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["abstentions"], 2)
        self.assertEqual(result["selective_accuracy"], 1.0)
        self.assertEqual(result["false_approval_rate"], 0.0)
        self.assertEqual(result["approval_rate"], 0.25)
        empty = score_selective_margins(examples, [0.1] * 4, threshold=0.5)
        self.assertEqual(empty["coverage"], 0.0)
        self.assertIsNone(empty["selective_accuracy"])
        self.assertEqual(empty["abstentions"], 4)
        with self.assertRaises(ValueError):
            score_selective_margins(examples, [float("nan")] * 4, threshold=0.0)
        with self.assertRaises(ValueError):
            score_selective_margins(examples, [0.1] * 4, threshold=-1.0)

    def test_binary_only_verdict_objective_must_be_explicitly_selected(self):
        config = CriticConfig(model_path="model", output_dir="new",
                              loss_kind="binary_logits_ce")
        config.validate_fields()
        self.assertEqual(config.loss_kind, "binary_logits_ce")
        with self.assertRaises(ValueError):
            CriticConfig(model_path="model", output_dir="new",
                         loss_kind="unknown").validate_fields()

    def test_config_bounds_no_auto_download(self):
        config = CriticConfig(model_path="/not-a-real-model", output_dir="/tmp/not-a-run")
        with self.assertRaises((FileNotFoundError, ValueError)):
            config.validate()
        with self.assertRaises(ValueError):
            CriticConfig(model_path="x", output_dir="y", steps=0).validate_fields()
        with self.assertRaises(ValueError):
            CriticConfig(model_path="x", output_dir="y", device="remote").validate_fields()

    def test_reload_rejects_source_mismatch_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "critic_report.json").write_text(json.dumps({
                "schema": SCHEMA, "source": {"source_tree_sha256": "invalid"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source tree changed"):
                reload_critic(tmp, str(Path(tmp) / "reload.json"))

    def test_reload_requires_exact_decisions_but_tolerates_bounded_margin_roundoff(self):
        expected = {"dev": {"accuracy": 0.5, "predictions_sha256": "a" * 64,
                            "verdict_margin_diagnostic": {
                                "qualification": "NOT_ESTABLISHED",
                                "margin_mean": 3.0, "margin_min": 2.0, "margin_max": 4.0,
                                "mean_margin_on_correct": 3.2,
                                "mean_margin_on_incorrect": 2.8,
                                "predeclared_thresholds": {
                                    "0.0": {"false_approval_rate": 1.0, "coverage": 1.0},
                                }}}}
        import copy
        tiny = copy.deepcopy(expected)
        tiny["dev"]["verdict_margin_diagnostic"]["margin_mean"] += 0.000006
        verify_reload_contract(training_pid=42, reload_pid=43, measured=tiny,
                               expected=expected)
        for change in ("accuracy", "prediction_hash", "risk", "margin_big"):
            altered = copy.deepcopy(tiny)
            if change == "accuracy":
                altered["dev"]["accuracy"] = 1.0
            elif change == "prediction_hash":
                altered["dev"]["predictions_sha256"] = "b" * 64
            elif change == "risk":
                altered["dev"]["verdict_margin_diagnostic"]["predeclared_thresholds"]["0.0"]["false_approval_rate"] = 0
            else:
                altered["dev"]["verdict_margin_diagnostic"]["margin_mean"] += 0.01
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                verify_reload_contract(training_pid=42, reload_pid=43, measured=altered,
                                       expected=expected)

    def test_reload_contract_fails_closed_on_same_process_or_changed_predictions(self):
        with self.assertRaisesRegex(RuntimeError, "fresh process"):
            verify_reload_contract(training_pid=12, reload_pid=12,
                                   measured={"accuracy": 0.5}, expected={"accuracy": 0.5})
        with self.assertRaisesRegex(RuntimeError, "differ"):
            verify_reload_contract(training_pid=12, reload_pid=13,
                                   measured={"accuracy": 0.0}, expected={"accuracy": 0.5})
        verify_reload_contract(training_pid=12, reload_pid=13,
                               measured={"accuracy": 0.5}, expected={"accuracy": 0.5})


if __name__ == "__main__":
    unittest.main()
