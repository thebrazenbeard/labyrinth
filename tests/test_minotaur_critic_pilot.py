"""Contract tests for the opt-in pretrained Minotaur correctness-critic pilot."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from labyrinth.minotaur_critic_pilot import (
    CriticConfig, critic_examples, training_schedule, score_binary,
    encode_supervised_verdict, validate_dataset_separation, reload_critic, SCHEMA,
    verify_reload_contract, prepare_verdict_pair,
    verdict_logit_diagnostics, score_pairwise_margins,
    hybrid_margin_loss_and_gradients,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        if text == " Yes":
            return [11]
        if text == " No":
            return [12]
        return [ord(char) % 200 + 20 for char in text]


class CriticPilotTests(unittest.TestCase):
    def test_logit_diagnostics_expose_all_yes_and_class_margin_separation(self):
        examples = critic_examples(3, "critic_dev", 2, (1,))
        result = verdict_logit_diagnostics(examples, [(3, 1), (4, 1), (2, 1), (5, 1)])
        self.assertEqual([row["gold"] for row in result["examples"]],
                         ["Yes", "No", "Yes", "No"])
        self.assertEqual([row["margin_yes_minus_no"] for row in result["examples"]],
                         [2, 3, 1, 4])
        self.assertEqual(result["by_gold"]["Yes"]["mean_margin_yes_minus_no"], 1.5)
        self.assertEqual(result["by_gold"]["No"]["mean_margin_yes_minus_no"], 3.5)
        self.assertEqual(result["by_gold"]["No"]["predicted_yes"], 2)
        self.assertEqual(result["metrics"]["false_approval_rate"], 1.0)
        with self.assertRaises(ValueError):
            verdict_logit_diagnostics(examples, [(3, 1)])
        with self.assertRaises(ValueError):
            verdict_logit_diagnostics(examples, [(float("nan"), 1)] * 4)

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

    def test_pairwise_margin_rank_keeps_same_task_discrimination_separate(self):
        examples = critic_examples(3, "critic_dev", 2, (1,))
        rank = score_pairwise_margins(examples, [3.0, 2.0, 4.0, 1.0])
        self.assertEqual(rank["pairs"], 2)
        self.assertEqual(rank["pairwise_accuracy"], 1.0)
        self.assertEqual(rank["ties"], 0)
        with self.assertRaises(ValueError):
            score_pairwise_margins(examples, [0.0, 1.0, 2.0])
        CriticConfig(model_path="model", output_dir="new",
                     loss_kind="pairwise_margin_rank").validate_fields()

    def test_hybrid_margin_loss_penalizes_all_yes_all_no_and_reversed_pairs(self):
        good = hybrid_margin_loss_and_gradients(4.0, -4.0)
        all_no = hybrid_margin_loss_and_gradients(-8.0, -8.0)
        all_yes = hybrid_margin_loss_and_gradients(8.0, 8.0)
        reversed_pair = hybrid_margin_loss_and_gradients(-4.0, 4.0)
        self.assertLess(good[0], all_no[0])
        self.assertLess(good[0], all_yes[0])
        self.assertLess(all_no[0], reversed_pair[0])
        self.assertLess(all_no[1] + all_no[2], 0.0)  # shift both up helps all-No
        self.assertGreater(all_yes[1] + all_yes[2], 0.0)  # shift both down helps all-Yes
        self.assertLess(all_no[1], 0.0)
        self.assertGreater(all_yes[2], 0.0)
        CriticConfig(model_path="model", output_dir="new",
                     loss_kind="hybrid_margin_bce_rank").validate_fields()

    def test_hybrid_uses_equal_weight_mean_binary_ce_and_pair_rank_ce(self):
        # For zero margins both mean binary CE and ranking CE are ln(2).
        # A prior unnormalized sum returned 1.5*ln(2), over-weighting BCE.
        loss, correct_grad, incorrect_grad = hybrid_margin_loss_and_gradients(0.0, 0.0)
        self.assertAlmostEqual(loss, math.log(2), places=12)
        self.assertAlmostEqual(correct_grad, -0.375, places=12)
        self.assertAlmostEqual(incorrect_grad, 0.375, places=12)

    def test_hybrid_margin_analytic_grads_match_independent_central_differences(self):
        for correct, incorrect in [(-8.0, -8.0), (8.0, 8.0), (-2.0, 1.0), (2.0, -1.0)]:
            loss, grad_correct, grad_incorrect = hybrid_margin_loss_and_gradients(
                correct, incorrect)
            eps = 1e-5
            numeric_good = (hybrid_margin_loss_and_gradients(correct + eps, incorrect)[0]
                            - hybrid_margin_loss_and_gradients(correct - eps, incorrect)[0]) / (2 * eps)
            numeric_bad = (hybrid_margin_loss_and_gradients(correct, incorrect + eps)[0]
                           - hybrid_margin_loss_and_gradients(correct, incorrect - eps)[0]) / (2 * eps)
            self.assertAlmostEqual(grad_correct, numeric_good, places=6)
            self.assertAlmostEqual(grad_incorrect, numeric_bad, places=6)
            self.assertGreaterEqual(loss, 0.0)
        for values in [(float("nan"), 0.0), (0.0, float("inf"))]:
            with self.assertRaisesRegex(ValueError, "finite"):
                hybrid_margin_loss_and_gradients(*values)

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

    def test_reload_accepts_only_tiny_logit_drift_with_identical_verdicts(self):
        examples = critic_examples(3, "critic_dev", 2, (1,))
        original = [(1.5, 1.0), (0.5, 1.0), (1.4, 1.0), (0.4, 1.0)]
        def record(values):
            return {"dev": {"accuracy": 0.5, "predictions_sha256": "unchanged",
                            "logit_diagnostics": verdict_logit_diagnostics(examples, values)}}
        baseline = record(original)
        slight = record([(a + 0.00002, b) for a, b in original])
        verify_reload_contract(training_pid=12, reload_pid=13,
                               measured=slight, expected=baseline)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            verify_reload_contract(training_pid=12, reload_pid=13,
                                   measured=record([(a + 0.002, b) for a, b in original]),
                                   expected=baseline)
        changed_verdict = list(original)
        changed_verdict[1] = (2.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            verify_reload_contract(training_pid=12, reload_pid=13,
                                   measured=record(changed_verdict), expected=baseline)

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
