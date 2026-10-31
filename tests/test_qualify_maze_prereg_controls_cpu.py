"""Opt-in controls are bounded TRAIN experiments, never a final examination."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualify_maze_conditioned_minilm_cpu as probe


class PreregisteredControlsTests(unittest.TestCase):
    def test_default_three_arm_contract_unchanged_and_opt_in_has_five(self):
        self.assertEqual(probe.ARMS, ("correct", "zero_update", "inverted"))
        self.assertEqual(probe.configured_arms(False), probe.ARMS)
        self.assertEqual(probe.configured_arms(True), (
            "correct", "zero_update", "inverted", "zero_lr", "randomized_labels"))

    def test_randomized_labels_are_deterministic_balanced_not_identity(self):
        cycles = probe.build_training_cycles(
            probe.default_scope(), weak_cycle_one=("negative_polarity",))
        for cycle in cycles:
            original = {row.exposure_id: row.label for row in cycle.examples}
            first = probe.preregistered_randomized_labels(cycle, probe.default_scope().seed)
            second = probe.preregistered_randomized_labels(cycle, probe.default_scope().seed)
            self.assertEqual(first, second)
            self.assertEqual(set(first), set(original))
            self.assertEqual(Counter(first.values()), Counter(original.values()))
            self.assertNotEqual(first, original)
            self.assertNotEqual(first, {key: 1-label for key, label in original.items()})
            self.assertTrue(any(first[row.exposure_id] != row.label
                                for row in cycle.examples[:4]))
            self.assertTrue(all(label in (0, 1) for label in first.values()))

    def test_zero_learning_rate_performs_real_optimizer_step_with_zero_lr(self):
        class CounterOptimizer:
            def __init__(self):
                self.calls = 0

            def step(self):
                self.calls += 1

        self.assertEqual(probe.optimizer_learning_rate("zero_lr", 0.002), 0.0)
        for arm in probe.configured_arms(True):
            optimizer = CounterOptimizer()
            for _ in range(4):
                probe.perform_optimizer_step(arm, optimizer)
            self.assertEqual(optimizer.calls, 0 if arm == "zero_update" else 4)
            if arm != "zero_lr":
                self.assertEqual(probe.optimizer_learning_rate(arm, 0.002), 0.002)
        with self.assertRaises(ValueError):
            probe.optimizer_learning_rate("unsupported", 0.002)
        with self.assertRaises(ValueError):
            probe.optimizer_learning_rate("correct", "0.002")

    def test_physical_loss_label_selector_has_five_distinct_contracts(self):
        cycle = probe.build_training_cycles(probe.default_scope())[0]
        batch = probe.select_training_batch(cycle, 0)
        good = tuple(row.label for row in batch)
        randomized = probe.preregistered_randomized_labels(cycle, probe.default_scope().seed)
        self.assertEqual(probe.training_batch_labels(batch, "correct"), good)
        self.assertEqual(probe.training_batch_labels(batch, "zero_update"), good)
        self.assertEqual(probe.training_batch_labels(batch, "zero_lr"), good)
        self.assertEqual(probe.training_batch_labels(batch, "inverted"),
                         tuple(1-label for label in good))
        self.assertEqual(probe.training_batch_labels(batch, "randomized_labels", randomized),
                         tuple(randomized[row.exposure_id] for row in batch))
        self.assertNotEqual(probe.training_batch_labels(batch, "randomized_labels", randomized),
                            good)
        with self.assertRaises(ValueError):
            probe.training_batch_labels(batch, "randomized_labels")
        with self.assertRaises(ValueError):
            probe.training_batch_labels(batch, "correct", randomized)

    def test_control_summary_requires_all_five_and_treats_public_as_public(self):
        scores = {"correct": .8, "zero_update": .5, "inverted": .2,
                  "zero_lr": .5, "randomized_labels": .4}
        result = probe.preregistered_control_comparison(scores)
        self.assertTrue(result["correct_exceeds_every_control"])
        self.assertEqual(result["qualification"], "PUBLIC_DEVELOPMENT_ONLY")
        self.assertFalse(result["independent_generalization_qualified"])
        with self.assertRaises(ValueError):
            probe.preregistered_control_comparison({**scores, "extra": 1.0})

    def test_opt_in_cli_dry_run_is_read_only_and_declares_no_graduation(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("builtins.print") as printed:
                self.assertEqual(probe.main([
                    "--output-parent", temp, "--run-name", "prereg-dry",
                    "--preregistered-controls",
                ]), 0)
            observed = json.loads(printed.call_args.args[0])
            self.assertFalse(observed["execute"])
            self.assertFalse(observed["graduation_authorized"])
            self.assertEqual(tuple(observed["arms"]), probe.configured_arms(True))
            self.assertEqual(observed["schema"], probe.PREREG_SCHEMA)
            self.assertEqual(observed["control_preregistration"],
                             "OPERATOR_REQUESTED_UNSEALED_SOURCE_ONLY")
            self.assertFalse((Path(temp) / "prereg-dry").exists())


if __name__ == "__main__":
    unittest.main()
