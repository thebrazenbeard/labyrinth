"""Optional reasoning-gym integration must never turn development into final evidence."""

import importlib
import io
import math
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class FakeDataset:
    def __init__(self):
        self.entries = [
            {"question": "How many legs has one deer?", "answer": "4", "metadata": {"secret": "not-for-prompt"}},
            {"question": "What is 8 minus 3?", "answer": "5", "metadata": {}},
        ]
        self.scored = []

    def __getitem__(self, index):
        return self.entries[index]

    def score_answer(self, answer, entry):
        self.scored.append((answer, entry))
        return 1.0 if answer.strip() == entry["answer"] else 0.0


class RecordingAdapter:
    model_id = "test-recording-adapter"

    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or ["4", "5"]

    def generate(self, prompt, *, seed=None):
        self.calls.append((prompt, seed))
        return self.answers[len(self.calls) - 1]


class ReasoningGymBridgeTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset()
        self.factory_calls = []

        def create_dataset(name, **options):
            self.factory_calls.append((name, options))
            return self.dataset

        self.fake_module = types.SimpleNamespace(create_dataset=create_dataset)

    def invoke(self, adapter=None, **kwargs):
        from labyrinth.reasoning_gym_bridge import benchmark_reasoning_gym

        with patch("labyrinth.reasoning_gym_bridge.importlib.import_module", return_value=self.fake_module):
            return benchmark_reasoning_gym(adapter or RecordingAdapter(),
                                           dataset_name="leg_counting", size=2, seed=37, **kwargs)

    def test_uses_package_scorer_with_questions_only_and_no_gold_leak(self):
        adapter = RecordingAdapter()
        result = self.invoke(adapter)
        self.assertEqual(self.factory_calls, [("leg_counting", {"size": 2, "seed": 37})])
        self.assertEqual(adapter.calls, [("How many legs has one deer?", 37),
                                         ("What is 8 minus 3?", 38)])
        self.assertEqual([a for a, _ in self.dataset.scored], ["4", "5"])
        self.assertEqual(result["correctness_mean"], 1.0)
        self.assertEqual(result["status"], "development_only")
        self.assertEqual(result["model_id"], "test-recording-adapter")
        self.assertNotIn("not-for-prompt", str(adapter.calls))
        self.assertNotIn("secret", str(result))
        self.assertNotIn("How many legs", str(result))
        self.assertNotIn('"answer"', str(result))

    def test_wrong_predictions_are_scored_without_fabricated_ground_truth(self):
        adapter = RecordingAdapter(["0", "5"])
        report = self.invoke(adapter)
        self.assertEqual(report["correctness_mean"], 0.5)
        self.assertEqual(report["per_case_scores"], [0.0, 1.0])

    def test_bad_inputs_fail_before_import(self):
        from labyrinth.reasoning_gym_bridge import benchmark_reasoning_gym

        for name, size, seed in [("", 2, 0), ("pkg.foo", 2, 0),
                                 ("leg_counting", 0, 0), ("leg_counting", 99999, 0),
                                 ("leg_counting", 2, -1), ("leg_counting", 2, True),
                                 ("leg_counting", 2, 2**31 - 1)]:
            with self.subTest(name=name, size=size, seed=seed):
                with patch("labyrinth.reasoning_gym_bridge.importlib.import_module", side_effect=AssertionError("premature import")):
                    with self.assertRaises(ValueError):
                        benchmark_reasoning_gym(RecordingAdapter(), name, size=size, seed=seed)

    def test_scorer_cannot_return_invalid_scores_or_inputs(self):
        from labyrinth.reasoning_gym_bridge import benchmark_reasoning_gym

        for value in (float("nan"), float("inf"), -0.1, 1.2, "1.0", True):
            with self.subTest(value=value):
                self.dataset.score_answer = lambda **_: value
                with patch("labyrinth.reasoning_gym_bridge.importlib.import_module", return_value=self.fake_module):
                    with self.assertRaises(ValueError):
                        benchmark_reasoning_gym(RecordingAdapter(), "leg_counting", size=2, seed=37)

    def test_missing_dependency_is_reported_not_installed(self):
        from labyrinth.reasoning_gym_bridge import benchmark_reasoning_gym

        with patch("labyrinth.reasoning_gym_bridge.importlib.import_module", side_effect=ModuleNotFoundError("reasoning_gym")):
            with self.assertRaisesRegex(RuntimeError, "reasoning-gym"):
                benchmark_reasoning_gym(RecordingAdapter(), "leg_counting", size=2, seed=37)

    def test_cli_runs_optional_benchmark_with_standard_negative_control(self):
        import json
        from labyrinth.cli import main

        output = io.StringIO()
        with patch("labyrinth.reasoning_gym_bridge.importlib.import_module", return_value=self.fake_module):
            with redirect_stdout(output):
                status = main(["benchmark-rg", "--dataset", "leg_counting", "--size", "2", "--seed", "37"])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "development_only")
        self.assertEqual(report["model_id"], "replay:constant-zero")
        self.assertEqual(report["correctness_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
