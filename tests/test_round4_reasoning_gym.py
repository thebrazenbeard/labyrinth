"""A development dataset must not switch its question after validation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.reasoning_gym_bridge import benchmark_reasoning_gym


class ReasoningGymQuestionBoundaryTests(unittest.TestCase):
    def test_question_validation_cannot_be_bypassed_by_dict_subclass(self):
        class InconsistentCase(dict):
            def get(self, key, default=None):
                if key == "question":
                    return "Public question: what is 6 times 7?"
                return super().get(key, default)

            def __getitem__(self, key):
                if key == "question":
                    return "Public question ANSWER: 42 GOLD-PRIVATE-ANSWER"
                return super().__getitem__(key)

        class FakeDataset:
            def __getitem__(self, _index):
                return InconsistentCase(answer="42", metadata={"private": True})

            def score_answer(self, answer, entry):
                return 1.0 if answer == entry["answer"] else 0.0

        class RecordingModel:
            model_id = "round4-test-model"

            def __init__(self):
                self.prompts = []

            def generate(self, prompt, *, seed=None):
                self.prompts.append(prompt)
                return "42"

        model = RecordingModel()
        module = SimpleNamespace(create_dataset=lambda *args, **kwargs: FakeDataset())
        with patch("labyrinth.reasoning_gym_bridge.importlib.import_module",
                   return_value=module):
            benchmark_reasoning_gym(model, "leg_counting", size=1, seed=3)
        self.assertTrue(model.prompts)
        self.assertNotIn("GOLD-PRIVATE-ANSWER", str(model.prompts), model.prompts)


if __name__ == "__main__":
    unittest.main()
