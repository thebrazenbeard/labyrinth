"""Dependency-free scoring smoke for frozen raw-vs-chat calibration.

The tiny fakes exercise the complete scoring path, not model quality.  This
does not instantiate transformers, download weights or open sealed eval cases.
"""

from contextlib import nullcontext
import unittest

from labyrinth.hf_pilot import _score_model
from labyrinth.maze import generate_maze


class TinyTensor:
    def __init__(self, values):
        self.values = [list(values)]
        self.shape = (1, len(values))

    def to(self, _device):
        return self

    def __getitem__(self, key):
        row, column = key
        return self.values[row][column]


class TinyTokenizer:
    chat_template = "fake role-delimited template"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.raw_prompts = []
        self.chat_prompts = []

    def __call__(self, raw, *, add_special_tokens, return_tensors):
        self.raw_prompts.append((raw, add_special_tokens, return_tensors))
        return {"input_ids": TinyTensor([11, 12]),
                "attention_mask": TinyTensor([1, 1])}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            return_tensors, return_dict):
        self.chat_prompts.append((messages, tokenize, add_generation_prompt,
                                  return_tensors, return_dict))
        return {"input_ids": TinyTensor([21, 22, 23]),
                "attention_mask": TinyTensor([1, 1, 1])}

    def decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(chr(value) for value in values)


class TinyFrozenModel:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.evaluations = 0

    def eval(self):
        self.evaluations += 1

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = kwargs["input_ids"].values[0] + [ord(c) for c in
                                                  f"ANSWER: {self.answer}"]
        return TinyTensor(output)


class TinyTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


class CalibrationOfflineSmokeTests(unittest.TestCase):
    def test_raw_and_chat_exercise_same_scoring_and_generation_budget(self):
        cases = generate_maze(seed=42, count=1, tiers=(1,), split="train")
        self.assertEqual({case.split for case in cases}, {"train"})
        tokenizer = TinyTokenizer()
        model = TinyFrozenModel(cases[0].answer)
        reports = {}
        for style in ("raw", "chat"):
            reports[style] = _score_model(model, tokenizer, TinyTorch, cases,
                                          device="cpu", max_tokens=128,
                                          max_new_tokens=64, prompt_style=style)
            self.assertEqual(reports[style]["correct"], 1)
            self.assertEqual(reports[style]["format_valid"], 1)
        self.assertEqual(model.evaluations, 2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([call["max_new_tokens"] for call in model.calls], [64, 64])
        self.assertEqual([call["do_sample"] for call in model.calls], [False, False])
        self.assertEqual(len(tokenizer.raw_prompts), 1)
        self.assertEqual(len(tokenizer.chat_prompts), 1)
        self.assertIn("Assistant:", tokenizer.raw_prompts[0][0])
        self.assertEqual(tokenizer.chat_prompts[0][0][0]["role"], "user")

    def test_context_bound_checked_before_model_generation(self):
        cases = generate_maze(seed=42, count=1, tiers=(1,), split="train")
        model = TinyFrozenModel(cases[0].answer)
        with self.assertRaisesRegex(ValueError, "context budget"):
            _score_model(model, TinyTokenizer(), TinyTorch, cases,
                         device="cpu", max_tokens=64, max_new_tokens=63,
                         prompt_style="chat")
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
