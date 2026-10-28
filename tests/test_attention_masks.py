import unittest
from contextlib import nullcontext
from labyrinth.hf_pilot import _score_model
from labyrinth.maze import Challenge

class Tensor:
    shape = (1, 3)
    def to(self, device): return self
    def __getitem__(self, key): return self

class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def __call__(self, text, **kwargs):
        return {"input_ids": Tensor(), "attention_mask": Tensor()}
    def decode(self, x, **kwargs): return "ANSWER: 2"

class Torch:
    @staticmethod
    def no_grad(): return nullcontext()

class Model:
    def eval(self): pass
    def generate(self, **kwargs):
        assert "attention_mask" in kwargs
        assert kwargs["attention_mask"].shape == (1, 3)
        return Tensor()

class NeuralMaskTests(unittest.TestCase):
    def test_explicit_attention_mask(self):
        cases = [Challenge("test-id", "arithmetic", 1, "eval", "1+1?", "2")]
        result = _score_model(Model(), Tokenizer(), Torch(), cases,
                              device="cpu", max_tokens=32, max_new_tokens=4)
        self.assertEqual(result["correct"], 1)

class MissingMask(Tokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": Tensor()}

class WrongSizeMask(Tokenizer):
    def __call__(self, text, **kwargs):
        class ShortTensor(Tensor):
            shape = (1, 2)
        return {"input_ids": Tensor(), "attention_mask": ShortTensor()}

class NeuralMaskRejections(unittest.TestCase):
    def test_missing_mask_fails_before_model_generation(self):
        with self.assertRaisesRegex(ValueError, "aligned attention mask"):
            _score_model(Model(), MissingMask(), Torch(),
                         [Challenge("x", "arithmetic", 1, "eval", "1+1?", "2")],
                         device="cpu", max_tokens=32, max_new_tokens=4)

    def test_mismatched_mask_fails_before_model_generation(self):
        with self.assertRaisesRegex(ValueError, "aligned attention mask"):
            _score_model(Model(), WrongSizeMask(), Torch(),
                         [Challenge("x", "arithmetic", 1, "eval", "1+1?", "2")],
                         device="cpu", max_tokens=32, max_new_tokens=4)
