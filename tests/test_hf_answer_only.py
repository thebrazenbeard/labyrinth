"""Answer-only semantic SFT: the ANSWER scaffold is an input, not a target."""
from contextlib import nullcontext
import unittest

from labyrinth.hf_pilot import _encode_pair, _score_model
from labyrinth.maze import Challenge


class Tokenizer:
    eos_token = "~"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        if return_tensors is None:
            return {"input_ids": ([1] if add_special_tokens else []) + [ord(x) for x in text]}
        return {"input_ids": Tensor(), "attention_mask": Tensor()}

    def decode(self, tokens, *, skip_special_tokens=True):
        return " 2"


class Tensor:
    shape = (1, 3)
    def to(self, device): return self
    def __getitem__(self, slice_): return self


class Torch:
    @staticmethod
    def no_grad(): return nullcontext()


class Model:
    def eval(self): return self
    def generate(self, **kwargs): return Tensor()


class AnswerOnlyObjectiveTests(unittest.TestCase):
    def test_answer_scaffold_is_masked_before_supervised_answer_tokens(self):
        tok = Tokenizer()
        ids, labels = _encode_pair(tok, "Compute 1+1", "2", 256, answer_only=True)
        prefix = "Task:\nCompute 1+1\nAssistant:\nANSWER:"
        n = len(prefix) + 1
        self.assertEqual(labels[:n], [-100] * n)
        self.assertEqual(labels[n:], ids[n:])
        self.assertNotIn(ord("A"), labels[n:])

    def test_evaluation_prepends_unlearned_scaffold_for_semantic_score(self):
        case = Challenge("eval:1", "arithmetic", 1, "eval", "1+1", "2")
        score = _score_model(Model(), Tokenizer(), Torch(), [case],
                             device="cpu", max_tokens=256, max_new_tokens=8,
                             answer_only=True)
        self.assertEqual(score["correct"], 1)
        self.assertEqual(score["trials"][0]["response_preview"], "ANSWER: 2")


if __name__ == "__main__":
    unittest.main()
