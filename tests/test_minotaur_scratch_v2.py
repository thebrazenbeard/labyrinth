"""TDD coverage for the isolated random-only Minotaur V2 experiment."""
import unittest

from labyrinth.guardian_benchmark import generate_process_cases
from labyrinth.minotaur_scratch_v2 import (
    CLS, OBSERVED_STATE, STEP_BASE, V2Config, tokenize_observed_trace_v2,
)


class V2RepresentationTests(unittest.TestCase):
    def test_every_trace_step_has_explicit_ordered_observed_state_tokens(self):
        case = generate_process_cases(101, 1)[0]
        tokens = tokenize_observed_trace_v2(case.question, case.steps)
        self.assertEqual(tokens[0], CLS)
        markers = [i for i, token in enumerate(tokens) if token == OBSERVED_STATE]
        self.assertEqual(len(markers), len(case.steps))
        for step, marker in enumerate(markers):
            self.assertEqual(tokens[marker - 5], STEP_BASE + step)
            self.assertTrue(all(token >= 32 for token in tokens[marker + 1:marker + 3]))

    def test_trace_order_is_not_collapsed_by_representation(self):
        case = generate_process_cases(102, 1)[0]
        original = tokenize_observed_trace_v2(case.question, case.steps)
        reordered_steps = (case.steps[1], case.steps[0]) + case.steps[2:]
        with self.assertRaisesRegex(ValueError, "order"):
            tokenize_observed_trace_v2(case.question, reordered_steps)
        # A valid later-state mutation remains at its own step position, not bagged.
        changed = list(case.steps)
        prefix, rest = changed[1].split("A=", 1)
        observed_a, observed_b = rest.split(", B=", 1)
        changed[1] = f"{prefix}A={(int(observed_a) + 1) % 97}, B={observed_b}"
        changed_tokens = tokenize_observed_trace_v2(case.question, tuple(changed))
        self.assertNotEqual(original, changed_tokens)
        first_marker = original.index(OBSERVED_STATE)
        second_marker = original.index(OBSERVED_STATE, first_marker + 1)
        self.assertEqual(original[first_marker + 1:first_marker + 3],
                         changed_tokens[first_marker + 1:first_marker + 3])
        self.assertNotEqual(original[second_marker + 1:second_marker + 3],
                            changed_tokens[second_marker + 1:second_marker + 3])


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class V2GradientSmokeTests(unittest.TestCase):
    def test_real_gradient_smoke_updates_random_weights_within_30_seconds(self):
        from labyrinth.minotaur_scratch_v2 import gradient_smoke_v2
        result = gradient_smoke_v2(V2Config(width=16, heads=2, layers=1), max_seconds=30.0)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["lineage"], "random_from_seed")
        self.assertLessEqual(result["elapsed_seconds"], 30.0)
        self.assertNotEqual(result["initial_weight_sha256"], result["trained_weight_sha256"])


@unittest.skipUnless(HAS_TORCH, "optional local PyTorch is absent")
class V2NeuralTests(unittest.TestCase):
    def setUp(self):
        from labyrinth.minotaur_scratch_v2 import build_v2_model
        self.config = V2Config(seed=12, width=16, heads=2, layers=1)
        torch.manual_seed(self.config.seed)
        self.model = build_v2_model(self.config)

    def test_two_heads_have_bounded_shapes_and_padding_is_masked(self):
        from labyrinth.minotaur_scratch_v2 import pad_v2_tokens
        case = generate_process_cases(103, 1)[0]
        tokens = tokenize_observed_trace_v2(case.question, case.steps)
        short = tokens[:]
        padded = pad_v2_tokens([short, tokens], self.config, torch)
        # Compare the same sequence alone and with trailing zero padding.
        alone = torch.tensor([tokens], dtype=torch.long)
        extended = torch.nn.functional.pad(alone, (0, 5))
        self.model.eval()
        with torch.no_grad():
            a = self.model(alone)
            b = self.model(extended)
            batch = self.model(padded)
        self.assertEqual(tuple(batch["correctness_logits"].shape), (2, 2))
        self.assertEqual(tuple(batch["earliest_error_logits"].shape), (2, 6))
        self.assertTrue(torch.allclose(a["correctness_logits"], b["correctness_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(a["earliest_error_logits"], b["earliest_error_logits"], atol=1e-6))

    def test_advisory_is_bounded_and_cannot_claim_authority(self):
        from labyrinth.minotaur_scratch_v2 import advisory_v2
        advice = advisory_v2({"correctness_logits": torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
                              "earliest_error_logits": torch.zeros((2, 6))}, threshold=0.70)
        self.assertEqual(advice[0]["disposition"], "ADVISE_ALL_CORRECT")
        self.assertEqual(advice[1]["disposition"], "ABSTAIN")
        for item in advice:
            self.assertGreaterEqual(item["confidence"], 0.0)
            self.assertLessEqual(item["confidence"], 1.0)
            self.assertEqual(item["graduation_authority"], "none")

if __name__ == "__main__":
    unittest.main()
