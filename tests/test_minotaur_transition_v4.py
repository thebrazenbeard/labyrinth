import unittest
from dataclasses import replace

from labyrinth.minotaur_combinatorial_curriculum import (
    CombinatorialCase,
    CurriculumConfig,
    build_combinatorial_splits,
)
from labyrinth.minotaur_transition_v4 import (
    TransitionConfig,
    build_transition_model,
    derive_first_mismatch,
    encode_transition_cases,
)


class TransitionV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is optional")
        cls.torch = torch
        cls.cases = build_combinatorial_splits(
            CurriculumConfig(seed=91, train_cases=10, dev_cases=10, transfer_cases=10)
        )["train"].cases

    def test_config_is_bounded_and_strict(self):
        self.assertIs(TransitionConfig(width=32).validate().width, 32)
        for config in (
            TransitionConfig(seed=True), TransitionConfig(width=15),
            TransitionConfig(max_steps=7), TransitionConfig(modulus=96),
            TransitionConfig(temperature=1),
        ):
            with self.assertRaises(ValueError):
                config.validate()

    def test_encoding_has_ordered_per_instruction_features_and_oracle_states(self):
        config = TransitionConfig(width=32)
        features, expected = encode_transition_cases(self.cases[:4], config, self.torch)
        self.assertEqual(tuple(features.initial_states.shape), (4, 2))
        self.assertEqual(tuple(features.operations.shape), (4, 6))
        self.assertEqual(tuple(expected.shape), (4, 6, 2))
        self.assertEqual(features.mask.dtype, self.torch.bool)
        self.assertTrue(self.torch.equal(features.step_numbers[0, features.mask[0]],
                                         self.torch.arange(1, int(features.mask[0].sum()) + 1)))
        self.assertTrue(self.torch.all(expected[~features.mask] == 0))

    def test_observed_faults_are_never_model_features_or_oracle_predictors(self):
        clean = next(case for case in self.cases if case.first_error_step is None)
        lines = list(clean.steps)
        first = lines[0]
        a = int(first.split("A=")[1].split(",")[0])
        lines[0] = first.replace(f"A={a}", f"A={(a + 19) % 97}")
        faulty = replace(clean, steps=tuple(lines), first_error_step=1, fault_family="test")
        clean_features, clean_expected = encode_transition_cases([clean], TransitionConfig(), self.torch)
        bad_features, bad_expected = encode_transition_cases([faulty], TransitionConfig(), self.torch)
        for name in clean_features.__dataclass_fields__:
            self.assertTrue(self.torch.equal(getattr(clean_features, name),
                                             getattr(bad_features, name)), name)
        self.assertTrue(self.torch.equal(clean_expected, bad_expected))

    def test_oracle_chains_true_arithmetic_not_faulty_observed_state(self):
        clean = next(case for case in self.cases
                     if case.first_error_step is None and len(case.steps) > 1)
        lines = list(clean.steps)
        lines[0] = "Step 1: A=0, B=0" if lines[0] != "Step 1: A=0, B=0" else "Step 1: A=1, B=1"
        faulty = replace(clean, steps=tuple(lines), first_error_step=1, fault_family="test")
        _, expected_clean = encode_transition_cases([clean], TransitionConfig(), self.torch)
        _, expected_faulty = encode_transition_cases([faulty], TransitionConfig(), self.torch)
        self.assertTrue(self.torch.equal(expected_clean, expected_faulty))

    def test_model_is_per_state_neural_and_has_real_gradients(self):
        config = TransitionConfig(seed=7, width=32)
        features, expected = encode_transition_cases(self.cases[:6], config, self.torch)
        model = build_transition_model(config)
        self.assertFalse(hasattr(model, "correctness_head"))
        self.assertFalse(hasattr(model, "earliest_error_head"))
        output = model(features, teacher_states=expected)
        self.assertEqual(tuple(output["state_logits"].shape), (6, 6, 2, 97))
        selected = output["state_logits"][features.mask]
        targets = expected[features.mask]
        loss = self.torch.nn.functional.cross_entropy(selected.reshape(-1, 97),
                                                       targets.reshape(-1))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and bool((g != 0).any()) for g in gradients))

    def test_free_running_never_requires_targets(self):
        config = TransitionConfig(width=32)
        features, _ = encode_transition_cases(self.cases[:3], config, self.torch)
        output = build_transition_model(config)(features)
        self.assertEqual(tuple(output["predicted_states"].shape), (3, 6, 2))
        self.assertTrue(self.torch.all(output["predicted_states"][~features.mask] == 0))

    def test_first_mismatch_handles_step_one_later_and_all_correct(self):
        case = next(case for case in self.cases
                    if case.first_error_step is None and len(case.steps) >= 3)
        _, expected = encode_transition_cases([case], TransitionConfig(), self.torch)
        length = len(case.steps)
        predictions = expected[0, :length].clone()
        self.assertIsNone(derive_first_mismatch(predictions, case.steps))
        first = predictions.clone(); first[0, 0] = (first[0, 0] + 1) % 97
        later = predictions.clone(); later[2, 1] = (later[2, 1] + 1) % 97
        self.assertEqual(derive_first_mismatch(first, case.steps), 1)
        self.assertEqual(derive_first_mismatch(later, case.steps), 3)

    def test_empty_and_malformed_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_transition_cases([], TransitionConfig(), self.torch)
        malformed = CombinatorialCase("x", "train", "bad", ("Step 1: A=1, B=2",),
                                      None, "x", "none")
        with self.assertRaises(ValueError):
            encode_transition_cases([malformed], TransitionConfig(), self.torch)
        with self.assertRaises(ValueError):
            derive_first_mismatch(self.torch.zeros((1, 2), dtype=self.torch.long),
                                  ("Step 2: A=0, B=0",))


if __name__ == "__main__":
    unittest.main()
