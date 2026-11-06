"""Hostile controls for exact hidden permutation versus naive model averaging."""

from fractions import Fraction
from itertools import product
import random
import unittest

from labyrinth.boolean_neural_compiler import NeuralArtifact, compile_boolean
from labyrinth.boolean_neural_specs import build
from labyrinth.weight_space_permutation import (
    ablate_permutation,
    align_identical_neurons,
    coordinate_mean_forward,
    permute_hidden,
)


class WeightSpacePermutationAblationTests(unittest.TestCase):
    def setUp(self):
        self.parity, self.majority, self.full_adder = build()

    def test_permutation_preserves_function_while_naive_mean_destroys_it(self):
        original = self.full_adder
        reversed_neurons = tuple(reversed(range(len(original.w1))))
        report = ablate_permutation(original, reversed_neurons)
        self.assertEqual(report["exhaustive_cases"], 8)
        self.assertGreater(report["changed_parameter_coordinates"], 0)
        self.assertEqual(report["permuted_function_mismatches"], 0)
        self.assertEqual(report["naive_mean_function_mismatches"], 7)
        self.assertEqual(report["aligned_mean_function_mismatches"], 0)
        self.assertIs(report["model_training_performed"], False)
        self.assertIs(report["qwen_model_edited_or_executed"], False)
        changed = permute_hidden(original, reversed_neurons)
        self.assertNotEqual(original.sha256(), changed.sha256())
        for x in product((0, 1), repeat=3):
            self.assertEqual(original.evaluate(x), changed.evaluate(x))
        self.assertEqual(coordinate_mean_forward(original, changed, (1, 0, 0)),
                         (Fraction(0), Fraction(0)))
        self.assertEqual(original.evaluate((1, 0, 0)), (1, 0))

    def test_exact_alignment_recovers_original_tensor_coordinates(self):
        source = self.full_adder
        rng = random.Random(80451)
        for _ in range(40):
            order = list(range(len(source.w1)))
            rng.shuffle(order)
            candidate = permute_hidden(source, tuple(order))
            aligned = align_identical_neurons(source, candidate)
            self.assertEqual((aligned.w1, aligned.b1, aligned.w2, aligned.b2),
                             (source.w1, source.b1, source.w2, source.b2))
            for bits in product((0, 1), repeat=3):
                self.assertEqual(coordinate_mean_forward(source, aligned, bits),
                                 source.evaluate(bits))

    def test_distinct_task_weights_are_not_silently_declared_alignable(self):
        source = self.parity
        rows = [list(row) for row in source.w2]
        rows[0][0] = 1 - rows[0][0]
        changed_task = NeuralArtifact(source.name, source.input_names,
                                      source.output_names, source.w1, source.b1,
                                      tuple(tuple(row) for row in rows), source.b2)
        with self.assertRaisesRegex(ValueError, "no exact hidden"):
            align_identical_neurons(source, changed_task)

    def test_malformed_permutations_and_abis_rejected(self):
        model = self.parity
        order = tuple(range(len(model.w1)))
        for invalid in (list(order), order[:-1], order[:-1] + (True,),
                        order[:-1] + (order[0],), order[:-1] + (999,)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "bijection"):
                permute_hidden(model, invalid)
        changed_abi = NeuralArtifact(model.name, model.input_names,
                                     ("different_output",), model.w1,
                                     model.b1, model.w2, model.b2)
        with self.assertRaisesRegex(ValueError, "identical tensor shapes and ABI"):
            coordinate_mean_forward(model, changed_abi, (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "Boolean ABI"):
            coordinate_mean_forward(model, model, (0, False, 1))

    def test_identity_is_necessary_negative_control(self):
        source = self.full_adder
        report = ablate_permutation(source, tuple(range(len(source.w1))))
        self.assertEqual(report["changed_parameter_coordinates"], 0)
        self.assertEqual(report["naive_mean_function_mismatches"], 0)
        self.assertEqual(report["aligned_mean_function_mismatches"], 0)

    def test_full_truth_table_fails_closed_before_expensive_fraction_work(self):
        names = tuple(f"b{i}" for i in range(10))
        model = compile_boolean("parity10", names, ("parity",),
                                lambda bits: (sum(bits) % 2,))
        self.assertEqual(len(model.w1), 1024)
        with self.assertRaisesRegex(ValueError, "CPU budget"):
            ablate_permutation(model, tuple(range(len(model.w1))))


if __name__ == "__main__":
    unittest.main()
