"""Research-only tests that distinguish fitted coefficients from supplied priors."""
from __future__ import annotations

import importlib.util
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch optional")
class AlgebraicAblationTests(unittest.TestCase):
    def test_equal_parameter_xy_ablation_disables_multiplication_transfer(self):
        # The public seam is absent until the opt-in ablation is implemented.
        self.assertIsNotNone(importlib.util.find_spec(
            "labyrinth.minotaur_algebraic_ablation"))
        from labyrinth.minotaur_algebraic_ablation import run_ablation

        result = run_ablation(torch, seeds=(47,), program_seed=808,
                              program_count=128)
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
        self.assertFalse(result["minotaur_graduated"])
        self.assertEqual(result["parameter_count_per_arm"], 12)
        self.assertEqual(result["method"], "float64_least_squares_representation_test")
        row = result["runs"][0]
        bilinear = row["arms"]["bilinear"]
        no_xy = row["arms"]["no_xy"]
        self.assertEqual(bilinear["full_correct_by_op"], [9409, 9409, 9409])
        self.assertEqual(bilinear["withheld_correct_by_op"], [9397, 9397, 9397])
        self.assertEqual(bilinear["withheld_total_by_op"], [9397, 9397, 9397])
        self.assertEqual(bilinear["wrap_total_by_op"], [4656, 4656, 8757])
        self.assertEqual(bilinear["wrap_correct_by_op"], [4656, 4656, 8757])
        self.assertEqual(bilinear["programs"]["exact"], 128)
        self.assertEqual(bilinear["programs"]["states_correct"], 416)
        self.assertEqual(no_xy["full_correct_by_op"][:2], [9409, 9409])
        self.assertLess(no_xy["full_correct_by_op"][2], 200)
        self.assertLess(no_xy["programs"]["exact"], 128)
        self.assertEqual(sum(n for _, n in bilinear["programs"]["by_length"]), 128)

    def test_three_frozen_dataset_seeds_are_reproducible(self):
        from labyrinth.minotaur_algebraic_ablation import run_ablation

        first = run_ablation(torch)
        self.assertEqual(first, run_ablation(torch))
        self.assertEqual([row["seed"] for row in first["runs"]], [47, 42, 2026])
        self.assertEqual(len(first["source_manifest"]), 2)
        for row in first["runs"]:
            self.assertEqual(row["training_examples"], 36)
            self.assertEqual(row["arms"]["bilinear"]["full_correct_by_op"],
                             [9409, 9409, 9409])
            self.assertEqual(row["arms"]["bilinear"]["programs"]["exact"], 128)
            self.assertEqual(row["arms"]["no_xy"]["full_correct_by_op"][:2],
                             [9409, 9409])
            self.assertLess(row["arms"]["no_xy"]["full_correct_by_op"][2], 200)

    def test_rejects_invalid_seed_collections_and_program_bounds(self):
        from labyrinth.minotaur_algebraic_ablation import run_ablation

        for bad in ((47, 47), ([47],), (True,), (), (2**31,)):
            with self.subTest(seeds=bad):
                try:
                    run_ablation(torch, seeds=bad)
                except Exception as error:
                    self.assertIsInstance(error, ValueError)
                else:
                    self.fail("malformed seeds were accepted")
        with self.assertRaisesRegex(ValueError, "bounded distinct seeds"):
            run_ablation(torch, program_count=0)


if __name__ == "__main__":
    unittest.main()
