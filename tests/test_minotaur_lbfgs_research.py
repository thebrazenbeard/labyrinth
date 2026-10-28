"""Opt-in L-BFGS research checks; these are not graduation evidence."""
from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch optional")
class MinotaurLbfgsResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(), 2))

    def test_equal_parameter_basis_variants_are_real_neural_heads(self):
        from labyrinth.minotaur_algebraic_rules import build_algebraic_head
        arms = {basis: build_algebraic_head(torch, seed=19, basis=basis)
                for basis in ("bilinear", "no_xy")}
        for head in arms.values():
            self.assertEqual(sum(p.numel() for p in head.parameters()), 12)
            with torch.no_grad():
                head.coefficients.zero_()
                head.coefficients[:, 3] = 1.0
        inputs = (torch.tensor(2), torch.tensor(3), torch.tensor(5))
        self.assertEqual(float(arms["bilinear"].raw(*inputs)), 15.0)
        self.assertEqual(float(arms["no_xy"].raw(*inputs)), 9.0)
        with self.assertRaises(ValueError):
            build_algebraic_head(torch, basis="unrecognized")

    def test_explicit_closure_budget_is_hard_and_rolls_back(self):
        from labyrinth.minotaur_algebraic_rules import (
            build_algebraic_head, incomplete_training_atoms,
            train_rule_head_lbfgs,
        )
        examples = incomplete_training_atoms(47, 12)
        head = build_algebraic_head(torch, seed=19)
        original = head.coefficients.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "closure budget exhausted"):
            train_rule_head_lbfgs(head, examples, torch, max_iter=200,
                                  closure_budget=1, require_fit=False)
        self.assertTrue(torch.equal(head.coefficients.detach(), original))
        self.assertIsNone(head.coefficients.grad)
        self.assertFalse(head.training)
        with self.assertRaises(ValueError):
            train_rule_head_lbfgs(build_algebraic_head(torch), examples, torch,
                                  closure_budget=0)
        success = build_algebraic_head(torch, seed=19)
        stats = train_rule_head_lbfgs(success, examples, torch, max_iter=200,
                                      closure_budget=500)
        self.assertLessEqual(stats["closure_evaluations"], 500)
        self.assertEqual(stats["closure_budget"], 500)

    def test_predeclared_dataset_seed_sweep(self):
        from labyrinth.minotaur_algebraic_rules import (
            MOD, all_atom_accuracy, build_algebraic_head,
            incomplete_training_atoms, train_rule_head_lbfgs,
        )
        # Development seeds previously probed with an external L-BFGS script;
        # this fixed list is not a blinded/private qualification.
        for seed in (3, 47, 521, 1234, 2026, 42, 7, 89, 314, 1024):
            with self.subTest(dataset_seed=seed):
                head = build_algebraic_head(torch, seed=19)
                stats = train_rule_head_lbfgs(
                    head, incomplete_training_atoms(seed, 12), torch,
                    max_iter=200, learning_rate=1.0,
                )
                self.assertEqual(stats["optimizer"], "LBFGS_strong_wolfe")
                self.assertTrue(stats["training_fit"])
                self.assertLessEqual(stats["final_loss"], 1e-12)
                self.assertLessEqual(stats["optimizer_iterations"], 200)
                self.assertGreaterEqual(stats["closure_evaluations"], stats["optimizer_iterations"])
                self.assertGreater(stats["parameter_delta_l2"], 0.0)
                self.assertEqual(stats["design_rank_by_operation"], [4,4,4])
                self.assertTrue(stats["identifiable_all_operations"])
                self.assertEqual(sum(p.numel() for p in head.parameters()), 12)
                self.assertEqual(all_atom_accuracy(head, torch)["correct"], 3*MOD*MOD)

    def test_fresh_dataset_and_neural_initialization_seeds(self):
        from labyrinth.minotaur_algebraic_rules import (
            MOD, all_atom_accuracy, build_algebraic_head,
            incomplete_training_atoms, train_rule_head_lbfgs,
        )
        for dataset_seed, neural_seed in ((0, 33), (11, 402), (91, 7), (314159, 29)):
            with self.subTest(dataset_seed=dataset_seed, neural_seed=neural_seed):
                head = build_algebraic_head(torch, seed=neural_seed)
                train_rule_head_lbfgs(head, incomplete_training_atoms(dataset_seed, 12),
                                      torch, max_iter=200)
                self.assertEqual(all_atom_accuracy(head, torch)["correct"], 3*MOD*MOD)

    def test_zero_update_and_corrupted_label_controls(self):
        from labyrinth.minotaur_algebraic_rules import (
            all_atom_accuracy, build_algebraic_head,
            corrupt_labels_within_op, incomplete_training_atoms,
            train_rule_head, train_rule_head_lbfgs,
        )
        examples = incomplete_training_atoms(47, 12)
        sham = build_algebraic_head(torch, seed=19)
        before = sham.coefficients.detach().clone()
        train_rule_head(sham, examples, torch, steps=200, learning_rate=0.0)
        self.assertTrue(torch.equal(sham.coefficients.detach(), before))

        candidate = build_algebraic_head(torch, seed=19)
        train_rule_head_lbfgs(candidate, examples, torch)
        corrupted = build_algebraic_head(torch, seed=19)
        stats = train_rule_head_lbfgs(
            corrupted, corrupt_labels_within_op(examples, 59), torch,
            require_fit=False,
        )
        self.assertFalse(stats["training_fit"])
        self.assertGreater(stats["final_loss"], 1e-12)
        self.assertLess(all_atom_accuracy(corrupted, torch)["correct"],
                        all_atom_accuracy(candidate, torch)["correct"])

    def test_nonconvergence_and_invalid_configuration_fail_closed(self):
        from labyrinth.minotaur_algebraic_rules import (
            all_atom_accuracy, build_algebraic_head,
            incomplete_training_atoms, train_rule_head_lbfgs,
        )
        examples = incomplete_training_atoms(47, 12)
        with self.assertRaisesRegex(RuntimeError, "did not fit"):
            train_rule_head_lbfgs(build_algebraic_head(torch, seed=19), examples,
                                  torch, max_iter=1)
        with self.assertRaises(ValueError):
            train_rule_head_lbfgs(build_algebraic_head(torch), examples, torch,
                                  max_iter=501)
        with self.assertRaises(ValueError):
            train_rule_head_lbfgs(build_algebraic_head(torch), examples, torch,
                                  max_eval=0)
        with self.assertRaises(ValueError):
            train_rule_head_lbfgs(build_algebraic_head(torch), examples, torch,
                                  learning_rate=0.0)

        # This distinct data split demonstrably does not fit within the 200
        # iteration budget; failure must be explicit and weights restored.
        difficult = incomplete_training_atoms(31, 12)
        rejected = build_algebraic_head(torch, seed=19)
        original = rejected.coefficients.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "did not fit"):
            train_rule_head_lbfgs(rejected, difficult, torch)
        self.assertTrue(torch.equal(rejected.coefficients.detach(), original))
        diagnostic = build_algebraic_head(torch, seed=19)
        stats = train_rule_head_lbfgs(diagnostic, difficult, torch,
                                      require_fit=False)
        self.assertFalse(stats["training_fit"])
        self.assertGreater(stats["closure_evaluations"], stats["max_iter"])
        self.assertLess(all_atom_accuracy(diagnostic, torch)["correct"], 3*97*97)

        # A larger, explicitly measured evaluation budget may rescue this
        # split without using full-domain labels to adjust the coefficients.
        extended = build_algebraic_head(torch, seed=19)
        extended_stats = train_rule_head_lbfgs(
            extended, difficult, torch, max_iter=500, max_eval=500,
        )
        self.assertTrue(extended_stats["training_fit"])
        self.assertEqual(extended_stats["max_eval"], 500)
        self.assertLessEqual(extended_stats["closure_evaluations"], 501)
        self.assertEqual(all_atom_accuracy(extended, torch)["correct"], 3*97*97)

    def test_minimum_sample_rank_deficiency_is_reported(self):
        from labyrinth.minotaur_algebraic_rules import (
            all_atom_accuracy, build_algebraic_head,
            incomplete_training_atoms, train_rule_head_lbfgs,
        )
        # Five examples/op is a valid interface input but not necessarily an
        # identifiable four-coefficient experiment for every operation.
        samples = incomplete_training_atoms(12, 5)
        head = build_algebraic_head(torch, seed=19)
        stats = train_rule_head_lbfgs(head, samples, torch,
                                      max_iter=500, max_eval=500,
                                      require_fit=False)
        self.assertLess(stats["design_rank_by_operation"][1], 4)
        self.assertFalse(stats["identifiable_all_operations"])
        self.assertLess(all_atom_accuracy(head, torch)["correct"], 3*97*97)
        self.assertEqual(stats["qualification"], "NOT_ESTABLISHED")


if __name__ == "__main__":
    unittest.main()
