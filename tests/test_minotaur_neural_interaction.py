"""Research-only checks: no multiplication feature and no promotion claims."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy optional research dependency")
class NeuralInteractionTests(unittest.TestCase):
    def test_operand_family_separation_and_forward_inputs(self):
        from labyrinth.minotaur_neural_interaction import NeuralInteraction, split_pairs

        training, withheld, outside = split_pairs()
        train_ids = {operand for pair in training for operand in pair}
        heldout_ids = {operand for pair in withheld for operand in pair}
        self.assertEqual(len(training), 121)
        self.assertEqual(len(withheld), 100)
        self.assertEqual(len(outside), 144)
        self.assertFalse(train_ids & heldout_ids)
        self.assertTrue(all(a > 20 and b > 20 for a, b in outside))
        model = NeuralInteraction(seed=41)
        self.assertEqual(len(model.weights()), 1185)
        # The forward interface accepts precisely two scalar input channels;
        # it is not given a third product or arithmetic-operation feature.
        output, _ = model.forward(np.asarray([[0.2, 0.4], [0.4, 0.2]]))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_matched_controls_and_transfer_limitations_two_seeds(self):
        from labyrinth.minotaur_neural_interaction import study

        for seed in (41, 42):
            with self.subTest(seed=seed):
                result = study(seed=seed, steps=10000)
                self.assertFalse(result["minotaur_graduated"])
                self.assertFalse(result["deployed"])
                self.assertEqual(result["primary_holdout_operand_identity_overlap"], 0)
                arms = result["arms"]
                for name in ("frozen", "zero_lr_sham", "corrupted_labels", "trained"):
                    self.assertEqual(arms[name]["withheld_odd_operands"]["total"], 100)
                    self.assertEqual(arms[name]["outside_21_32"]["total"], 144)
                self.assertEqual(arms["zero_lr_sham"], arms["frozen"])
                learned = arms["trained"]["withheld_odd_operands"]["correct"]
                controls = [arms[name]["withheld_odd_operands"]["correct"]
                            for name in ("frozen", "zero_lr_sham", "corrupted_labels")]
                self.assertGreaterEqual(learned, 50)
                self.assertGreater(learned, max(controls) + 40)
                # A good interpolation score is not a claim of extrapolation.
                self.assertLessEqual(arms["trained"]["outside_21_32"]["correct"], 5)

    def test_initialization_and_zero_lr_reproducible(self):
        from labyrinth.minotaur_neural_interaction import NeuralInteraction, split_pairs, train

        a = NeuralInteraction(seed=41)
        b = NeuralInteraction(seed=41)
        c = NeuralInteraction(seed=42)
        self.assertTrue(np.array_equal(a.weights(), b.weights()))
        self.assertFalse(np.array_equal(a.weights(), c.weights()))
        examples, _, _ = split_pairs()
        labels = np.asarray([x * y for x, y in examples])
        train(b, examples, labels, steps=50, learning_rate=0.0)
        self.assertTrue(np.array_equal(a.weights(), b.weights()))


if __name__ == "__main__":
    unittest.main()
