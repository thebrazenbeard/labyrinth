"""Contracts for leakage-resistant V6 multiplication holdouts."""
import hashlib
import unittest

from labyrinth.minotaur_v6_holdouts import (
    MOD97,
    MOD101,
    build_mod101_evaluation,
    build_mod97_splits,
    model_features,
    verify_multiplication_label,
)


class MinotaurV6HoldoutTests(unittest.TestCase):
    def test_grouped_allocator_meets_exact_requested_sizes(self):
        """Diagonal singleton groups make the stated default sizes attainable."""
        expected_sizes = (7527, 941, 941)
        for challenge, options in (
            ("none", {}),
            ("withheld_value", {"withheld_value": 7}),
            ("cartesian_block", {"block_size": 12}),
        ):
            with self.subTest(challenge=challenge):
                data = build_mod97_splits(seed=23, challenge=challenge, **options)
                self.assertEqual(tuple(map(len, (data.train, data.dev, data.test))), expected_sizes)

    def test_reproducible_grouped_partition_and_digests(self):
        data = build_mod97_splits(seed=23)
        self.assertEqual(data, build_mod97_splits(seed=23))
        self.assertEqual(data.modulus, MOD97)
        examples = data.train + data.dev + data.test
        self.assertEqual(len(examples), MOD97 * MOD97)
        self.assertEqual(len({row[:2] for row in examples}), MOD97 * MOD97)
        locations = {row[:2]: name for name, rows in
                     (("train", data.train), ("dev", data.dev), ("test", data.test))
                     for row in rows}
        for x, y in locations:
            self.assertEqual(locations[x, y], locations[y, x])
        for name, rows in (("train", data.train), ("dev", data.dev), ("test", data.test)):
            payload = "\n".join(f"{x}\t{y}\t{label}" for x, y, label in rows)
            self.assertEqual(data.digests[name], hashlib.sha256(payload.encode()).hexdigest())

    def test_labels_are_independently_checked_and_hidden_from_features(self):
        data = build_mod97_splits(seed=24)
        for rows in (data.train, data.dev, data.test):
            self.assertEqual(model_features(rows), tuple(row[:2] for row in rows))
            altered_labels = tuple((x, y, (label + 1) % MOD97) for x, y, label in rows)
            self.assertEqual(model_features(rows), model_features(altered_labels))
            for x, y, label in rows:
                self.assertEqual(label, verify_multiplication_label(x, y, MOD97))

    def test_withheld_value_never_appears_in_train_or_dev(self):
        data = build_mod97_splits(seed=25, challenge="withheld_value")
        value = data.withheld_value
        self.assertIsNotNone(value)
        self.assertTrue(all(value not in row[:2] for rows in (data.train, data.dev) for row in rows))
        expected = {(value, other) for other in range(MOD97)} | {(other, value) for other in range(MOD97)}
        self.assertTrue(expected.issubset({row[:2] for row in data.test}))

    def test_cartesian_block_is_test_only_and_reverse_safe(self):
        data = build_mod97_splits(seed=26, challenge="cartesian_block", block_size=12)
        block = set(data.block_values)
        self.assertEqual(len(block), 12)
        self.assertTrue(all(not (x in block and y in block)
                            for rows in (data.train, data.dev) for x, y, _ in rows))
        test_pairs = {row[:2] for row in data.test}
        self.assertTrue(all((x, y) in test_pairs for x in block for y in block))

    def test_singleton_cartesian_block_is_a_valid_diagonal_challenge(self):
        data = build_mod97_splits(seed=26, challenge="cartesian_block", block_size=1)
        self.assertEqual(len(data.block_values), 1)
        value = data.block_values[0]
        self.assertIn((value, value), {row[:2] for row in data.test})
        self.assertTrue(all((value, value) != row[:2]
                            for rows in (data.train, data.dev) for row in rows))

    def test_mod101_is_separately_labeled_out_of_domain_evaluation(self):
        evaluation = build_mod101_evaluation(seed=27)
        self.assertEqual(evaluation, build_mod101_evaluation(seed=27))
        self.assertEqual(evaluation.modulus, MOD101)
        self.assertEqual(evaluation.domain, "out_of_domain_mod101")
        self.assertEqual(len(evaluation.examples), MOD101 * MOD101)
        payload = "\n".join(f"{x}\t{y}\t{label}" for x, y, label in evaluation.examples)
        self.assertEqual(evaluation.digest, hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(model_features(evaluation.examples), tuple(row[:2] for row in evaluation.examples))
        self.assertTrue(all(label == verify_multiplication_label(x, y, MOD101)
                            for x, y, label in evaluation.examples))


if __name__ == "__main__":
    unittest.main()
