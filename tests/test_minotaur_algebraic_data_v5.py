"""Dependency-free contract tests for the V5 algebraic data owner."""
import hashlib
import unittest

from labyrinth.minotaur_algebraic_data_v5 import (
    ATOM_COUNT, MODULUS, atomic_features, atomic_split, program_composition_split,
    verify_mod97,
)


class AlgebraicDataV5Tests(unittest.TestCase):
    def test_exhaustive_deterministic_committed_partition(self):
        data = atomic_split()
        self.assertEqual(data, atomic_split())
        self.assertEqual(sum(map(len, (data.train, data.val, data.holdout))), ATOM_COUNT)
        all_keys = [entry[:3] for group in (data.train, data.val, data.holdout) for entry in group]
        self.assertEqual(len(all_keys), ATOM_COUNT)
        self.assertEqual(len(set(all_keys)), ATOM_COUNT)
        for name, group in (("train", data.train), ("val", data.val), ("holdout", data.holdout)):
            payload = "\n".join(f"{op}\t{x}\t{y}\t{label}" for op, x, y, label in group)
            self.assertEqual(data.commitments[name], hashlib.sha256(payload.encode()).hexdigest())

    def test_labels_independent_oracle_and_features_do_not_contain_label(self):
        data = atomic_split(42)
        semantic = {opaque: operation for opaque, operation in zip(data.op_ids, ("ADD", "SUB", "MUL"))}
        for group in (data.train, data.val, data.holdout):
            self.assertEqual(atomic_features(group), tuple(entry[:3] for entry in group))
            for op, x, y, label in group:
                self.assertEqual(label, verify_mod97(semantic[op], x, y))

    def test_no_holdout_alias_leakage_and_balanced_zero_strata(self):
        data = atomic_split(20261028)
        locations = {entry[:3]: name for name, group in
                     (("train", data.train), ("val", data.val), ("holdout", data.holdout)) for entry in group}
        for op, x, y in locations:
            self.assertEqual(locations[(op, x, y)], locations[(op, y, x)])
        for group in (data.train, data.val, data.holdout):
            self.assertEqual({entry[0] for entry in group}, set(data.op_ids))
            for op in data.op_ids:
                self.assertTrue(any(x == 0 for current, x, y, label in group if current == op))
                self.assertTrue(any(y == 0 for current, x, y, label in group if current == op))

    def test_stronger_value_and_diagonal_holdouts(self):
        value = atomic_split(77, challenge="value")
        self.assertIsNotNone(value.withheld_value)
        held = {(op, x, y) for op, x, y, _ in value.holdout}
        for op in value.op_ids:
            for other in range(MODULUS):
                self.assertIn((op, value.withheld_value, other), held)
                self.assertIn((op, other, value.withheld_value), held)
        diagonal = atomic_split(77, challenge="diagonal")
        held_diagonal = {(op, x, y) for op, x, y, _ in diagonal.holdout}
        self.assertTrue(all((op, value, value) in held_diagonal
                            for op in diagonal.op_ids for value in range(MODULUS)))
        # Non-default fractions retain the same reversible-group invariant.
        self.assertEqual(sum(map(len, atomic_split(78, .7, challenge="diagonal"))), ATOM_COUNT)

    def test_separate_causal_composition_structures_and_commitments(self):
        split = program_composition_split(93, train_cases=20, val_cases=20, holdout_cases=20)
        self.assertEqual(split, program_composition_split(93, train_cases=20, val_cases=20, holdout_cases=20))
        structures = [{case.canonical_sequence for case in group}
                      for group in (split.train, split.val, split.holdout)]
        self.assertTrue(all(structures[i].isdisjoint(structures[j])
                            for i in range(3) for j in range(i + 1, 3)))
        self.assertEqual(set(split.commitments), {"train", "val", "holdout"})


if __name__ == "__main__":
    unittest.main()
