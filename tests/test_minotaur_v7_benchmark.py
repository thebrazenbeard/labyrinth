"""Pure-Python contracts for the V7 leakage-resistant multiplier benchmark."""
import hashlib
import inspect
import unittest

from labyrinth.minotaur_v7_benchmark import (
    ABSENT_LABEL,
    MOD97,
    MOD101,
    build_absent_output_stress_variants,
    build_mod101_evaluation,
    build_mod97_dataset,
    data_oracle,
    evaluate_baselines,
    model_features,
)


class MinotaurV7BenchmarkTests(unittest.TestCase):
    def _assert_digest(self, rows, digest):
        payload = "\n".join(f"{left}\t{right}\t{target}" for left, right, target in rows)
        self.assertEqual(digest, hashlib.sha256(payload.encode("ascii")).hexdigest())

    def _assert_group_disjoint(self, dataset):
        seen = {}
        for split_name, rows in dataset.splits.items():
            for left, right, _target in rows:
                group = (min(left, right), max(left, right))
                if group in seen:
                    self.assertEqual(seen[group], split_name,
                                     f"{group} aliases {seen[group]} and {split_name}")
                seen[group] = split_name

    def test_base_partition_is_seeded_group_disjoint_and_traceable(self):
        first = build_mod97_dataset(seed=20261028)
        second = build_mod97_dataset(seed=20261028)
        self.assertEqual(first, second)
        self.assertEqual(first.modulus, MOD97)
        self.assertEqual(set(first.splits), {"train", "dev", "test"})
        self._assert_group_disjoint(first)
        self.assertEqual(sum(len(rows) for rows in first.splits.values()), MOD97 ** 2)
        self.assertEqual(sum(counts["groups"] for counts in first.counts.values()), MOD97 * (MOD97 + 1) // 2)
        for split_name, rows in first.splits.items():
            self.assertEqual(first.counts[split_name]["rows"], len(rows))
            self.assertEqual(first.counts[split_name]["groups"], len({(min(x, y), max(x, y)) for x, y, _ in rows}))
            self._assert_digest(rows, first.digests[split_name])
            self.assertTrue(all(target == data_oracle(x, y, MOD97) for x, y, target in rows))

    def test_reversed_pairs_never_cross_train_dev_or_test(self):
        dataset = build_mod97_dataset(seed=9)
        locations = {
            (left, right): split_name
            for split_name, rows in dataset.splits.items()
            for left, right, _ in rows
        }
        self.assertEqual(len(locations), MOD97 ** 2)
        self.assertTrue(all(locations[left, right] == locations[right, left] for left, right in locations))

    def test_absent_output_53_is_not_learnable_from_train_or_dev(self):
        dataset = build_mod97_dataset(seed=10, variant="absent_output")
        self.assertEqual(dataset.absent_label, ABSENT_LABEL)
        self.assertTrue(all(target != ABSENT_LABEL for name in ("train", "dev")
                            for _x, _y, target in dataset.splits[name]))
        self.assertTrue(dataset.splits["test"])
        self.assertTrue(all(target == ABSENT_LABEL for _x, _y, target in dataset.splits["test"]))
        self._assert_group_disjoint(dataset)

    def test_operand_and_output_53_are_both_absent_from_train_and_dev(self):
        dataset = build_mod97_dataset(seed=11, variant="absent_operand_and_output")
        for name in ("train", "dev"):
            self.assertTrue(all(ABSENT_LABEL not in (x, y, target) for x, y, target in dataset.splits[name]))
        self.assertTrue(all(ABSENT_LABEL in (x, y, target) for x, y, target in dataset.splits["test"]))
        self._assert_group_disjoint(dataset)

    def test_all_97_absent_output_stresses_are_deterministic(self):
        stresses = build_absent_output_stress_variants(seed=12)
        self.assertEqual(set(stresses), set(range(MOD97)))
        self.assertEqual(stresses, build_absent_output_stress_variants(seed=12))
        for label, dataset in stresses.items():
            self.assertEqual(dataset.absent_label, label)
            self.assertTrue(all(target != label for name in ("train", "dev")
                                for _x, _y, target in dataset.splits[name]))
            self.assertTrue(all(target == label for _x, _y, target in dataset.splits["test"]))
            self._assert_group_disjoint(dataset)

    def test_features_exclude_targets_and_oracle_is_builder_only(self):
        dataset = build_mod97_dataset(seed=13)
        rows = dataset.splits["train"]
        self.assertEqual(model_features(rows), tuple((x, y) for x, y, _ in rows))
        modified = tuple((x, y, (target + 1) % MOD97) for x, y, target in rows)
        self.assertEqual(model_features(rows), model_features(modified))
        source = inspect.getsource(__import__("labyrinth.minotaur_v7_benchmark", fromlist=["*"]))
        self.assertNotIn("data_oracle(", source[source.index("def evaluate_baselines"):])

    def test_baselines_are_explicit_and_test_is_never_a_fit_source(self):
        dataset = build_mod97_dataset(seed=14)
        report = evaluate_baselines(dataset, seed=88)
        self.assertEqual(set(report), {"zero", "constant", "copy_left", "random"})
        for baseline, values in report.items():
            self.assertEqual(set(values), {"train", "dev", "test"})
            self.assertTrue(all(0.0 <= result["accuracy"] <= 1.0 for result in values.values()))
            self.assertEqual(values["train"]["fit_source"], "train" if baseline == "constant" else None)
        self.assertEqual(report, evaluate_baselines(dataset, seed=88))

    def test_mod101_is_evaluation_only_and_has_no_train_split(self):
        evaluation = build_mod101_evaluation(seed=15)
        self.assertEqual(evaluation.modulus, MOD101)
        self.assertEqual(evaluation.domain, "evaluation_only_mod101")
        self.assertFalse(hasattr(evaluation, "train"))
        self.assertEqual(len(evaluation.rows), MOD101 ** 2)
        self._assert_digest(evaluation.rows, evaluation.digest)
        self.assertTrue(all(target == data_oracle(x, y, MOD101) for x, y, target in evaluation.rows))


if __name__ == "__main__":
    unittest.main()
