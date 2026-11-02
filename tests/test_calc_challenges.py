"""Development-only Calc challenge integration, never a sealed evaluation."""

from __future__ import annotations

import re
from fractions import Fraction
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.evaluation import score_response
from labyrinth.calc_challenges import (
    CALC_COMMIT,
    CALC_ALGEBRA_SHA256,
    generate_calc_training_challenges,
)


class CalcChallengeTests(unittest.TestCase):
    def _cases(self, *, seed: int, count: int):
        try:
            return generate_calc_training_challenges(seed=seed, count=count)
        except RuntimeError as exc:
            if str(exc) == "optional Continuum Calc algebra is not installed":
                self.skipTest(f"pinned Calc source not installed: {exc}")
            raise

    def test_contract_rejects_bad_counts_seeds_and_nontraining_splits(self):
        for seed in (True, -1, 1.2, "4"):
            with self.assertRaises(ValueError):
                generate_calc_training_challenges(seed=seed, count=2)
        for count in (True, 0, 129, 1.5):
            with self.assertRaises(ValueError):
                generate_calc_training_challenges(seed=4, count=count)
        for split in ("dev", "eval", "private", "control", 5):
            with self.assertRaises(ValueError):
                generate_calc_training_challenges(seed=4, count=2, split=split)

    def test_exact_math_and_score_response(self):
        cases = self._cases(seed=41, count=32)
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c.id for c in cases}), 32)
        self.assertEqual(len({c.prompt for c in cases}), 32)
        derivative_re = re.compile(
            r"^For p\(x\) = (-?\d+) \+ \((-?\d+)\)x \+ \((-?\d+)\)x\^2 "
            r"\+ \((-?\d+)\)x\^3, compute p'\((-?\d+)\)\.\n"
        )
        determinant_re = re.compile(
            r"^Compute the determinant of the integer matrix "
            r"\[\[(-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+)\]\]\.\n"
        )
        for case in cases:
            self.assertEqual(case.split, "train")
            self.assertEqual(case.tier, 5)
            self.assertIn(("calc_source_commit", CALC_COMMIT), case.metadata)
            self.assertIn(("calc_algebra_sha256", CALC_ALGEBRA_SHA256), case.metadata)
            if case.kind == "calc_polynomial_derivative":
                m = derivative_re.match(case.prompt)
                self.assertIsNotNone(m)
                _, b, c, d, x = map(int, m.groups())
                independent_answer = b + 2 * c * x + 3 * d * x * x
            elif case.kind == "calc_matrix_determinant":
                m = determinant_re.match(case.prompt)
                self.assertIsNotNone(m)
                a, b, c, d = map(int, m.groups())
                independent_answer = a * d - b * c
            else:
                self.fail(f"unsupported kind {case.kind!r}")
            self.assertEqual(case.answer, str(independent_answer))
            self.assertTrue(score_response(case, f"ANSWER: {independent_answer}").correct)
            self.assertFalse(score_response(case, f"ANSWER: {independent_answer + 1}").correct)

    def test_replay_determinism_and_seed_separation(self):
        first = self._cases(seed=19, count=12)
        self.assertEqual(first, self._cases(seed=19, count=12))
        second = self._cases(seed=20, count=12)
        self.assertTrue(set(c.prompt for c in first).isdisjoint(c.prompt for c in second))

    def test_maximum_count_does_not_collide_at_seed_17(self):
        cases = self._cases(seed=17, count=128)
        self.assertEqual(len(set(c.prompt for c in cases)), 128)

    def test_no_eval_path_is_exposed(self):
        with self.assertRaises(ValueError):
            generate_calc_training_challenges(seed=3, count=1, split="eval")

    def test_forged_calc_module_source_is_rejected_before_cases(self):
        with tempfile.NamedTemporaryFile(mode="wb") as forged:
            forged.write(b"this is not the audited Calc algebra module")
            forged.flush()
            impostor = SimpleNamespace(
                __file__=forged.name,
                Polynomial=lambda *_: None,
                determinant=lambda *_: 0,
            )
            with patch("labyrinth.calc_challenges.importlib.import_module", return_value=impostor):
                with self.assertRaisesRegex(RuntimeError, "differs from reviewed source pin"):
                    generate_calc_training_challenges(seed=3, count=1)

    def test_in_process_calc_result_tampering_is_independently_rejected(self):
        try:
            from calc import algebra
        except ImportError:
            self.skipTest("optional Continuum Calc algebra is not installed")
        with patch.object(algebra, "determinant", return_value=Fraction(999999)):
            with self.assertRaisesRegex(RuntimeError, "independent 2x2 formula"):
                generate_calc_training_challenges(seed=41, count=2)


if __name__ == "__main__":
    unittest.main()
