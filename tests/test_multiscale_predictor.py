"""Causal, train-only, deterministic tests for the opt-in diagnostic."""

import math
import unittest

from labyrinth.curriculum import (
    CurriculumSpec,
    CurriculumState,
    TrainingObservation,
    next_training_case,
)
from labyrinth.multiscale_predictor import (
    MultiScaleSpec,
    forecast_training_score,
    rank_training_challenges,
)


def sample(scores, *, level="basic", challenge="a", split="train"):
    return tuple(
        TrainingObservation(challenge_id=challenge, seed=i, level=level, score=value, split=split)
        for i, value in enumerate(scores)
    )


class MultiScalePredictorTests(unittest.TestCase):
    def predict(self, observations, **kwargs):
        return forecast_training_score(observations, level="basic", challenge_id="a", **kwargs)

    def test_empty_history_is_neutral(self):
        forecast = self.predict(())
        self.assertEqual(forecast.expected_score, 0.5)
        self.assertEqual(forecast.uncertainty, 0.5)
        self.assertEqual(forecast.sample_count, 0)
        self.assertEqual(forecast.dyadic_roughness, 0)
        self.assertEqual(forecast.expert_weights, (0.25,) * 4)

    def test_constant_training_sequences_predict_bounded_extremes(self):
        high = self.predict(sample([1.0] * 64))
        low = self.predict(sample([0.0] * 64))
        self.assertGreater(high.expected_score, 0.8)
        self.assertLess(low.expected_score, 0.2)
        self.assertEqual(high.dyadic_roughness, 0)
        self.assertEqual(low.dyadic_roughness, 0)

    def test_online_update_reacts_to_shift_without_future_lookahead(self):
        before = sample([0.0] * 40)
        after = sample([0.0] * 40 + [1.0] * 12)
        self.assertGreater(self.predict(after).expected_score, self.predict(before).expected_score)
        self.assertNotEqual(self.predict(before), self.predict(after))
        self.assertEqual(self.predict(before), self.predict(before[:]))

    def test_future_changes_cannot_rewrite_prefix_prediction(self):
        prefix = sample([0, 1, 0, 1, 0, 1])
        two_alternatives = prefix + sample([0] * 20) + sample([1] * 20)
        frozen = self.predict(prefix)
        self.assertEqual(frozen, self.predict(two_alternatives[: len(prefix)]))

    def test_prequential_synthetic_shift_compared_with_nonadaptive_control(self):
        # Every forecast is computed *before* observing its target. This is a
        # mathematical fixture, not a held-out LLM improvement benchmark.
        scores = [0.0] * 25 + [1.0] * 25 + [0.0] * 25 + [1.0] * 25
        observed = sample(scores)
        adaptive_loss = 0.0
        neutral_loss = 0.0
        for index in range(1, len(observed)):
            estimate = self.predict(observed[:index]).expected_score
            truth = observed[index].score
            adaptive_loss += (estimate - truth) ** 2
            neutral_loss += (0.5 - truth) ** 2
        self.assertLess(adaptive_loss, neutral_loss)

    def test_eval_and_control_contamination_fail_closed_even_other_level(self):
        for split in ("eval", "placebo", "shuffled", "control"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.predict(sample([0.5]) + sample([1.0], level="advanced", split=split))

    def test_non_observations_fail_closed(self):
        with self.assertRaises(ValueError):
            self.predict([object()])
        with self.assertRaises(ValueError):
            self.predict(iter(sample([1.0] * 10)))

    def test_mutated_observation_cannot_inject_nonfinite_score(self):
        observation = TrainingObservation("a", 1, "basic", 0.8)
        object.__setattr__(observation, "score", float("nan"))
        with self.assertRaises(ValueError):
            self.predict([observation])

    def test_mutated_spec_cannot_inject_nonfinite_weights(self):
        spec = MultiScaleSpec()
        object.__setattr__(spec, "learning_rate", float("nan"))
        with self.assertRaises(ValueError):
            self.predict((), spec=spec)

    def test_level_isolation(self):
        basic = sample([0.25] * 10)
        advanced = sample([1.0] * 20, level="advanced")
        self.assertEqual(self.predict(basic), self.predict(basic + advanced))

    def test_deterministic_bounded_mixture_and_roughness(self):
        observed = sample([float(i % 2) for i in range(50)])
        forecast = self.predict(observed)
        self.assertEqual(forecast, self.predict(observed))
        self.assertGreater(forecast.dyadic_roughness, 0)
        self.assertTrue(0 <= forecast.expected_score <= 1)
        self.assertTrue(0 <= forecast.uncertainty <= 1)
        self.assertTrue(all(math.isfinite(weight) and weight > 0 for weight in forecast.expert_weights))
        self.assertAlmostEqual(sum(forecast.expert_weights), 1.0)

    def test_bounded_history_is_explicit(self):
        spec = MultiScaleSpec(history_limit=32)
        observed = sample([1] * 60 + [0] * 32)
        truncated = sample([0] * 32)
        self.assertEqual(self.predict(observed, spec=spec), self.predict(truncated, spec=spec))

    def test_weak_first_suggestions_do_not_modify_fairness_scheduler(self):
        history = tuple(
            TrainingObservation("a" if i % 2 == 0 else "b", i, "basic", float(i % 2))
            for i in range(64)
        )
        spec = CurriculumSpec()
        state = CurriculumState(observations=history)
        before = next_training_case(spec, state, {"basic": ("a", "b", "c")}, run_seed=5)
        rankings = rank_training_challenges(history, level="basic", challenge_ids=("b", "a"))
        self.assertEqual(tuple(item.challenge_id for item in rankings), ("a", "b"))
        self.assertEqual(before, next_training_case(spec, state, {"basic": ("a", "b", "c")}, run_seed=5))

    def test_bad_spec_and_candidates_rejected(self):
        for spec in (
            {"scales": ()},
            {"scales": (4, 3)},
            {"scales": (4, 4)},
            {"scales": (1, 2)},
            {"scales": (4, 8), "history_limit": 4},
            {"learning_rate": float("nan")},
            {"learning_rate": True},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                MultiScaleSpec(**spec)
        for candidates in (("a", "a"), (), ("",), "a", iter(("a", "b"))):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                rank_training_challenges((), level="basic", challenge_ids=candidates)


if __name__ == "__main__":
    unittest.main()
