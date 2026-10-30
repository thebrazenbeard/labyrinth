"""Opt-in, causal multi-scale forecasting of *training* scores.

This is a curriculum diagnostic, not a neural optimizer or a claim that model
reasoning improves. Neither evaluation/control observations nor future scores
may inform a forecast. No default curriculum routing or promotion rule changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .curriculum import TrainingObservation


@dataclass(frozen=True)
class MultiScaleSpec:
    """Dyadic trailing-window experts and bounded online error adaptation."""

    scales: tuple[int, ...] = (4, 8, 16, 32)
    learning_rate: float = 4.0
    history_limit: int = 256

    def __post_init__(self) -> None:
        if (
            not self.scales
            or any(type(k) is not int or k < 2 or k & (k - 1) for k in self.scales)
            or tuple(sorted(set(self.scales))) != self.scales
        ):
            raise ValueError("scales must be distinct, ascending powers of two >= 2")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (float, int))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.learning_rate > 100
        ):
            raise ValueError("learning_rate must be finite and in (0, 100]")
        if type(self.history_limit) is not int or not self.scales[-1] <= self.history_limit <= 4096:
            raise ValueError("history_limit must include the longest scale and be <= 4096")


@dataclass(frozen=True)
class TrainingForecast:
    """Bounded score estimate; uncertainty is a heuristic, not a calibrated CI."""

    level: str
    challenge_id: str
    expected_score: float
    uncertainty: float
    dyadic_roughness: float
    sample_count: int
    scales: tuple[int, ...]
    expert_weights: tuple[float, ...]


def _expert_estimate(
    prior: Sequence[TrainingObservation], challenge_id: str, window: int
) -> float:
    recent = prior[-window:]
    if not recent:
        return 0.5
    # Beta(1,1) shrinkage for the level-wide rate. Challenge-level history
    # shrinks toward that rate, rather than overfitting one observation.
    level_mean = (1.0 + sum(o.score for o in recent)) / (len(recent) + 2)
    matching = [o.score for o in recent if o.challenge_id == challenge_id]
    return (2.0 * level_mean + sum(matching)) / (2.0 + len(matching))


def _roughness(prior: Sequence[TrainingObservation], longest: int) -> float:
    scores = [o.score for o in prior[-longest:]]
    if len(scores) < 2:
        return 0.0
    # Mean absolute score change at octave-separated lags. This is NOT a
    # Higuchi fractal dimension or evidence of self-similar dynamics.
    lags = (1, 2, 4, 8)
    changes = [
        sum(abs(scores[i] - scores[i - lag]) for i in range(lag, len(scores)))
        / (len(scores) - lag)
        for lag in lags
        if lag < len(scores)
    ]
    return sum(changes) / len(changes)


def forecast_training_score(
    observations: Sequence[TrainingObservation],
    *,
    level: str,
    challenge_id: str,
    spec: MultiScaleSpec | None = None,
) -> TrainingForecast:
    """Forecast the *next* training score using past same-level observations.

    All input observations are validated before filtering; any eval/placebo/
    control item fails closed even if it belongs to another level. Order is the
    caller's chronological training order. Mixture weights replay experts on
    prior observations only, with no hindsight or holdout-derived tuning.
    """
    if not isinstance(level, str) or not level or not isinstance(challenge_id, str) or not challenge_id:
        raise ValueError("level and challenge_id must be nonempty strings")
    if spec is None:
        spec = MultiScaleSpec()
    if not isinstance(spec, MultiScaleSpec):
        raise ValueError("spec must be MultiScaleSpec")
    MultiScaleSpec.__post_init__(spec)
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("observations must be a sequence of training records")
    for observation in observations:
        if not isinstance(observation, TrainingObservation) or observation.split != "train":
            raise ValueError("forecasts accept training observations only")
        # Dataclass freezing is not a security boundary: restored or mutated
        # objects can bypass construction-time checks. Recheck all values
        # before any arithmetic to reject NaN/inf and malformed identifiers.
        TrainingObservation.__post_init__(observation)

    history = [o for o in observations if o.level == level][-spec.history_limit :]
    losses = [0.0] * len(spec.scales)
    prior: list[TrainingObservation] = []
    for observation in history:
        for index, scale in enumerate(spec.scales):
            estimate = _expert_estimate(prior, observation.challenge_id, scale)
            # Discount stale mistakes so an expert can recover after a shift.
            losses[index] = 0.9 * losses[index] + (estimate - observation.score) ** 2
        prior.append(observation)

    minimum_loss = min(losses)
    unnormalized = [math.exp(-spec.learning_rate * (loss - minimum_loss)) for loss in losses]
    total = sum(unnormalized)
    weights = tuple(weight / total for weight in unnormalized)
    estimates = [_expert_estimate(prior, challenge_id, scale) for scale in spec.scales]
    expected = sum(weight * estimate for weight, estimate in zip(weights, estimates))
    disagreement = sum(weight * (estimate - expected) ** 2 for weight, estimate in zip(weights, estimates))
    residual = sum(weight * loss for weight, loss in zip(weights, losses))
    uncertainty = math.sqrt(min(1.0, disagreement + residual)) if prior else 0.5
    return TrainingForecast(
        level=level,
        challenge_id=challenge_id,
        expected_score=expected,
        uncertainty=uncertainty,
        dyadic_roughness=_roughness(prior, spec.scales[-1]),
        sample_count=len(prior),
        scales=spec.scales,
        expert_weights=weights,
    )


def rank_training_challenges(
    observations: Sequence[TrainingObservation],
    *,
    level: str,
    challenge_ids: Sequence[str],
    spec: MultiScaleSpec | None = None,
) -> tuple[TrainingForecast, ...]:
    """Read-only weak-first suggestions; does not change curriculum fairness."""
    if (
        not isinstance(challenge_ids, Sequence)
        or isinstance(challenge_ids, (str, bytes))
        or not challenge_ids
        or any(not isinstance(item, str) or not item for item in challenge_ids)
        or len(set(challenge_ids)) != len(challenge_ids)
    ):
        raise ValueError("challenge_ids must be distinct nonempty strings")
    forecasts = [
        forecast_training_score(observations, level=level, challenge_id=item, spec=spec)
        for item in challenge_ids
    ]
    return tuple(sorted(forecasts, key=lambda item: (item.expected_score, item.challenge_id)))
