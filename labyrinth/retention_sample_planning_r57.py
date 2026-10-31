"""Source-only, preregistration-oriented sample planning for prior-skill retention.

The existing retention decision uses independent semantic situations as fixed
binomial units, with one-sided exact Clopper-Pearson bounds and familywise
Bonferroni alpha. A policy minimum of 64 cases is *not* a guarantee that a
given number of prior skills can pass even with zero deteriorations.

This module calculates a fixed-size necessary sample count for a stated loss
budget. The caller must separately preregister it BEFORE evaluation. Inputs
and results are untrusted planning data: no bank, evaluator, checkpoint,
signer, model execution, or host authority is accessed or authenticated.
"""

from __future__ import annotations

from collections import namedtuple
import math
import re

from labyrinth.retention_qualification import one_sided_exact_deterioration_upper


SCHEMA = "labyrinth.prior-skill-retention-source-sample-plan.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class RetentionSamplePlanningError(ValueError):
    """Invalid or impossible fixed-size retention sample requirements."""


def _derive_sample_numbers(
    prior_skill_ids: tuple[str, ...],
    anticipated_max_deteriorations: tuple[int, ...],
    significance_alpha: float,
    max_deterioration_rate: float,
    minimum_units_per_skill: int,
    maximum_units_per_skill: int,
) -> tuple[tuple[int, ...], float]:
    """Validate source policy and independently derive its exact numerical claims.

    Keep this free of sample-plan construction: __post_init__ and the public
    factory must compute the same result without constructor recursion.
    """
    if (type(prior_skill_ids) is not tuple
            or not 1 <= len(prior_skill_ids) <= 12
            or any(type(skill) is not str or _IDENTIFIER.fullmatch(skill) is None
                   for skill in prior_skill_ids)
            or len(set(prior_skill_ids)) != len(prior_skill_ids)):
        raise RetentionSamplePlanningError("one to 12 distinct prior skill IDs required")
    if (type(anticipated_max_deteriorations) is not tuple
            or len(anticipated_max_deteriorations) != len(prior_skill_ids)
            or any(type(loss) is not int or loss < 0
                   for loss in anticipated_max_deteriorations)):
        raise RetentionSamplePlanningError("exact nonnegative per-skill loss budgets required")
    if (type(minimum_units_per_skill) is not int
            or type(maximum_units_per_skill) is not int
            or not 64 <= minimum_units_per_skill <= maximum_units_per_skill <= 512):
        raise RetentionSamplePlanningError("sample limits must be integers within [64, 512]")
    for value, name in ((significance_alpha, "significance alpha"),
                        (max_deterioration_rate, "deterioration margin")):
        if (type(value) not in (float, int) or not math.isfinite(value)
                or not 0 < value <= 0.05):
            raise RetentionSamplePlanningError("invalid or weakened " + name)

    per_skill_alpha = significance_alpha / len(prior_skill_ids)
    required: list[int] = []
    for loss_budget in anticipated_max_deteriorations:
        count = next((n for n in range(minimum_units_per_skill,
                                       maximum_units_per_skill + 1)
                      if loss_budget <= n and one_sided_exact_deterioration_upper(
                          n, loss_budget, per_skill_alpha
                      ) <= max_deterioration_rate), None)
        if count is None:
            raise RetentionSamplePlanningError(
                "no fixed-size sample satisfies the loss budget within the bounded policy"
            )
        required.append(count)
    return tuple(required), per_skill_alpha


_RetentionPolicy = namedtuple(
    "_RetentionPolicy",
    ("prior_skill_ids", "anticipated_max_deteriorations", "significance_alpha",
     "max_deterioration_rate", "minimum_units_per_skill", "maximum_units_per_skill"),
)


class RetentionSamplePlan(_RetentionPolicy):
    """Immutable source proposal: only validated policy data are stored.

    A frozen dataclass is not sufficient here: Python's object.__setattr__
    can mutate its numeric and purported authority fields. A tuple has no
    writable instance state. Derived counts are computed from validated
    policy whenever requested; privilege assertions are constant properties,
    never constructor arguments or serialized member values.
    """

    __slots__ = ()

    def __new__(
        cls,
        prior_skill_ids: tuple[str, ...],
        anticipated_max_deteriorations: tuple[int, ...],
        significance_alpha: float = 0.05,
        max_deterioration_rate: float = 0.05,
        minimum_units_per_skill: int = 64,
        maximum_units_per_skill: int = 512,
    ):
        _derive_sample_numbers(
            prior_skill_ids, anticipated_max_deteriorations,
            significance_alpha, max_deterioration_rate,
            minimum_units_per_skill, maximum_units_per_skill,
        )
        return super().__new__(
            cls, prior_skill_ids, anticipated_max_deteriorations,
            float(significance_alpha), float(max_deterioration_rate),
            minimum_units_per_skill, maximum_units_per_skill,
        )

    def _replace(self, **kwargs):
        # Not a ledger of independently approved amendments. Regenerate a new
        # source proposal explicitly rather than treating a copy as attested.
        raise RetentionSamplePlanningError("retention plan cannot be replaced in place")

    def _numbers(self) -> tuple[tuple[int, ...], float]:
        return _derive_sample_numbers(
            self.prior_skill_ids, self.anticipated_max_deteriorations,
            self.significance_alpha, self.max_deterioration_rate,
            self.minimum_units_per_skill, self.maximum_units_per_skill,
        )

    @property
    def schema(self) -> str:
        return SCHEMA

    @property
    def required_units_per_skill(self) -> tuple[int, ...]:
        return self._numbers()[0]

    @property
    def total_required_units(self) -> int:
        return sum(self._numbers()[0])

    @property
    def per_skill_alpha(self) -> float:
        return self._numbers()[1]

    @property
    def status(self) -> str:
        return "SOURCE_ONLY_UNATTESTED_SAMPLE_PLAN"

    @property
    def precommitted_before_training(self) -> bool:
        return False

    @property
    def independent_case_custody_verified(self) -> bool:
        return False

    @property
    def model_execution_authenticated(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def plan_prior_skill_retention_samples(
    prior_skill_ids: tuple[str, ...],
    anticipated_max_deteriorations: tuple[int, ...],
    *,
    significance_alpha: float = 0.05,
    max_deterioration_rate: float = 0.05,
    minimum_units_per_skill: int = 64,
    maximum_units_per_skill: int = 512,
) -> RetentionSamplePlan:
    """Return minimum *fixed* independent units for each planned loss budget.

    For each skill, choose the first n in the bounded policy-supported range
    whose one-sided exact deterioration upper confidence bound at the
    *declared budget* is at most the margin. Bonferroni divides the alpha
    among all prior skills. If none exists, fail closed, not a false PASS.

    Choosing n after seeing examination results (optional stopping), using
    repeated/related semantic situations as independent units, or supplying
    the expected signatures from the same untrusted producer invalidates the
    resulting inference. This planner cannot authenticate those conditions.
    """
    return RetentionSamplePlan(
        prior_skill_ids=prior_skill_ids,
        anticipated_max_deteriorations=anticipated_max_deteriorations,
        significance_alpha=significance_alpha,
        max_deterioration_rate=max_deterioration_rate,
        minimum_units_per_skill=minimum_units_per_skill,
        maximum_units_per_skill=maximum_units_per_skill,
    )
