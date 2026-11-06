"""Independent numerical oracles for preregistered retention sample planning."""

import math
import unittest
from dataclasses import replace
from decimal import Decimal

from labyrinth.retention_sample_planning_r57 import (
    RetentionSamplePlan,
    RetentionSamplePlanningError,
    plan_prior_skill_retention_samples,
)


class RetentionSamplePlanningTests(unittest.TestCase):
    def test_single_prior_skill_default_floor_64_is_sufficient_with_zero_losses(self):
        plan = plan_prior_skill_retention_samples(("reasoning",), (0,))
        self.assertEqual(plan.required_units_per_skill, (64,))
        self.assertEqual(plan.total_required_units, 64)
        self.assertEqual(plan.per_skill_alpha, 0.05)

    def test_two_prior_skills_need_exactly_72_not_73_zero_loss_units(self):
        plan = plan_prior_skill_retention_samples(("math", "language"), (0, 0))
        self.assertEqual(plan.required_units_per_skill, (72, 72))
        self.assertEqual(plan.total_required_units, 144)
        # Independent closed-form zero-loss binomial upper bound.
        self.assertGreater(1 - math.pow(0.05 / 2, 1 / 71), 0.05)
        self.assertLess(1 - math.pow(0.05 / 2, 1 / 72), 0.05)

    def test_twelve_prior_skills_use_exact_bonferroni_not_uncorrected_alpha(self):
        plan = plan_prior_skill_retention_samples(
            tuple(f"skill{i}" for i in range(12)), (0,) * 12,
        )
        self.assertEqual(plan.required_units_per_skill, (107,) * 12)
        self.assertEqual(plan.total_required_units, 1284)

    def test_loss_budget_one_uses_exact_binomial_tail_independent_oracle(self):
        plan = plan_prior_skill_retention_samples(("math", "language"), (0, 1))
        n = plan.required_units_per_skill[1]
        self.assertGreater(n, plan.required_units_per_skill[0])
        def cdf_at_margin(units: int) -> float:
            return (0.95 ** units + units * 0.05 * 0.95 ** (units - 1))
        self.assertGreater(cdf_at_margin(n - 1), 0.025)
        self.assertLessEqual(cdf_at_margin(n), 0.025)

    def test_no_feasible_plan_fails_closed_at_bounded_cap(self):
        with self.assertRaises(RetentionSamplePlanningError):
            plan_prior_skill_retention_samples(("math", "language"), (0, 0),
                                               max_deterioration_rate=0.001)

    def test_invalid_skill_names_and_duplicates_rejected(self):
        for skills in ((), ("math", "math"), ("",), ("a/b",), ["math"]):
            with self.subTest(skills=skills), self.assertRaises(RetentionSamplePlanningError):
                plan_prior_skill_retention_samples(skills, (0,) * len(skills))

    def test_bad_loss_budgets_rejected_without_coercion(self):
        for losses in ((0,), (False, 0), (-1, 0), (1.0, 0), [0, 0], (600, 0)):
            with self.subTest(losses=losses), self.assertRaises(RetentionSamplePlanningError):
                plan_prior_skill_retention_samples(("math", "language"), losses)

    def test_invalid_alpha_margin_floor_ceiling_rejected(self):
        invalid = (
            {"significance_alpha": 0.06}, {"significance_alpha": float("nan")},
            {"significance_alpha": True},
            {"significance_alpha": Decimal("0.025")},
            {"max_deterioration_rate": Decimal("0.025")},
            {"max_deterioration_rate": 0.06},
            {"max_deterioration_rate": 0.0}, {"minimum_units_per_skill": 63},
            {"minimum_units_per_skill": True}, {"maximum_units_per_skill": 513},
            {"maximum_units_per_skill": 63},
        )
        for option in invalid:
            with self.subTest(option=option), self.assertRaises(RetentionSamplePlanningError):
                plan_prior_skill_retention_samples(("math",), (0,), **option)

    def test_non_default_stricter_policy_is_supported_but_not_weakened(self):
        plan = plan_prior_skill_retention_samples(
            ("math",), (0,), significance_alpha=0.025,
            max_deterioration_rate=0.025,
        )
        self.assertGreater(plan.required_units_per_skill[0], 64)
        self.assertEqual(plan.significance_alpha, 0.025)

    def test_result_is_frozen_nonauthorizing_and_does_not_claim_precommit(self):
        plan = plan_prior_skill_retention_samples(("math", "language"), (0, 0))
        self.assertEqual(plan.status, "SOURCE_ONLY_UNATTESTED_SAMPLE_PLAN")
        self.assertFalse(plan.model_execution_authenticated)
        self.assertFalse(plan.graduation_authorized)
        self.assertFalse(plan.independent_case_custody_verified)
        self.assertFalse(plan.precommitted_before_training)
        with self.assertRaises(AttributeError):
            plan.graduation_authorized = True
        for forged in (
            {"graduation_authorized": True},
            {"model_execution_authenticated": True},
            {"independent_case_custody_verified": True},
            {"precommitted_before_training": True},
            {"status": "PHYSICAL_RETENTION_VERIFIED"},
        ):
            with self.subTest(forged=forged), self.assertRaises(RetentionSamplePlanningError):
                plan._replace(**forged)
            with self.subTest(dataclass_forgery=forged), self.assertRaises(TypeError):
                replace(plan, **forged)

    def test_frozen_plan_recomputes_derived_numbers_and_policy_before_accepting(self):
        plan = plan_prior_skill_retention_samples(("math", "language"), (0, 0))
        self.assertEqual(plan.required_units_per_skill, (72, 72))
        attacks = (
            {"required_units_per_skill": (1, 1), "total_required_units": 2,
             "per_skill_alpha": 0.05},
            {"required_units_per_skill": (72, 73), "total_required_units": 145},
            {"total_required_units": 99},
            {"per_skill_alpha": 0.05},
            {"per_skill_alpha": float("nan")},
            {"minimum_units_per_skill": 80},
            {"maximum_units_per_skill": 70},
            {"prior_skill_ids": ("math", "math")},
            {"anticipated_max_deteriorations": (0, 1)},
            {"significance_alpha": 0.025},
            {"max_deterioration_rate": 0.025},
            {"required_units_per_skill": [72, 72]},
            {"total_required_units": True},
        )
        for forged in attacks:
            with self.subTest(forged=forged), self.assertRaises(RetentionSamplePlanningError):
                plan._replace(**forged)
            with self.subTest(dataclass_forgery=forged), self.assertRaises(TypeError):
                replace(plan, **forged)

    def test_direct_constructor_has_no_derived_or_privileged_fields(self):
        plan = RetentionSamplePlan(("math", "language"), (0, 0))
        self.assertEqual(plan.required_units_per_skill, (72, 72))
        self.assertEqual(plan.total_required_units, 144)
        for forged in (
            {"graduation_authorized": True},
            {"status": "QUALIFIED"},
            {"total_required_units": 1},
            {"required_units_per_skill": (1, 1)},
            {"per_skill_alpha": 0.05},
        ):
            with self.subTest(forged=forged), self.assertRaises(TypeError):
                RetentionSamplePlan(("math", "language"), (0, 0), **forged)
        with self.assertRaises(RetentionSamplePlanningError):
            RetentionSamplePlan(("math", "math"), (0, 0))
        with self.assertRaises(RetentionSamplePlanningError):
            RetentionSamplePlan(("math", "language"), (0, 0), 0.06)

    def test_object_setattr_cannot_forge_authority_or_sample_numbers(self):
        plan = plan_prior_skill_retention_samples(("math", "language"), (0, 0))
        before = (plan.status, plan.graduation_authorized,
                  plan.model_execution_authenticated,
                  plan.independent_case_custody_verified,
                  plan.precommitted_before_training,
                  plan.required_units_per_skill, plan.total_required_units,
                  plan.per_skill_alpha)
        for field, value in (
            ("status", "VERIFIED"),
            ("graduation_authorized", True),
            ("model_execution_authenticated", True),
            ("independent_case_custody_verified", True),
            ("precommitted_before_training", True),
            ("required_units_per_skill", (1, 1)),
            ("total_required_units", 1),
            ("per_skill_alpha", 0.05),
            ("prior_skill_ids", ("math", "math")),
        ):
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                object.__setattr__(plan, field, value)
        self.assertEqual((plan.status, plan.graduation_authorized,
                          plan.model_execution_authenticated,
                          plan.independent_case_custody_verified,
                          plan.precommitted_before_training,
                          plan.required_units_per_skill, plan.total_required_units,
                          plan.per_skill_alpha), before)


if __name__ == "__main__":
    unittest.main()
