"""Public-only formal objective metamorphic negative tests, no private labels."""

from dataclasses import replace
import unittest

from labyrinth.trainingday_objective_metamorphic import (
    FormalObjectiveContract, FormalSkillCriterion, PublicDevelopmentObservation,
    compare_public_objective_metamorphs,
)


SKILLS = (
    FormalSkillCriterion("program_steps", "exact_state", 2, 2),
    FormalSkillCriterion("source_priority", "source_rank", 2, 1),
)
FROZEN = FormalObjectiveContract(
    "objective-algebra", "skill_mastery", SKILLS, ("private_memory",),
    "a" * 64, "b" * 64, "c" * 64,
    "Practice program steps and source precedence",
)
CASES = (
    PublicDevelopmentObservation("case-a", "program_steps", True, True),
    PublicDevelopmentObservation("case-b", "program_steps", True, True),
    PublicDevelopmentObservation("case-c", "source_priority", True, True),
    PublicDevelopmentObservation("case-d", "source_priority", False, True),
)


def check(variant=FROZEN, original=CASES, observed=CASES):
    return compare_public_objective_metamorphs(FROZEN, variant, original, observed)


class ObjectiveMetamorphicPublicTests(unittest.TestCase):
    def test_permuted_formal_skill_and_case_order_preserve_public_mastery_only(self):
        changed = replace(FROZEN, skills=tuple(reversed(SKILLS)),
                          presentation_label="Different prose; formal IDs unchanged")
        v = check(variant=changed, observed=tuple(reversed(CASES)))
        self.assertEqual(v.status, "PUBLIC_FORMAL_INVARIANT_ONLY")
        self.assertTrue(v.original_public_mastery)
        self.assertEqual(v.original_public_mastery, v.variant_public_mastery)
        self.assertFalse(v.graduation_authorized)
        self.assertFalse(v.promotion_allowed)
        self.assertIsNotNone(v.formal_fingerprint)

    def test_objective_expansion_rejected_even_with_reused_scores(self):
        added = FormalSkillCriterion("empathy", "public_roleplay", 2, 2)
        v = check(variant=replace(FROZEN, skills=SKILLS + (added,)))
        self.assertEqual(v.status, "SCOPE_OR_RUBRIC_CHANGED")
        self.assertIsNone(v.variant_public_mastery)

    def test_hidden_rubric_or_evaluator_change_requires_new_protocol(self):
        variants = (
            replace(FROZEN, skills=(replace(SKILLS[0], min_correct=1), SKILLS[1])),
            replace(FROZEN, evaluator_sha256="d" * 64),
            replace(FROZEN, conditions_sha256="e" * 64),
            replace(FROZEN, objective_kind="training_effect"),
            replace(FROZEN, excluded_skill_ids=()),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(check(variant).status, "SCOPE_OR_RUBRIC_CHANGED")

    def test_public_case_grade_mutation_is_not_paraphrase_invariance(self):
        switched = list(CASES)
        switched[0] = replace(CASES[0], correct=False)
        switched[1] = replace(CASES[1], correct=True)
        v = check(observed=tuple(switched))
        self.assertEqual(v.status, "PUBLIC_CASE_MUTATION")
        self.assertFalse(v.graduation_authorized)

    def test_private_or_training_split_cannot_be_passed_off_as_public(self):
        for split in ("train", "private_exam", "heldout"):
            with self.subTest(split=split):
                altered = (replace(CASES[0], split=split),) + CASES[1:]
                self.assertEqual(check(observed=altered).status,
                                 "UNSUPPORTED_OBSERVATION_SPLIT")

    def test_duplicates_and_ambiguous_criteria_fail_closed(self):
        repeat_skill = replace(FROZEN, skills=(SKILLS[0], SKILLS[0]))
        self.assertEqual(check(repeat_skill).status, "MALFORMED_FORMAL_CONTRACT")
        repeated_case = CASES[:2] + (replace(CASES[2], case_id="case-b"),) + CASES[3:]
        self.assertEqual(check(observed=repeated_case).status,
                         "MALFORMED_PUBLIC_OBSERVATIONS")

    def test_public_practice_failure_remains_non_graduation(self):
        failed = (replace(CASES[0], correct=False),) + CASES[1:]
        v = check(original=failed, observed=tuple(reversed(failed)))
        self.assertEqual(v.status, "PUBLIC_FORMAL_INVARIANT_ONLY")
        self.assertFalse(v.original_public_mastery)
        self.assertFalse(v.variant_public_mastery)
        self.assertFalse(v.graduation_authorized)
        self.assertFalse(v.promotion_allowed)

    def test_reassigned_case_labels_rejected_despite_equal_aggregate(self):
        switched = list(CASES)
        switched[2] = replace(CASES[2], correct=False)
        switched[3] = replace(CASES[3], correct=True)
        # The total and skill-level public scores did not change. That does
        # not turn different case-level outcomes into an equivalent exam.
        self.assertEqual(check(observed=tuple(switched)).status,
                         "PUBLIC_CASE_MUTATION")

    def test_unrecognized_skill_and_case_subclass_do_not_pass(self):
        other = (replace(CASES[0], skill_id="unrelated"),) + CASES[1:]
        self.assertEqual(check(observed=other).status,
                         "MALFORMED_PUBLIC_OBSERVATIONS")

        class MutableCase(PublicDevelopmentObservation):
            pass

        subclass = (MutableCase("case-a", "program_steps", True, True),) + CASES[1:]
        self.assertEqual(check(observed=subclass).status,
                         "MALFORMED_PUBLIC_OBSERVATIONS")

    def test_ambiguous_exclusions_and_metric_threshold_types_rejected(self):
        bad = (
            replace(FROZEN, excluded_skill_ids=("program_steps",)),
            replace(FROZEN, excluded_skill_ids=("private_memory", "private_memory")),
            replace(FROZEN, skills=(replace(SKILLS[0], min_cases=True), SKILLS[1])),
            replace(FROZEN, skills=(replace(SKILLS[0], min_correct=10**1000), SKILLS[1])),
            replace(FROZEN, public_context_sha256="not-a-hash"),
        )
        for variant in bad:
            with self.subTest(variant=variant):
                self.assertEqual(check(variant=variant).status,
                                 "MALFORMED_FORMAL_CONTRACT")

    def test_overloaded_objective_kind_cannot_be_used_as_enum_or_json_value(self):
        class AnyEqual:
            def __eq__(self, other):
                return True

        class SpoofStr(str):
            def __eq__(self, other):
                return True

        for kind in (AnyEqual(), SpoofStr("not-a-real-kind"), True, None):
            with self.subTest(kind=repr(kind)):
                # A caller-supplied exotic object must be a controlled HOLD,
                # not an uncaught serialization error after enum membership.
                report = check(variant=replace(FROZEN, objective_kind=kind))
                self.assertEqual(report.status, "MALFORMED_FORMAL_CONTRACT")
                self.assertFalse(report.graduation_authorized)
                self.assertFalse(report.promotion_allowed)


if __name__ == "__main__":
    unittest.main()
