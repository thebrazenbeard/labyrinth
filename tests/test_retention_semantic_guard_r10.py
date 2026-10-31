"""Red/green tests for a narrow arithmetic-only retention preflight."""

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.retention_qualification import (
    PROMPT_COMMITMENT_VERSION,
    RetentionObservation,
    RetentionPair,
    RetentionPolicy,
    decide_retention,
    normalized_prompt_sha256,
    retention_roster_sha256,
    sign_retention_development_evidence,
)
from labyrinth.retention_semantic_guard import (
    StrictArithmeticRetentionError,
    require_distinct_arithmetic_retention,
)


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def pair(number: int, before: str, after: str) -> RetentionPair:
    return RetentionPair(
        skill_id="arithmetic",
        situation_id=f"situation-{number:04d}",
        structure_family_id=f"family-{number:04d}",
        before_case_id=f"before-{number:04d}",
        after_case_id=f"after-{number:04d}",
        before_prompt_text=before,
        after_prompt_text=after,
        before_prompt_sha256=normalized_prompt_sha256(before),
        after_prompt_sha256=normalized_prompt_sha256(after),
    )


def campaign(roster: tuple[RetentionPair, ...], training: str = "What is 99 plus 99?"):
    policy = RetentionPolicy(
        objective_id="arithmetic-retention",
        scope_sha256=digest("scope"),
        before_checkpoint_sha256=digest("before"),
        after_checkpoint_sha256=digest("after"),
        evaluator_sha256=digest("evaluator"),
        conditions_sha256=digest("conditions"),
        session_nonce="f" * 64,
        prior_skill_ids=("arithmetic",),
        expected_roster_sha256=retention_roster_sha256(roster),
        prompt_commitment_version=PROMPT_COMMITMENT_VERSION,
        excluded_training_prompts=(training,),
        excluded_training_prompt_sha256s=(normalized_prompt_sha256(training),),
        excluded_training_structure_family_ids=("training",),
    )
    return policy


def diverse_roster() -> tuple[RetentionPair, ...]:
    return tuple(pair(i, f"What is 0 plus {i}?", f"Compute 0 + {i}.")
                 for i in range(64))


def same_fact_roster() -> tuple[RetentionPair, ...]:
    """128 distinct supported wordings; all still mean 2+3."""
    variants = []
    for left, right in (("2", "3"), ("2", "three"), ("two", "3"),
                        ("two", "three"), ("3", "2"), ("three", "2"),
                        ("3", "two"), ("three", "two")):
        for form in (
            f"What is {left} plus {right}",
            f"What is {left} + {right}",
            f"Compute {left} plus {right}",
            f"Compute {left} + {right}",
            f"Add {left} to {right}",
            f"{left} plus {right}",
            f"{left} + {right}",
        ):
            for suffix in ("", "?", "."):
                variants.append(form + suffix)
    assert len(set(variants)) >= 128
    return tuple(pair(i, variants[2 * i], variants[2 * i + 1])
                 for i in range(64))


class ArithmeticRetentionGuardTests(unittest.TestCase):
    def test_prior_source_can_pass_sixty_four_reference_labels_on_one_fact(self):
        roster = tuple(pair(
            i, f"What is 2 plus 2? Reference {i}",
            f"Compute 2 + 2. Edition {i}",
        ) for i in range(64))
        policy = campaign(roster)
        observations = tuple(RetentionObservation(p.situation_id, True, True)
                             for p in roster)
        key = b"controlled-development-evaluator-12345678"
        receipt = sign_retention_development_evidence(
            policy, roster, observations, independent_evaluator_key=key)
        self.assertEqual(decide_retention(
            policy, roster, receipt, independent_evaluator_key=key).disposition,
            "PASS")
        with self.assertRaises(StrictArithmeticRetentionError):
            require_distinct_arithmetic_retention(policy, roster)

    def test_paraphrased_commutative_same_fact_is_not_sixty_four_units(self):
        roster = same_fact_roster()
        policy = campaign(roster)
        with self.assertRaisesRegex(StrictArithmeticRetentionError, "duplicate"):
            require_distinct_arithmetic_retention(policy, roster)

    def test_sixty_four_distinct_supported_facts_pass_only_source_preflight(self):
        roster = diverse_roster()
        result = require_distinct_arithmetic_retention(campaign(roster), roster)
        self.assertEqual(result.semantic_case_count, 64)
        self.assertEqual(len(result.semantic_inventory_sha256), 64)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.independent_semantics_authenticated)

    def test_training_overlap_rejected_by_semantic_key(self):
        roster = diverse_roster()
        policy = campaign(roster, "Add 0 to 9?")
        with self.assertRaisesRegex(StrictArithmeticRetentionError, "training"):
            require_distinct_arithmetic_retention(policy, roster)

    def test_before_after_must_describe_same_computation(self):
        roster = diverse_roster()
        revised = (replace(roster[0], after_prompt_text="Compute 1 + 1.",
                           after_prompt_sha256=normalized_prompt_sha256("Compute 1 + 1.")),
                   *roster[1:])
        policy = campaign(revised)
        with self.assertRaisesRegex(StrictArithmeticRetentionError, "paired"):
            require_distinct_arithmetic_retention(policy, revised)

    def test_unapproved_prior_skill_does_not_inherit_arithmetic_authority(self):
        roster = diverse_roster()
        policy = campaign(roster)
        with self.assertRaisesRegex(StrictArithmeticRetentionError, "arithmetic"):
            require_distinct_arithmetic_retention(
                replace(policy, prior_skill_ids=("empathy",)), roster)


if __name__ == "__main__":
    unittest.main()
