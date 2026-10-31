"""Strict arithmetic retention is an explicit development-only entrypoint."""

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.retention_family_registry import (
    decide_retention_with_family_custody,
    sign_family_taxonomy,
)
from labyrinth.retention_qualification import (
    PROMPT_COMMITMENT_VERSION,
    RetentionObservation,
    RetentionPair,
    RetentionPolicy,
    _policy_sha256,
    normalized_prompt_sha256,
    retention_roster_sha256,
    sign_retention_development_evidence,
)
from labyrinth.retention_semantic_guard import (
    require_distinct_arithmetic_retention,
)
from labyrinth.strict_retention_development import (
    StrictRetentionDevelopmentError,
    evaluate_strict_arithmetic_retention,
)


EVAL_KEY = b"evaluator-independent-custody-key-1234567"
TAXONOMY_KEY = b"taxonomy-independent-custody-key-1234567"
NONCE = "a" * 64
SIGNER = "review-taxonomy"


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def roster_pair(i: int, before: str, after: str) -> RetentionPair:
    return RetentionPair(
        "arithmetic", f"situation-{i:04d}", f"family-{i:04d}",
        f"before-{i:04d}", f"after-{i:04d}", before, after,
        normalized_prompt_sha256(before), normalized_prompt_sha256(after),
    )


def distinct_roster() -> tuple[RetentionPair, ...]:
    return tuple(roster_pair(i, f"What is 0 plus {i}?", f"Compute 0 + {i}.")
                 for i in range(64))


def duplicate_fact_roster() -> tuple[RetentionPair, ...]:
    forms = []
    for left, right in (("2", "3"), ("2", "three"), ("two", "3"),
                        ("two", "three"), ("3", "2"), ("three", "2"),
                        ("3", "two"), ("three", "two")):
        for stem in (
            f"What is {left} plus {right}", f"What is {left} + {right}",
            f"Compute {left} plus {right}", f"Compute {left} + {right}",
            f"Add {left} to {right}", f"{left} plus {right}", f"{left} + {right}",
        ):
            for end in ("", "?", "."):
                forms.append(stem + end)
    return tuple(roster_pair(i, forms[2 * i], forms[2 * i + 1])
                 for i in range(64))


def campaign(roster: tuple[RetentionPair, ...]):
    training = "What is 99 plus 99?"
    policy = RetentionPolicy(
        "prior-arithmetic", digest("scope"), digest("before"), digest("after"),
        digest("evaluator"), digest("conditions"), "f" * 64,
        ("arithmetic",), retention_roster_sha256(roster),
        PROMPT_COMMITMENT_VERSION, (training,),
        (normalized_prompt_sha256(training),), ("training-family",),
    )
    observations = tuple(RetentionObservation(p.situation_id, True, True)
                         for p in roster)
    evidence = sign_retention_development_evidence(
        policy, roster, observations, independent_evaluator_key=EVAL_KEY)
    taxonomy = sign_family_taxonomy(
        policy, roster, signer_id=SIGNER, precommitted_nonce_sha256=NONCE,
        independent_taxonomy_key=TAXONOMY_KEY,
        independent_evaluator_key=EVAL_KEY)
    return policy, evidence, taxonomy


class StrictRetentionDevelopmentTests(unittest.TestCase):
    def evaluate(self, policy, roster, evidence, taxonomy, *, semantic=None,
                 policy_pin=None, nonce=NONCE):
        preflight = require_distinct_arithmetic_retention(policy, roster)
        return evaluate_strict_arithmetic_retention(
            policy, roster, evidence, taxonomy,
            expected_policy_sha256=policy_pin or _policy_sha256(policy),
            expected_roster_sha256=retention_roster_sha256(roster),
            expected_semantic_inventory_sha256=(
                semantic or preflight.semantic_inventory_sha256),
            expected_signer_id=SIGNER, expected_nonce_sha256=nonce,
            independent_taxonomy_key=TAXONOMY_KEY,
            independent_evaluator_key=EVAL_KEY,
        )

    def test_valid_diverse_arithmetic_returns_only_provisional_development(self):
        roster = distinct_roster()
        policy, evidence, taxonomy = campaign(roster)
        result = self.evaluate(policy, roster, evidence, taxonomy)
        self.assertEqual(result.development_decision.disposition, "PASS")
        self.assertEqual(result.semantic_case_count, 64)
        self.assertEqual(result.policy_sha256, _policy_sha256(policy))
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertFalse(result.independent_semantics_authenticated)
        self.assertFalse(result.examiner_custody_verified)

    def test_sixty_four_distinct_aliases_of_one_fact_rejected_before_scoring(self):
        roster = duplicate_fact_roster()
        policy, evidence, taxonomy = campaign(roster)
        # The legacy signed development scorer can pass this synthetic result.
        self.assertEqual(decide_retention_with_family_custody(
            policy, roster, evidence, taxonomy, expected_signer_id=SIGNER,
            expected_nonce_sha256=NONCE,
            independent_taxonomy_key=TAXONOMY_KEY,
            independent_evaluator_key=EVAL_KEY).disposition, "PASS")
        with self.assertRaises(StrictRetentionDevelopmentError):
            evaluate_strict_arithmetic_retention(
                policy, roster, evidence, taxonomy,
                expected_policy_sha256=_policy_sha256(policy),
                expected_roster_sha256=retention_roster_sha256(roster),
                expected_semantic_inventory_sha256=digest("not-a-real-semantic-inventory"),
                expected_signer_id=SIGNER, expected_nonce_sha256=NONCE,
                independent_taxonomy_key=TAXONOMY_KEY,
                independent_evaluator_key=EVAL_KEY)

    def test_operator_pinned_semantic_and_policy_commitments_cannot_change(self):
        roster = distinct_roster()
        policy, evidence, taxonomy = campaign(roster)
        with self.assertRaisesRegex(StrictRetentionDevelopmentError, "semantic"):
            self.evaluate(policy, roster, evidence, taxonomy,
                          semantic=digest("wrong semantic inventory"))
        with self.assertRaisesRegex(StrictRetentionDevelopmentError, "policy"):
            self.evaluate(policy, roster, evidence, taxonomy,
                          policy_pin=digest("wrong policy"))

        with self.assertRaisesRegex(StrictRetentionDevelopmentError, "roster"):
            evaluate_strict_arithmetic_retention(
                policy, roster, evidence, taxonomy,
                expected_policy_sha256=_policy_sha256(policy),
                expected_roster_sha256=digest("wrong roster"),
                expected_semantic_inventory_sha256=(
                    require_distinct_arithmetic_retention(
                        policy, roster).semantic_inventory_sha256),
                expected_signer_id=SIGNER, expected_nonce_sha256=NONCE,
                independent_taxonomy_key=TAXONOMY_KEY,
                independent_evaluator_key=EVAL_KEY)

    def test_wrong_signer_nonce_and_evaluator_key_fail_closed(self):
        roster = distinct_roster()
        policy, evidence, taxonomy = campaign(roster)
        with self.assertRaises(ValueError):
            self.evaluate(policy, roster, evidence, taxonomy, nonce="b" * 64)
        with self.assertRaises(ValueError):
            evaluate_strict_arithmetic_retention(
                policy, roster, evidence, taxonomy,
                expected_policy_sha256=_policy_sha256(policy),
                expected_roster_sha256=retention_roster_sha256(roster),
                expected_semantic_inventory_sha256=(
                    require_distinct_arithmetic_retention(
                        policy, roster).semantic_inventory_sha256),
                expected_signer_id=SIGNER, expected_nonce_sha256=NONCE,
                independent_taxonomy_key=TAXONOMY_KEY,
                independent_evaluator_key=b"different-evaluator-key-123456789012345",
            )

    def test_mutated_retention_policy_cannot_widen_signed_threshold(self):
        roster = distinct_roster()
        policy, evidence, taxonomy = campaign(roster)
        original_sha = _policy_sha256(policy)
        object.__setattr__(policy, "max_deterioration_rate", 1.0)
        with self.assertRaises(ValueError):
            evaluate_strict_arithmetic_retention(
                policy, roster, evidence, taxonomy,
                expected_policy_sha256=original_sha,
                expected_roster_sha256=retention_roster_sha256(roster),
                expected_semantic_inventory_sha256=digest("arbitrary"),
                expected_signer_id=SIGNER, expected_nonce_sha256=NONCE,
                independent_taxonomy_key=TAXONOMY_KEY,
                independent_evaluator_key=EVAL_KEY,
            )

    def test_signed_receipts_cannot_claim_execution_or_graduation(self):
        roster = distinct_roster()
        policy, evidence, taxonomy = campaign(roster)
        with self.assertRaises(ValueError):
            self.evaluate(policy, roster,
                          replace(evidence, graduation_authorized=True), taxonomy)
        with self.assertRaises(ValueError):
            self.evaluate(policy, roster, evidence,
                          replace(taxonomy, model_execution_authenticated=True))


if __name__ == "__main__":
    unittest.main()
