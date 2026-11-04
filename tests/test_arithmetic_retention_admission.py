"""Narrow arithmetic semantic veto: source integrity is not host custody."""

from dataclasses import replace
import unittest

from labyrinth.arithmetic_retention_admission import (
    arithmetic_training_inventory_sha256,
    evaluate_arithmetic_retention_admission,
)
from labyrinth.retention_entrypoint_guard import evaluate_retention_admission
from labyrinth.retention_qualification import (
    RetentionObservation, normalized_prompt_sha256, retention_roster_sha256,
    sign_retention_development_evidence,
)
from labyrinth.retention_family_registry import sign_family_taxonomy
from tests import test_retention_entrypoint_guard as retention_fixtures


class ArithmeticRetentionAdmissionTests(unittest.TestCase):
    def fixture(self, *, overlap=False, duplicate=False, wrong_pair=False,
                unsupported=False, training_duplicate=False):
        donor = retention_fixtures.RetentionEntrypointGuardTests(
            "test_valid_distinct_signed_development_evidence_passes_only_veto")
        request, policy, roster, _, _, options = donor.campaign()
        training = (("What is seven plus five?", "Compute 7 + 5.")
                    if training_duplicate else ("What is seven plus five?",))
        policy = replace(
            policy,
            excluded_training_prompts=training,
            excluded_training_prompt_sha256s=tuple(
                normalized_prompt_sha256(prompt) for prompt in training),
        )
        rebuilt = []
        for i, original in enumerate(roster):
            before = f"What is {i} plus 2?"
            after = f"Compute 2 + {i}."
            if overlap and i == 0:
                before, after = "Compute 7 + 5.", "Add five to seven."
            if duplicate and i == 1:
                before, after = "Add zero to two.", "What is 2 + zero?"
            if wrong_pair and i == 0:
                after = "Compute 3 + 9."
            if unsupported and i == 0:
                before = "Explain addition with seven and five."
            rebuilt.append(replace(
                original, before_prompt_text=before, after_prompt_text=after,
                before_prompt_sha256=normalized_prompt_sha256(before),
                after_prompt_sha256=normalized_prompt_sha256(after),
            ))
        roster = tuple(rebuilt)
        policy = replace(policy, expected_roster_sha256=retention_roster_sha256(roster))
        evidence = sign_retention_development_evidence(
            policy, roster, tuple(
                RetentionObservation(pair.situation_id, True, True)
                for pair in roster),
            independent_evaluator_key=donor.EVALUATOR_KEY,
        )
        taxonomy = sign_family_taxonomy(
            policy, roster, signer_id="independent-taxonomy-custodian",
            precommitted_nonce_sha256=donor.TAXONOMY_NONCE,
            independent_taxonomy_key=donor.TAXONOMY_KEY,
            independent_evaluator_key=donor.EVALUATOR_KEY,
        )
        options.update(
            trusted_training_inventory_sha256=arithmetic_training_inventory_sha256(training),
            trusted_retention_roster_sha256=retention_roster_sha256(roster),
        )
        return request, policy, roster, evidence, taxonomy, options

    def evaluate(self, **changes):
        overlap = changes.pop("overlap", False)
        duplicate = changes.pop("duplicate", False)
        wrong_pair = changes.pop("wrong_pair", False)
        unsupported = changes.pop("unsupported", False)
        training_duplicate = changes.pop("training_duplicate", False)
        request, policy, roster, evidence, taxonomy, options = self.fixture(
            overlap=overlap, duplicate=duplicate, wrong_pair=wrong_pair,
            unsupported=unsupported, training_duplicate=training_duplicate,
        )
        options.update(changes)
        return evaluate_arithmetic_retention_admission(
            request, policy, roster, evidence, taxonomy, **options)

    def test_disjoint_signed_64_pair_development_veto_passes_only_source_gate(self):
        outcome = self.evaluate()
        self.assertEqual(outcome.status, "PASS")
        self.assertFalse(outcome.graduation_authorized)
        self.assertFalse(outcome.model_execution_authenticated)

    def test_semantic_train_retention_overlap_rejected_despite_different_families(self):
        request, policy, roster, evidence, taxonomy, options = self.fixture(overlap=True)
        legacy_options = {key: value for key, value in options.items()
                          if not key.startswith("trusted_training_inventory")
                          and key != "trusted_retention_roster_sha256"}
        # The old signed family/taxonomy gate accepts distinct text/family IDs
        # even though the underlying arithmetic instance is identical.
        self.assertEqual(evaluate_retention_admission(
            request, policy, roster, evidence, taxonomy,
            **legacy_options).status, "PASS")
        outcome = self.evaluate(overlap=True)
        self.assertEqual(outcome.status, "BLOCKED")
        self.assertIn("semantic", outcome.reason)

    def test_duplicate_semantic_situation_across_family_aliases_rejected(self):
        self.assertEqual(self.evaluate(duplicate=True).status, "BLOCKED")

    def test_paired_forms_must_represent_same_arithmetic_situation(self):
        self.assertEqual(self.evaluate(wrong_pair=True).status, "BLOCKED")

    def test_unrecognized_arithmetic_grammar_fails_closed(self):
        self.assertEqual(self.evaluate(unsupported=True).status, "BLOCKED")

    def test_duplicate_training_semantic_case_cannot_hide_under_aliases(self):
        self.assertEqual(self.evaluate(training_duplicate=True).status, "BLOCKED")

    def test_missing_or_wrong_independent_inventory_precommit_blocks(self):
        for changes in (
            {"trusted_training_inventory_sha256": None},
            {"trusted_retention_roster_sha256": None},
            {"trusted_training_inventory_sha256": "0" * 64},
            {"trusted_retention_roster_sha256": "0" * 64},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.evaluate(**changes).status, "BLOCKED")

    def test_signed_score_tamper_is_still_blocked(self):
        request, policy, roster, evidence, taxonomy, options = self.fixture()
        tampered = replace(evidence, signature_hex="f" * 64)
        outcome = evaluate_arithmetic_retention_admission(
            request, policy, roster, tampered, taxonomy, **options)
        self.assertEqual(outcome.status, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
