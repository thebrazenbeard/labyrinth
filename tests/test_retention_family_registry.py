"""Source-only hostile checks for an opt-in retention family custody companion."""

from dataclasses import replace
from hashlib import sha256
from unittest import TestCase

from labyrinth.retention_family_registry import (
    FamilyTaxonomyError,
    SignedFamilyTaxonomy,
    canonical_family_id,
    canonical_prompt_alias_sha256,
    decide_retention_with_family_custody,
    sign_family_taxonomy,
    verify_family_taxonomy,
)
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


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class FamilyRegistryTests(TestCase):
    EVALUATOR = b"independent-retention-exam-evaluator-key-1234567"
    TAXONOMY = b"independent-retention-taxonomy-signer-key-1234567"
    NONCE = "b" * 64

    def campaign(self):
        roster = tuple(RetentionPair(
            skill_id="arithmetic", situation_id=f"situation-{i:04d}",
            structure_family_id=f"review-family-{i:04d}",
            before_case_id=f"before-{i:04d}", after_case_id=f"after-{i:04d}",
            before_prompt_text=f"before arithmetic {i}",
            after_prompt_text=f"after arithmetic {i}",
            before_prompt_sha256=normalized_prompt_sha256(f"before arithmetic {i}"),
            after_prompt_sha256=normalized_prompt_sha256(f"after arithmetic {i}"),
        ) for i in range(64))
        policy = RetentionPolicy(
            objective_id="arithmetic-retention", scope_sha256=digest("scope"),
            before_checkpoint_sha256=digest("before"),
            after_checkpoint_sha256=digest("after"),
            evaluator_sha256=digest("evaluator"),
            conditions_sha256=digest("conditions"),
            session_nonce="a" * 64, prior_skill_ids=("arithmetic",),
            expected_roster_sha256=retention_roster_sha256(roster),
            prompt_commitment_version=PROMPT_COMMITMENT_VERSION,
            excluded_training_prompts=("train-prompt",),
            excluded_training_prompt_sha256s=(normalized_prompt_sha256("train-prompt"),),
            excluded_training_structure_family_ids=("train-family",),
        )
        observations = tuple(RetentionObservation(
            situation_id=pair.situation_id,
            before_correct=True, after_correct=True,
        ) for pair in roster)
        return policy, roster, observations

    def sign(self, policy, roster):
        return sign_family_taxonomy(
            policy, roster, signer_id="external-taxonomy-1",
            precommitted_nonce_sha256=self.NONCE,
            independent_taxonomy_key=self.TAXONOMY,
            independent_evaluator_key=self.EVALUATOR,
        )

    def verify(self, signed, policy, roster, **overrides):
        params = dict(
            expected_signer_id="external-taxonomy-1",
            expected_nonce_sha256=self.NONCE,
            independent_taxonomy_key=self.TAXONOMY,
            independent_evaluator_key=self.EVALUATOR,
        )
        params.update(overrides)
        return verify_family_taxonomy(signed, policy, roster, **params)

    def test_canonical_family_aliases_fold_case_separators_and_hyphens(self):
        variants = ("Train-Family", "train_family", "train.family",
                    "train:family", " train family ", "train\u2010family",
                    "ＴＲＡＩＮ－ＦＡＭＩＬＹ")
        self.assertEqual({canonical_family_id(item) for item in variants},
                         {"train-family"})

    def test_rejects_invisible_unicode_non_ascii_homoglyph_and_invalid_family(self):
        for bad in ("train\u200bfamily", "train\u2060family", "tr\u0430in-family",
                    "", "__", "../escape", "a/b", "family\x00name"):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                canonical_family_id(bad)

    def test_clean_taxonomy_is_signed_and_development_only(self):
        policy, roster, observations = self.campaign()
        signed = self.sign(policy, roster)
        self.assertIsInstance(signed, SignedFamilyTaxonomy)
        self.assertFalse(signed.graduation_authorized)
        self.assertFalse(signed.model_execution_authenticated)
        body = self.verify(signed, policy, roster)
        self.assertEqual(body["objective_id"], policy.objective_id)
        self.assertNotIn(b"before arithmetic", signed.receipt_bytes)
        evidence = sign_retention_development_evidence(
            policy, roster, observations, independent_evaluator_key=self.EVALUATOR)
        guarded = decide_retention_with_family_custody(
            policy, roster, evidence, signed,
            expected_signer_id="external-taxonomy-1",
            expected_nonce_sha256=self.NONCE,
            independent_taxonomy_key=self.TAXONOMY,
            independent_evaluator_key=self.EVALUATOR,
        )
        self.assertEqual(guarded.disposition, "PASS")
        self.assertFalse(guarded.graduation_authorized)

    def test_training_family_alias_bypass_is_rejected_before_retention_pass(self):
        policy, roster, observations = self.campaign()
        modified = (replace(roster[0], structure_family_id="Train_Family"),
                    *roster[1:])
        modified_policy = replace(policy, expected_roster_sha256=retention_roster_sha256(modified))
        evidence = sign_retention_development_evidence(
            modified_policy, modified, observations, independent_evaluator_key=self.EVALUATOR)
        self.assertEqual(decide_retention(
            modified_policy, modified, evidence,
            independent_evaluator_key=self.EVALUATOR).disposition, "PASS")
        with self.assertRaisesRegex(FamilyTaxonomyError, "overlap"):
            self.sign(modified_policy, modified)

    def test_unicode_hyphen_prompt_alias_bypass_is_rejected(self):
        policy, roster, observations = self.campaign()
        alternate = "train\u2010prompt"
        self.assertNotEqual(normalized_prompt_sha256(alternate),
                            normalized_prompt_sha256("train-prompt"))
        self.assertEqual(canonical_prompt_alias_sha256(alternate),
                         canonical_prompt_alias_sha256("train-prompt"))
        modified = (replace(roster[0], before_prompt_text=alternate,
                            before_prompt_sha256=normalized_prompt_sha256(alternate)),
                    *roster[1:])
        modified_policy = replace(policy, expected_roster_sha256=retention_roster_sha256(modified))
        evidence = sign_retention_development_evidence(
            modified_policy, modified, observations, independent_evaluator_key=self.EVALUATOR)
        self.assertEqual(decide_retention(
            modified_policy, modified, evidence,
            independent_evaluator_key=self.EVALUATOR).disposition, "PASS")
        with self.assertRaisesRegex(FamilyTaxonomyError, "prompt alias"):
            self.sign(modified_policy, modified)

    def test_two_retention_families_that_are_only_aliases_are_rejected(self):
        policy, roster, _ = self.campaign()
        modified = (roster[0], replace(roster[1],
                       structure_family_id="REVIEW_FAMILY_0000"), *roster[2:])
        modified_policy = replace(policy, expected_roster_sha256=retention_roster_sha256(modified))
        with self.assertRaisesRegex(FamilyTaxonomyError, "duplicate canonical"):
            self.sign(modified_policy, modified)

    def test_signatures_and_independent_pins_reject_swap_and_forgery(self):
        policy, roster, _ = self.campaign()
        signed = self.sign(policy, roster)
        self.verify(signed, policy, roster)
        with self.assertRaises(FamilyTaxonomyError):
            self.verify(replace(signed, signature_hex="0" * 64), policy, roster)
        with self.assertRaises(FamilyTaxonomyError):
            self.verify(signed, policy, roster, expected_nonce_sha256="c" * 64)
        with self.assertRaises(FamilyTaxonomyError):
            self.verify(signed, policy, roster, expected_signer_id="other-taxon")
        with self.assertRaises(ValueError):
            sign_family_taxonomy(policy, roster, signer_id="external-taxonomy-1",
                                 precommitted_nonce_sha256=self.NONCE,
                                 independent_taxonomy_key=self.EVALUATOR,
                                 independent_evaluator_key=self.EVALUATOR)

    def test_unsigned_or_wrong_scope_taxonomy_cannot_guard_retention(self):
        policy, roster, observations = self.campaign()
        signed = self.sign(policy, roster)
        evidence = sign_retention_development_evidence(
            policy, roster, observations, independent_evaluator_key=self.EVALUATOR)
        with self.assertRaises(FamilyTaxonomyError):
            decide_retention_with_family_custody(
                policy, roster, evidence, replace(signed, graduation_authorized=True),
                expected_signer_id="external-taxonomy-1",
                expected_nonce_sha256=self.NONCE,
                independent_taxonomy_key=self.TAXONOMY,
                independent_evaluator_key=self.EVALUATOR)
        with self.assertRaises(FamilyTaxonomyError):
            self.verify(signed, replace(policy, conditions_sha256=digest("other")), roster)
