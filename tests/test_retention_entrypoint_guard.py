"""Fail-closed source-level tests; no independent examiner is installed."""

from dataclasses import replace
from hashlib import sha256
from unittest import TestCase

from labyrinth.journey_qualification import (
    JourneyQualificationPrecommit, PublicQualificationRequest,
)
from labyrinth.retention_entrypoint_guard import evaluate_retention_admission
from labyrinth.retention_family_registry import sign_family_taxonomy
from labyrinth.retention_qualification import (
    PROMPT_COMMITMENT_VERSION,
    RetentionObservation, RetentionPair, RetentionPolicy,
    normalized_prompt_sha256, retention_roster_sha256,
    sign_retention_development_evidence,
)


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class RetentionEntrypointGuardTests(TestCase):
    TAXONOMY_KEY = b"independent-exam-taxonomy-key-20261030-abc"
    EVALUATOR_KEY = b"independent-exam-evaluator-key-20261030-xyz"
    TAXONOMY_NONCE = digest("precommitted-taxonomy-nonce")

    def campaign(self, *, losses=0):
        roster = tuple(RetentionPair(
            skill_id="arithmetic", situation_id=f"situation-{i:03d}",
            structure_family_id=f"prior-review-family-{i:03d}",
            before_case_id=f"before-{i:03d}", after_case_id=f"after-{i:03d}",
            before_prompt_text=f"prior before {i}",
            after_prompt_text=f"prior after {i}",
            before_prompt_sha256=normalized_prompt_sha256(f"prior before {i}"),
            after_prompt_sha256=normalized_prompt_sha256(f"prior after {i}"),
        ) for i in range(64))
        scope = digest("scope")
        prior = digest("prior-checkpoint")
        candidate = digest("candidate-checkpoint")
        policy = RetentionPolicy(
            objective_id="arithmetic", scope_sha256=scope,
            before_checkpoint_sha256=prior,
            after_checkpoint_sha256=candidate,
            evaluator_sha256=digest("evaluator-source"),
            conditions_sha256=digest("conditions"),
            session_nonce="a" * 64,
            prior_skill_ids=("arithmetic",),
            expected_roster_sha256=retention_roster_sha256(roster),
            prompt_commitment_version=PROMPT_COMMITMENT_VERSION,
            excluded_training_prompts=("train prompt",),
            excluded_training_prompt_sha256s=(normalized_prompt_sha256("train prompt"),),
            excluded_training_structure_family_ids=("train-family",),
        )
        request = PublicQualificationRequest(
            objective_id="arithmetic", scope_fingerprint=scope,
            candidate_checkpoint_sha256=candidate,
            entrant_checkpoint_sha256=digest("initial-entrant"),
            development_cycle_index=1, development_exam_id="development-1",
        )
        precommit = JourneyQualificationPrecommit(
            scope_fingerprint=scope, development_exam_id="development-1",
            entrant_checkpoint_sha256=digest("initial-entrant"),
            development_cycle_index=1,
        )
        observations = tuple(RetentionObservation(
            pair.situation_id, True, i >= losses
        ) for i, pair in enumerate(roster))
        evidence = sign_retention_development_evidence(
            policy, roster, observations,
            independent_evaluator_key=self.EVALUATOR_KEY,
        )
        taxonomy = sign_family_taxonomy(
            policy, roster, signer_id="independent-taxonomy-custodian",
            precommitted_nonce_sha256=self.TAXONOMY_NONCE,
            independent_taxonomy_key=self.TAXONOMY_KEY,
            independent_evaluator_key=self.EVALUATOR_KEY,
        )
        options = dict(
            trusted_journey_precommit=precommit,
            trusted_prior_checkpoint_sha256=prior,
            expected_taxonomy_signer_id="independent-taxonomy-custodian",
            expected_taxonomy_nonce_sha256=self.TAXONOMY_NONCE,
            independent_taxonomy_key=self.TAXONOMY_KEY,
            independent_evaluator_key=self.EVALUATOR_KEY,
        )
        return request, policy, roster, evidence, taxonomy, options

    def verdict(self, *, losses=0, replace_args=None, **change_options):
        request, policy, roster, evidence, taxonomy, options = self.campaign(losses=losses)
        args = [request, policy, roster, evidence, taxonomy]
        if replace_args:
            for index, substitute in replace_args.items():
                args[index] = substitute
        options.update(change_options)
        return evaluate_retention_admission(*args, **options)

    def test_valid_distinct_signed_development_evidence_passes_only_veto(self):
        result = self.verdict()
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.retention.disposition, "PASS")
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)

    def test_one_true_loss_causes_statistical_hold(self):
        result = self.verdict(losses=1)
        self.assertEqual(result.status, "HOLD")
        self.assertEqual(result.retention.disposition, "HOLD")
        self.assertFalse(result.graduation_authorized)

    def test_missing_signed_taxonomy_or_retention_blocks(self):
        for index in (3, 4):
            with self.subTest(missing=index):
                result = self.verdict(replace_args={index: None})
                self.assertEqual(result.status, "BLOCKED")
                self.assertIsNone(result.retention)

    def test_missing_precommit_or_keys_block(self):
        self.assertEqual(self.verdict(trusted_journey_precommit=None).status, "BLOCKED")
        self.assertEqual(self.verdict(independent_taxonomy_key=None).status, "BLOCKED")
        self.assertEqual(self.verdict(independent_evaluator_key=None).status, "BLOCKED")

    def test_wrong_frozen_signer_or_nonce_block(self):
        self.assertEqual(self.verdict(expected_taxonomy_signer_id="other-signer").status,
                         "BLOCKED")
        self.assertEqual(self.verdict(expected_taxonomy_nonce_sha256="f" * 64).status,
                         "BLOCKED")

    def test_request_mismatch_is_blocked_before_verification(self):
        request, policy, roster, evidence, taxonomy, options = self.campaign()
        for changed in (
            replace(request, objective_id="other"),
            replace(request, scope_fingerprint=digest("different-scope")),
            replace(request, candidate_checkpoint_sha256=digest("other-candidate")),
            replace(request, development_exam_id="other-development"),
            replace(request, entrant_checkpoint_sha256=digest("other-entrant")),
        ):
            with self.subTest(changed=changed):
                result = evaluate_retention_admission(
                    changed, policy, roster, evidence, taxonomy, **options)
                self.assertEqual(result.status, "BLOCKED")

    def test_retention_predecessor_is_independently_pinned(self):
        self.assertEqual(self.verdict(
            trusted_prior_checkpoint_sha256=digest("different-prior")).status,
            "BLOCKED")

    def test_tampering_receipt_or_claiming_authority_blocks(self):
        request, policy, roster, evidence, taxonomy, options = self.campaign()
        for manipulated in (
            replace(evidence, signature_hex="0" * 64),
            replace(evidence, graduation_authorized=True),
        ):
            with self.subTest(manipulated=manipulated):
                self.assertEqual(evaluate_retention_admission(
                    request, policy, roster, manipulated, taxonomy,
                    **options).status, "BLOCKED")
        self.assertEqual(evaluate_retention_admission(
            request, policy, roster, evidence,
            replace(taxonomy, model_execution_authenticated=True),
            **options).status, "BLOCKED")

    def test_same_keys_rejected_even_when_both_signatures_exist(self):
        self.assertEqual(self.verdict(
            independent_taxonomy_key=self.EVALUATOR_KEY).status, "BLOCKED")
