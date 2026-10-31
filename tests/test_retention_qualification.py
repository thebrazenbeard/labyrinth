"""Source-level tests; no private examiner or neural inference is simulated."""

from dataclasses import replace
from hashlib import sha256
import json
from unittest import TestCase

from labyrinth.retention_qualification import (
    RetentionError, RetentionObservation, RetentionPair, RetentionPolicy,
    PROMPT_COMMITMENT_VERSION, decide_retention, normalized_prompt_sha256,
    retention_roster_sha256,
    sign_retention_development_evidence,
)


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class RetentionQualificationTests(TestCase):
    KEY = b"independent-retention-evaluator-key-12345678"

    def campaign(self, skills=("arithmetic",), count=64):
        roster = tuple(
            RetentionPair(
                skill_id=skill,
                situation_id=f"{skill}-situation-{index:04d}",
                structure_family_id=f"{skill}-family-{index:04d}",
                before_case_id=f"{skill}-before-{index:04d}",
                after_case_id=f"{skill}-after-{index:04d}",
                before_prompt_text=f"before-{skill}-{index}",
                after_prompt_text=f"after-{skill}-{index}",
                before_prompt_sha256=normalized_prompt_sha256(f"before-{skill}-{index}"),
                after_prompt_sha256=normalized_prompt_sha256(f"after-{skill}-{index}"),
            )
            for skill in skills for index in range(count)
        )
        policy = RetentionPolicy(
            objective_id="sentiment-development-v1",
            scope_sha256=digest("declared-scope"),
            before_checkpoint_sha256=digest("learned-before"),
            after_checkpoint_sha256=digest("learned-after"),
            evaluator_sha256=digest("external-evaluator-code"),
            conditions_sha256=digest("external-evaluation-conditions"),
            session_nonce="a" * 64,
            prior_skill_ids=tuple(skills),
            expected_roster_sha256=retention_roster_sha256(roster),
            prompt_commitment_version=PROMPT_COMMITMENT_VERSION,
            excluded_training_prompts=("train-prompt",),
            excluded_training_prompt_sha256s=(normalized_prompt_sha256("train-prompt"),),
            excluded_training_structure_family_ids=("train-family",),
        )
        observed = tuple(RetentionObservation(
            situation_id=pair.situation_id, before_correct=True,
            after_correct=True,
        ) for pair in roster)
        return policy, roster, observed

    def evaluate(self, policy, roster, observed):
        evidence = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=self.KEY,
        )
        return decide_retention(
            policy, roster, evidence,
            independent_evaluator_key=self.KEY,
        )

    def test_zero_losses_with_64_independent_units_pass_conservative_bound(self):
        policy, roster, observed = self.campaign()
        result = self.evaluate(policy, roster, observed)
        self.assertEqual(result.disposition, "PASS")
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)
        self.assertEqual(result.groups[0].deteriorations, 0)
        self.assertEqual(result.groups[0].total, 64)
        self.assertGreater(result.groups[0].upper_deterioration_rate, 0.04)
        self.assertLess(result.groups[0].upper_deterioration_rate, 0.05)

    def test_one_loss_from_64_units_fails_five_point_noninferiority(self):
        policy, roster, observed = self.campaign()
        observed = (replace(observed[0], after_correct=False), *observed[1:])
        result = self.evaluate(policy, roster, observed)
        self.assertEqual(result.disposition, "HOLD")
        self.assertGreater(result.groups[0].upper_deterioration_rate, 0.05)

    def test_observable_improvements_do_not_hide_unbounded_forgetting(self):
        policy, roster, observed = self.campaign()
        observed = tuple(replace(item, before_correct=False, after_correct=True)
                         if 1 <= index < 10 else
                         replace(item, after_correct=False) if index == 0 else item
                         for index, item in enumerate(observed))
        self.assertEqual(self.evaluate(policy, roster, observed).disposition, "HOLD")

    def test_never_mastered_prior_skill_cannot_pass_vacuous_no_loss(self):
        policy, roster, observed = self.campaign()
        observed = tuple(replace(item, before_correct=False, after_correct=False)
                         for item in observed)
        self.assertEqual(self.evaluate(policy, roster, observed).disposition, "HOLD")

    def test_every_prior_skill_requires_full_distinct_retention_form(self):
        # Two simultaneous skill claims split the family-wise alpha; 64
        # perfect cases per skill no longer suffice for a 5pp upper bound.
        policy, roster, observed = self.campaign(("arithmetic", "program"), 80)
        self.assertEqual(self.evaluate(policy, roster, observed).disposition, "PASS")
        with self.assertRaisesRegex(RetentionError, "missing prior-skill coverage"):
            self.evaluate(replace(policy, expected_roster_sha256=
                                  retention_roster_sha256(roster[:80])),
                          roster[:80], observed[:80])

    def test_familywise_alpha_correction_can_hold_two_perfect_small_skills(self):
        policy, roster, observed = self.campaign(("arithmetic", "program"), 64)
        verdict = self.evaluate(policy, roster, observed)
        self.assertEqual(verdict.disposition, "HOLD")
        self.assertTrue(all(group.upper_deterioration_rate > 0.05
                            for group in verdict.groups))

    def test_short_campaign_cannot_pass_even_if_perfect(self):
        policy, roster, observed = self.campaign(count=32)
        with self.assertRaisesRegex(RetentionError, "insufficient independent"):
            self.evaluate(policy, roster, observed)

    def test_reused_prompt_between_forms_or_training_fails(self):
        policy, roster, observed = self.campaign()
        contaminated = (replace(roster[0],
                                after_prompt_text=roster[0].before_prompt_text,
                                after_prompt_sha256=roster[0].before_prompt_sha256),
                        *roster[1:])
        contaminated_policy = replace(policy, expected_roster_sha256=
                                      retention_roster_sha256(contaminated))
        with self.assertRaisesRegex(RetentionError, "repeated or contaminated prompt"):
            self.evaluate(contaminated_policy, contaminated, observed)

    def test_normalized_training_prompt_variant_cannot_pass(self):
        policy, roster, observed = self.campaign()
        equivalent = " \u2003ＴＲＡＩＮ－ＰＲＯＭＰＴ  "
        self.assertEqual(normalized_prompt_sha256(equivalent),
                         normalized_prompt_sha256("train-prompt"))
        contaminated = (replace(roster[0], before_prompt_text=equivalent,
                                before_prompt_sha256=normalized_prompt_sha256(equivalent)),
                        *roster[1:])
        contaminated_policy = replace(policy, expected_roster_sha256=
                                      retention_roster_sha256(contaminated))
        with self.assertRaisesRegex(RetentionError, "repeated or contaminated prompt"):
            self.evaluate(contaminated_policy, contaminated, observed)

    def test_invisible_unicode_format_characters_are_rejected_before_hashing(self):
        for invisible in ("\u200b", "\u2060", "\u200d", "\ufeff"):
            with self.subTest(codepoint=ord(invisible)):
                with self.assertRaisesRegex(ValueError, "invisible Unicode format"):
                    normalized_prompt_sha256("train-" + invisible + "prompt")

    def test_old_equivalent_digest_cannot_mask_invisible_eval_or_training_text(self):
        policy, roster, observed = self.campaign()
        for invisible in ("\u200b", "\u2060"):
            with self.subTest(codepoint=ord(invisible)):
                altered_pair = replace(
                    roster[0], before_prompt_text="train-" + invisible + "prompt",
                    before_prompt_sha256=normalized_prompt_sha256("train-prompt"),
                )
                altered_roster = (altered_pair, *roster[1:])
                policy_for_altered_roster = replace(
                    policy, expected_roster_sha256=retention_roster_sha256(altered_roster),
                )
                with self.assertRaisesRegex(ValueError, "invisible Unicode format"):
                    self.evaluate(policy_for_altered_roster, altered_roster, observed)
                with self.assertRaisesRegex(ValueError, "invisible Unicode format"):
                    replace(policy, excluded_training_prompts=("train-" + invisible + "prompt",))

    def test_training_structure_family_overlap_refuses(self):
        policy, roster, observed = self.campaign()
        contaminated = (replace(roster[0], structure_family_id="train-family"),
                        *roster[1:])
        contaminated_policy = replace(policy, expected_roster_sha256=
                                      retention_roster_sha256(contaminated))
        with self.assertRaisesRegex(RetentionError, "training structure family"):
            self.evaluate(contaminated_policy, contaminated, observed)

    def test_legacy_raw_hash_or_missing_version_refuses(self):
        policy, roster, observed = self.campaign()
        with self.assertRaises(ValueError):
            replace(policy, prompt_commitment_version="legacy-raw-sha256-v1")
        with self.assertRaisesRegex(ValueError, "normalized versioned"):
            replace(policy, excluded_training_prompt_sha256s=(digest("train-prompt"),))
        legacy = (replace(roster[0], before_prompt_sha256=digest(roster[0].before_prompt_text)),
                  *roster[1:])
        legacy_policy = replace(policy, expected_roster_sha256=
                                retention_roster_sha256(legacy))
        with self.assertRaisesRegex(RetentionError, "normalized prompt commitment"):
            self.evaluate(legacy_policy, legacy, observed)
        contaminated = (replace(roster[0], before_prompt_text="train-prompt",
                                before_prompt_sha256=
                                policy.excluded_training_prompt_sha256s[0]), *roster[1:])
        contaminated_policy = replace(policy, expected_roster_sha256=
                                      retention_roster_sha256(contaminated))
        with self.assertRaisesRegex(RetentionError, "repeated or contaminated prompt"):
            self.evaluate(contaminated_policy, contaminated, observed)

    def test_reused_semantic_family_with_unique_prompts_fails(self):
        policy, roster, observed = self.campaign()
        contaminated = (roster[0], replace(roster[1], structure_family_id=roster[0].structure_family_id),
                        *roster[2:])
        contaminated_policy = replace(policy, expected_roster_sha256=
                                      retention_roster_sha256(contaminated))
        with self.assertRaisesRegex(RetentionError, "duplicate semantic situation"):
            self.evaluate(contaminated_policy, contaminated, observed)

    def test_roster_hash_and_model_context_tampering_fail_closed(self):
        policy, roster, observed = self.campaign()
        with self.assertRaisesRegex(RetentionError, "roster commitment"):
            self.evaluate(replace(policy, expected_roster_sha256="f" * 64),
                          roster, observed)
        evidence = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=self.KEY,
        )
        with self.assertRaisesRegex(RetentionError, "policy or model"):
            decide_retention(replace(policy, after_checkpoint_sha256=digest("other")),
                             roster, evidence, independent_evaluator_key=self.KEY)

    def test_signature_and_receipt_mutation_fail(self):
        policy, roster, observed = self.campaign()
        evidence = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=self.KEY,
        )
        with self.assertRaisesRegex(RetentionError, "signature"):
            decide_retention(policy, roster, replace(evidence, signature_hex="0" * 64),
                             independent_evaluator_key=self.KEY)
        forged = evidence.receipt_bytes.replace(b'"after_correct":true',
                                                b'"after_correct":false', 1)
        with self.assertRaises(RetentionError):
            decide_retention(policy, roster, replace(evidence, receipt_bytes=forged),
                             independent_evaluator_key=self.KEY)

    def test_extra_or_duplicate_observation_fails(self):
        policy, roster, observed = self.campaign()
        with self.assertRaisesRegex(RetentionError, "observation identity"):
            self.evaluate(policy, roster, (*observed, observed[0]))
        with self.assertRaisesRegex(RetentionError, "observation identity"):
            self.evaluate(policy, roster, observed[1:])

    def test_policy_cannot_be_weakened_to_zero_margin_or_no_prior_skill(self):
        policy, _, _ = self.campaign()
        with self.assertRaises(ValueError):
            replace(policy, max_deterioration_rate=0.50)
        with self.assertRaises(ValueError):
            replace(policy, prior_skill_ids=())
        with self.assertRaises(ValueError):
            replace(policy, min_units_per_skill=16)
        with self.assertRaises(ValueError):
            replace(policy, min_before_accuracy=0.20)

    def test_receipt_is_development_only_and_canonical(self):
        policy, roster, observed = self.campaign()
        evidence = sign_retention_development_evidence(
            policy, roster, observed, independent_evaluator_key=self.KEY,
        )
        body = json.loads(evidence.receipt_bytes)
        self.assertEqual(body["schema"], "labyrinth.prior-skill-retention-development.v2")
        self.assertIs(body["graduation_authorized"], False)
        self.assertIs(body["model_execution_authenticated"], False)
        self.assertEqual(len(body["observations"]), 64)
        self.assertNotIn(b"train-prompt", evidence.receipt_bytes)
        self.assertNotIn(b"before-arithmetic-0", evidence.receipt_bytes)
