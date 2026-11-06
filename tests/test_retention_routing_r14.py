"""R14: adversarial arithmetic routing checks, not a private examination.

These fixtures use test-local signing keys, so a source PASS never attests
actual learner inference, independent custody, or graduation eligibility.
"""

from dataclasses import replace
import unittest

from labyrinth.arithmetic_retention_admission import (
    arithmetic_training_inventory_sha256,
    evaluate_arithmetic_retention_admission,
)
from labyrinth.arithmetic_semantic_case_key import arithmetic_semantic_case_key
from labyrinth.retention_entrypoint_guard import evaluate_retention_admission
from labyrinth.retention_family_registry import sign_family_taxonomy
from labyrinth.retention_qualification import (
    RetentionObservation,
    normalized_prompt_sha256,
    retention_roster_sha256,
    sign_retention_development_evidence,
)
from tests import test_retention_entrypoint_guard as retention_fixtures


def _seven_plus_five_forms() -> tuple[str, ...]:
    """At least 128 distinct normalized texts all expressing one ADD case."""
    templates = (
        "what is {left} plus {right}",
        "what is {left} + {right}",
        "compute {left} plus {right}",
        "compute {left} + {right}",
        "add {left} to {right}",
        "{left} plus {right}",
        "{left} + {right}",
    )
    operand_pairs = (
        ("7", "5"), ("7", "five"), ("seven", "5"),
        ("seven", "five"), ("5", "7"), ("5", "seven"),
        ("five", "7"), ("five", "seven"),
    )
    forms = tuple(
        template.format(left=left, right=right) + punctuation
        for template in templates
        for left, right in operand_pairs
        for punctuation in ("", ".", "?")
    )
    assert len({normalized_prompt_sha256(text) for text in forms}) == len(forms)
    assert {arithmetic_semantic_case_key(text) for text in forms} == {"add:5:7"}
    return forms


class RetentionRoutingR14Tests(unittest.TestCase):
    def campaign(self):
        donor = retention_fixtures.RetentionEntrypointGuardTests(
            "test_valid_distinct_signed_development_evidence_passes_only_veto")
        request, policy, roster, _, _, options = donor.campaign()
        texts = _seven_plus_five_forms()
        self.assertGreaterEqual(len(texts), len(roster) * 2)
        roster = tuple(replace(
            old,
            before_prompt_text=texts[i * 2],
            after_prompt_text=texts[i * 2 + 1],
            before_prompt_sha256=normalized_prompt_sha256(texts[i * 2]),
            after_prompt_sha256=normalized_prompt_sha256(texts[i * 2 + 1]),
        ) for i, old in enumerate(roster))
        training = ("What is three plus four?",)
        policy = replace(
            policy,
            excluded_training_prompts=training,
            excluded_training_prompt_sha256s=tuple(
                normalized_prompt_sha256(text) for text in training),
            expected_roster_sha256=retention_roster_sha256(roster),
        )
        observations = tuple(
            RetentionObservation(pair.situation_id, True, True) for pair in roster
        )
        evidence = sign_retention_development_evidence(
            policy, roster, observations,
            independent_evaluator_key=donor.EVALUATOR_KEY,
        )
        taxonomy = sign_family_taxonomy(
            policy, roster,
            signer_id="independent-taxonomy-custodian",
            precommitted_nonce_sha256=donor.TAXONOMY_NONCE,
            independent_taxonomy_key=donor.TAXONOMY_KEY,
            independent_evaluator_key=donor.EVALUATOR_KEY,
        )
        return request, policy, roster, evidence, taxonomy, options

    def test_sixty_four_signed_family_aliases_one_fact_legacy_bypass_strict_blocks(self):
        request, policy, roster, evidence, taxonomy, custody = self.campaign()
        self.assertEqual(len(roster), 64)
        self.assertEqual(len({p.structure_family_id for p in roster}), 64)
        self.assertEqual(len({arithmetic_semantic_case_key(p.before_prompt_text)
                              for p in roster}), 1)

        # A public legacy API remains callable; its PASS is not semantic
        # independence, physical execution, or host-enforced route evidence.
        legacy = evaluate_retention_admission(
            request, policy, roster, evidence, taxonomy, **custody)
        self.assertEqual(legacy.status, "PASS")
        self.assertFalse(legacy.graduation_authorized)
        self.assertFalse(legacy.model_execution_authenticated)

        strict = evaluate_arithmetic_retention_admission(
            request, policy, roster, evidence, taxonomy,
            trusted_training_inventory_sha256=arithmetic_training_inventory_sha256(
                policy.excluded_training_prompts),
            trusted_retention_roster_sha256=retention_roster_sha256(roster),
            **custody,
        )
        self.assertEqual(strict.status, "BLOCKED")
        self.assertIsNone(strict.retention)
        self.assertFalse(strict.graduation_authorized)
        self.assertFalse(strict.model_execution_authenticated)

    def test_unpinned_strict_source_route_blocks_even_when_legacy_passes(self):
        request, policy, roster, evidence, taxonomy, custody = self.campaign()
        strict = evaluate_arithmetic_retention_admission(
            request, policy, roster, evidence, taxonomy,
            trusted_training_inventory_sha256="0" * 64,
            trusted_retention_roster_sha256=retention_roster_sha256(roster),
            **custody,
        )
        self.assertEqual(strict.status, "BLOCKED")
        self.assertIsNone(strict.retention)


if __name__ == "__main__":
    unittest.main()
