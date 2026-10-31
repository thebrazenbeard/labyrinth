"""PUBLIC TOY metamorphic practice only; no independent neural examination."""

from dataclasses import replace
import json
import re
import unittest
from unittest.mock import patch

import labyrinth.r34_public_metamorphic_challenges as metamorphic
from labyrinth.r34_public_metamorphic_challenges import (
    materialize_public_metamorphic_challenges,
    serialize_public_metamorphic_challenge_jsonl,
    verify_public_metamorphic_challenges,
    _variant_question,
)
from labyrinth.r31_public_challenge_synth import _question
from tests.test_r31_public_challenge_synth import three_skills


class R34PublicMetamorphicChallenges(unittest.TestCase):
    def test_all_three_skills_have_real_different_but_equivalent_wordings(self):
        contract, receipt = three_skills()
        pack = materialize_public_metamorphic_challenges(contract, receipt)
        self.assertEqual(pack, materialize_public_metamorphic_challenges(contract, receipt))
        self.assertEqual(len(pack.development), 6)
        self.assertEqual(len(pack.challenge), 6)
        self.assertEqual({x.skill_id for x in pack.development},
                         {"arithmetic", "source_priority", "program_reasoning"})
        for pair in pack.development + pack.challenge:
            self.assertNotEqual(pair.original_prompt, pair.variant_prompt)
            self.assertNotEqual(pair.original_prompt_sha256, pair.variant_prompt_sha256)
            self.assertEqual(len(pair.semantic_sha256), 64)
            self.assertEqual(len(pair.pair_sha256), 64)
            self.assertIn("Public toy", pair.original_prompt)
            self.assertIn("Public toy", pair.variant_prompt)
            self.assertEqual(pair.expected_answer is None,
                             pair.split == "public_challenge")
        self.assertEqual(verify_public_metamorphic_challenges(
            contract, receipt, pack).status, "PUBLIC_TOY_METAMORPHIC_CONSISTENT_ONLY")
        self.assertFalse(pack.graduation_authorized)
        self.assertFalse(pack.training_authorized)
        self.assertFalse(pack.promotion_allowed)

    def test_variants_preserve_expected_arithmetic_program_and_source_answers(self):
        params = (11, 17, 4, 3)
        for style in ("hint_free", "fresh_examples", "structure_variation"):
            for skill in ("arithmetic", "source_priority", "program_reasoning"):
                with self.subTest(style=style, skill=skill):
                    original, answer, _ = _question(skill, params, style, "X-1")
                    variation, equivalent_answer = _variant_question(skill, params, style, "X-1")
                    self.assertEqual(answer, equivalent_answer)
                    self.assertNotEqual(original, variation)
                    self.assertNotIn(f"Answer: {answer}", variation)
        for style in ("hint_free", "fresh_examples", "structure_variation"):
            variant, answer = _variant_question("source_priority", params, style, "Y-2")
            original, _, _ = _question("source_priority", params, style, "Y-2")
            self.assertIn(answer, variant)
            self.assertIn("higher-priority", variant)
            self.assertIn("lower-priority", variant)
            self.assertEqual(re.findall(r"CODE-\d\d", original),
                             list(reversed(re.findall(r"CODE-\d\d", variant))))

    def test_challenge_export_is_answer_free_and_source_bound(self):
        contract, receipt = three_skills()
        pack = materialize_public_metamorphic_challenges(contract, receipt)
        encoded = serialize_public_metamorphic_challenge_jsonl(contract, receipt, pack)
        rows = [json.loads(line) for line in encoded.splitlines()]
        self.assertEqual(len(rows), len(pack.challenge))
        self.assertEqual({row["split"] for row in rows}, {"public_challenge"})
        self.assertTrue(all(not any("answer" in k.lower() or "expected" in k.lower()
                                    for k in row) for row in rows))
        self.assertEqual(len({row["pair_sha256"] for row in rows}), len(rows))
        self.assertTrue({x.pair_sha256 for x in pack.development}.isdisjoint(
            {x.pair_sha256 for x in pack.challenge}))
        bad = replace(pack.challenge[0], variant_prompt="The answer is obvious: 42")
        poisoned = replace(pack, challenge=(bad,) + pack.challenge[1:])
        self.assertEqual(verify_public_metamorphic_challenges(
            contract, receipt, poisoned).status, "PUBLIC_TOY_METAMORPHIC_INVALID")
        with self.assertRaises(ValueError):
            serialize_public_metamorphic_challenge_jsonl(contract, receipt, poisoned)
        wrong_receipt = replace(receipt, commitment_sha256="a" * 64)
        with self.assertRaises(ValueError):
            materialize_public_metamorphic_challenges(contract, wrong_receipt)

    def test_untrusted_nested_field_cannot_execute_custom_equality(self):
        contract, receipt = three_skills()
        pack = materialize_public_metamorphic_challenges(contract, receipt)

        class CustomComparator:
            def __eq__(self, other):
                raise AssertionError("unsafe caller comparator invoked")

        bad_pair = replace(pack.development[0], variant_prompt=CustomComparator())
        bad_pack = replace(pack, development=(bad_pair,) + pack.development[1:])
        self.assertEqual(verify_public_metamorphic_challenges(
            contract, receipt, bad_pack).status, "PUBLIC_TOY_METAMORPHIC_INVALID")
        bad_commit = replace(pack, commitment_sha256=CustomComparator())
        self.assertEqual(verify_public_metamorphic_challenges(
            contract, receipt, bad_commit).status, "PUBLIC_TOY_METAMORPHIC_INVALID")

    def test_deleted_frozen_fields_fail_closed_before_verification_or_export(self):
        contract, receipt = three_skills()
        for location, field in (("pair", "variant_prompt"),
                                ("bundle", "commitment_sha256")):
            with self.subTest(location=location, field=field):
                pack = materialize_public_metamorphic_challenges(contract, receipt)
                victim = pack.challenge[0] if location == "pair" else pack
                object.__delattr__(victim, field)
                verdict = verify_public_metamorphic_challenges(contract, receipt, pack)
                self.assertEqual(verdict.status, "PUBLIC_TOY_METAMORPHIC_INVALID")
                self.assertFalse(verdict.training_authorized)
                self.assertFalse(verdict.graduation_authorized)
                with self.assertRaises(ValueError):
                    serialize_public_metamorphic_challenge_jsonl(contract, receipt, pack)

    def test_deleted_default_authority_flags_are_not_inherited_from_class(self):
        contract, receipt = three_skills()
        for flag in ("training_authorized", "graduation_authorized",
                     "independent_exam_verified", "promotion_allowed"):
            with self.subTest(flag=flag):
                pack = materialize_public_metamorphic_challenges(contract, receipt)
                object.__delattr__(pack, flag)
                # Frozen dataclass default is still readable as a class
                # attribute; an attestation must require the instance field.
                self.assertIs(getattr(pack, flag), False)
                verdict = verify_public_metamorphic_challenges(contract, receipt, pack)
                self.assertEqual(verdict.status, "PUBLIC_TOY_METAMORPHIC_INVALID")
                self.assertFalse(verdict.promotion_allowed)
                with self.assertRaises(ValueError):
                    serialize_public_metamorphic_challenge_jsonl(contract, receipt, pack)

    def test_unexpected_instance_metadata_is_not_silently_admitted(self):
        contract, receipt = three_skills()
        pack = materialize_public_metamorphic_challenges(contract, receipt)
        object.__setattr__(pack, "unattested_override", "yes")
        verdict = verify_public_metamorphic_challenges(contract, receipt, pack)
        self.assertEqual(verdict.status, "PUBLIC_TOY_METAMORPHIC_INVALID")
        with self.assertRaises(ValueError):
            serialize_public_metamorphic_challenge_jsonl(contract, receipt, pack)

    def test_serialization_reconstructs_after_validation_mutates_caller(self):
        contract, receipt = three_skills()
        pack = materialize_public_metamorphic_challenges(contract, receipt)
        original = metamorphic.verify_public_metamorphic_challenges

        def mutate_after_verified(*args, **kwargs):
            result = original(*args, **kwargs)
            object.__setattr__(pack.challenge[0], "variant_prompt", "Leaked expected answer")
            return result

        with patch.object(metamorphic, "verify_public_metamorphic_challenges",
                          side_effect=mutate_after_verified):
            lines = serialize_public_metamorphic_challenge_jsonl(contract, receipt, pack)
        self.assertNotIn("Leaked expected answer", lines)
        self.assertEqual(len(lines.splitlines()), len(pack.challenge))


if __name__ == "__main__":
    unittest.main()
