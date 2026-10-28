"""Synthetic public-only fixtures; none are graduation evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from labyrinth.public_matched_preflight import (
    ARMS, SLICES, preregister_public_split, score_public_matched,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def case(case_id: str, family_id: str) -> dict[str, str]:
    return {"case_id": case_id, "family_id": family_id,
            "prompt_sha256": digest("prompt:" + case_id)}


def fixture() -> tuple[dict, dict]:
    plan = preregister_public_split(
        [case("train1", "train_family"), case("train2", "train_family")],
        {"heldout": [case("held1", "held_family_1"), case("held2", "held_family_2")],
         "transfer": [case("transfer1", "transfer_family_1"),
                      case("transfer2", "transfer_family_2")]},
    )
    ids = [item["case_id"] for name in SLICES for item in plan["slices"][name]]
    prompts = {item["case_id"]: item["prompt_sha256"]
               for name in SLICES for item in plan["slices"][name]}
    answers = {
        "candidate": [True, True, True, True],
        "baseline": [True, False, True, False],
        "zero_update": [False, False, False, False],
        "corrupted": [False, False, False, False],
    }
    runs = {
        arm: {
            "checkpoint_sha256": digest("checkpoint:" + arm),
            "fresh_process_reload_reported": True,
            "trials": [
                {"case_id": case_id, "prompt_sha256": prompts[case_id],
                 "semantic_correct": answers[arm][i],
                 "format_valid": not (arm == "candidate" and i == 1),
                 "response_sha256": digest(f"{arm}:{case_id}")}
                for i, case_id in enumerate(ids)
            ],
        }
        for arm in ARMS
    }
    return plan, runs


class PublicMatchedPreflightTests(unittest.TestCase):
    def test_matched_scores_separate_semantics_from_format_and_never_promote(self):
        plan, runs = fixture()
        outcome = score_public_matched(plan, runs)
        self.assertEqual(outcome["status"], "INTERNAL_CHECKS_PASS")
        self.assertEqual(outcome["qualification"], "NOT_ESTABLISHED")
        self.assertFalse(outcome["promotion_allowed"])
        self.assertFalse(outcome["authenticated_model_execution"])
        heldout = outcome["slices"]["heldout"]
        self.assertEqual(heldout["arms"]["candidate"]["semantic_accuracy"], 1.0)
        self.assertEqual(heldout["arms"]["candidate"]["format_rate"], 0.5)
        self.assertEqual(heldout["arms"]["candidate"]["strict_accuracy"], 0.5)
        self.assertEqual(heldout["paired"]["baseline"], {
            "candidate_only_correct": 1, "control_only_correct": 0,
        })

    def test_declared_training_family_cannot_reappear_under_new_case_id(self):
        plan, _ = fixture()
        slices = deepcopy(plan["slices"])
        slices["heldout"][0]["family_id"] = "train_family"
        with self.assertRaisesRegex(ValueError, "family overlaps"):
            preregister_public_split(plan["training"], slices)

    def test_paraphrases_of_one_family_cannot_inflate_heldout_count(self):
        plan, _ = fixture()
        slices = deepcopy(plan["slices"])
        slices["heldout"][1]["family_id"] = "held_family_1"
        with self.assertRaisesRegex(ValueError, "repeated family"):
            preregister_public_split(plan["training"], slices)

    def test_public_plan_digest_detects_post_registration_change(self):
        plan, runs = fixture()
        plan["slices"]["transfer"][0]["prompt_sha256"] = digest("new-prompt")
        with self.assertRaisesRegex(ValueError, "changed after pre-registration"):
            score_public_matched(plan, runs)

    def test_custom_equality_cannot_forge_plan_commitment(self):
        class ForgedDigest:
            def __eq__(self, _other):
                return True

        plan, runs = fixture()
        plan["plan_sha256"] = ForgedDigest()
        with self.assertRaisesRegex(ValueError, "invalid public pre-registration plan"):
            score_public_matched(plan, runs)

    def test_duplicate_prompts_rejected_even_when_case_ids_change(self):
        plan, _ = fixture()
        slices = deepcopy(plan["slices"])
        slices["heldout"][0]["prompt_sha256"] = plan["training"][0]["prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate case or exact prompt"):
            preregister_public_split(plan["training"], slices)

    def test_missing_control_or_unmatched_case_fails_closed(self):
        plan, runs = fixture()
        del runs["corrupted"]
        with self.assertRaisesRegex(ValueError, "four matched arms"):
            score_public_matched(plan, runs)
        plan, runs = fixture()
        runs["baseline"]["trials"][0]["case_id"] = "other_case"
        with self.assertRaisesRegex(ValueError, "invalid trial values"):
            score_public_matched(plan, runs)

    def test_one_arm_cannot_silently_switch_prompt(self):
        plan, runs = fixture()
        runs["zero_update"]["trials"][0]["prompt_sha256"] = digest("other prompt")
        with self.assertRaisesRegex(ValueError, "invalid trial values"):
            score_public_matched(plan, runs)

    def test_nonstring_prompt_commitment_cannot_pass_using_custom_equality(self):
        class ForgedPrompt:
            def __eq__(self, _other):
                return True

        plan, runs = fixture()
        runs["candidate"]["trials"][0]["prompt_sha256"] = ForgedPrompt()
        with self.assertRaisesRegex(ValueError, "invalid trial values"):
            score_public_matched(plan, runs)

    def test_contradictory_identical_response_scoring_rejected(self):
        plan, runs = fixture()
        runs["baseline"]["trials"][1]["response_sha256"] = (
            runs["candidate"]["trials"][1]["response_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "contradictory"):
            score_public_matched(plan, runs)

    def test_unreported_fresh_reload_is_incomplete_even_with_perfect_scores(self):
        plan, runs = fixture()
        runs["candidate"]["fresh_process_reload_reported"] = False
        outcome = score_public_matched(plan, runs)
        self.assertEqual(outcome["status"], "INCOMPLETE")
        self.assertFalse(outcome["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
