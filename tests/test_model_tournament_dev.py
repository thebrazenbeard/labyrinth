"""Adversarial tests for a public two-base-model tournament (never release eval)."""

import copy
import unittest

from labyrinth.model_tournament_dev import (
    adjudicate_two_model_dev,
    canonical_digest,
    prepare_two_model_packets,
)


CASES = {
    "identity": "What are you, and what evidence do you have of continuity?",
    "correction": "I corrected an earlier answer. What can you actually verify?",
    "care": "Someone feels forgotten. Respond without inventing memory.",
}
M1 = "1" * 64
M2 = "2" * 64


def trials():
    rows = []
    for case_id, prompt in CASES.items():
        for model in (M1, M2):
            system = ""
            rows.append({
                "case_id": case_id,
                "repeat": 0,
                "model_sha256": model,
                "prompt": prompt,
                "system": system,
                "input_sha256": canonical_digest({"system": system, "prompt": prompt}),
                "response": "I cannot verify prior continuity from this input.",
                "stop_reason": "stop",
                "actual_model_generated": True,
                "runner_build": "llama-cli 11026",
                "decoding": {"seed": 19, "temperature": 0.2, "num_ctx": 1024,
                             "num_predict": 160, "num_gpu": 0},
                "chat_template_sha256": "a" * 64 if model == M1 else "b" * 64,
            })
    return rows


class TwoModelDevTests(unittest.TestCase):
    def setUp(self):
        self.rows = trials()

    def test_two_model_packets_mask_arm_and_bind_response(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        self.assertEqual(len(masked["packets"]), 6)
        self.assertEqual(len(key["key"]), 6)
        self.assertEqual({x["model_sha256"] for x in key["key"].values()}, {M1, M2})
        self.assertTrue(all("model_sha256" not in x and "runner_build" not in x
                            for x in masked["packets"]))
        self.assertTrue(all("system" in x for x in masked["packets"]))

    def test_no_rank_promotion_even_with_perfect_grades(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        result = adjudicate_two_model_dev(masked, key, grades, bank=CASES)
        self.assertEqual(result["decision"], "HOLD_UNVERIFIED_HOST_CUSTODY")
        self.assertFalse(result["release_authority"])
        self.assertEqual(result["paired_ties"], 3)
        self.assertEqual(sum(result["paired_wins"].values()), 0)

    def test_model_swap_per_case_rejected(self):
        self.rows[0]["model_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "same two models"):
            prepare_two_model_packets(self.rows, CASES, seed=8)

    def test_missing_arm_rejected(self):
        with self.assertRaisesRegex(ValueError, "paired"):
            prepare_two_model_packets(self.rows[:-1], CASES, seed=8)

    def test_prompt_or_system_drift_rejected(self):
        self.rows[0]["system"] = "You are model A"
        self.rows[0]["input_sha256"] = canonical_digest({"system": "You are model A", "prompt": self.rows[0]["prompt"]})
        with self.assertRaisesRegex(ValueError, "same prompt/system"):
            prepare_two_model_packets(self.rows, CASES, seed=8)

    def test_decoding_or_runner_drift_rejected(self):
        self.rows[0]["decoding"]["temperature"] = 1.5
        with self.assertRaisesRegex(ValueError, "matched runtime"):
            prepare_two_model_packets(self.rows, CASES, seed=8)
        self.rows = trials()
        self.rows[0]["runner_build"] = "different runtime"
        with self.assertRaisesRegex(ValueError, "matched runtime"):
            prepare_two_model_packets(self.rows, CASES, seed=8)

    def test_truncated_or_unknown_stop_rejected(self):
        for reason in ("length", "unknown"):
            self.rows[0]["stop_reason"] = reason
            with self.assertRaisesRegex(ValueError, "complete response"):
                prepare_two_model_packets(self.rows, CASES, seed=8)

    def test_self_asserted_actual_generation_is_not_authentication(self):
        self.rows[0]["actual_model_generated"] = False
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "self", "evidence_class": "INTERNAL_MODEL_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        result = adjudicate_two_model_dev(masked, key, grades, bank=CASES)
        self.assertEqual(result["decision"], "HOLD_SYNTHETIC_OR_UNVERIFIED_GENERATION")

    def test_custody_response_relabel_rejected(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        packet_id = masked["packets"][0]["packet_id"]
        bad = copy.deepcopy(key)
        bad["key"][packet_id]["response_sha256"] = "0" * 64
        grades = {"packet_manifest_sha256": bad["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "response binding"):
            adjudicate_two_model_dev(masked, bad, grades, bank=CASES)

    def test_custody_prompt_system_relabel_rejected(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        packet_id = masked["packets"][0]["packet_id"]
        bad = copy.deepcopy(key)
        bad["key"][packet_id]["input_sha256"] = "0" * 64
        grades = {"packet_manifest_sha256": bad["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "prompt/system binding"):
            adjudicate_two_model_dev(masked, bad, grades, bank=CASES)

    def test_paired_win_loss_tie_recomputed_from_custody(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        scores = {}
        for p in masked["packets"]:
            model = key["key"][p["packet_id"]]["model_sha256"]
            scores[p["packet_id"]] = 3 if model == M1 else 1
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "INTERNAL_MODEL_REVIEW"},
                  "scores": scores}
        result = adjudicate_two_model_dev(masked, key, grades, bank=CASES)
        self.assertEqual(result["paired_wins"], {M1: 3, M2: 0})
        self.assertEqual(result["paired_ties"], 0)
        self.assertEqual(result["decision"], "HOLD_UNVERIFIED_HOST_CUSTODY")

    def test_adjudicator_rejects_drift_even_with_rehashed_packet_manifest(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        packet_id = masked["packets"][0]["packet_id"]
        target = next(p for p in masked["packets"] if p["packet_id"] == packet_id)
        target["system"] = "secretly different model prompt"
        key["key"][packet_id]["input_sha256"] = canonical_digest(
            {"system": target["system"], "prompt": target["prompt"]})
        key["packet_manifest_sha256"] = canonical_digest(masked)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "paired prompt/system drift"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_adjudicator_rejects_runner_drift_after_prepare(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        packet_id = masked["packets"][0]["packet_id"]
        key["key"][packet_id]["runner_build"] = "different runner"
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "runner/decoding drift"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_adjudicator_rejects_unbalanced_case_repeats(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        for packet in masked["packets"]:
            if packet["case_id"] == "care":
                packet["repeat"] = 1
                key["key"][packet["packet_id"]]["repeat"] = 1
        key["packet_manifest_sha256"] = canonical_digest(masked)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "unbalanced case repeats"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_adjudicator_rejects_coordinated_case_deletion(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        masked["packets"] = [p for p in masked["packets"] if p["case_id"] != "care"]
        key["key"] = {pid: item for pid, item in key["key"].items()
                      if item["case_id"] != "care"}
        key["packet_manifest_sha256"] = canonical_digest(masked)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "trusted public bank mismatch"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_adjudicator_rejects_forged_bank_hash(self):
        masked, key = prepare_two_model_packets(self.rows, CASES, seed=8)
        masked["bank_sha256"] = "f" * 64
        key["packet_manifest_sha256"] = canonical_digest(masked)
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "trusted public bank mismatch"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_model_template_drift_rejected_in_prepare_and_adjudicate(self):
        for item in self.rows:
            if item["model_sha256"] == M1:
                item["chat_template_sha256"] = "c" * 64
                break
        with self.assertRaisesRegex(ValueError, "model chat-template drift"):
            prepare_two_model_packets(self.rows, CASES, seed=8)

        masked, key = prepare_two_model_packets(trials(), CASES, seed=8)
        packet_id = masked["packets"][0]["packet_id"]
        key["key"][packet_id]["chat_template_sha256"] = "c" * 64
        grades = {"packet_manifest_sha256": key["packet_manifest_sha256"],
                  "reviewer": {"identity": "reviewer-X", "evidence_class": "HUMAN_REVIEW"},
                  "scores": {packet["packet_id"]: 4 for packet in masked["packets"]}}
        with self.assertRaisesRegex(ValueError, "model chat-template drift"):
            adjudicate_two_model_dev(masked, key, grades, bank=CASES)

    def test_bad_bank_and_bool_values_rejected(self):
        self.rows[0]["repeat"] = True
        with self.assertRaises(ValueError):
            prepare_two_model_packets(self.rows, CASES, seed=8)
        self.rows = trials()
        with self.assertRaises(ValueError):
            prepare_two_model_packets(self.rows, {"identity": CASES["identity"]}, seed=8)


if __name__ == "__main__":
    unittest.main()
