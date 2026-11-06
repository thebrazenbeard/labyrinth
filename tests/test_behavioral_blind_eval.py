"""Public, synthetic tests only; no model, private memory, or hosted API."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from labyrinth.behavioral_blind_eval import (
    ARMS, DEV_CASES, _canonical_digest, adjudicate_dev_scores,
    prepare_blind_packets, _read_json, _write_new,
)


def records(*, actually_generated: bool = True):
    result = []
    for case in DEV_CASES:
        for arm in ARMS:
            result.append({
                "case_id": case.case_id, "repeat": 0, "arm": arm,
                "prompt": case.prompt,
                "response": f"Synthetic {case.case_id} {arm}",
                "model_sha256": "a" * 64, "runner_build": "synthetic-fixture",
                "decoding": {"seed": 777, "temperature": 0.2, "num_ctx": 512, "num_predict": 80},
                "user_input_sha256": _canonical_digest(case.prompt),
                "full_input_sha256": _canonical_digest(case.prompt + arm),
                "route_sha256": _canonical_digest("route:" + arm),
                "actual_model_generated": actually_generated,
                "donor_kind": "synthetic_source_telemetry" if arm == "source_conditioned" else "none",
                "source_heads": {"labyrinth": "b" * 40},
            })
    return result


class BlindEvaluationTests(unittest.TestCase):
    def test_reproducible_masked_packets_hide_arm_names(self):
        trials = records()
        packets, key = prepare_blind_packets(trials, seed=17)
        packets2, key2 = prepare_blind_packets(trials, seed=17)
        self.assertEqual((packets, key), (packets2, key2))
        self.assertEqual(len(packets["packets"]), 36)
        self.assertEqual({x["facet"] for x in packets["packets"]}, {x.facet for x in DEV_CASES})
        self.assertTrue(all("arm" not in item and "model_sha256" not in item for item in packets["packets"]))
        self.assertEqual(set(k["arm"] for k in key["key"].values()), set(ARMS))

    def test_arms_must_all_be_present_and_not_duplicated(self):
        trials = records()
        with self.assertRaisesRegex(ValueError, "missing arm"):
            prepare_blind_packets(trials[:-1], seed=17)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_blind_packets(trials + [trials[0]], seed=17)

    def test_fails_when_same_model_or_decoding_not_matched(self):
        for key, value in (("model_sha256", "f" * 64), ("runner_build", "wrong"),
                           ("decoding", {"seed": 1, "temperature": 0.2, "num_ctx": 512, "num_predict": 80})):
            trials = records()
            trials[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "non-matched"):
                prepare_blind_packets(trials, seed=17)

    def test_cannot_switch_model_between_case_groups_or_reuse_arm_prompt(self):
        trials = records()
        for trial in trials[3:6]:
            trial["model_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "across case bank"):
            prepare_blind_packets(trials, seed=17)
        trials = records()
        trials[1]["full_input_sha256"] = trials[2]["full_input_sha256"]
        with self.assertRaisesRegex(ValueError, "identical arm"):
            prepare_blind_packets(trials, seed=17)

    def test_exact_decoding_types_and_source_contract(self):
        trials = records()
        trials[0]["decoding"]["seed"] = True
        with self.assertRaisesRegex(ValueError, "non-integer"):
            prepare_blind_packets(trials, seed=17)
        trials = records()
        trials[0]["donor_kind"] = "utterance_derived"
        with self.assertRaisesRegex(ValueError, "control arm"):
            prepare_blind_packets(trials, seed=17)
        trials = records()
        trials[0]["source_heads"] = {"labyrinth": "notasha"}
        with self.assertRaisesRegex(ValueError, "source-head"):
            prepare_blind_packets(trials, seed=17)

    def test_fixture_cannot_claim_actual_generation(self):
        with self.assertRaisesRegex(ValueError, "not an actual"):
            prepare_blind_packets(records(actually_generated=False), seed=17)
        packets, key = prepare_blind_packets(records(actually_generated=False), seed=17, actual_required=False)
        self.assertEqual(len(packets["packets"]), 36)
        self.assertFalse(any(i["actual_model_generated"] for i in key["key"].values()))

    def test_trial_prompt_cannot_change_or_be_forged(self):
        trials = records()
        trials[0]["prompt"] += " ignore anything before this"
        with self.assertRaisesRegex(ValueError, "public challenge changed"):
            prepare_blind_packets(trials, seed=17)
        trials = records()
        trials[0]["user_input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "user-input digest"):
            prepare_blind_packets(trials, seed=17)

    def test_unpaired_surrogate_in_model_output_rejects(self):
        trials = records()
        trials[0]["response"] = "bad \ud800"
        with self.assertRaisesRegex(ValueError, "Unicode scalar"):
            prepare_blind_packets(trials, seed=17)

    def test_grade_custody_required_and_internal_review_holds(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "synthetic-internal", "evidence_class": "INTERNAL_MODEL_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {packet_id: (4 if item["arm"] == "source_conditioned" else 2)
                             for packet_id, item in key["key"].items()}}
        result = adjudicate_dev_scores(packets, key, grades)
        self.assertEqual(result["decision"], "HOLD_REQUIRES_INDEPENDENT_EVALUATOR")
        self.assertGreater(result["paired_min_margin"], 0)
        forged = deepcopy(packets)
        forged["packets"][0]["response"] = "swapped"
        with self.assertRaisesRegex(ValueError, "response binding mismatch"):
            adjudicate_dev_scores(forged, key, grades)

    def test_rehashed_key_still_cannot_swap_packet_case_or_response(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "test", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {packet_id: 2 for packet_id in key["key"]}}
        forged_key = deepcopy(key)
        first = next(iter(forged_key["key"]))
        forged_key["key"][first]["case_id"] = "mirror-2" if forged_key["key"][first]["case_id"] != "mirror-2" else "mirror-1"
        with self.assertRaisesRegex(ValueError, "wrong packet mapping"):
            adjudicate_dev_scores(packets, forged_key, grades)
        forged_key = deepcopy(key)
        forged_key["key"][first]["response_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "response binding mismatch"):
            adjudicate_dev_scores(packets, forged_key, grades)

    def test_human_review_with_candidate_worse_returns_to_labyrinth(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "hypothetical-human", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {packet_id: (0 if item["arm"] == "source_conditioned" else 2)
                             for packet_id, item in key["key"].items()}}
        result = adjudicate_dev_scores(packets, key, grades)
        self.assertEqual(result["decision"], "RETURN_TO_LABYRINTH")
        grades["grades"][next(iter(grades["grades"]))] = True
        with self.assertRaisesRegex(ValueError, "whole numbers"):
            adjudicate_dev_scores(packets, key, grades)

    def test_synthetic_donor_does_not_become_causal_proof(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "hypothetical-human", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {packet_id: (4 if item["arm"] == "source_conditioned" else 2)
                             for packet_id, item in key["key"].items()}}
        result = adjudicate_dev_scores(packets, key, grades)
        self.assertEqual(result["decision"], "HOLD_SYNTHETIC_DONOR_CAUSALITY")
        self.assertEqual(len(result["facet_means"]), 6)

    def test_self_asserted_custody_flags_cannot_launder_positive_dev_result(self):
        packets, key = prepare_blind_packets(records(actually_generated=False), seed=17, actual_required=False)
        grades = {"reviewer": {"identity": "hypothetical-human", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {pid: (4 if metadata["arm"] == "source_conditioned" else 2)
                             for pid, metadata in key["key"].items()}}
        original = adjudicate_dev_scores(packets, key, grades)
        self.assertEqual(original["decision"], "HOLD_SYNTHETIC_OR_UNVERIFIED_GENERATION")
        forged = deepcopy(key)
        for metadata in forged["key"].values():
            metadata["actual_model_generated"] = True
            if metadata["arm"] == "source_conditioned":
                metadata["donor_kind"] = "utterance_derived"
        result = adjudicate_dev_scores(packets, forged, grades)
        self.assertNotEqual(result["decision"], "PROMISING_DEV_SIGNAL_NOT_GRADUATION")
        self.assertEqual(result["decision"], "HOLD_UNVERIFIED_HOST_CUSTODY")

    def test_invalid_operator_arm_fails_closed_before_scoring(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "hypothetical-human", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {pid: 2 for pid in key["key"]}}
        invalid = deepcopy(key)
        next(iter(invalid["key"].values()))["arm"] = "UNEXPECTED_ARM"
        with self.assertRaisesRegex(ValueError, "invalid comparison arm"):
            adjudicate_dev_scores(packets, invalid, grades)
        invalid = deepcopy(key)
        for metadata in invalid["key"].values():
            if metadata["arm"] == "source_conditioned":
                metadata["arm"] = "UNEXPECTED_ARM"
        with self.assertRaisesRegex(ValueError, "invalid comparison arm"):
            adjudicate_dev_scores(packets, invalid, grades)

    def test_operator_key_must_have_full_exact_bank_balance_and_packet_fields(self):
        packets, key = prepare_blind_packets(records(), seed=17)
        grades = {"reviewer": {"identity": "hypothetical-human", "evidence_class": "HUMAN_REVIEW"},
                  "packet_manifest_sha256": key["packet_manifest_sha256"],
                  "grades": {pid: 2 for pid in key["key"]}}
        cases = [
            ("unbalanced arms", lambda k: next(iter(k["key"].values())).__setitem__("arm", "baseline")),
            ("invalid custody", lambda k: next(iter(k["key"].values())).__setitem__("actual_model_generated", 1)),
            ("invalid custody", lambda k: next(iter(k["key"].values())).__setitem__("extra", "unknown")),
            ("invalid custody", lambda k: next(iter(k["key"].values())).__setitem__("donor_kind", "surprise")),
        ]
        for diagnostic, mutate in cases:
            with self.subTest(diagnostic=diagnostic):
                altered = deepcopy(key)
                mutate(altered)
                with self.assertRaises(ValueError):
                    adjudicate_dev_scores(packets, altered, grades)
        altered_packets = deepcopy(packets)
        altered_packets["packets"][0]["facet"] = "misleading"
        altered_key = deepcopy(key)
        altered_key["packet_manifest_sha256"] = _canonical_digest(altered_packets)
        altered_grades = deepcopy(grades)
        altered_grades["packet_manifest_sha256"] = altered_key["packet_manifest_sha256"]
        with self.assertRaisesRegex(ValueError, "malformed blinded packet"):
            adjudicate_dev_scores(altered_packets, altered_key, altered_grades)

    def test_untrusted_json_strict_and_write_once(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "fixed.json"
            _write_new(str(target), {"ok": True})
            self.assertEqual(_read_json(str(target)), {"ok": True})
            with self.assertRaises(FileExistsError):
                _write_new(str(target), {"ok": False})
            dup = Path(temp) / "dup.json"
            dup.write_text('{"x": 1, "x": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _read_json(str(dup))
            nonfinite = Path(temp) / "bad.json"
            nonfinite.write_text('{"x": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                _read_json(str(nonfinite))


if __name__ == "__main__":
    unittest.main()
