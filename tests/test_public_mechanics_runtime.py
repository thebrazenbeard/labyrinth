"""Hybrid runtime: public rules can rescue model plans but never evidence learning."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labyrinth import public_mechanics_runtime as hybrid_module

from labyrinth.blind_structural_exam import prepare_exam, score_exam, RESPONSE_SCHEMA
from labyrinth.public_mechanics_runtime import run_public_mechanics


def world(*, goal_enabled=True, short_B=False):
    n = 8
    tables = {a: [[-1] * n for _ in range(2)] for a in "ABC"}
    tables["A"][0][0] = 3
    tables["B"][1][3] = 5
    if goal_enabled:
        tables["C"][0][5] = 7
    if short_B:
        tables["B"][0][0] = 7
    return {
        "state_count": n, "start": 0, "target": 7, "max_moves": 3,
        "action_tables": tables,
    }


class PublicMechanicsHybridTests(unittest.TestCase):
    def setUp(self):
        td = TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        self.private = self.root / "private.json"
        self.public = self.root / "public.json"
        self.precommit = self.root / "precommit.json"
        self.candidate = self.root / "candidate.json"
        self.hybrid = self.root / "hybrid.json"
        self.report = self.root / "hybrid-report.json"
        self.score = self.root / "score.json"
        self.plan = prepare_exam(self.private, self.public, self.precommit,
                                 worlds=[world(), world(goal_enabled=False)],
                                 rng=random.Random(460))
        self.public_bytes = self.public.read_bytes()
        self.original_public_sha = sha256(self.public_bytes).hexdigest()
        self.public_doc = json.loads(self.public_bytes)

    def make_candidate(self, answers):
        result = {
            "schema": RESPONSE_SCHEMA, "exam_id": self.public_doc["exam_id"],
            "responses": [
                {"trial_id": item["trial_id"], "response": response}
                for item, response in zip(self.public_doc["trials"], answers)
            ],
        }
        self.candidate.write_text(json.dumps(result), encoding="utf-8")

    def run_hybrid(self, **kwargs):
        return run_public_mechanics(
            self.public, self.candidate, self.hybrid, self.report,
            expected_public_sha256=self.original_public_sha, **kwargs)

    def test_runtime_uses_public_only_and_explicitly_labels_symbolic_fallback(self):
        self.make_candidate(["PLAN: A B C", "PLAN: A B C"])
        # Prove the runtime can execute with NO private case file available.
        hidden = self.private.with_suffix(".sealed")
        self.private.rename(hidden)
        try:
            report = self.run_hybrid()
        finally:
            hidden.rename(self.private)
        actual = json.loads(self.hybrid.read_text(encoding="utf-8"))
        self.assertEqual([r["response"] for r in actual["responses"]],
                         ["PLAN: A B C", "UNSAT"])
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["candidate_accepted_count"], 1)
        self.assertEqual(report["symbolic_fallback_count"], 1)
        self.assertEqual(
            [r["decision_source"] for r in report["case_results"]],
            ["CANDIDATE_PUBLICLY_VERIFIED", "DETERMINISTIC_PUBLIC_FALLBACK"])
        self.assertEqual(
            [r["candidate_publicly_valid"] for r in report["case_results"]],
            [True, False])
        self.assertEqual(report["candidate_invocation_evidence"], "NOT_SUPPLIED")
        self.assertFalse(report["neural_learning_verified"])
        self.assertFalse(report["independent_custody_verified"])
        self.assertFalse(report["served_weight_bytes_attested"])
        self.assertEqual(report["selected_responses_file_sha256"],
                         sha256(self.hybrid.read_bytes()).hexdigest())
        self.assertNotIn("oracle_witness", self.report.read_text(encoding="utf-8"))
        self.assertNotIn("source_class", self.report.read_text(encoding="utf-8"))
        scored = score_exam(
            self.private, self.public, self.precommit, self.hybrid,
            self.score, expected_precommit_sha256=self.plan["receipt_sha256"])
        self.assertEqual(scored["correct"], 2)
        self.assertEqual(scored["case_count"], 2)
        self.assertEqual(scored["invocation_evidence"], "NOT_SUPPLIED")

    def test_candidate_false_unsat_on_satisfiable_case_is_replaced_without_neural_credit(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        report = self.run_hybrid()
        self.assertEqual(report["candidate_accepted_count"], 1)
        self.assertEqual(report["symbolic_fallback_count"], 1)
        self.assertEqual(report["case_results"][0]["selected_response"],
                         "PLAN: A B C")
        self.assertEqual(report["case_results"][0]["decision_source"],
                         "DETERMINISTIC_PUBLIC_FALLBACK")
        self.assertEqual(report["case_results"][1]["decision_source"],
                         "CANDIDATE_PUBLICLY_VERIFIED")

    def test_valid_nonshortest_candidate_plan_is_preserved(self):
        self.private.unlink()
        self.public.unlink()
        self.precommit.unlink()
        self.plan = prepare_exam(
            self.private, self.public, self.precommit,
            worlds=[world(short_B=True)], rng=random.Random(461))
        self.public_bytes = self.public.read_bytes()
        self.public_doc = json.loads(self.public_bytes)
        self.original_public_sha = sha256(self.public_bytes).hexdigest()
        self.make_candidate(["PLAN: A B C"])
        report = self.run_hybrid()
        self.assertEqual(report["candidate_accepted_count"], 1)
        self.assertEqual(report["symbolic_fallback_count"], 0)
        self.assertEqual(report["case_results"][0]["selected_response"],
                         "PLAN: A B C")
        self.assertEqual(report["case_results"][0]["public_action_trace"],
                         [{"turn": 0, "action": "A", "phase": 0,
                           "state_before": 0, "state_after": 3},
                          {"turn": 1, "action": "B", "phase": 1,
                           "state_before": 3, "state_after": 5},
                          {"turn": 2, "action": "C", "phase": 0,
                           "state_before": 5, "state_after": 7}])

    def test_malformed_candidate_is_fallback_not_model_success(self):
        self.make_candidate(["PLAN: X", "unparsable"])
        report = self.run_hybrid()
        self.assertEqual(report["candidate_accepted_count"], 0)
        self.assertEqual(report["symbolic_fallback_count"], 2)
        self.assertFalse(any(x["candidate_publicly_valid"]
                             for x in report["case_results"]))

    def test_changed_original_public_bytes_rejected_before_outputs(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        data = json.loads(self.public.read_text(encoding="utf-8"))
        data["trials"][0]["world"]["target"] = 6
        self.public.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public|SHA|source|digest"):
            self.run_hybrid()
        self.assertFalse(self.hybrid.exists())
        self.assertFalse(self.report.exists())

    def test_semantically_equal_public_reformat_still_breaks_held_source_pin(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        public = json.loads(self.public.read_text(encoding="utf-8"))
        # Source semantics and prompt remain unchanged; only original bytes
        # differ. This catches disabled public-SHA checks independently.
        self.public.write_text(json.dumps(public, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public|source|SHA|digest"):
            self.run_hybrid()
        self.assertFalse(self.hybrid.exists())
        self.assertFalse(self.report.exists())

    def test_fallback_count_and_model_credit_cannot_be_swapped(self):
        self.make_candidate(["PLAN: A B C", "PLAN: A B C"])
        report = self.run_hybrid()
        self.assertEqual(report["status"],
                         "PUBLIC_RULE_HYBRID_NOT_NEURAL_LEARNING")
        self.assertEqual(report["candidate_accepted_count"], 1)
        self.assertEqual(report["symbolic_fallback_count"], 1)
        self.assertEqual(sum(x["decision_source"] ==
                             "DETERMINISTIC_PUBLIC_FALLBACK"
                             for x in report["case_results"]), 1)
        self.assertEqual(sum(x["decision_source"] ==
                             "CANDIDATE_PUBLICLY_VERIFIED"
                             for x in report["case_results"]), 1)
        self.assertFalse(report["neural_learning_verified"])
        self.assertNotIn("model_wins", report)

    def test_duplicate_reordered_or_extra_candidate_rows_fail_closed(self):
        for variant in ("reordered", "extra", "missing"):
            with self.subTest(variant=variant):
                self.make_candidate(["UNSAT", "UNSAT"])
                obj = json.loads(self.candidate.read_text())
                if variant == "reordered":
                    obj["responses"].reverse()
                elif variant == "extra":
                    obj["responses"][0]["private_witness"] = ["A", "B", "C"]
                else:
                    obj["responses"].pop()
                self.candidate.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "candidate|roster|response"):
                    self.run_hybrid()
                self.assertFalse(self.hybrid.exists())
                self.assertFalse(self.report.exists())

    def test_duplicate_candidate_json_key_cannot_silently_override_answer(self):
        self.make_candidate(["PLAN: A B C", "UNSAT"])
        raw = self.candidate.read_text(encoding="utf-8")
        self.assertIn('"response": "PLAN: A B C"', raw)
        raw = raw.replace('"response": "PLAN: A B C"',
                          '"response": "UNSAT", "response": "PLAN: A B C"', 1)
        self.candidate.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate|candidate"):
            self.run_hybrid()
        self.assertFalse(self.hybrid.exists())
        self.assertFalse(self.report.exists())

    def test_requires_both_original_observation_and_held_pin(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        with self.assertRaises(ValueError):
            self.run_hybrid(observation_path=self.root / "no-observation.json")
        with self.assertRaises(ValueError):
            self.run_hybrid(expected_observation_sha256="0" * 64)
        self.assertFalse(self.hybrid.exists())
        self.assertFalse(self.report.exists())

    def test_missing_or_fake_original_observation_rejected_before_fallback(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_hybrid(
                observation_path=self.root / "missing.json",
                expected_observation_sha256="0" * 64)
        self.assertFalse(self.hybrid.exists())
        self.assertFalse(self.report.exists())

    def test_failed_provenance_receipt_cannot_publish_unlabeled_corrected_answers(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        unreachable = self.root / "missing-parent" / "provenance.json"
        with self.assertRaises((FileNotFoundError, OSError)):
            run_public_mechanics(
                self.public, self.candidate, self.hybrid, unreachable,
                expected_public_sha256=self.original_public_sha)
        self.assertFalse(self.hybrid.exists(),
                         "corrected response cannot appear without provenance receipt")
        self.assertFalse(unreachable.exists())

    def test_receipt_write_failure_after_preflight_never_emits_corrected_roster(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        original_write = hybrid_module._write_once

        def fail_receipt_only(path, doc):
            if Path(path) == self.report:
                raise OSError("injected provenance receipt write failure")
            return original_write(path, doc)

        with patch.object(hybrid_module, "_write_once",
                          side_effect=fail_receipt_only):
            with self.assertRaisesRegex(OSError, "injected provenance"):
                self.run_hybrid()
        self.assertFalse(self.hybrid.exists(),
                         "corrected response cannot be published before receipt")
        self.assertFalse(self.report.exists())

    def test_cannot_overwrite_hybrid_or_receipt(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        original = self.run_hybrid()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_hybrid()
        self.assertEqual(original["receipt_sha256"],
                         json.loads(self.report.read_text())["receipt_sha256"])

    def test_candidate_input_file_cannot_be_one_of_output_paths(self):
        self.make_candidate(["UNSAT", "UNSAT"])
        with self.assertRaises(ValueError):
            run_public_mechanics(
                self.public, self.candidate, self.candidate, self.report,
                expected_public_sha256=self.original_public_sha)
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
