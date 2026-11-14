"""A corrected hybrid roster is not a raw model answer; require its full proof."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from labyrinth.blind_structural_exam import prepare_exam, score_exam, RESPONSE_SCHEMA
from labyrinth.public_mechanics_runtime import run_public_mechanics
from labyrinth.hybrid_artifact_verifier import verify_hybrid_artifacts


def simple_world(*, reachable=True):
    n = 6
    tables = {a: [[-1] * n for _ in range(2)] for a in "ABC"}
    tables["A"][0][0] = 2
    tables["B"][1][2] = 4
    if reachable:
        tables["C"][0][4] = 5
    return {
        "state_count": n, "start": 0, "target": 5,
        "max_moves": 3, "action_tables": tables,
    }


class HybridVerifierTests(unittest.TestCase):
    def setUp(self):
        t = TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.private = self.root / "private.json"
        self.public = self.root / "public.json"
        self.precommit = self.root / "precommit.json"
        self.candidate = self.root / "original-candidate.json"
        self.selected = self.root / "selected.json"
        self.report = self.root / "provenance.json"
        self.score = self.root / "raw-model-score.json"
        self.plan = prepare_exam(
            self.private, self.public, self.precommit,
            worlds=[simple_world(), simple_world(reachable=False)],
            rng=random.Random(4701))
        source = json.loads(self.public.read_bytes())
        self.pubsha = sha256(self.public.read_bytes()).hexdigest()
        responses = {
            "schema": RESPONSE_SCHEMA, "exam_id": source["exam_id"],
            "responses": [
                {"trial_id": item["trial_id"], "response": "UNSAT"}
                for item in source["trials"]
            ],
        }
        self.candidate.write_text(json.dumps(responses), encoding="utf-8")
        self.generated = run_public_mechanics(
            self.public, self.candidate, self.selected, self.report,
            expected_public_sha256=self.pubsha)
        self.pin = self.generated["receipt_sha256"]

    def verify(self, **kwargs):
        return verify_hybrid_artifacts(
            self.public, self.candidate, self.selected, self.report,
            expected_public_sha256=kwargs.pop("expected_public_sha256", self.pubsha),
            expected_report_sha256=kwargs.pop("expected_report_sha256", self.pin),
            **kwargs)

    def test_v2_verifier_accepts_bound_complete_artifact_without_private_file(self):
        sealed = self.private.with_suffix(".sealed")
        self.private.rename(sealed)
        try:
            result = self.verify()
        finally:
            sealed.rename(self.private)
        self.assertEqual(result["status"],
                         "PUBLIC_HYBRID_SOURCE_VERIFIED_NOT_MODEL_ATTESTED")
        self.assertEqual(result["candidate_accepted_count"], 1)
        self.assertEqual(result["symbolic_fallback_count"], 1)
        self.assertEqual(result["selected_correct_count"], 2)
        self.assertFalse(result["neural_learning_verified"])
        self.assertFalse(result["independent_custody_verified"])
        self.assertFalse(result["served_weight_bytes_attested"])
        self.assertEqual(result["report_receipt_sha256"], self.pin)

    def test_raw_candidate_examiner_rejects_v2_corrected_answer_schema(self):
        self.assertEqual(json.loads(self.selected.read_text())["schema"],
                         "labyrinth.public-mechanics-selected.v2")
        with self.assertRaisesRegex(ValueError, "candidate|response|schema"):
            score_exam(self.private, self.public, self.precommit,
                       self.selected, self.score,
                       expected_precommit_sha256=self.plan["receipt_sha256"])
        self.assertFalse(self.score.exists())

    def test_missing_provenance_report_must_not_be_a_hybrid_success(self):
        self.report.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.verify()

    def test_wrong_original_provenance_receipt_pin_fails(self):
        with self.assertRaisesRegex(ValueError, "report|receipt|pin|SHA"):
            self.verify(expected_report_sha256="0" * 64)

    def test_altered_selected_answer_rejected_even_when_mechanically_correct(self):
        doc = json.loads(self.selected.read_text())
        doc["responses"][0]["response"] = "PLAN: A B C"  # same mechanical answer
        self.selected.write_text(json.dumps(doc, indent=4))
        with self.assertRaises(ValueError):
            self.verify()

    def test_altered_candidate_after_original_run_rejected(self):
        doc = json.loads(self.candidate.read_text())
        doc["responses"][0]["response"] = "PLAN: A B C"
        self.candidate.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            self.verify()

    def test_rehashed_self_signed_report_cannot_override_held_original_pin(self):
        doc = json.loads(self.report.read_text())
        doc["candidate_accepted_count"] = 2
        doc.pop("receipt_sha256")
        canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode()
        doc["receipt_sha256"] = sha256(canonical).hexdigest()
        self.report.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            self.verify()

    def test_old_v1_candidate_schema_is_not_accepted_as_v2_hybrid(self):
        doc = json.loads(self.selected.read_text())
        doc["schema"] = RESPONSE_SCHEMA
        self.selected.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            self.verify()

    def test_mutated_original_public_source_detected(self):
        self.public.write_bytes(self.public.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "public|source|SHA"):
            self.verify()

    def test_missing_original_observation_pin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(observation_path=self.root / "missing-model.json")
        with self.assertRaises(ValueError):
            self.verify(expected_observation_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
