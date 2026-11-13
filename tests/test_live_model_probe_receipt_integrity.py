"""Adversarial receipt-integrity tests: source-comparison cannot trust self-signed claims.

These tests deliberately construct observations without calling the model transport.
They demonstrate the limited authenticity of unkeyed receipts; they do NOT
assert that a forged receipt came from any running neural model.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.live_model_probe import compare_reports


def _receipt(payload):
    """Test-owned canonical hashing, independent of production receipt helpers."""
    value = dict(payload)
    value.pop("receipt_sha256", None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    value["receipt_sha256"] = sha256(raw).hexdigest()
    return value


def _observation():
    response = "ANSWER: 4"
    return {
        "schema": "labyrinth.live-probe-observation.v1",
        "status": "OBSERVED_SINGLE_MODEL_ONLY",
        "commitment_sha256": "a" * 64,
        "cases_file_sha256": "b" * 64,
        "model": {
            "path": "/tmp/fake-model.gguf",
            "sha256": "c" * 64,
            "server_advertised_path_match": True,
            "server_loaded_weight_bytes_independently_attested": False,
        },
        "endpoint": "http://127.0.0.1:11435",
        "case_count": 1,
        "correct": 1,
        "format_valid": 1,
        "trials": [{
            "case_id": "trial-1",
            "prompt_sha256": "d" * 64,
            "response": response,
            "response_sha256": sha256(response.encode("utf-8")).hexdigest(),
            "format_valid": True,
            "correct": True,
            "elapsed_ms": 1,
        }],
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "limitations": [],
    }


class ObservationReceiptHostileTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _compare(self, after, *, before=None):
        before_path = self.root / "before.json"
        after_path = self.root / "after.json"
        before_path.write_text(json.dumps(_receipt(_observation() if before is None else before)), encoding="utf-8")
        after_path.write_text(json.dumps(_receipt(after)), encoding="utf-8")
        return compare_reports(before_path, after_path, self.root / "paired.json")

    def test_consistent_observations_still_compare_without_training_claim(self):
        result = self._compare(_observation())
        self.assertEqual(result["before_correct"], 1)
        self.assertEqual(result["after_correct"], 1)
        self.assertEqual(result["status"], "SAME_CHECKPOINT_NO_WEIGHT_UPDATE")
        self.assertFalse(result["neural_learning_verified"])

    def test_reject_rehashed_aggregate_that_disagrees_with_trial_outcomes(self):
        forged = _observation()
        forged["correct"] = 0
        with self.assertRaisesRegex(ValueError, "observation|aggregate|score"):
            self._compare(forged)

    def test_reject_rehashed_response_hash_mismatch(self):
        forged = _observation()
        forged["trials"][0]["response_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "observation|response|hash"):
            self._compare(forged)

    def test_reject_rehashed_format_flag_inconsistent_with_model_text(self):
        forged = _observation()
        forged["trials"][0]["format_valid"] = False
        forged["format_valid"] = 0
        with self.assertRaisesRegex(ValueError, "observation|format"):
            self._compare(forged)

    def test_reject_nonboolean_trial_correctness(self):
        forged = _observation()
        forged["trials"][0]["correct"] = "false"
        with self.assertRaisesRegex(ValueError, "observation|correct|boolean"):
            self._compare(forged)

    def test_reject_duplicate_trial_identity_with_rehashed_receipt(self):
        forged = _observation()
        forged["case_count"] = 2
        forged["trials"].append(dict(forged["trials"][0]))
        forged["correct"] = 2
        forged["format_valid"] = 2
        with self.assertRaisesRegex(ValueError, "observation|duplicate"):
            self._compare(forged, before=forged)


if __name__ == "__main__":
    unittest.main()
