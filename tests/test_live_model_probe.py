"""Red-first verification of bounded local model observation."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from labyrinth.live_model_probe import commit_cases, observe_cases, compare_reports


class _LocalLlamaFixture(BaseHTTPRequestHandler):
    advertised_model = ""
    prompts = []

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._respond({"data": [{"id": self.advertised_model}]})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size))
        self.prompts.append(body["messages"][0]["content"])
        self._respond({"choices": [{"message": {"content": "ANSWER: 4"}}],
                       "usage": {"completion_tokens": 3}})

    def _respond(self, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


class LiveModelProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cases = self.root / "cases.json"
        self.commitment = self.root / "commit.json"
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(b"test-only fake weights")
        self.cases.write_text(json.dumps({
            "schema": "labyrinth.live-probe-cases.v1",
            "cases": [
                {"case_id": "c1", "prompt": "2+2? Reply ANSWER: 4", "answer": "4"},
                {"case_id": "c2", "prompt": "3+3? Reply ANSWER: 6", "answer": "6"},
            ],
        }), encoding="utf-8")
        self.fixtures = type("Fixture", (_LocalLlamaFixture,), {
            "advertised_model": str(self.model),
            "prompts": [],
        })
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.fixtures)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # Cleanup executes in reverse order: stop server, close, then join.
        self.addCleanup(lambda: self.thread.join(timeout=3))
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = "http://127.0.0.1:" + str(self.server.server_port)

    def test_precommit_is_write_once_and_detects_case_change(self):
        result = commit_cases(self.cases, self.commitment)
        self.assertEqual(result["case_count"], 2)
        with self.assertRaises(FileExistsError):
            commit_cases(self.cases, self.commitment)
        source = json.loads(self.cases.read_text())
        source["cases"][0]["answer"] = "5"
        self.cases.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "commitment"):
            observe_cases(self.cases, self.commitment, self.model,
                          self.endpoint, self.root / "out.json")
        self.assertEqual(self.fixtures.prompts, [])

    def test_actual_loopback_transport_and_exact_weight_digest(self):
        commit_cases(self.cases, self.commitment)
        out = self.root / "out.json"
        result = observe_cases(self.cases, self.commitment, self.model,
                               self.endpoint, out, max_tokens=24)
        self.assertEqual(result["status"], "OBSERVED_SINGLE_MODEL_ONLY")
        self.assertEqual(result["model"]["sha256"],
                         hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(result["correct"], 1)
        self.assertEqual(len(self.fixtures.prompts), 2)
        self.assertEqual(result["trials"][0]["response"], "ANSWER: 4")
        self.assertEqual(result["trials"][1]["correct"], False)
        self.assertFalse(result["neural_learning_verified"])
        self.assertFalse(result["independent_custody_verified"])
        self.assertEqual(json.loads(out.read_text())["receipt_sha256"],
                         result["receipt_sha256"])
        with self.assertRaises(FileExistsError):
            observe_cases(self.cases, self.commitment, self.model,
                          self.endpoint, out)

    def test_reject_untrusted_endpoint_and_mismatched_server_model(self):
        commit_cases(self.cases, self.commitment)
        with self.assertRaisesRegex(ValueError, "loopback"):
            observe_cases(self.cases, self.commitment, self.model,
                          "http://example.com:11435", self.root / "out.json")
        self.fixtures.advertised_model = str(self.root / "wrong.gguf")
        with self.assertRaisesRegex(ValueError, "advertised"):
            observe_cases(self.cases, self.commitment, self.model,
                          self.endpoint, self.root / "out.json")
        self.assertEqual(self.fixtures.prompts, [])

    def test_comparison_is_not_neural_qualification(self):
        commit_cases(self.cases, self.commitment)
        before = self.root / "before.json"
        after = self.root / "after.json"
        observe_cases(self.cases, self.commitment, self.model,
                      self.endpoint, before)
        other_model = self.root / "different.gguf"
        other_model.write_bytes(b"other fixture weights")
        self.fixtures.advertised_model = str(other_model)
        observe_cases(self.cases, self.commitment, other_model,
                      self.endpoint, after)
        compared = compare_reports(before, after, self.root / "comparison.json")
        self.assertEqual(compared["status"], "PAIRED_BEHAVIOR_NOT_TRAINING_ATTESTED")
        self.assertEqual(compared["before_correct"], 1)
        self.assertEqual(compared["after_correct"], 1)
        self.assertFalse(compared["neural_learning_verified"])

    def test_forged_or_mismatched_comparison_receipts_are_rejected(self):
        commit_cases(self.cases, self.commitment)
        before = self.root / "before.json"
        after = self.root / "after.json"
        observe_cases(self.cases, self.commitment, self.model,
                      self.endpoint, before)
        observe_cases(self.cases, self.commitment, self.model,
                      self.endpoint, after)
        forged = json.loads(after.read_text(encoding="utf-8"))
        forged["correct"] = 2
        after.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            compare_reports(before, after, self.root / "comparison.json")

    def test_reject_duplicate_case_ids_before_precommit(self):
        raw = json.loads(self.cases.read_text(encoding="utf-8"))
        raw["cases"][1]["case_id"] = raw["cases"][0]["case_id"]
        self.cases.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            commit_cases(self.cases, self.commitment)
        self.assertFalse(self.commitment.exists())

    def test_comparison_same_weights_cannot_claim_update(self):
        commit_cases(self.cases, self.commitment)
        before = self.root / "before.json"
        after = self.root / "after.json"
        observe_cases(self.cases, self.commitment, self.model,
                      self.endpoint, before)
        observe_cases(self.cases, self.commitment, self.model,
                      self.endpoint, after)
        compared = compare_reports(before, after, self.root / "comparison.json")
        self.assertEqual(compared["status"], "SAME_DECLARED_FILE_HASH_NOT_TRAINING_ATTESTED")
        self.assertFalse(compared["neural_learning_verified"])


if __name__ == "__main__":
    unittest.main()
