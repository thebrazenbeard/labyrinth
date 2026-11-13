"""Local blind-exam HTTP evidence: real loopback transport, no trusted model attestation."""
from __future__ import annotations

from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import threading
import unittest

from labyrinth.blind_structural_exam import prepare_exam, score_exam
from labyrinth.local_neural_observer import observe_public_exam, verify_local_observation


class LocalModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private = self.root / "private.json"
        self.public = self.root / "public.json"
        self.precommit = self.root / "precommit.json"
        self.responses = self.root / "responses.json"
        self.observation = self.root / "observation.json"
        self.score_file = self.root / "score.json"
        self.plan = prepare_exam(self.private, self.public, self.precommit,
                                 count=2, rng=random.Random(39))
        self.model = "qwen3.5:0.8b"
        self.state = {"gets": 0, "posts": [], "first_digest": "b" * 64,
                      "last_digest": "b" * 64, "reply_model": self.model,
                      "reply": "PLAN: A", "redirect": False}
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, value, status=200, headers=None):
                raw = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for key, item in (headers or {}).items():
                    self.send_header(key, item)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path != "/api/tags":
                    return self._send({"error": "bad route"}, 404)
                state["gets"] += 1
                digest = (state["first_digest"] if state["gets"] == 1
                          else state["last_digest"])
                self._send({"models": [{
                    "name": "qwen3.5:0.8b", "digest": digest, "size": 5000
                }]})

            def do_POST(self):
                if self.path != "/api/generate":
                    return self._send({"error": "bad route"}, 404)
                if state["redirect"]:
                    return self._send({"redirect": True}, 302,
                                      {"Location": "http://198.51.100.2:9999/"})
                size = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(size))
                state["posts"].append(req)
                self._send({"model": state["reply_model"],
                            "response": state["reply"],
                            "done": True, "done_reason": "stop",
                            "eval_count": 3})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def run_observe(self):
        return observe_public_exam(self.public, self.responses,
                                   self.observation, self.base, self.model)

    def test_actual_transport_records_exact_public_prompts_responses_and_tag_digest(self):
        observed = self.run_observe()
        public = json.loads(self.public.read_text(encoding="utf-8"))
        answers = json.loads(self.responses.read_text(encoding="utf-8"))
        self.assertEqual(self.state["gets"], 2)
        self.assertEqual(len(self.state["posts"]), 2)
        self.assertEqual([call["prompt"] for call in self.state["posts"]],
                         [row["prompt"] for row in public["trials"]])
        self.assertTrue(all(call["stream"] is False and call["think"] is False
                            and call["options"]["num_gpu"] == 0
                            for call in self.state["posts"]))
        self.assertEqual([x["response"] for x in answers["responses"]],
                         ["PLAN: A", "PLAN: A"])
        self.assertEqual(observed["declared_model_digest"], "b" * 64)
        self.assertEqual(observed["public_file_sha256"], sha256(self.public.read_bytes()).hexdigest())
        self.assertEqual(observed["responses_file_sha256"],
                         sha256(self.responses.read_bytes()).hexdigest())
        self.assertEqual([t["prompt_sha256"] for t in observed["trials"]],
                         [sha256(p["prompt"].encode()).hexdigest()
                          for p in public["trials"]])
        self.assertFalse(observed["independent_custody_verified"])
        self.assertFalse(observed["served_weight_bytes_attested"])

    def test_readback_rechecks_raw_response_file_and_original_receipt_pin(self):
        receipt = self.run_observe()
        verified = verify_local_observation(self.observation, receipt["receipt_sha256"],
                                            self.public.read_bytes(),
                                            self.responses.read_bytes())
        self.assertEqual(verified["evidence"], "LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED")
        self.assertEqual(verified["declared_model_digest"], "b" * 64)
        self.assertFalse(verified["served_weight_bytes_attested"])
        with self.assertRaisesRegex(ValueError, "pin|observation"):
            verify_local_observation(self.observation, "0" * 64,
                                     self.public.read_bytes(), self.responses.read_bytes())
        self.assertFalse(self.score_file.exists())

    def test_score_can_check_original_http_observation_digest_without_promoting_custody(self):
        receipt = self.run_observe()
        result = score_exam(
            self.private, self.public, self.precommit, self.responses,
            self.score_file, expected_precommit_sha256=self.plan["receipt_sha256"],
            observation_path=self.observation,
            expected_observation_sha256=receipt["receipt_sha256"])
        self.assertEqual(result["invocation_evidence"],
                         "LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED")
        self.assertEqual(result["observed_model_digest"], "b" * 64)
        self.assertFalse(result["independent_custody_verified"])
        self.assertFalse(result["neural_learning_verified"])

    def test_rehashed_responses_are_rejected_against_original_http_receipt(self):
        receipt = self.run_observe()
        data = json.loads(self.responses.read_text(encoding="utf-8"))
        data["responses"][0]["response"] = "UNSAT"
        self.responses.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "response|observation"):
            verify_local_observation(self.observation, receipt["receipt_sha256"],
                                     self.public.read_bytes(), self.responses.read_bytes())

    def test_model_digest_changes_mid_exam_rejected_without_outputs(self):
        self.state["last_digest"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "digest|model"):
            self.run_observe()
        self.assertFalse(self.responses.exists())
        self.assertFalse(self.observation.exists())

    def test_different_server_reply_model_rejected(self):
        self.state["reply_model"] = "other-model"
        with self.assertRaisesRegex(ValueError, "model|reply"):
            self.run_observe()
        self.assertFalse(self.responses.exists())

    def test_private_answer_in_public_bundle_is_rejected_before_model_request(self):
        public = json.loads(self.public.read_text(encoding="utf-8"))
        public["trials"][0]["oracle_witness"] = ["A"]
        self.public.write_text(json.dumps(public), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public|trial|case"):
            self.run_observe()
        self.assertEqual(len(self.state["posts"]), 0)

    def test_remote_endpoints_redirects_and_ambiguous_loopback_names_are_rejected(self):
        for bad in ("http://localhost:11434", "http://198.51.100.10:1234",
                    "https://127.0.0.1:443", "http://127.0.0.1:1234/a",
                    "http://foo@127.0.0.1:1234"):
            with self.subTest(endpoint=bad), self.assertRaisesRegex(ValueError, "endpoint|loopback"):
                observe_public_exam(self.public, self.responses, self.observation,
                                    bad, self.model)
        self.state["redirect"] = True
        with self.assertRaises(Exception):
            self.run_observe()
        self.assertFalse(self.responses.exists())

    def test_both_response_and_observation_outputs_require_distinct_new_paths(self):
        with self.assertRaisesRegex(ValueError, "distinct|fresh|output"):
            observe_public_exam(self.public, self.responses, self.responses,
                                self.base, self.model)
        self.responses.write_text("reserved", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distinct|fresh|output"):
            self.run_observe()
        self.assertEqual(len(self.state["posts"]), 0)


if __name__ == "__main__":
    unittest.main()
