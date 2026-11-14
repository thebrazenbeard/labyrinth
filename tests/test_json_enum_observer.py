"""Hostile actual-TCP tests for distinct constrained JSON model observations."""
from __future__ import annotations

from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import threading
import unittest

from labyrinth.blind_structural_exam import prepare_exam, score_exam, _public_trial
from labyrinth.json_enum_observer import observe_json_enum_exam
from labyrinth.local_neural_observer import verify_local_observation


class JsonEnumModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private = self.root / "private.json"
        self.public = self.root / "public.json"
        self.precommit = self.root / "precommit.json"
        self.responses = self.root / "responses.json"
        self.observation = self.root / "observation.json"
        self.score = self.root / "score.json"
        self.plan = prepare_exam(self.private, self.public, self.precommit,
                                 count=2, rng=random.Random(45))
        self.model = "qwen3:4b-instruct"
        self.state = {
            "gets": 0, "posts": [], "digest": "d" * 64,
            "last_digest": "d" * 64, "reply_model": self.model,
            "raw": '{"answer":"UNSAT"}', "done": True,
            "done_reason": "stop", "eval_count": 11,
        }
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body, status=200):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path != "/api/tags":
                    return self._send({"error": "invalid"}, 404)
                state["gets"] += 1
                digest = (state["digest"] if state["gets"] <= 2
                          else state["last_digest"])
                self._send({"models": [{"name": "qwen3:4b-instruct",
                                        "digest": digest}]})

            def do_POST(self):
                if self.path != "/api/generate":
                    return self._send({"error": "invalid"}, 404)
                size = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(size))
                state["posts"].append(req)
                self._send({
                    "model": state["reply_model"], "response": state["raw"],
                    "done": state["done"], "done_reason": state["done_reason"],
                    "eval_count": state["eval_count"],
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def run_observer(self):
        return observe_json_enum_exam(
            self.public, self.responses, self.observation, self.base, self.model)

    def test_real_loopback_constrains_only_syntactic_plan_choices_without_answers(self):
        witness = self.run_observer()
        public = json.loads(self.public.read_text())
        output = json.loads(self.responses.read_text())
        self.assertEqual(self.state["gets"], 5)
        self.assertEqual(len(self.state["posts"]), 2)
        for call, trial in zip(self.state["posts"], public["trials"]):
            self.assertEqual(call["prompt"], trial["prompt"])
            self.assertFalse(call["stream"])
            self.assertFalse(call["think"])
            self.assertEqual(call["options"]["num_gpu"], 0)
            self.assertEqual(call["options"]["seed"], 17)
            self.assertEqual(call["options"]["temperature"], 0)
            choice = call["format"]["properties"]["answer"]["enum"]
            self.assertIn("UNSAT", choice)
            self.assertIn("PLAN: A B C", choice)
            self.assertIn("PLAN: C", choice)
            self.assertEqual(len(choice), 40)
            self.assertEqual(set(call["format"]), {
                "type", "properties", "required", "additionalProperties"})
            # Source-defined PUBLIC prompt mentions the word "verdict".
            # Only the JSON constraint may not carry private verdict data.
            self.assertNotIn("oracle_witness", json.dumps(call))
            self.assertNotIn("oracle_witness", json.dumps(call["format"]))
            self.assertNotIn("verdict", json.dumps(call["format"]))
        self.assertEqual([r["response"] for r in output["responses"]],
                         ["UNSAT", "UNSAT"])
        self.assertEqual(witness["declared_model_digest"], "d" * 64)
        self.assertEqual(witness["model_tag_check_count"], 5)
        self.assertTrue(all(row["raw_json_response"] == '{"answer":"UNSAT"}'
                            for row in witness["trials"]))
        self.assertFalse(witness["served_weight_bytes_attested"])
        self.assertFalse(witness["independent_custody_verified"])
        self.assertFalse(witness["neural_learning_verified"])

    def test_original_json_and_response_file_replay_through_existing_scorer(self):
        observation = self.run_observer()
        verified = verify_local_observation(
            self.observation, observation["receipt_sha256"],
            self.public.read_bytes(), self.responses.read_bytes())
        self.assertEqual(verified["evidence"],
                         "LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED")
        self.assertFalse(verified["served_weight_bytes_attested"])
        scored = score_exam(
            self.private, self.public, self.precommit, self.responses, self.score,
            expected_precommit_sha256=self.plan["receipt_sha256"],
            observation_path=self.observation,
            expected_observation_sha256=observation["receipt_sha256"])
        self.assertEqual(scored["invocation_evidence"],
                         "LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED")
        self.assertFalse(scored["neural_learning_verified"])
        self.assertFalse(scored["independent_custody_verified"])

    def test_invalid_answer_not_in_syntactic_enum_rejected_before_output(self):
        self.state["raw"] = '{"answer":"WIN"}'
        with self.assertRaises(ValueError):
            self.run_observer()
        self.assertFalse(self.responses.exists())
        self.assertFalse(self.observation.exists())

    def test_duplicate_json_keys_rejected_instead_of_overwriting_choice(self):
        self.state["raw"] = '{"answer":"UNSAT","answer":"PLAN: A B C"}'
        with self.assertRaises(ValueError):
            self.run_observer()
        self.assertFalse(self.responses.exists())

    def test_extra_json_property_rejected(self):
        self.state["raw"] = '{"answer":"UNSAT","private_witness":"LEAK"}'
        with self.assertRaises(ValueError):
            self.run_observer()
        self.assertFalse(self.responses.exists())

    def test_token_limit_completion_cannot_be_scored_as_model_response(self):
        self.state["done_reason"] = "length"
        with self.assertRaises(ValueError):
            self.run_observer()
        self.assertFalse(self.responses.exists())
        self.assertFalse(self.observation.exists())

    def test_different_server_model_rejected(self):
        self.state["reply_model"] = "not-requested"
        with self.assertRaises(ValueError):
            self.run_observer()
        self.assertFalse(self.responses.exists())

    def test_changed_advertised_model_digest_rejected(self):
        self.state["last_digest"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "digest|model"):
            self.run_observer()
        self.assertFalse(self.observation.exists())

    def test_held_original_observation_pin_blocks_rewritten_answers(self):
        evidence = self.run_observer()
        output = json.loads(self.responses.read_text())
        output["responses"][0]["response"] = "PLAN: A"
        self.responses.write_text(json.dumps(output))
        with self.assertRaises(ValueError):
            verify_local_observation(
                self.observation, evidence["receipt_sha256"],
                self.public.read_bytes(), self.responses.read_bytes())
        with self.assertRaises(ValueError):
            score_exam(
                self.private, self.public, self.precommit, self.responses,
                self.score, expected_precommit_sha256=self.plan["receipt_sha256"],
                observation_path=self.observation,
                expected_observation_sha256=evidence["receipt_sha256"])
        self.assertFalse(self.score.exists())

    def test_rehashed_forged_json_report_fails_original_pin(self):
        evidence = self.run_observer()
        report = json.loads(self.observation.read_text())
        report["trials"][0]["raw_json_response"] = '{"answer":"PLAN: A"}'
        report.pop("receipt_sha256")
        report["receipt_sha256"] = sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.observation.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            verify_local_observation(
                self.observation, evidence["receipt_sha256"],
                self.public.read_bytes(), self.responses.read_bytes())

    def test_four_move_question_rejected_before_any_model_request(self):
        # Keep enum enumeration bounded to <=40 strings; never silently
        # broaden an unverified small-model format protocol.
        bundle = json.loads(self.public.read_text(encoding="utf-8"))
        row = bundle["trials"][0]
        world = row["world"]
        world["max_moves"] = 4
        row["prompt"] = _public_trial({"trial_id": row["trial_id"],
                                       "world": world})["prompt"]
        self.public.write_text(json.dumps(bundle), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "max_moves"):
            self.run_observer()
        self.assertEqual(self.state["gets"], 0)
        self.assertEqual(self.state["posts"], [])
        self.assertFalse(self.responses.exists())
        self.assertFalse(self.observation.exists())

    def test_reused_output_paths_and_nonloopback_endpoint_rejected(self):
        self.run_observer()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_observer()
        with self.assertRaises(ValueError):
            observe_json_enum_exam(
                self.public, self.root / "new-responses.json",
                self.root / "new-observation.json",
                "http://127.0.0.2:11580", self.model)
        self.assertFalse((self.root / "new-responses.json").exists())


if __name__ == "__main__":
    unittest.main()
