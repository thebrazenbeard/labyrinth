"""End-to-end local-model child protocol; no model installation or GPU required."""
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest

from labyrinth.minotaur_candidate_bridge import (
    SubprocessCandidateAdapter, audit_receipt, run_candidate,
)
from labyrinth.timegate_exam import CandidateSession, candidate_view, make_case

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "minotaur_ollama_child.py"
ACTION = '{"kind":"ACTION","action":"wait"}'


@contextmanager
def fake_local_ollama(*, response=ACTION, done=True):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n > 64000:
                self.send_error(413)
                return
            data = json.loads(self.rfile.read(n))
            calls.append((self.path, data))
            payload = json.dumps({"response": response, "done": done}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class OllamaChildBridgeTests(unittest.TestCase):
    def test_local_child_performs_actual_http_protocol_through_process_boundary(self):
        case = make_case(7, sealed=True)
        public = candidate_view(case)
        factory = lambda: CandidateSession(case)
        with fake_local_ollama() as (url, calls):
            child = SubprocessCandidateAdapter(
                (sys.executable, str(SCRIPT), "synthetic:local", url),
                model_id="ollama:synthetic:local:unattested", timeout_seconds=6)
            receipt = run_candidate(
                public, factory, child, seed=17, authorize_local_adapter=True,
                require_process_separation=True, max_model_calls=1)
        self.assertEqual(receipt.turns[0].disposition, "ACTION")
        self.assertEqual(receipt.adapter_provenance, "MODEL_SUBPROCESS_UNATTESTED")
        self.assertTrue(audit_receipt(receipt, public, factory))
        self.assertEqual(len(calls), 1)
        path, request = calls[0]
        self.assertEqual(path, "/api/generate")
        self.assertEqual(request["model"], "synthetic:local")
        self.assertIs(request["stream"], False)
        self.assertEqual(request["options"]["temperature"], 0)
        self.assertEqual(request["options"]["seed"], 17)
        self.assertIs(request["think"], False)
        self.assertIn("public_start", request["prompt"])
        self.assertNotIn(case.case_id, request["prompt"])

    def test_remote_endpoint_is_rejected_before_any_model_call(self):
        input_json = json.dumps({"prompt": {"response_instructions": "one JSON",
                                              "public_start": {}, "history": [], "turn": 0},
                                 "seed": 1})
        result = subprocess.run(
            (sys.executable, str(SCRIPT), "synthetic:local",
             "http://192.0.2.1:11434"),
            input=input_json, text=True, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_incomplete_model_reply_records_unreplayable_call_failure(self):
        case = make_case(7, sealed=True)
        public = candidate_view(case)
        factory = lambda: CandidateSession(case)
        with fake_local_ollama(done=False) as (url, calls):
            child = SubprocessCandidateAdapter(
                (sys.executable, str(SCRIPT), "synthetic:local", url),
                model_id="ollama:synthetic:local:unattested", timeout_seconds=6)
            receipt = run_candidate(
                public, factory, child, seed=17, authorize_local_adapter=True,
                require_process_separation=True, max_model_calls=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(receipt.final_verdict, "MODEL_CALL_ERROR")
        self.assertFalse(audit_receipt(receipt, public, factory))

    def test_malformed_child_request_fails_without_contacting_model(self):
        with fake_local_ollama() as (url, calls):
            result = subprocess.run(
                (sys.executable, str(SCRIPT), "synthetic:local", url),
                input='{"prompt":"not a JSON object","seed":3}',
                text=True, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
