"""Real-process candidate boundary probes; not an OS-credential isolation claim."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from labyrinth.timegate_exam import CandidateSession, candidate_view, make_case
from labyrinth.minotaur_candidate_bridge import audit_receipt, run_candidate


def probe(command, *, seconds=2.0):
    from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter
    return SubprocessCandidateAdapter(command, model_id="synthetic:child", timeout_seconds=seconds)


class ProcessCandidateBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case(7, sealed=True)
        self.public = candidate_view(self.case)
        self.factory = lambda: CandidateSession(self.case)

    def test_process_adapter_instance_cannot_shadow_generate(self):
        adapter = probe((sys.executable, "-c", "print('invalid')"))
        with self.assertRaises(AttributeError):
            adapter.generate = lambda prompt, seed=None: '{"kind":"ACTION","action":"leak"}'

    def test_process_only_ignores_monkeypatched_adapter_class_method(self):
        script = ("import json,sys;json.load(sys.stdin);"
                  "print(json.dumps({'kind':'ACTION','action':'wait'}))")
        adapter = probe((sys.executable, "-c", script))
        host_calls = []
        def hostile(self, prompt, *, seed=None):
            host_calls.append("HOST_PYTHON_INVOKED")
            return '{"kind":"ACTION","action":"leak"}'
        with mock.patch.object(type(adapter), "generate", hostile):
            receipt = run_candidate(self.public, self.factory, adapter,
                                    authorize_local_adapter=True,
                                    require_process_separation=True, max_model_calls=1)
        self.assertEqual(host_calls, [])
        self.assertEqual(json.loads(receipt.turns[0].response_text)["action"], "wait")

    def test_process_only_mode_rejects_python_callback_even_if_authorized(self):
        class InProcessSpy:
            model_id = "python:spy"
            def generate(self, prompt, *, seed=None):
                return '{"kind":"ACTION","action":"wait"}'
        with self.assertRaises(PermissionError):
            run_candidate(self.public, self.factory, InProcessSpy(),
                          authorize_local_adapter=True, require_process_separation=True)
    def test_child_cannot_inspect_host_frames_or_inherit_private_environment(self):
        script = (
            "import inspect,json,os,sys;"
            "request=json.load(sys.stdin);"
            "frames=inspect.stack();"
            "visible=any(any(type(v).__name__=='CandidateSession' "
            "for v in f.frame.f_locals.values()) for f in frames);"
            "leak=visible or bool(os.environ.get('LABYRINTH_EXAM_SECRET')) "
            "or 'case_id' in json.dumps(request);"
            "print(json.dumps({'kind':'ACTION','action':'leak' if leak else 'wait'}))"
        )
        with mock.patch.dict(os.environ, {"LABYRINTH_EXAM_SECRET": self.case.case_id}):
            receipt = run_candidate(self.public, self.factory,
                                    probe((sys.executable, "-c", script)),
                                    authorize_local_adapter=True, require_process_separation=True,
                                    max_model_calls=1)
        self.assertEqual(receipt.adapter_provenance, "MODEL_SUBPROCESS_UNATTESTED")
        self.assertEqual(receipt.turns[0].disposition, "ACTION")
        self.assertEqual(json.loads(receipt.turns[0].response_text)["action"], "wait")
        self.assertNotIn(self.case.case_id, receipt.turns[0].prompt_json)
        self.assertTrue(audit_receipt(receipt, self.public, self.factory))

    def test_malformed_stdout_and_timeout_fail_closed(self):
        malformed = run_candidate(self.public, self.factory,
                                  probe((sys.executable, "-c", "print('not JSON')")),
                                  authorize_local_adapter=True, require_process_separation=True,
                                  max_model_calls=1)
        self.assertEqual(malformed.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertTrue(audit_receipt(malformed, self.public, self.factory))
        sleeping = run_candidate(self.public, self.factory,
                                 probe((sys.executable, "-c", "import time;time.sleep(5)"),
                                       seconds=0.25),
                                 authorize_local_adapter=True, require_process_separation=True,
                                 max_model_calls=1)
        self.assertEqual(sleeping.final_verdict, "MODEL_CALL_ERROR")
        self.assertTrue(audit_receipt(sleeping, self.public, self.factory))

    def test_oversize_output_is_cut_off_before_hanging_child_timeout(self):
        script = ("import sys,time;"
                  "sys.stdout.write('X'*200000);sys.stdout.flush();time.sleep(5)")
        receipt = run_candidate(self.public, self.factory,
                                probe((sys.executable, "-c", script), seconds=1.0),
                                authorize_local_adapter=True, require_process_separation=True,
                                max_model_calls=1)
        self.assertEqual(receipt.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertTrue(audit_receipt(receipt, self.public, self.factory))

    def test_same_user_process_can_still_read_known_private_file(self):
        """Hostile falsifier: process separation does NOT establish denied file access."""
        with tempfile.TemporaryDirectory() as root:
            secret = Path(root) / "examiner.txt"
            secret.write_text("private", encoding="utf-8")
            script = ("import json,pathlib,sys;"
                      "json.load(sys.stdin);"
                      "readable=pathlib.Path(sys.argv[1]).read_text()=='private';"
                      "print(json.dumps({'kind':'ACTION',"
                      "'action':'leak' if readable else 'wait'}))")
            receipt = run_candidate(self.public, self.factory,
                                    probe((sys.executable, "-c", script, str(secret))),
                                    authorize_local_adapter=True, require_process_separation=True,
                                    max_model_calls=1)
        self.assertEqual(json.loads(receipt.turns[0].response_text)["action"], "leak")
        self.assertEqual(receipt.adapter_provenance, "MODEL_SUBPROCESS_UNATTESTED")
        self.assertFalse(receipt.verified_escape)

    def test_posix_timeout_kills_stdout_holding_grandchild(self):
        """A timed-out child cannot leave a same-process-group writer alive."""
        if os.name != "posix":
            return  # Windows has no demonstrated process-tree containment.
        with tempfile.TemporaryDirectory(prefix="minotaur_descendant_") as directory:
            marker = Path(directory, "survivor.txt")
            grandchild = ("import pathlib,sys,time;time.sleep(0.75);"
                          "pathlib.Path(sys.argv[1]).write_text('escaped')")
            parent = ("import subprocess,sys;"
                      "subprocess.Popen((sys.executable,'-c',sys.argv[2],sys.argv[1]));"
                      "print('not JSON',flush=True)")
            receipt = run_candidate(self.public, self.factory,
                                    probe((sys.executable, "-c", parent,
                                           str(marker), grandchild), seconds=0.3),
                                    authorize_local_adapter=True,
                                    require_process_separation=True, max_model_calls=1)
            self.assertEqual(receipt.final_verdict, "MODEL_CALL_ERROR")
            time.sleep(0.9)  # Exceeds the descendant's known marker deadline.
            self.assertFalse(marker.exists(),
                             "Timed-out candidate left a live grandchild behind")



    def test_detached_stdout_holder_does_not_stall_cleanup(self):
        """A detached stdout-holder cannot extend the examiner's output deadline."""
        if os.name != "posix":
            return  # Windows process-tree deadline remains unqualified.
        # A synthetic grandchild escapes the process group and exits naturally.
        grandchild = "import time;time.sleep(2.5)"
        parent = ("import subprocess,sys;"
                  "subprocess.Popen((sys.executable,'-c',sys.argv[1]),"
                  "start_new_session=True);"
                  "print('not JSON',flush=True)")
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            probe((sys.executable, "-c", parent, grandchild),
                  seconds=0.25).generate('{"public":true}')
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.4,
                        "detached stdout holder blocked bounded candidate cleanup")


if __name__ == "__main__":
    unittest.main()
