"""Source contract for opt-in MiniLM post-checkpoint process-kill probe."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.qualify_minilm_fault_cpu import (
    MARKER_SCHEMA, SCHEMA, _snapshot_digest, main, verify_forced_kill,
    wait_for_marker_and_kill,
)


def _marker(**overrides):
    marker = {
        "schema": MARKER_SCHEMA, "pid": 303, "run_id": "run-1",
        "seed": 20261033, "steps": 8, "split_at": 4,
        "post_checkpoint_update_index": 1,
        "checkpoint_sha256": "a" * 64,
        "pre_update_sha256": "b" * 64,
        "post_update_sha256": "c" * 64,
    }
    marker.update(overrides)
    return marker


def _verify(marker, **overrides):
    arguments = {
        "root_name": "run-1", "expected_checkpoint_sha256": "a" * 64,
        "worker_pid": 303, "other_pids": (101, 202), "returncode": -9,
        "seed": 20261033, "steps": 8, "split_at": 4,
    }
    arguments.update(overrides)
    return verify_forced_kill(marker, **arguments)


class MiniLMForcedCrashContractTests(unittest.TestCase):
    def test_readonly_preflight_and_no_run_dir_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = StringIO()
            with redirect_stdout(out):
                self.assertEqual(main([
                    "--output-parent", temporary, "--run-name", "fault-safe-run",
                    "--steps", "8", "--split-at", "4",
                ]), 0)
            plan = json.loads(out.getvalue())
            self.assertEqual(plan["schema"], SCHEMA)
            self.assertFalse(plan["execute"])
            self.assertFalse(plan["graduation_authorized"])
            self.assertFalse(Path(plan["run_root"]).exists())

    def test_verified_kill_requires_changed_step_and_distinct_process(self):
        self.assertIsNone(_verify(_marker()))
        bad = (
            (_marker(post_update_sha256="b" * 64), {}),
            (_marker(pre_update_sha256="not-sha"), {}),
            (_marker(checkpoint_sha256="d" * 64), {}),
            (_marker(pid=101), {}),
            (_marker(run_id="other"), {}),
            (_marker(post_checkpoint_update_index=0), {}),
            (_marker(seed=0), {}),
            (_marker(), {"worker_pid": 101}),
            (_marker(), {"other_pids": (101, 101)}),
            (_marker(), {"returncode": 0}),
            (_marker(extra="unrecognized"), {}),
        )
        for marker, arguments in bad:
            with self.subTest(marker=marker, arguments=arguments):
                with self.assertRaises(ValueError):
                    _verify(marker, **arguments)

    def test_checkpoint_byte_commitment_detects_mutation_and_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text("{}\n", encoding="utf-8")
            weights = snapshot / "state.safetensors"
            weights.write_bytes(b"mock-weights")
            before = _snapshot_digest(root)
            weights.write_bytes(b"mutated-weights")
            self.assertNotEqual(before, _snapshot_digest(root))
            weights.unlink()
            with self.assertRaises(ValueError):
                _snapshot_digest(root)
            try:
                weights.symlink_to(snapshot / "manifest.json")
            except (NotImplementedError, OSError):
                pass
            else:
                with self.assertRaises(ValueError):
                    _snapshot_digest(root)

    def test_parent_actually_terminates_exact_spawned_child_after_marker(self):
        # This is a standard-library process-kill test, NOT a neural experiment.
        child_source = (
            "import json,os,pathlib,sys,time\n"
            "p=pathlib.Path(sys.argv[1])/'kill_ready.json'\n"
            "temp=p.with_suffix('.pending')\n"
            "temp.write_text(json.dumps({'pid':os.getpid(),'ready':True}))\n"
            "os.replace(temp,p)\n"
            "while True: time.sleep(0.1)\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with subprocess.Popen(
                [sys.executable, "-c", child_source, temporary],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ) as worker:
                marker, returncode = wait_for_marker_and_kill(
                    Path(temporary), worker, timeout_seconds=5)
                self.assertTrue(marker["ready"])
                self.assertEqual(marker["pid"], worker.pid)
                self.assertNotEqual(returncode, 0)
                self.assertIsNotNone(worker.poll())

    def test_missing_marker_terminates_only_owned_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            with subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ) as worker:
                with self.assertRaisesRegex(RuntimeError, "never reached"):
                    wait_for_marker_and_kill(
                        Path(temporary), worker, timeout_seconds=0.2)
                self.assertIsNotNone(worker.poll())


if __name__ == "__main__":
    unittest.main()
