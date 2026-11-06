"""Fresh-process public CLI contracts for regenerative toy journeys.

These tests deliberately use ``python -m labyrinth`` rather than an in-process
``main`` invocation: a successful unit call does not establish that another
process can import, execute, and persist the same journey safely.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_SCOPE = REPO_ROOT / "examples" / "scopes" / "arithmetic_toy_declaration.json"


class JourneySubprocessContractTests(unittest.TestCase):
    def _run(self, command: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "labyrinth", command, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_journey_toy_is_byte_deterministic_across_fresh_processes(self):
        arguments = ("--scope", str(TOY_SCOPE), "--learner-seed", "7",
                     "--epochs-by-cycle", "4,4,4")
        first = self._run("journey-toy", *arguments)
        second = self._run("journey-toy", *arguments)

        self.assertEqual((first.returncode, second.returncode), (0, 0),
                         (first.stderr, second.stderr))
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual((first.stderr, second.stderr), ("", ""))
        report = json.loads(first.stdout)
        self.assertEqual(report["schema"], "labyrinth.toy-journey.v1")
        self.assertEqual(len(report["cycles"]), 3)
        self.assertFalse(report["formal_graduation"])
        self.assertEqual(report["qualification"], "NOT_ESTABLISHED")

    def test_journey_controls_are_byte_deterministic_across_fresh_processes(self):
        arguments = ("--scope", str(TOY_SCOPE), "--learner-seed", "7",
                     "--epochs-by-cycle", "4,4,4")
        first = self._run("journey-toy-controls", *arguments)
        second = self._run("journey-toy-controls", *arguments)

        self.assertEqual((first.returncode, second.returncode), (0, 0),
                         (first.stderr, second.stderr))
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual((first.stderr, second.stderr), ("", ""))
        report = json.loads(first.stdout)
        self.assertEqual(report["schema"], "labyrinth.toy-journey-controls.v1")
        self.assertEqual(report["cycle_count"], 3)
        self.assertEqual(set(report["arms"]),
                         {"baseline", "correct_labels", "zero_update", "corrupted_labels"})
        self.assertFalse(report["formal_graduation"])
        self.assertEqual(report["qualification"], "NOT_ESTABLISHED")

    def test_malformed_epoch_schedules_fail_closed_before_report_creation(self):
        invalid_schedules = ("4", "4,0,4", "4,4,NaN", "129,1,1", "86,86,86")
        with tempfile.TemporaryDirectory() as temporary:
            for command in ("journey-toy", "journey-toy-controls"):
                for index, schedule in enumerate(invalid_schedules):
                    with self.subTest(command=command, schedule=schedule):
                        report_path = Path(temporary) / f"{command}-{index}.json"
                        result = self._run(
                            command, "--scope", str(TOY_SCOPE),
                            "--epochs-by-cycle", schedule, "--report", str(report_path),
                        )
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("epochs-by-cycle", result.stderr)
                        self.assertFalse(report_path.exists())

    def test_all_journey_reports_are_write_once_across_fresh_processes(self):
        commands = (
            ("journey-plan", ("--scope", str(TOY_SCOPE))),
            ("journey-toy", ("--scope", str(TOY_SCOPE), "--epochs-by-cycle", "4,4,4")),
            ("journey-toy-controls", ("--scope", str(TOY_SCOPE),
                                      "--epochs-by-cycle", "4,4,4")),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for command, args in commands:
                with self.subTest(command=command):
                    report_path = Path(temporary) / f"{command}.json"
                    options = (*args, "--report", str(report_path))
                    first = self._run(command, *options)
                    self.assertEqual(first.returncode, 0, first.stderr)
                    self.assertEqual(first.stdout, "")
                    self.assertTrue(report_path.is_file())
                    original = report_path.read_bytes()
                    self.assertTrue(json.loads(original))

                    second = self._run(command, *options)
                    self.assertEqual(second.returncode, 2, second.stderr)
                    self.assertEqual(second.stdout, "")
                    self.assertIn("exist", second.stderr.lower())
                    self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
