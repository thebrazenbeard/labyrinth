"""CLI smoke and fail-closed contracts for regenerative journey commands."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.cli import main


REPO = Path(__file__).resolve().parents[1]
TOY_SCOPE = REPO / "examples" / "scopes" / "arithmetic_toy_declaration.json"
EMPATHY_SCOPE = REPO / "examples" / "scopes" / "empathy_declaration.json"


class JourneyCliTests(unittest.TestCase):
    def _invoke(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(list(args))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_empathy_scope_previews_declaration_without_training(self):
        status, output, _ = self._invoke("journey-plan", "--scope", str(EMPATHY_SCOPE))
        self.assertEqual(status, 0)
        report = json.loads(output)
        self.assertEqual(report["schema"], "labyrinth.journey-plan.v1")
        self.assertEqual(report["scope"]["domain"], "empathy")
        self.assertGreater(len(report["plan"]["rooms"]), 0)

    def test_toy_default_epochs_runs_complete_cycle_budget(self):
        status, output, error = self._invoke("journey-toy", "--scope", str(TOY_SCOPE))
        self.assertEqual(status, 0, error)
        report = json.loads(output)
        self.assertEqual(report["schema"], "labyrinth.toy-journey.v1")
        self.assertGreaterEqual(len(report["cycles"]), 1)
        self.assertEqual(report["cycles"][0]["epochs"], 4)
        self.assertFalse(report["formal_graduation"])

    def test_partial_epoch_schedule_rejected(self):
        status, _, error = self._invoke("journey-toy", "--scope", str(TOY_SCOPE),
                                        "--epochs-by-cycle", "4")
        self.assertEqual(status, 2)
        self.assertIn("every cycle", error)

    def test_empathy_cannot_be_misrepresented_as_toy_learning(self):
        status, _, error = self._invoke("journey-toy", "--scope", str(EMPATHY_SCOPE))
        self.assertEqual(status, 2)
        self.assertIn("arithmetic-only", error)

    def test_reports_are_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "plan.json"
            args = ("journey-plan", "--scope", str(TOY_SCOPE), "--report", str(out))
            first, _, _ = self._invoke(*args)
            self.assertEqual(first, 0)
            original = out.read_bytes()
            second, _, error = self._invoke(*args)
            self.assertEqual(second, 2)
            self.assertEqual(out.read_bytes(), original)
            self.assertIn("exist", error.lower())


if __name__ == "__main__":
    unittest.main()
