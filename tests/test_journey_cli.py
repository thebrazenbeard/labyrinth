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
EMPATHY_CONTRACT = REPO / "examples" / "objectives" / "empathy_observable_declaration.json"
PROGRAM_SCOPE = REPO / "examples" / "scopes" / "program_specificity5_declaration.json"


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

    def test_materialized_taskmaster_route_changes_training_prompts_without_gold(self):
        status, output, error = self._invoke("journey-plan", "--scope", str(TOY_SCOPE),
                                             "--materialize-training")
        self.assertEqual(status, 0, error)
        report = json.loads(output)
        self.assertEqual(len(report["training_views"]), len(report["plan"]["rooms"]))
        self.assertTrue(all("prompt" in view and "answer" not in view
                            for view in report["training_views"]))
        self.assertTrue(all(view["tactic"] in ("practice", "contrast", "transfer")
                            for view in report["training_views"]))

    def test_materialize_unimplemented_empathy_provider_fails_closed(self):
        status, _, error = self._invoke("journey-plan", "--scope", str(EMPATHY_SCOPE),
                                        "--materialize-training")
        self.assertEqual(status, 2)
        self.assertIn("no approved training provider", error)

    def test_matched_toy_controls_stay_unqualified(self):
        status, output, error = self._invoke("journey-toy-controls", "--scope", str(TOY_SCOPE),
                                             "--epochs-by-cycle", "4,4,4")
        self.assertEqual(status, 0, error)
        report = json.loads(output)
        self.assertEqual(report["schema"], "labyrinth.toy-journey-controls.v1")
        self.assertEqual(report["cycle_count"], 3)
        self.assertFalse(report["formal_graduation"])
        self.assertEqual(set(report["arms"]), {"baseline", "correct_labels", "zero_update", "corrupted_labels"})

    def test_objective_contract_predeclares_observable_semantic_focus(self):
        status, output, error = self._invoke("journey-plan", "--objective-contract", str(EMPATHY_CONTRACT))
        self.assertEqual(status, 0, error)
        result = json.loads(output)
        self.assertEqual(result["scope"]["domain"], "empathy")
        self.assertEqual(result["selected_focus"]["compassionate_empathy"], ["helpfulness", "honesty"])
        self.assertEqual(result["objective_contract"]["targets"][3]["rubric_id"], "compassion_v1")
        self.assertNotIn("training_views", result)

    def test_program_specificity_materializes_structurally_focused_rooms(self):
        status, output, error = self._invoke("journey-plan", "--scope", str(PROGRAM_SCOPE),
                                             "--materialize-training")
        self.assertEqual(status, 0, error)
        result = json.loads(output)
        self.assertEqual(result["training_provider"], "approved_skill_providers_with_program_focus")
        self.assertTrue(result["training_views"])
        self.assertTrue(all(view["semantic_focus"] == "dependency_path_wrapping_both_registers"
                            and "answer" not in view for view in result["training_views"]))

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
