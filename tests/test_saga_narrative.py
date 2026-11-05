"""Chronicles must reflect development receipts without claiming graduation."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.cli import main
from labyrinth.journey_toy import run_toy_journey
from labyrinth.runner import save_json_new
from labyrinth.saga_narrative import render_journey_chronicle
from labyrinth.training_scope import TrainingScopeSpec


class SagaNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = TrainingScopeSpec(
            objective_id="saga-arithmetic", domain="arithmetic",
            skills=("arithmetic",), specificity=2, scale=1,
            max_cycles=2, seed=17,
        )
        cls.report = run_toy_journey(scope, epochs_by_cycle=(4, 4))

    def test_real_toy_trajectory_has_continuity_and_truthful_verdicts(self):
        story = render_journey_chronicle(self.report)
        self.assertIn("THE LABYRINTH — saga-arithmetic", story)
        self.assertIn("Ariadne's thread", story)
        self.assertIn("Minotaur:", story)
        self.assertIn("reported checkpoint digests", story)
        self.assertIn("no graduation", story.lower())
        self.assertNotIn(self.report["cycles"][0]["exam_case_ids"][0], story)
        self.assertEqual(story.count("Cycle "), len(self.report["cycles"]))

    def test_refuses_fabricated_graduation_or_invalid_score(self):
        forged = deepcopy(self.report)
        forged["formal_graduation"] = True
        with self.assertRaisesRegex(ValueError, "cannot assert formal graduation"):
            render_journey_chronicle(forged)
        forged = deepcopy(self.report)
        forged["cycles"][0]["exam_candidate_correct"] = (
            forged["cycles"][0]["exam_case_count"] + 1
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            render_journey_chronicle(forged)

    def test_refuses_broken_checkpoint_chain(self):
        forged = deepcopy(self.report)
        forged["cycles"][0]["checkpoint_before_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "broken or unchanged"):
            render_journey_chronicle(forged)

    def test_cli_reads_existing_report_without_changing_it(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "journey.json"
            save_json_new(str(path), self.report)
            original = path.read_bytes()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(["journey-chronicle", "--input", str(path)])
            self.assertEqual(result, 0)
            self.assertIn("At the exit:", stdout.getvalue())
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
