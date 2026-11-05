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
from labyrinth.journey import DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION
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
        self.assertIn("reported in-memory model digests", story)
        self.assertIn("no saved checkpoint is established", story)
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

    def test_development_pass_is_not_formal_graduation(self):
        synthetic = deepcopy(self.report)
        # Legacy reports without an event chain remain displayable; a rewritten
        # chain-bearing report must pass the separate Ariadne replay check.
        for field in ("ariadne_thread", "ariadne_thread_schema", "ariadne_thread_head_sha256"):
            synthetic.pop(field, None)
        synthetic["cycles"][-1]["exam_status"] = "PASS"
        synthetic["cycles"][-1]["weak_skills"] = []
        synthetic["final_phase"] = DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION
        rendered = render_journey_chronicle(synthetic)
        self.assertIn("Development criteria passed", rendered)
        self.assertIn("graduation remains unestablished", rendered)
        self.assertIn("no Ariadne event chain; event integrity unverified", rendered)
        synthetic["cycles"][-1]["exam_status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            render_journey_chronicle(synthetic)

    def test_chronicle_checks_ariadne_chain_if_present(self):
        self.assertIn("local event chain verified internally", render_journey_chronicle(self.report))
        forged = deepcopy(self.report)
        forged["ariadne_thread"][1]["status"] = "TRAINING"
        with self.assertRaisesRegex(ValueError, "Ariadne event chain/content mismatch"):
            render_journey_chronicle(forged)

        partial = deepcopy(self.report)
        partial.pop("ariadne_thread_head_sha256")
        with self.assertRaisesRegex(ValueError, "Ariadne thread head mismatch"):
            render_journey_chronicle(partial)

    def test_nonterminal_pass_cannot_regenerate_a_new_cycle(self):
        self.assertGreaterEqual(len(self.report["cycles"]), 2)
        forged = deepcopy(self.report)
        forged["cycles"][0]["exam_status"] = "PASS"
        with self.assertRaisesRegex(ValueError, "only a failed development exam"):
            render_journey_chronicle(forged)

    def test_rejects_impossible_toy_producer_counters(self):
        for field, value in (
            ("training_case_count", 1_000_000),
            ("parameter_updates", 1_000_000),
            ("exam_case_count", 1_000_000),
            ("epochs", 1_001),
        ):
            with self.subTest(field=field):
                forged = deepcopy(self.report)
                forged["cycles"][0][field] = value
                with self.assertRaisesRegex(ValueError, "impossible toy journey"):
                    render_journey_chronicle(forged)

    def test_partial_updates_are_valid_for_zero_residual_case_visits(self):
        # The actual toy learner counts a parameter update only when an SGD
        # step changes its values; not every case visit changes a parameter.
        partial = deepcopy(self.report)
        visit_count = (partial["cycles"][0]["training_case_count"]
                       * partial["cycles"][0]["epochs"])
        self.assertGreater(visit_count, 1)
        partial["cycles"][0]["parameter_updates"] = visit_count - 1
        # This fabricated adjustment is a legacy shape-validation fixture;
        # leave the cryptographically committed real fixture untouched.
        for field in ("ariadne_thread", "ariadne_thread_schema", "ariadne_thread_head_sha256"):
            partial.pop(field, None)
        self.assertIn("Cycle 1", render_journey_chronicle(partial))

    def test_real_sgd_convergence_does_not_break_the_chronicle(self):
        scope = TrainingScopeSpec(
            objective_id="sgd-steps", domain="arithmetic",
            skills=("arithmetic",), specificity=2, scale=1,
            max_cycles=1, seed=0,
        )
        report = run_toy_journey(scope, epochs_by_cycle=(100,))
        cycle = report["cycles"][0]
        self.assertLess(cycle["parameter_updates"], cycle["training_case_count"] * cycle["epochs"])
        self.assertIn("At the exit:", render_journey_chronicle(report))

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
