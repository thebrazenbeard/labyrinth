"""Source-only story replay must be observable and falsifiable."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout

from labyrinth.saga_replay_v1 import (
    ReplayIntegrityError, interactive_chronicle, main,
    render_markdown, reverify_public_chronicle,
)
from labyrinth.source_only_saga_demo_v1 import run_fixture_saga


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = TemporaryDirectory(prefix="labyrinth-chronicle-")
        cls.report = run_fixture_saga(Path(cls._temporary.name).resolve() / "saga")

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_factual_public_reexecution_produces_two_maps_and_a_narrative(self):
        verified = reverify_public_chronicle(self.report)
        text = render_markdown(verified)
        self.assertEqual(len(verified["verified_cycles"]), 2)
        self.assertEqual([c["difficulty"] for c in verified["verified_cycles"]], [1, 2])
        self.assertIn("Ariadne's Chronicle", text)
        self.assertEqual(text.count("```mermaid"), 2)
        self.assertIn("MINOTAUR", json.dumps(verified))
        self.assertIn("Ariadne's thread", text)
        self.assertIn("not independent grading", text)
        self.assertIn("verbatim dialogue replay", text)
        self.assertFalse("private_corpus" in text)
        first = verified["verified_cycles"][0]
        self.assertGreater(len(first["discovered_edges"]), 4)
        self.assertIn("minotaur_sanctum", first["discovered_rooms"])
        for cycle in verified["verified_cycles"]:
            self.assertEqual(cycle["steps"][-1]["state_sha256"], cycle["final_state_sha256"])

    def test_changed_action_or_observed_consequence_is_detected(self):
        dishonest = deepcopy(self.report)
        dishonest["cycles"][0]["actions"][2] = "move:minotaur_sanctum"
        with self.assertRaises(ReplayIntegrityError):
            reverify_public_chronicle(dishonest)
        dishonest = deepcopy(self.report)
        dishonest["cycles"][1]["outcomes"][0] = "GRADUATED"
        with self.assertRaisesRegex(ReplayIntegrityError, "consequence"):
            reverify_public_chronicle(dishonest)

    def test_changed_hash_or_unqualified_report_is_detected(self):
        dishonest = deepcopy(self.report)
        dishonest["cycles"][1]["final_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayIntegrityError, "end state"):
            reverify_public_chronicle(dishonest)
        dishonest = deepcopy(self.report)
        dishonest["evidence_class"] = "NEURAL_GENERALIZATION_VERIFIED"
        with self.assertRaisesRegex(ReplayIntegrityError, "source-only saga class"):
            reverify_public_chronicle(dishonest)

    def test_human_can_step_through_replayed_story_without_retraining(self):
        verified = reverify_public_chronicle(self.report)
        emitted = []
        count = interactive_chronicle(verified, write=emitted.append,
                                      prompt=lambda _: "q")
        self.assertEqual(count, 1)
        self.assertEqual(len(emitted), 2)
        self.assertIn("Ariadne", emitted[1])

    def test_cli_replays_existing_report_without_running_trainer(self):
        with TemporaryDirectory(prefix="chronicle-input-") as temp:
            source = Path(temp).resolve() / "report.json"
            source.write_text(json.dumps(self.report), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                main(["--input", str(source), "--format", "json"])
            data = json.loads(output.getvalue())
            self.assertTrue(data["source_public_experience_reexecuted"])
            self.assertIn("not SGD", data["trainer_runtime"])


if __name__ == "__main__":
    unittest.main()
