"""Ariadne chronology integrity is a local replay check, not attestation."""

from __future__ import annotations

from copy import deepcopy
import unittest

from labyrinth.ariadne_thread import THREAD_SCHEMA, verify_ariadne_thread
from labyrinth.journey_toy import run_toy_journey
from labyrinth.training_scope import TrainingScopeSpec


def scope() -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id="ariadne-toy", domain="arithmetic", skills=("arithmetic",),
        specificity=2, scale=4, max_cycles=2, seed=7,
    )


class AriadneThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_toy_journey(scope(), epochs_by_cycle=(1, 1))

    def test_chronology_covers_real_transitions_without_gold(self):
        report = self.report
        self.assertTrue(verify_ariadne_thread(report))
        self.assertEqual(report["ariadne_thread_schema"], THREAD_SCHEMA)
        events = report["ariadne_thread"]
        self.assertEqual([event["kind"] for event in events], [
            "entrance", "train_complete", "minotaur_verdict",
            "maze_regenerated", "train_complete", "minotaur_verdict", "exit",
        ])
        self.assertEqual(events[0]["checkpoint_before_sha256"], report["baseline_sha256"])
        self.assertEqual(events[-1]["checkpoint_after_sha256"], report["final_checkpoint_sha256"])
        self.assertEqual(events[3]["checkpoint_before_sha256"], events[1]["checkpoint_after_sha256"])
        self.assertEqual(events[4]["checkpoint_before_sha256"], events[3]["checkpoint_after_sha256"])
        self.assertEqual(events[3]["cycle_index"], 1)
        for event in events:
            self.assertNotIn("gold", repr(event).lower())
            self.assertNotIn("case_id", event)
            self.assertNotIn("prompt", event)
            self.assertNotIn("answer", event)

    def test_replay_rejects_edits_reorder_parent_and_report_mismatch(self):
        for name, alter in (
            ("edit", lambda report: report["ariadne_thread"][1].__setitem__("status", "FAKE")),
            ("reorder", lambda report: report["ariadne_thread"].__setitem__(
                slice(1, 3), report["ariadne_thread"][2:0:-1])),
            ("parent", lambda report: report["ariadne_thread"][2].__setitem__(
                "parent_sha256", "a" * 64)),
            ("extra_gold", lambda report: report["ariadne_thread"][1].__setitem__(
                "gold_answer", "42")),
            ("missing", lambda report: report["ariadne_thread"].pop(3)),
            ("checkpoint", lambda report: report["cycles"][1].__setitem__(
                "checkpoint_before_sha256", "f" * 64)),
            ("objective", lambda report: report.__setitem__("objective_id", "another")),
            ("head", lambda report: report.__setitem__(
                "ariadne_thread_head_sha256", "b" * 64)),
        ):
            with self.subTest(name=name):
                report = deepcopy(self.report)
                alter(report)
                with self.assertRaises(ValueError):
                    verify_ariadne_thread(report)


if __name__ == "__main__":
    unittest.main()
