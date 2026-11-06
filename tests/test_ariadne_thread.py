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
        for index, cycle in enumerate(report["cycles"]):
            train = events[1 + index * 3]
            verdict = events[2 + index * 3]
            self.assertEqual(train["summary"], {
                "training_case_count": cycle["training_case_count"],
                "training_room_count": len(cycle["training_room_ids"]),
                "parameter_updates": cycle["parameter_updates"],
                "epochs": cycle["epochs"],
            })
            self.assertEqual(verdict["summary"], {
                "exam_case_count": cycle["exam_case_count"],
                "exam_candidate_correct": cycle["exam_candidate_correct"],
                "exam_frozen_baseline_correct": cycle["exam_frozen_baseline_correct"],
                "weak_skills": cycle["weak_skills"],
            })
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
            ("train_summary", lambda report: report["ariadne_thread"][1]["summary"].__setitem__(
                "parameter_updates", 999)),
            ("verdict_summary", lambda report: report["ariadne_thread"][2]["summary"].__setitem__(
                "exam_candidate_correct", 16)),
            ("weak_skills", lambda report: report["ariadne_thread"][2]["summary"].__setitem__(
                "weak_skills", ["not_the_reported_skill"])),
            ("training_count_report", lambda report: report["cycles"][0].__setitem__(
                "training_case_count", 9999)),
            ("score_report", lambda report: report["cycles"][0].__setitem__(
                "exam_candidate_correct", 16)),
            ("updates_report", lambda report: report["cycles"][0].__setitem__(
                "parameter_updates", 999)),
        ):
            with self.subTest(name=name):
                report = deepcopy(self.report)
                alter(report)
                with self.assertRaises(ValueError):
                    verify_ariadne_thread(report)


if __name__ == "__main__":
    unittest.main()
