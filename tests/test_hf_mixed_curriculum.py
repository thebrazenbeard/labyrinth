"""Dependency-free tests of the opt-in mixed-task LoRA pilot curriculum."""

import unittest
import hashlib

from labyrinth.hf_pilot import (
    PilotConfig, SCHEMA, _prompt_commitment, _training_visit_indices,
    derange_answers, training_cases,
)
from labyrinth.evaluation import score_response
from labyrinth.hf_report_audit import audit_hf_report
from tests.test_hf_report_audit import pilot_fixture


class PairedMixedCurriculumTests(unittest.TestCase):
    def _config(self, **kwargs):
        defaults = dict(
            model_path="unused-until-train", output_dir="unused-until-train",
            seed=2810, train_count=8, eval_count=8, tiers=(1, 2, 3, 4),
            steps=32, source_counterfactual_pairs=True,
        )
        return PilotConfig(**{**defaults, **kwargs})

    def test_balanced_deterministic_and_split_separated(self):
        config = self._config()
        train, evaluation = training_cases(config)
        self.assertEqual((32, 32), (len(train), len(evaluation)))
        self.assertEqual([8] * 4, [sum(c.tier == tier for c in train) for tier in range(1, 5)])
        self.assertEqual(train, training_cases(config)[0])
        self.assertEqual(evaluation, training_cases(config)[1])
        self.assertFalse({c.id for c in train} & {c.id for c in evaluation})
        self.assertFalse({c.prompt for c in train} & {c.prompt for c in evaluation})
        train_structures = {dict(c.metadata)["structure_fingerprint"] for c in train if c.tier == 4}
        heldout_structures = {dict(c.metadata)["structure_fingerprint"] for c in evaluation if c.tier == 4}
        self.assertTrue(train_structures)
        self.assertTrue(heldout_structures)
        self.assertFalse(train_structures & heldout_structures)

        visits = _training_visit_indices(train, seed=config.seed, steps=config.steps)
        self.assertEqual([8] * 4, [sum(train[i].tier == tier for i in visits) for tier in range(1, 5)])

    def test_priority_swap_changes_answer_not_reports(self):
        train, _ = training_cases(self._config())
        sources = [case for case in train if case.tier == 3]
        self.assertEqual(8, len(sources))
        for i in range(0, len(sources), 2):
            original, swapped = sources[i:i + 2]
            self.assertNotEqual(original.answer, swapped.answer)
            self.assertEqual(original.id + "-swap", swapped.id)
            self.assertIn(("pair", "priority_swap"), swapped.metadata)
            original_reports = [row for row in original.prompt.splitlines() if row.startswith("- source=")]
            swapped_reports = [row for row in swapped.prompt.splitlines() if row.startswith("- source=")]
            self.assertEqual(original_reports, swapped_reports)

    def test_explicit_opt_in_and_invalid_pair_sizes(self):
        ordinary, _ = training_cases(self._config(source_counterfactual_pairs=False))
        self.assertFalse(any("-swap" in c.id for c in ordinary))
        with self.assertRaisesRegex(ValueError, "even train_count"):
            from labyrinth.hf_pilot import validate_config
            validate_config(self._config(train_count=7))
        with self.assertRaisesRegex(ValueError, "tier 3"):
            from labyrinth.hf_pilot import validate_config
            validate_config(self._config(tiers=(1, 2, 4)))

    def test_report_auditor_reconstructs_paired_training_ids(self):
        config = self._config(train_count=2, eval_count=2, steps=8)
        train, evaluation = training_cases(config)
        report = pilot_fixture()
        report["schema"] = SCHEMA
        report["config"].update({
            "seed": config.seed, "train_count": config.train_count,
            "eval_count": config.eval_count, "tiers": list(config.tiers),
            "steps": config.steps, "answer_only": False,
            "source_counterfactual_pairs": True,
        })
        report["initial_persistent_buffers_sha256"] = "0" * 64
        report["training_case_ids"] = [c.id for c in train]
        report["heldout_case_ids"] = [c.id for c in evaluation]
        report["heldout_public_prompt_sha256"] = _prompt_commitment(evaluation)
        visits = [train[i].id for i in _training_visit_indices(
            train, seed=config.seed, steps=config.steps,
        )]
        record = {"steps": 8, "example_visits": 8, "visited_case_ids": visits,
                  "input_tokens_seen": 400, "losses": [2.0] * 8}
        report["training"] = {name: dict(record) for name in ("sham", "candidate", "corrupted")}
        report["control_data"] = {
            "corrupted_label_collisions": sum(c.answer == answer for c, answer in zip(train, derange_answers(train))),
            "sham_updates": 0, "sham_optimizer_steps": 8,
            "sham_train_case_visits": 8,
            "sham_evaluation_before_updates": False,
            "sham_is_compute_matched": True,
            "answer_scaffold_supplied": False,
        }
        trials = []
        for case in evaluation:
            response = "ANSWER: 0"
            score = score_response(case, response)
            trials.append({"id": case.id, "correct": score.correct,
                           "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                           "response_preview": response, "score_reason": score.reason})
        correct = sum(t["correct"] for t in trials)
        summary = {"correct": correct, "total": len(trials),
                   "accuracy": correct / len(trials), "format_valid": len(trials),
                   "format_rate": 1.0, "trials": trials}
        report["evaluations"] = {name: dict(summary) for name in
                                  ("baseline", "sham", "candidate", "corrupted")}
        self.assertEqual("PASS", audit_hf_report(report)["status"])


if __name__ == "__main__":
    unittest.main()
