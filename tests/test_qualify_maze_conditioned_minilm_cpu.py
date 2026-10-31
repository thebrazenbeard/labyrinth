"""Source-only proof of real Maze room -> labeled TRAIN example binding.

No torch dependency, model download, filesystem mutation, or final examination.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec
from scripts import qualify_maze_conditioned_minilm_cpu as probe


class MazeConditionedMiniLMTests(unittest.TestCase):
    def test_frozen_pretrained_weight_byte_identity_rejects_backbone_swaps(self):
        class FakeTensor:
            dtype = "torch.float32"
            shape = (1, 2)

            def __init__(self, content: bytes):
                self.content = content

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def numpy(self):
                return self

            def tobytes(self):
                return self.content

        original = probe.hash_named_weights([
            ("embeddings.weight", FakeTensor(b"abcd")),
            ("encoder.weight", FakeTensor(b"efgh")),
        ])
        reordered = probe.hash_named_weights([
            ("encoder.weight", FakeTensor(b"efgh")),
            ("embeddings.weight", FakeTensor(b"abcd")),
        ])
        mutated = probe.hash_named_weights([
            ("embeddings.weight", FakeTensor(b"abcd")),
            ("encoder.weight", FakeTensor(b"efgX")),
        ])
        self.assertEqual(original, reordered)
        self.assertNotEqual(original, mutated)
        probe.require_frozen_identity(original, reordered)
        with self.assertRaisesRegex(RuntimeError, "backbone identity changed"):
            probe.require_frozen_identity(original, mutated)
        with self.assertRaises(ValueError):
            probe.hash_named_weights([])

    def test_final_checkpoint_readback_gate_requires_weights_and_per_case_equivalence(self):
        keys = (
            "trainable_sha256_after", "lora_sha256_after",
            "pretrained_base_sha256", "frozen_peft_sha256_after",
            "adapter_file_sha256", "head_file_sha256",
        )
        saved = {key: "a" * 64 for key in keys}
        expected = {
            "public_development": tuple(label for _, label in probe.PUBLIC_DEVELOPMENT),
            "public_transfer": tuple(label for _, label in probe.PUBLIC_TRANSFER),
        }
        successful = probe.validate_final_checkpoint_readback(
            saved, dict(saved), expected, dict(expected),
        )
        self.assertEqual(successful["status"], "SELF_REPORTED_FRESH_INSTANCE_PASS")
        self.assertEqual(successful["public_development_accuracy_after_reload"], 1.0)
        self.assertEqual(successful["public_transfer_accuracy_after_reload"], 1.0)
        for key in keys:
            altered = dict(saved)
            altered[key] = "b" * 64
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                probe.validate_final_checkpoint_readback(
                    saved, altered, expected, dict(expected),
                )
        # Two per-case prediction tuples with *identical* aggregate accuracy:
        # flipping the error location must still fail deterministic readback.
        development = expected["public_development"]
        one_error = (1-development[0],) + development[1:]
        other_error = (development[0], 1-development[1]) + development[2:]
        self.assertEqual(
            probe.score_predictions(one_error, probe.PUBLIC_DEVELOPMENT),
            probe.score_predictions(other_error, probe.PUBLIC_DEVELOPMENT),
        )
        with self.assertRaisesRegex(RuntimeError, "prediction mismatch"):
            probe.validate_final_checkpoint_readback(
                saved, dict(saved),
                {**expected, "public_development": one_error},
                {**expected, "public_development": other_error},
            )
        with self.assertRaises(ValueError):
            probe.score_predictions((1,), probe.PUBLIC_DEVELOPMENT)

    def test_default_scope_generates_new_real_rooms_and_training_rows(self):
        scope = probe.default_scope()
        cycles = probe.build_training_cycles(scope)
        self.assertEqual(cycles, probe.build_training_cycles(scope))
        self.assertEqual(len(cycles), 2)
        self.assertEqual(len(cycles[0].examples), 12)
        for index, cycle in enumerate(cycles):
            self.assertEqual(cycle.plan, generate_maze_plan(scope, index))
            self.assertEqual(
                [row.room_id for row in cycle.examples],
                [room.room_id for room in cycle.plan.rooms],
            )
            self.assertEqual(
                [row.case_seed for row in cycle.examples],
                [room.case_seed for room in cycle.plan.rooms],
            )
            self.assertEqual(
                {row.label for row in cycle.examples}, {0, 1},
            )
            self.assertTrue(all(row.skill in probe.SKILL_LABEL for row in cycle.examples))
            self.assertTrue(all(row.to_dict()["split"] == "train"
                                for row in cycle.examples))
            self.assertEqual(
                probe._rows(cycle),
                tuple((row.prompt, row.label) for row in cycle.examples),
            )
        validation = probe.validate_cycles(cycles)
        self.assertEqual(validation["unique_room_ids"], 24)
        self.assertEqual(validation["unique_case_seeds"], 24)
        self.assertEqual(validation["unique_training_prompts"], 24)
        self.assertEqual(validation["unique_exposure_ids"], 24)

    def test_weakness_hint_reweights_regenerated_room_content_not_initial_cycle(self):
        scope = probe.default_scope()
        balanced = probe.build_training_cycles(scope)
        weak = probe.build_training_cycles(
            scope, weak_cycle_one=("negative_polarity",),
        )
        self.assertEqual(balanced[0], weak[0])
        self.assertNotEqual(balanced[1], weak[1])
        self.assertEqual(len(balanced[1].examples), len(weak[1].examples))
        self.assertGreater(weak[1].plan.counts_by_skill["negative_polarity"],
                           balanced[1].plan.counts_by_skill["negative_polarity"])
        self.assertTrue(all(row.label == probe.SKILL_LABEL[row.skill]
                            for row in weak[1].examples))

    def test_scope_seed_changes_generated_cases_and_identifiers(self):
        baseline = probe.build_training_cycles(probe.default_scope())
        changed = probe.build_training_cycles(replace(probe.default_scope(), seed=123))
        self.assertNotEqual(baseline, changed)
        self.assertNotEqual(baseline[0].examples[0].room_id,
                            changed[0].examples[0].room_id)
        self.assertNotEqual(baseline[0].examples[0].prompt,
                            changed[0].examples[0].prompt)

    def test_physical_training_batch_selector_consumes_generated_cases(self):
        cycles = probe.build_training_cycles(probe.default_scope())
        for cycle in cycles:
            for step in range(7):
                selected = probe.select_training_batch(cycle, step)
                expected = tuple(cycle.examples[(4 * step + offset) % len(cycle.examples)]
                                 for offset in range(4))
                self.assertEqual(selected, expected)
                self.assertEqual({row.cycle_index for row in selected},
                                 {cycle.plan.cycle_index})
                self.assertTrue(all(row.room_id in {r.room_id for r in cycle.plan.rooms}
                                    for row in selected))
        with self.assertRaises(ValueError):
            probe.select_training_batch(cycles[0], -1)

    def test_unapproved_objective_or_unbounded_scope_fails_closed(self):
        scope = probe.default_scope()
        invalid = (
            replace(scope, objective_id="empathy"),
            replace(scope, domain="reasoning"),
            replace(scope, skills=("positive_polarity", "empathy")),
            replace(scope, max_cycles=1),
            replace(scope, scale=1, specificity=1),
            replace(scope, scale=5, specificity=5),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                probe.build_training_cycles(candidate)

    def test_tampering_with_room_binding_exposure_prompt_and_label_rejected(self):
        original = probe.build_training_cycles(probe.default_scope())
        first = original[0]
        row = first.examples[0]
        altered = (
            replace(row, label=1-row.label),
            replace(row, prompt="The fake case was not dispatched."),
            replace(row, exposure_id="0" * 64),
            replace(row, case_seed=row.case_seed + 1),
            replace(row, room_id="0" * 32),
        )
        for candidate in altered:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                forged_first = replace(first, examples=(candidate,) + first.examples[1:])
                probe.validate_cycles((forged_first, original[1]))

    def test_duplicate_generated_text_is_not_hidden_by_unique_room_ids(self):
        with patch.object(probe, "SUBJECTS", ("service",)), \
             patch.object(probe, "POSITIVES", ("excellent",)), \
             patch.object(probe, "NEGATIVES", ("terrible",)), \
             patch.object(probe, "TEMPLATES", ("The {subject} was {adjective}.",)):
            with self.assertRaisesRegex(ValueError, "repeated training case"):
                probe.build_training_cycles(probe.default_scope())

    def test_training_and_public_development_overlap_fails_closed(self):
        cycles = probe.build_training_cycles(probe.default_scope())
        example = cycles[0].examples[0]
        with patch.object(probe, "PUBLIC_DEVELOPMENT",
                          ((example.prompt, example.label),) + probe.PUBLIC_DEVELOPMENT):
            with self.assertRaisesRegex(ValueError, "overlap"):
                probe.validate_cycles(cycles)

    def test_execute_rejects_injected_training_case_before_loading_ml_or_writing(self):
        scope = probe.default_scope()
        cycles = probe.build_training_cycles(scope)
        original = cycles[0].examples[0]
        injected = replace(original, prompt="Injected unrelated training fixture")
        forged_first = replace(cycles[0], examples=(injected,) + cycles[0].examples[1:])
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "no-execution"
            with self.assertRaisesRegex(ValueError, "diverged"):
                probe.execute(run_root, scope=scope, cycles=(forged_first, cycles[1]),
                              steps_per_cycle=1, learning_rate=0.002)
            self.assertFalse(run_root.exists())

    def test_read_only_cli_dry_run_and_manual_prior_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("builtins.print") as printed:
                self.assertEqual(probe.main([
                    "--output-parent", temp, "--run-name", "labyrinth-l13",
                    "--weak-cycle-one", "negative_polarity",
                ]), 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertFalse(payload["execute"])
            self.assertFalse(payload["graduation_authorized"])
            self.assertTrue(payload["public_development_only"])
            self.assertEqual(len(payload["cycles"]), 2)
            self.assertGreater(
                payload["cycles"][1]["train_label_counts"]["negative_polarity"],
                payload["cycles"][1]["train_label_counts"]["positive_polarity"],
            )
            self.assertFalse((Path(temp) / "labyrinth-l13").exists())

    def test_strict_external_scope_is_read_only_and_rejects_wrong_intention(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scope.json"
            path.write_text(json.dumps(probe.default_scope().to_dict()), encoding="utf-8")
            with patch("builtins.print"):
                self.assertEqual(probe.main([
                    "--output-parent", temp, "--run-name", "scope-ok",
                    "--scope-file", str(path),
                ]), 0)
            self.assertFalse((Path(temp) / "scope-ok").exists())
            bad = probe.default_scope().to_dict()
            bad["objective_id"] = "empathy"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(SystemExit) as error:
                probe.main([
                    "--output-parent", temp, "--run-name", "scope-bad",
                    "--scope-file", str(path),
                ])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(temp) / "scope-bad").exists())

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "occupied").mkdir()
            with self.assertRaises(SystemExit) as error:
                probe.main(["--output-parent", temp, "--run-name", "occupied"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
