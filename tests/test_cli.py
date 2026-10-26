"""End-to-end CLI and control-isolation regression tests (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from labyrinth.adapters import ReplayAdapter
from labyrinth.curriculum import (
    CurriculumSpec, CurriculumState, TrainingObservation,
    next_training_case, record_training_result,
)
from labyrinth.evaluation import score_response
from labyrinth.maze import Challenge, generate_maze
from labyrinth.runner import (
    build_training_artifact,
    digest,
    run_experiment,
    run_maze,
    validate_training_artifact,
)


ROOT = Path(__file__).resolve().parent.parent


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "labyrinth", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )


class RunnerTests(unittest.TestCase):
    def test_replay_correct_answers_and_private_answer_keys(self):
        cases = generate_maze(seed=21, count=2, tiers=(1, 2, 3), split="eval")
        model = ReplayAdapter({c.prompt: f"ANSWER: {c.answer}" for c in cases})
        report = run_maze(model, seed=21, count=2)
        self.assertEqual(report["evaluation"]["total"], 6)
        self.assertEqual(report["evaluation"]["points"], 6)
        self.assertEqual(report["evaluation"]["accuracy"], 1)
        self.assertEqual(report["evaluation"]["format_adherence"], 1)
        self.assertEqual([r["challenge_id"] for r in report["evaluation"]["trials"]], [c.id for c in cases])
        for trial, case in zip(report["evaluation"]["trials"], cases):
            self.assertNotIn("answer", trial)
            self.assertNotIn("prompt", trial)
            self.assertEqual(len(trial["model_prompt_sha256"]), 64)
            self.assertTrue(0 <= trial["model_seed"] <= 2**31 - 1)
        self.assertEqual(len(report["source"]["source_tree_sha256"]), 64)

    def test_format_adherence_is_not_accuracy(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 987654321"), seed=3, count=2, tiers=(1,))
        self.assertEqual(report["evaluation"]["accuracy"], 0)
        self.assertEqual(report["evaluation"]["format_adherence"], 1)

    def test_oversized_untrusted_response_is_bounded_in_report(self):
        response = "malicious-text " * 15000
        report = run_maze(ReplayAdapter({}, default=response), seed=3, count=1, tiers=(1,))
        trial = report["evaluation"]["trials"][0]
        self.assertEqual(len(trial["response_preview"]), 4096)
        self.assertEqual(trial["response_length"], len(response))
        self.assertEqual(trial["scoring_reason"], "invalid_response")
        self.assertEqual(len(trial["response_sha256"]), 64)

    def test_four_arms_same_heldout_cases_but_distinct_prompts(self):
        artifact = build_training_artifact(seed=8, count=2, tiers=(1, 2, 3))
        report = run_experiment(
            lambda: ReplayAdapter({}, default="ANSWER: 0"),
            seed=8, eval_count=2, tiers=(1, 2, 3), artifact=artifact,
        )
        arms = report["arms"]
        ids = [[t["challenge_id"] for t in arm["trials"]] for arm in arms.values()]
        seeds = [[t["model_seed"] for t in arm["trials"]] for arm in arms.values()]
        self.assertTrue(all(ids[0] == member for member in ids[1:]))
        self.assertTrue(all(seeds[0] == member for member in seeds[1:]))
        prompts = [arms[k]["trials"][0]["model_prompt_sha256"] for k in arms]
        self.assertEqual(len(set(prompts)), 4)
        self.assertEqual(report["mode"], "controlled-few-shot")
        self.assertEqual(report["shuffled_label_collisions"], 0)

    def test_shuffled_control_rejects_identical_training_answers(self):
        # A circular shift is not a negative control if every label is equal.
        artifact = build_training_artifact(seed=3, count=2, tiers=(2,))
        self.assertEqual([e["answer"] for e in artifact["examples"]], ["10", "10"])
        with self.assertRaisesRegex(ValueError, "shuffled.*control"):
            run_experiment(
                lambda: ReplayAdapter({}, default="ANSWER: 0"),
                seed=3, eval_count=2, tiers=(2,), artifact=artifact,
            )

    def test_stateful_adapter_instance_reuse_is_rejected(self):
        adapter = ReplayAdapter({}, default="ANSWER: 0")
        artifact = build_training_artifact(seed=8, count=2, tiers=(1,))
        with self.assertRaisesRegex(ValueError, "same instance"):
            run_experiment(lambda: adapter, seed=8, eval_count=1, tiers=(1,), artifact=artifact)

    def test_tampered_training_artifact_rehash_cannot_forge_generated_case(self):
        artifact = build_training_artifact(seed=10, count=2, tiers=(1,))
        artifact["examples"][0]["prompt"] += "\nForged answer: 55555"
        artifact["integrity_sha256"] = digest({k: v for k, v in artifact.items() if k != "integrity_sha256"})
        with self.assertRaisesRegex(ValueError, "canonical generator"):
            validate_training_artifact(artifact)

    def test_normalized_holdout_collision_rejected(self):
        artifact = build_training_artifact(seed=10, count=2, tiers=(1,))
        example = artifact["examples"][0]
        fake_eval = Challenge(
            id="separate-eval-id", kind="arithmetic", tier=1, split="eval",
            prompt="  " + example["prompt"].upper().replace(" ", "   ") + "  ",
            answer=example["answer"],
        )
        from labyrinth.maze import generate_maze as real_generate_maze

        def generate_with_overlap(*args, **kwargs):
            if kwargs.get("split") == "eval":
                return [fake_eval]
            return real_generate_maze(*args, **kwargs)

        with patch("labyrinth.runner.generate_maze", side_effect=generate_with_overlap):
            with self.assertRaisesRegex(ValueError, "split contamination"):
                run_experiment(lambda: ReplayAdapter({}, default="ANSWER: 0"),
                               seed=10, eval_count=1, tiers=(1,), artifact=artifact)

    def test_custom_protocol_adapter_with_optional_collision(self):
        class Custom:
            model_id = "custom:valid"
            max_output_tokens = 10  # Name collision, not an Ollama contract.

            def generate(self, prompt, *, seed=None):
                return "ANSWER: 0"

        report = run_maze(Custom(), seed=7, count=1, tiers=(1,))
        self.assertEqual(report["adapter"]["id"], "custom:valid")

    def test_curriculum_case_to_adapter_to_scoring_and_replay(self):
        generated = generate_maze(seed=84, count=3, tiers=(1, 2), split="train")
        lookup = {c.id: c for c in generated}
        catalog = {str(tier): tuple(c.id for c in generated if c.tier == tier) for tier in (1, 2)}
        spec = CurriculumSpec(levels=("1", "2"), window_size=3, distinct_challenges_required=3)
        state = CurriculumState()
        model = ReplayAdapter({c.prompt: f"ANSWER: {c.answer}" for c in generated})
        case = next_training_case(spec, state, catalog, run_seed=11)
        challenge = lookup[case.challenge_id]
        response = model.generate(challenge.prompt, seed=case.seed)
        result = score_response(challenge, response)
        self.assertTrue(result.correct)
        state, promoted = record_training_result(
            spec, state,
            TrainingObservation(case.challenge_id, case.seed, case.level, float(result.points)),
        )
        self.assertFalse(promoted)
        self.assertEqual(CurriculumState.from_dict(json.loads(json.dumps(state.to_dict()))), state)


class CLITests(unittest.TestCase):
    def test_offline_cli_smoke(self):
        proc = invoke("run", "--seed", "17", "--count", "2", "--tiers", "1,2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["evaluation"]["total"], 4)
        self.assertEqual(report["model_id"], "replay:constant-zero")

    def test_structural_holdout_maze_cli(self):
        proc = invoke("run", "--seed", "17", "--count", "2", "--tiers", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["evaluation"]["total"], 2)
        self.assertTrue(all(t["kind"] == "register_program" for t in report["evaluation"]["trials"]))

    def test_toy_parameter_training_checkpoint_reload_in_new_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "trained.json")
            first_report = str(Path(tmp) / "comparison.json")
            second_report = str(Path(tmp) / "reloaded.json")
            trained = invoke(
                "train-toy", "--seed", "0", "--train-count", "64", "--eval-count", "64",
                "--checkpoint-out", checkpoint, "--report", first_report,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            experiment = json.loads(Path(first_report).read_text())
            self.assertGreater(experiment["arms"]["trained"]["accuracy"], experiment["arms"]["baseline"]["accuracy"])
            self.assertEqual(experiment["arms"]["zero_update"]["model_hash"], experiment["arms"]["baseline"]["model_hash"])
            reloaded = invoke("toy-eval", "--checkpoint", checkpoint, "--seed", "0", "--eval-count", "64",
                              "--report", second_report)
            self.assertEqual(reloaded.returncode, 0, reloaded.stderr)
            after = json.loads(Path(second_report).read_text())
            self.assertEqual(after["model_hash"], experiment["arms"]["trained"]["model_hash"])
            self.assertEqual(after["correct"], experiment["arms"]["trained"]["correct"])
            self.assertEqual(invoke("train-toy", "--checkpoint-out", checkpoint).returncode, 2)

    def test_artifact_reloaded_in_fresh_process_and_reports_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = str(Path(tmp) / "memory.json")
            first = invoke("experiment", "--seed", "31", "--train-count", "2",
                           "--eval-count", "2", "--artifact-out", artifact)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = invoke("experiment", "--seed", "31", "--eval-count", "2", "--artifact-in", artifact)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))
            self.assertEqual(len(json.loads(Path(artifact).read_text())["training_trajectory"]), 6)

    def test_report_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "run.json")
            self.assertEqual(invoke("run", "--report", target).returncode, 0)
            self.assertEqual(invoke("run", "--report", target).returncode, 2)

    def test_untrusted_cli_flags_are_rejected(self):
        proc = invoke("run", "--adapter", "ollama")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--model is required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
