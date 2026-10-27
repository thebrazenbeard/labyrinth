"""Contract tests for train-only evidence and immutable CPU pilot inputs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.neural_cpu_pilot import (
    file_digest, read_training, supervised_target, training_prefix,
)
from labyrinth.program_maze import generate_program_challenges


class NeuralCpuPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "train.json"
        self.sample = [{"case_id": "train-1", "split": "train", "prompt": "Compute 2+2",
                        "answer": "4"}]

    def write(self, records):
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def test_valid_train_only_data_and_digest(self):
        self.write(self.sample)
        self.assertEqual(read_training(self.path), self.sample)
        self.assertEqual(file_digest(self.path), hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_reject_eval_gold_and_duplicate_cases(self):
        for bad in (
            [{**self.sample[0], "split": "final"}],
            self.sample * 2,
            [{**self.sample[0], "secret": "leak"}],
            [{**self.sample[0], "answer": "ANSWER: 4"}],
            [{**self.sample[0], "answer": "-4"}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.write(bad)
                read_training(self.path)

    def test_reject_empty_and_oversized_data(self):
        for bad in ([], [{}], self.sample * 257):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                self.write(bad)
                read_training(self.path)

    def test_generator_attestation_rejects_relabelled_holdout(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        seed = 1807
        generated = generate_program_challenges(seed, 4, "train")
        valid = [
            {"case_id": c.id, "split": c.split, "prompt": c.prompt,
             "answer": c.answer}
            for c in generated
        ]
        attest_generated_train_only(valid, generator_seed=seed)
        # Self-reported split used to be the only barrier. Changing it is
        # insufficient evidence that this is a train-generator example.
        holdout = generate_program_challenges(seed, 1, "dev")[0]
        forged = {"case_id": holdout.id, "split": "train",
                  "prompt": holdout.prompt, "answer": holdout.answer}
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only([forged], generator_seed=seed)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only([{**valid[0], "answer": "0"}], generator_seed=seed)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only(valid, generator_seed=seed + 1)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only(valid[::-1], generator_seed=seed)

    def test_existing_train_only_128_case_corpus_attests(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        repo = Path(__file__).resolve().parents[1]
        for name, seed, count in (
            ("train_cpu_pilot_seed1701.json", 1701, 12),
            ("train_cpu_pilot_seed1702_32.json", 1702, 32),
            ("train_cpu_pilot_seed1807_128.json", 1807, 128),
        ):
            with self.subTest(fixture=name):
                fixture = repo / "experiments" / "data" / name
                self.assertEqual(
                    len(attest_generated_train_only(
                        read_training(fixture), generator_seed=seed)), count,
                )

    def test_trace_target_and_corrupted_labels_are_distinct(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": ch.split,
                  "prompt": ch.prompt, "answer": ch.answer}
        answer = supervised_target(sample, policy="answer", arm="trained", step=0)
        trace = supervised_target(sample, policy="trace", arm="trained", step=0)
        corrupt = supervised_target(sample, policy="trace", arm="corrupted", step=0)
        self.assertEqual(answer, "ANSWER: " + ch.answer)
        self.assertIn("STATE 1:", trace)
        self.assertTrue(trace.endswith(answer))
        self.assertNotEqual(trace, corrupt)
        self.assertTrue(corrupt.endswith("ANSWER: " + str((int(ch.answer) + 1) % 97)))
        self.assertEqual(trace.count("STATE "), corrupt.count("STATE "))

    def test_invalid_policy_step_and_holdout_rejected(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": "train", "prompt": ch.prompt,
                  "answer": ch.answer}
        for kwargs in ({"policy": "unknown", "arm": "trained", "step": 0},
                       {"policy": "trace", "arm": "other", "step": 0},
                       {"policy": "trace", "arm": "trained", "step": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                supervised_target(sample, **kwargs)
        with self.assertRaises(ValueError):
            supervised_target({**sample, "split": "final"}, policy="trace",
                              arm="trained", step=0)

    def test_trace_policy_rewrites_one_line_instruction_to_trace_contract(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": ch.split,
                  "prompt": ch.prompt, "answer": ch.answer}
        self.assertEqual(training_prefix(sample, policy="answer"), ch.prompt + "\n")
        prompt = training_prefix(sample, policy="trace")
        self.assertIn("STATE N: A=<integer>, B=<integer>", prompt)
        self.assertNotIn("Respond with exactly one line:", prompt)
        self.assertIn("Initial registers:", prompt)
        self.assertNotIn(ch.answer, prompt.split("Then finish with exactly one line:")[-1])
        with self.assertRaises(ValueError):
            training_prefix({"prompt": "unexpected prompt"}, policy="trace")

    def test_repository_scripts_run_without_pythonpath_and_never_overwrite(self):
        repo = Path(__file__).resolve().parents[1]
        preflight = subprocess.run(
            [sys.executable, str(repo / "scripts" / "preflight_trace_prefix.py"), "--help"],
            cwd=self.tmp.name, capture_output=True, text=True,
        )
        self.assertEqual(preflight.returncode, 0, preflight.stderr)
        generator = repo / "scripts" / "generate_neural_train.py"
        output = Path(self.tmp.name) / "one.json"
        args = [sys.executable, str(generator), "--seed", "1901",
                "--count", "1", "--output", str(output)]
        first = subprocess.run(args, cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(args, cwd=self.tmp.name, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_train_cli_requires_generator_seed_before_any_neural_load(self):
        cmd = [sys.executable, "-m", "labyrinth.neural_cpu_pilot", "train",
               "--base", "missing", "--train-data", "missing", "--output", "missing",
               "--arm", "trained"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("--train-generator-seed", run.stderr)


if __name__ == "__main__":
    unittest.main()
