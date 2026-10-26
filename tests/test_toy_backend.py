"""Real CPU parameter updates and deliberately narrow lifecycle evidence."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from labyrinth.adapters import AdapterError, ModelAdapter
from labyrinth.maze import generate_maze
from labyrinth.toy_backend import ToyLinearAdapter, evaluate_toy_model, run_toy_experiment


class ToyBackendTests(unittest.TestCase):
    def test_real_update_noop_and_corruption(self):
        cases = generate_maze(12, tiers=(1,), count=64, split="train")
        baseline = ToyLinearAdapter(seed=12)
        self.assertIsInstance(baseline, ModelAdapter)
        initial = baseline.model_hash
        trained = ToyLinearAdapter(seed=12)
        result = trained.train(cases)
        self.assertNotEqual(initial, trained.model_hash)
        self.assertGreater(result["parameter_updates"], 0)
        self.assertEqual(trained.generate("Calculate 723 - 612.\nRespond with exactly one line: ANSWER: <integer>."), "ANSWER: 111")
        noop = baseline.train(cases, learning_rate=0)
        self.assertEqual(initial, baseline.model_hash)
        self.assertEqual(noop["parameter_updates"], 0)
        corrupted = ToyLinearAdapter(seed=12)
        sham = corrupted.train(cases, shuffled_labels=True)
        self.assertEqual(sham["changed_labels"], len(cases))
        self.assertEqual(sham["case_visits"], result["case_visits"])
        self.assertNotEqual(trained.model_hash, corrupted.model_hash)

    def test_paired_deterministic_heldout(self):
        report = run_toy_experiment(seed=12)
        self.assertEqual(report, run_toy_experiment(seed=12))
        arms = report["arms"]
        self.assertLess(arms["baseline"]["accuracy"], 0.1)
        self.assertGreaterEqual(arms["trained"]["accuracy"], 0.95)
        self.assertLess(arms["shuffled_labels"]["accuracy"], arms["trained"]["accuracy"])
        self.assertEqual(arms["baseline"]["model_hash"], arms["zero_update"]["model_hash"])
        for arm in arms.values():
            self.assertEqual(arm["training"]["case_visits"], 64 * 80)
            self.assertEqual([t["challenge_id"] for t in arm["trials"]], report["eval_challenge_ids"])
        self.assertTrue(set(report["train_challenge_ids"]).isdisjoint(report["eval_challenge_ids"]))
        self.assertTrue(set(report["train_prompt_hashes"]).isdisjoint(report["eval_prompt_hashes"]))

    def test_readonly_evaluation_matches_experiment(self):
        model = ToyLinearAdapter(seed=12)
        model.train(generate_maze(12, tiers=(1,), count=64, split="train"))
        before = model.model_hash
        with patch.object(model, "train", side_effect=AssertionError("evaluation must not train")):
            evaluation = evaluate_toy_model(model, seed=12, eval_count=64)
        paired = run_toy_experiment(seed=12)["arms"]["trained"]
        self.assertEqual(evaluation["trials"], paired["trials"])
        self.assertEqual(evaluation["accuracy"], paired["accuracy"])
        self.assertEqual(model.model_hash, before)

    def test_eval_labels_rejected_before_access(self):
        class Poison:
            split = "eval"
            @property
            def answer(self):
                raise AssertionError("evaluation gold accessed by training")
        with self.assertRaises(ValueError):
            ToyLinearAdapter().train([Poison(), Poison()])

    def test_overlap_fails_closed_and_invalid_train_batch_is_atomic(self):
        cases = generate_maze(12, tiers=(1,), count=64, split="train")
        with patch("labyrinth.toy_backend.generate_maze", side_effect=[cases, cases]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_toy_experiment(seed=12)
        model = ToyLinearAdapter(seed=12)
        before = model.model_hash
        for invalid in ([cases[0], cases[0]], [cases[0], replace(cases[1], answer="nan")],
                        [cases[0], replace(cases[1], split="eval")]):
            with self.assertRaises(ValueError):
                model.train(invalid)
            self.assertEqual(model.model_hash, before)

    def test_adapter_only_receives_public_prompt(self):
        original = ToyLinearAdapter.generate
        seen = []
        def public_only(model, prompt, *, seed=None):
            self.assertIsInstance(prompt, str)
            self.assertTrue(prompt.startswith("Calculate "))
            seen.append(prompt)
            return original(model, prompt, seed=seed)
        with patch.object(ToyLinearAdapter, "generate", public_only):
            report = run_toy_experiment(seed=12, eval_count=5)
        self.assertEqual(len(seen), 20)
        self.assertEqual(seen[:5], seen[5:10])
        self.assertEqual(seen[:5], seen[10:15])
        self.assertEqual(seen[:5], seen[15:])
        self.assertNotIn("answer", report["arms"]["trained"]["trials"][0])

    def test_fresh_process_checkpoint_and_overwrite_protection(self):
        model = ToyLinearAdapter(seed=12)
        model.train(generate_maze(12, tiers=(1,), count=64, split="train"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            saved_hash = model.save_checkpoint(path)
            self.assertEqual(saved_hash, model.model_hash)
            reloaded = ToyLinearAdapter.load_checkpoint(path)
            self.assertEqual(reloaded.model_hash, model.model_hash)
            script = ("from labyrinth.toy_backend import ToyLinearAdapter; import sys; "
                      "m=ToyLinearAdapter.load_checkpoint(sys.argv[1]); print(m.model_hash); "
                      "print(m.generate('Calculate 100 + 300.\\nRespond with exactly one line: ANSWER: <integer>.'))")
            result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(result.stdout.splitlines(), [model.model_hash, "ANSWER: 400"])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                model.save_checkpoint(path)
            self.assertEqual(path.read_bytes(), before)
            symlink = Path(directory) / "link.json"
            symlink.symlink_to(path)
            with self.assertRaises(FileExistsError):
                model.save_checkpoint(symlink)
            with self.assertRaises((OSError, ValueError)):
                ToyLinearAdapter.load_checkpoint(symlink)

    def test_strict_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            ToyLinearAdapter().save_checkpoint(path)
            valid = json.loads(path.read_text())
            cases = [b"not json", b"{" * 10000, b'{"schema":1,"schema":2}', b'[]']
            for field, value in (("seed", True), ("version", "unknown"),
                                 ("weights", {"+": [float("nan"), 0], "-": [0, 0]}),
                                 ("weights", {"+": [1e20, 0], "-": [0, 0]}),
                                 ("weights", {"+": [10**1000, 0], "-": [0, 0]}),
                                 ("weights", {"+": [True, 0], "-": [0, 0]}),
                                 ("weights", {"+": [0], "-": [0, 0]}),
                                 ("unexpected", "exec")):
                bad = {**valid, field: value}
                # Re-sign to prove structural validation is not just checksum validation.
                payload = {k: v for k, v in bad.items() if k != "integrity_sha256"}
                bad["integrity_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                cases.append(json.dumps(bad).encode())
            cases.append(json.dumps({**valid, "integrity_sha256": "0" * 64}).encode())
            for index, data in enumerate(cases):
                malformed = Path(directory) / f"bad-{index}.json"
                malformed.write_bytes(data)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    ToyLinearAdapter.load_checkpoint(malformed)

    def test_bounds_and_failure_atomicity(self):
        model = ToyLinearAdapter()
        initial = model.model_hash
        for prompt in ("Calculate 1 * 2.", "__import__('os').system('id')", "Calculate 1 + 2.\nANSWER: 3"):
            with self.assertRaises(AdapterError):
                model.generate(prompt)
        for kwargs in ({"epochs": True}, {"epochs": 0}, {"learning_rate": float("nan")}, {"learning_rate": -1}):
            with self.assertRaises(ValueError):
                model.train(generate_maze(1, tiers=(1,), count=2, split="train"), **kwargs)
        self.assertEqual(initial, model.model_hash)


if __name__ == "__main__":
    unittest.main()
