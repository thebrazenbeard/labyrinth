"""Executable synthetic protocol test; not evidence of model qualification."""
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.graduation_gate import GateState, GraduationPolicy, decide_graduation
from labyrinth.private_examiner import ExamCase, Arm, run_exam


class PrivateExaminerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arms = {}
        for name in ("candidate", "baseline", "zero_update", "corrupted"):
            path = self.root / name
            path.write_text(name, encoding="utf-8")
            self.arms[name] = Arm(path, (sys.executable, "-m", "labyrinth.synthetic_exam_adapter"),
                                  0 if name == "baseline" else 32)
        self.key = b"synthetic-custodian-test-key-not-for-use-outside-tests"
        self.cases = {
            name: [ExamCase(f"{name}-{i}", f"Synthetic example {name} {i}: answer is {i + 10}",
                            str(i + 10)) for i in range(16)]
            for name in ("heldout", "transfer")
        }
        self.training = [ExamCase("train-0", "synthetic training prompt", "0")]
        digest = lambda name: hashlib.sha256(name.encode()).hexdigest()
        self.policy = GraduationPolicy(
            "synthetic-exam", digest("candidate"), digest("baseline"), "12" * 16,
            digest("synthetic examiner v1"), digest("synthetic conditions v1"), max_attempts=2)

    def examine(self, **overrides):
        args = dict(policy=self.policy, state=GateState(), arms=self.arms,
                    cases=self.cases, training_cases=self.training, key=self.key,
                    run_id="synthetic-run-1", evaluator_sha256=self.policy.expected_evaluator_sha256,
                    conditions_sha256=self.policy.expected_conditions_sha256,
                    synthetic=True)
        args.update(overrides)
        return run_exam(**args)

    def test_actual_subprocess_predictions_are_scored_and_gate_accepts(self):
        raw, sig = self.examine()
        decision = decide_graduation(self.policy, GateState(), raw, sig,
                                     independent_evaluator_key=self.key)
        self.assertEqual(decision.disposition, "GRADUATED")
        self.assertNotIn(b"answer is", raw)
        self.assertNotIn(b"Synthetic example", raw)
        self.assertNotIn(b'"answer"', raw)

    def test_missing_key_or_oracle_fails_closed(self):
        for change in ({"key": None}, {"key": b"short"},
                       {"cases": {"heldout": self.cases["heldout"], "transfer": []}},
                       {"synthetic": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.examine(**change)

    def test_trainee_cannot_inject_correctness_boolean(self):
        # The adapter sees only prompt and an opaque checkpoint name; it cannot
        # declare correctness. A false answer remains wrong despite JSON text.
        bad = dict(self.cases)
        bad["heldout"] = [ExamCase(c.id, c.prompt, "999") for c in bad["heldout"]]
        raw, sig = self.examine(cases=bad)
        decision = decide_graduation(self.policy, GateState(), raw, sig,
                                     independent_evaluator_key=self.key)
        self.assertEqual(decision.disposition, "STAY")

    def test_subprocess_failure_does_not_sign(self):
        arms = dict(self.arms)
        arms["candidate"] = Arm(self.arms["candidate"].checkpoint,
                                (sys.executable, "-c", "import sys; sys.exit(3)"), 32)
        with self.assertRaises(RuntimeError):
            self.examine(arms=arms)

    def test_cli_key_custody_and_detached_signature(self):
        config = {
            "policy": asdict(self.policy), "attempt": 1, "run_id": "cli-run-1",
            "synthetic": True,
            "evaluator_sha256": self.policy.expected_evaluator_sha256,
            "conditions_sha256": self.policy.expected_conditions_sha256,
            "arms": {name: {"checkpoint": str(arm.checkpoint),
                            "command": list(arm.command), "training_steps": arm.training_steps}
                     for name, arm in self.arms.items()},
            "cases": {name: [asdict(c) for c in group] for name, group in self.cases.items()},
            "training_cases": [asdict(c) for c in self.training],
        }
        config_path = self.root / "private-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        prefix = self.root / "signed"
        command = [sys.executable, "-m", "labyrinth.private_examiner", str(config_path),
                   str(prefix)]
        missing = subprocess.run(command, capture_output=True, text=True,
                                 env={k: v for k, v in os.environ.items()
                                      if k != "EXAMINER_HMAC_KEY_HEX"})
        self.assertEqual(missing.returncode, 1)
        self.assertFalse(prefix.with_suffix(".receipt.json").exists())
        valid = subprocess.run(command, capture_output=True, text=True,
                               env={**os.environ, "EXAMINER_HMAC_KEY_HEX": self.key.hex()})
        self.assertEqual(valid.returncode, 0, valid.stderr)
        raw = prefix.with_suffix(".receipt.json").read_bytes()
        sig = prefix.with_suffix(".signature.txt").read_text().strip()
        self.assertEqual(decide_graduation(self.policy, GateState(), raw, sig,
                         independent_evaluator_key=self.key).disposition, "GRADUATED")
        self.assertNotIn(self.key.hex(), valid.stdout + valid.stderr)


if __name__ == "__main__":
    unittest.main()
