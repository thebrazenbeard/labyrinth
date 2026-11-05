"""Contract tests for train-only evidence and immutable CPU pilot inputs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.neural_cpu_pilot import file_digest, read_training


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


if __name__ == "__main__":
    unittest.main()
