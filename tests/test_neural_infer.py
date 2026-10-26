"""Inference receives public prompts and rejects any private answer fields."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.neural_infer import read_public_cases


class PublicCasesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "public.json"
        self.item = {"case_id": "final-1", "prompt": "Compute 2 + 2",
                     "prompt_sha256": hashlib.sha256(b"Compute 2 + 2").hexdigest()}

    def read(self, examples):
        self.path.write_text(json.dumps(examples))
        return read_public_cases(self.path)

    def test_valid_case(self):
        self.assertEqual(self.read([self.item]), [self.item])

    def test_no_gold_or_hidden_fields_allowed(self):
        for mutation in (
            {**self.item, "answer": "4"},
            {**self.item, "prompt_sha256": "0" * 64},
            {**self.item, "case_id": "../../escape"},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.read([mutation])
        with self.assertRaises(ValueError):
            self.read([self.item, self.item])


if __name__ == "__main__":
    unittest.main()
