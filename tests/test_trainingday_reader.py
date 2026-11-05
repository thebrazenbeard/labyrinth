"""Train-only consumer verification: no evaluation leakage, no tampered shards."""
import hashlib
import json
import pathlib
import tempfile
import unittest

from test_trainingday_corpus import make_repo, MIT
from labyrinth.trainingday_corpus import build_corpus
from labyrinth.trainingday_reader import iter_training_rows


class TrainReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)
        _, python = make_repo(root, "Python", {
            "code.py": "def fit(n):\n    return n * n + 2\n" * 8,
        })
        _, go = make_repo(root, "Go", {
            "code.go": "package a\nfunc Fit(n int) int { return n * n + 2 }\n" * 8,
        })
        self.output = root / "corpus"
        self.manifest = build_corpus([python, go], self.output, heldout_repo="TheAlgorithms/Go",
                                     reasoning_count=100, seed=43, rows_per_shard=15)

    def test_only_train_rows_without_eval_in_reader(self):
        got = list(iter_training_rows(self.output))
        self.assertEqual(len(got), self.manifest["counts"]["train"])
        self.assertTrue(all(row["kind"] in {"code", "verified_reasoning"} for row in got))
        self.assertFalse(any(row.get("source_repo") == "TheAlgorithms/Go" for row in got))
        eval_hashes = {json.loads(line)["content_sha256"]
                       for p in (self.output / "eval").glob("*.jsonl")
                       for line in p.read_text(encoding="utf-8").splitlines()}
        self.assertFalse({r["content_sha256"] for r in got} & eval_hashes)

    def test_shard_tampering_refused(self):
        target = sorted((self.output / "train").glob("*.jsonl"))[0]
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            list(iter_training_rows(self.output))

    def test_later_shard_tamper_is_rejected_before_first_training_row(self):
        shards = sorted((self.output / "train").glob("*.jsonl"))
        self.assertGreater(len(shards), 1)
        last = shards[-1]
        last.write_bytes(last.read_bytes() + b"\\n")
        # Consumer may take only the first K rows. It must not receive any
        # records from an unqualified corpus with a corrupt later shard.
        with self.assertRaisesRegex(ValueError, "checksum"):
            next(iter_training_rows(self.output))

    def test_missing_shard_refused(self):
        target = sorted((self.output / "train").glob("*.jsonl"))[0]
        target.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            list(iter_training_rows(self.output))

    def test_license_notice_copied_and_bound(self):
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["licenses"]), 2)
        for license in manifest["licenses"]:
            file = self.output / license["path"]
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), license["sha256"])
            self.assertIn("Permission is hereby granted", file.read_text(encoding="utf-8"))
        list(iter_training_rows(self.output))
        license = self.output / manifest["licenses"][0]["path"]
        license.write_text("SPOOFED MIT", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "license checksum"):
            list(iter_training_rows(self.output))


if __name__ == "__main__":
    unittest.main()
