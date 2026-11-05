"""Corpus review: measure structural overlap without test-data exposure."""
import hashlib
import json
import pathlib
import tempfile
import unittest

from test_trainingday_corpus import make_repo
from labyrinth.trainingday_corpus import build_corpus
from labyrinth.trainingday_audit import audit_corpus


class CorpusQualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = pathlib.Path(self.temp.name)
        _, python = make_repo(root, "Python", {
            "train.py": "def long_algorithm(x):\n    return x + 19\n" * 140,
        })
        _, go = make_repo(root, "Go", {
            "hold.go": "package algorithms\nfunc Twice(x int) int { return x*2 }\n" * 4
        })
        self.specs = [python, go]
        self.root = root / "output"
        self.report = build_corpus(
            [python, go], self.root, heldout_repo="TheAlgorithms/Go",
            reasoning_count=300, seed=29033, rows_per_shard=20,
        )

    def test_reports_correct_counts_and_in_distribution_overlap(self):
        audit = audit_corpus(self.root)
        self.assertEqual(audit["status"], "PASS_WITH_LIMITATIONS")
        self.assertEqual(audit["manifest_sha256"],
                         hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(audit["counts"], self.report["counts"])
        self.assertEqual(audit["eval_type"], "DEVELOPMENT_IN_DISTRIBUTION_NOT_BLIND")
        self.assertGreater(audit["reasoning_structure_overlap"]["shared_signatures"], 0)
        self.assertGreater(audit["reasoning_structure_overlap"]["eval_with_train_signature"], 0)
        self.assertGreater(audit["code"]["train_over_4096_characters"], 0)
        self.assertEqual(audit["code"]["eval_code"], 1)
        self.assertEqual(audit["verified_reasoning"]["total"], 300)

    def test_structural_holdout_has_zero_operator_pattern_overlap(self):
        target = self.root.parent / "structure_corpus"
        report = build_corpus(
            self.specs, target, heldout_repo="TheAlgorithms/Go",
            reasoning_count=2500, seed=29034, rows_per_shard=100,
            reasoning_split_policy="signature",
        )
        audit = audit_corpus(target)
        self.assertEqual(audit["counts"], report["counts"])
        self.assertEqual(audit["reasoning_structure_overlap"]["shared_signatures"], 0)
        self.assertEqual(audit["reasoning_structure_overlap"]["eval_with_train_signature"], 0)
        self.assertEqual(audit["eval_type"], "DEVELOPMENT_STRUCTURAL_HOLDOUT_NOT_BLIND")

    def test_rejects_corrupted_later_eval_shard(self):
        shards = sorted((self.root / "eval").glob("*.jsonl"))
        self.assertGreater(len(shards), 1)
        later = shards[-1]
        later.write_bytes(later.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            audit_corpus(self.root)

    def test_rejects_rehash_of_false_arithmetic_answer(self):
        # A compromised writer could recompute a shard digest and edit the manifest.
        # The audit must independently check the integer operation oracle.
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        target = sorted((self.root / "eval").glob("*.jsonl"))[0]
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        case = next(row for row in rows if row["kind"] == "verified_reasoning")
        case["answer"] += 1
        target.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        entry = next(x for x in manifest["shards"] if x["path"] == "eval/" + target.name)
        entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        entry["bytes"] = target.stat().st_size
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "oracle"):
            audit_corpus(self.root)

    def test_refuses_missing_eval_shards(self):
        sorted((self.root / "eval").glob("*.jsonl"))[-1].unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_corpus(self.root)


if __name__ == "__main__":
    unittest.main()
