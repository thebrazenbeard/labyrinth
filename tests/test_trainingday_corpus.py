"""VERA::TRAININGDAY corpus admission and leakage tests (offline)."""
import json
import pathlib
import subprocess
import tempfile
import unittest

from labyrinth.trainingday_corpus import build_corpus


MIT = """MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


def git(*cmd):
    subprocess.run(["git", *map(str, cmd)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_repo(root, name, files, license_text=MIT):
    target = root / name
    target.mkdir()
    git("init", target)
    git("-C", target, "config", "user.name", "Corpus Tester")
    git("-C", target, "config", "user.email", "corpus@example.invalid")
    git("-C", target, "remote", "add", "origin", f"https://github.com/TheAlgorithms/{name}.git")
    (target / "LICENSE").write_text(license_text, encoding="utf-8")
    for filename, contents in files.items():
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    git("-C", target, "add", ".")
    git("-C", target, "commit", "-m", "fixture")
    revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    import hashlib
    digest = hashlib.sha256((target / "LICENSE").read_bytes()).hexdigest()
    return target, {"path": str(target), "repo": f"TheAlgorithms/{name}", "revision": revision,
                    "license_sha256": digest}


def rows(output, split):
    data = []
    for path in sorted((output / split).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            data.append(json.loads(line))
    return data


class TrainingDayCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def sources(self):
        code = "def algorithm(value):\n    return value + 8\n" * 8
        _, train = make_repo(self.root, "Python", {
            "a.py": code,
            "duplicate.py": code,
            "secret.py": "-----BEGIN PRIVATE KEY-----\n" + code,
            "README.md": "not code\n",
            "tiny.py": "x=1\n",
        })
        _, held = make_repo(self.root, "Go", {"b.go": "package main\nfunc Twice(x int) int { return x * 2 }\n" * 4})
        return [train, held]

    def build(self, sources=None, count=100):
        specs = sources or self.sources()
        return build_corpus(specs, self.root / "out", heldout_repo="TheAlgorithms/Go",
                            reasoning_count=count, seed=29033, rows_per_shard=25)

    def test_corpus_is_licensed_deduped_and_split(self):
        report = self.build()
        train, held = rows(self.root / "out", "train"), rows(self.root / "out", "eval")
        self.assertEqual(report["counts"]["train"], len(train))
        self.assertEqual(report["counts"]["eval"], len(held))
        self.assertGreater(len(train), 75)
        self.assertGreater(len(held), 18)
        self.assertEqual(len([r for r in train if r["kind"] == "code"]), 1)
        self.assertEqual(len([r for r in held if r["kind"] == "code"]), 1)
        self.assertTrue(all(r["license"] == "MIT" for r in train if r["kind"] == "code"))
        self.assertFalse(any("PRIVATE KEY" in r["text"] for r in train))
        train_ids = {r["id"] for r in train}
        held_ids = {r["id"] for r in held}
        self.assertFalse(train_ids & held_ids)
        train_text = {r["content_sha256"] for r in train}
        self.assertFalse(train_text & {r["content_sha256"] for r in held})
        self.assertTrue(all("source_revision" in r and "source_path" in r for r in held if r["kind"] == "code"))
        self.assertEqual(report["sources"][0]["repo"], "TheAlgorithms/Go")  # held-out admitted first
        self.assertTrue(all(s["sha256"] and s["count"] <= 25 for s in report["shards"]))

    def test_reproducible_shard_digests_across_output_directories(self):
        specs = self.sources()
        one = self.build(specs)
        two = build_corpus(specs, self.root / "second", heldout_repo="TheAlgorithms/Go",
                           reasoning_count=100, seed=29033, rows_per_shard=25)
        self.assertEqual([(s["split"], s["sha256"], s["count"]) for s in one["shards"]],
                         [(s["split"], s["sha256"], s["count"]) for s in two["shards"]])

    def test_reasoning_signature_holdout_has_no_training_pattern_overlap(self):
        specs = self.sources()
        report = build_corpus(
            specs, self.root / "structural", heldout_repo="TheAlgorithms/Go",
            reasoning_count=2500, seed=29034, rows_per_shard=100,
            reasoning_split_policy="signature",
        )
        train = [row for row in rows(self.root / "structural", "train")
                 if row["kind"] == "verified_reasoning"]
        heldout = [row for row in rows(self.root / "structural", "eval")
                   if row["kind"] == "verified_reasoning"]
        sig = lambda r: tuple(op for op, operand in r["operations"])
        self.assertGreater(len(train), 300)
        self.assertGreater(len(heldout), 50)
        self.assertFalse({sig(r) for r in train} & {sig(r) for r in heldout})
        self.assertEqual(report["reasoning_split_policy"], "STRUCTURAL_SIGNATURE_HASH_V1")

    def test_invalid_reasoning_split_policy_rejected_without_creating_output(self):
        specs = self.sources()
        target = self.root / "bad_split"
        with self.assertRaisesRegex(ValueError, "reasoning_split_policy"):
            build_corpus(specs, target, heldout_repo="TheAlgorithms/Go",
                         reasoning_count=200, reasoning_split_policy="unknown")
        self.assertFalse(target.exists())

    def test_unlicensed_source_fails_before_any_dataset_emission(self):
        _, bad = make_repo(self.root, "Java", {"a.java": "class A { int add(int a) { return a + 1; } }\n" * 4},
                           license_text="GNU GENERAL PUBLIC LICENSE Version 3")
        _, good = make_repo(self.root, "Go", {"ok.go": "package a\nfunc f(x int) int { return x+1 }\n" * 4})
        with self.assertRaisesRegex(ValueError, "MIT license"):
            self.build([bad, good])
        self.assertFalse((self.root / "out").exists())

    def test_wrong_source_commit_rejected(self):
        specs = self.sources()
        specs[0]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "revision"):
            self.build(specs)
        self.assertFalse((self.root / "out").exists())

    def test_dirty_source_rejected(self):
        specs = self.sources()
        (self.root / "Python" / "changed.py").write_text("print('new')", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.build(specs)
        self.assertFalse((self.root / "out").exists())

    def test_refuses_overwrite_or_missing_holdout(self):
        specs = self.sources()
        with self.assertRaisesRegex(ValueError, "heldout"):
            build_corpus(specs, self.root / "out", heldout_repo="not-supplied", reasoning_count=20)
        self.build(specs)
        with self.assertRaises(FileExistsError):
            self.build(specs)

    def test_reasoning_answers_replay_with_independent_integer_interpreter(self):
        self.build(count=120)
        for row in rows(self.root / "out", "eval"):
            if row["kind"] != "verified_reasoning":
                continue
            value = row["start"]
            for op, operand in row["operations"]:
                if op == "add":
                    value += operand
                elif op == "subtract":
                    value -= operand
                elif op == "multiply":
                    value *= operand
                else:
                    self.fail("unexpected operation")
            self.assertEqual(value, row["answer"])
            self.assertIn(str(value), row["text"])


if __name__ == "__main__":
    unittest.main()
