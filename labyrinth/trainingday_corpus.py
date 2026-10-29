"""Streaming, fail-closed source-first corpus builder for VERA::TRAININGDAY.

Only curated, clean, pinned, locally cloned MIT repositories are accepted.
Code rows are *unverified code examples*; integer programs have independently
replayed exact answers. This module does not train or export model weights.
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import subprocess
from pathlib import Path


CODE_EXTENSIONS = frozenset({".py", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".cc", ".hpp", ".js", ".ts"})
SECRET = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|"
    r"(?i:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})"
)
SPDX = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9_.+/-]+)")
MAX_CODE_BYTES = 65536
SCHEMA = "vera.trainingday.corpus.v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    result = subprocess.run(("git", *map(str, args)), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def validate_source(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "repo", "revision", "license_sha256"}:
        raise ValueError("source must specify exactly path, repo, revision, license_sha256")
    name = spec["repo"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid repo identity")
    repo = Path(spec["path"]).resolve(strict=True)
    if not repo.is_dir() or repo.is_symlink():
        raise ValueError("source must be a regular directory")
    remote = command("-C", repo, "remote", "get-url", "origin")
    if remote.rstrip("/").removesuffix(".git").lower() != ("https://github.com/" + name).lower():
        raise ValueError("source remote does not match pinned GitHub repo")
    revision = command("-C", repo, "rev-parse", "HEAD")
    if not isinstance(spec["revision"], str) or revision != spec["revision"]:
        raise ValueError("source revision does not match pinned commit")
    if command("-C", repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source must be clean")
    candidates = [repo / "LICENSE", repo / "LICENSE.md"]
    found = [p for p in candidates if p.is_file() and not p.is_symlink()]
    if len(found) != 1:
        raise ValueError("source requires exactly one non-symlink LICENSE or LICENSE.md")
    license_path = found[0]
    body = license_path.read_bytes()
    if sha256(body) != spec["license_sha256"]:
        raise ValueError("license digest mismatch")
    text = body.decode("utf-8", errors="strict")
    if not all(term in text for term in ("MIT License", "Permission is hereby granted", 'THE SOFTWARE IS PROVIDED "AS IS"')):
        raise ValueError("source must contain a verified MIT license")
    return {"path": repo, "repo": name, "revision": revision, "license_file": license_path.name,
            "license_sha256": sha256(body), "license": "MIT"}


class ShardedWriter:
    def __init__(self, root, max_rows):
        self.root = root
        self.max_rows = max_rows
        self.streams = {}
        self.shards = []
        self.counts = {"train": 0, "eval": 0}

    def write(self, split, row):
        number = self.counts[split] // self.max_rows
        state = self.streams.get(split)
        if state is None or state["number"] != number:
            if state is not None:
                self.finish(split)
            target = self.root / split
            target.mkdir(parents=True, exist_ok=True)
            file = target / f"{number:06d}.jsonl"
            state = {"number": number, "path": file, "handle": file.open("wb"),
                     "hash": hashlib.sha256(), "count": 0}
            self.streams[split] = state
        data = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        state["handle"].write(data)
        state["hash"].update(data)
        state["count"] += 1
        self.counts[split] += 1

    def finish(self, split):
        state = self.streams.pop(split, None)
        if state is None:
            return
        state["handle"].close()
        self.shards.append({"split": split, "path": state["path"].relative_to(self.root).as_posix(),
                            "count": state["count"], "bytes": state["path"].stat().st_size,
                            "sha256": state["hash"].hexdigest()})

    def close(self):
        for split in list(self.streams):
            self.finish(split)
        self.shards.sort(key=lambda x: (x["split"], x["path"]))


def code_rows(source):
    base = source["path"]
    for file in sorted(base.rglob("*")):
        if not file.is_file() or file.is_symlink() or ".git" in file.relative_to(base).parts:
            continue
        if file.suffix.lower() not in CODE_EXTENSIONS:
            continue
        size = file.stat().st_size
        if size < 80 or size > MAX_CODE_BYTES:
            continue
        content = file.read_bytes()
        if b"\0" in content:
            continue
        try:
            text = content.decode("utf-8", errors="strict").replace("\r\n", "\n").strip()
        except UnicodeDecodeError:
            continue
        if SECRET.search(text) or len(text) < 80:
            continue
        file_licenses = SPDX.findall(text)
        if file_licenses and any(tag != "MIT" for tag in file_licenses):
            continue
        digest = sha256(text.encode("utf-8"))
        yield {"id": "code-" + digest, "kind": "code", "verification": "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED",
               "license": "MIT", "license_sha256": source["license_sha256"],
               "license_file": source["license_file"], "source_repo": source["repo"],
               "source_revision": source["revision"], "source_path": file.relative_to(base).as_posix(),
               "content_sha256": digest, "text": text}


def reasoning_rows(count, seed):
    rng = random.Random(seed)
    ops = ("add", "subtract", "multiply")
    for index in range(count):
        start = rng.randint(-1000, 1000)
        value = start
        operations = []
        for _ in range(rng.randint(2, 5)):
            name = rng.choice(ops)
            operand = rng.randint(-15, 15)
            if name == "add":
                value += operand
            elif name == "subtract":
                value -= operand
            else:
                value *= operand
            operations.append([name, operand])
        # Second calculation path, deliberately distinct control flow; no eval/exec.
        checked = start
        for name, operand in operations:
            checked = {"add": lambda x, y: x + y, "subtract": lambda x, y: x - y,
                       "multiply": lambda x, y: x * y}[name](checked, operand)
        if checked != value:
            raise AssertionError("independent integer interpreter mismatch")
        prompt = ("Start with " + str(start) + ". " +
                  " ".join(f"{name} {operand}." for name, operand in operations) +
                  " What is the final integer?")
        text = f"Question: {prompt}\nAnswer: {value}"
        digest = sha256(text.encode("utf-8"))
        yield "eval" if index % 5 == 0 else "train", {
            "id": "reasoning-" + digest, "kind": "verified_reasoning",
            "verification": "TWO_INTEGER_INTERPRETERS_MATCH", "license": "SYNTHETIC_ORIGINAL",
            "source_repo": "VERA_TRAININGDAY_GENERATOR", "source_revision": SCHEMA,
            "source_path": f"seed/{seed}/case/{index:010d}", "start": start,
            "operations": operations, "answer": value, "prompt": prompt,
            "content_sha256": digest, "text": text}


def build_corpus(sources, output_dir, *, heldout_repo, reasoning_count=1000, seed=29033, rows_per_shard=1000):
    """Build deterministic JSONL shards + manifest without changing any source repo.

    Output must not exist. Held-out source code and every fifth reasoning prompt
    are reserved in eval/ and must never be loaded by a training-only consumer.
    """
    if type(reasoning_count) is not int or not (5 <= reasoning_count <= 10_000_000):
        raise ValueError("reasoning_count outside 5..10,000,000")
    if type(seed) is not int or not (0 <= seed < 2**31):
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if type(rows_per_shard) is not int or not (1 <= rows_per_shard <= 100_000):
        raise ValueError("rows_per_shard outside 1..100,000")
    output = Path(output_dir).resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError("corpus output already exists")
    if not output.parent.is_dir():
        raise ValueError("output parent does not exist")
    validated = [validate_source(spec) for spec in sources]
    if len({s["repo"] for s in validated}) != len(validated):
        raise ValueError("duplicate repo")
    if heldout_repo not in {s["repo"] for s in validated}:
        raise ValueError("heldout repo must be a supplied independent source")
    if len(validated) < 2:
        raise ValueError("at least one training code repo and one heldout repo required")
    # Admit evaluation repository first so exact duplicates cannot leak into train.
    validated.sort(key=lambda s: (s["repo"] != heldout_repo, s["repo"]))
    stage = output.with_name(output.name + ".building")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError("unfinished staging area exists; inspect instead of overwrite")
    stage.mkdir()
    database = sqlite3.connect(stage / "dedup.sqlite3")
    try:
        database.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY, split TEXT NOT NULL)")
        writer = ShardedWriter(stage, rows_per_shard)
        dropped = {"code_duplicate": 0, "reasoning_duplicate": 0}
        for source in validated:
            split = "eval" if source["repo"] == heldout_repo else "train"
            for row in code_rows(source):
                before = database.total_changes
                database.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (row["content_sha256"], split))
                if database.total_changes == before:
                    dropped["code_duplicate"] += 1
                    continue
                writer.write(split, row)
        for split, row in reasoning_rows(reasoning_count, seed):
            before = database.total_changes
            database.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (row["content_sha256"], split))
            if database.total_changes == before:
                dropped["reasoning_duplicate"] += 1
                continue
            writer.write(split, row)
        writer.close()
        database.commit()
        report = {"schema": SCHEMA, "state": "CORPUS_BUILT_NO_MODEL_TRAINING",
                  "seed": seed, "requested_reasoning_cases": reasoning_count, "heldout_repo": heldout_repo,
                  "counts": writer.counts, "dropped": dropped,
                  "sources": [{k: v for k, v in s.items() if k != "path"} for s in validated],
                  "shards": writer.shards,
                  "qualification": "LICENSING_AND_EXACT_ANSWER_REPLAY_ONLY_NOT_REASONING_GENERALIZATION"}
        (stage / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    finally:
        database.close()
    stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description="Create licensed code + exact verified reasoning corpus")
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-repo", required=True)
    parser.add_argument("--reasoning-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29033)
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    args = parser.parse_args()
    specs = json.loads(args.sources_json.read_text(encoding="utf-8"))
    result = build_corpus(specs, args.output, heldout_repo=args.heldout_repo,
                          reasoning_count=args.reasoning_count, seed=args.seed,
                          rows_per_shard=args.rows_per_shard)
    print(json.dumps({"state": result["state"], "counts": result["counts"],
                      "shards": len(result["shards"]), "heldout_repo": result["heldout_repo"]}))


if __name__ == "__main__":
    main()
