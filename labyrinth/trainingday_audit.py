"""Read-only structural and answer-oracle audit for a TRAININGDAY corpus.

This development audit opens train and eval metadata, so it MUST NOT be used
inside a model-training process. It returns aggregates, not example plaintext.
"""
import collections
from contextlib import closing
import hashlib
import json
import operator
import sqlite3
import tempfile
from pathlib import Path

from .trainingday_reader import _inside, _digest, SCHEMA


OPS = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}


def _oracle(row):
    if row.get("kind") != "verified_reasoning" or row.get("verification") != "TWO_INTEGER_INTERPRETERS_MATCH":
        raise ValueError("unsupported reasoning oracle")
    start = row.get("start")
    actions = row.get("operations")
    if type(start) is not int or not isinstance(actions, list) or not (2 <= len(actions) <= 5):
        raise ValueError("reasoning oracle invalid operations")
    value = start
    names = []
    for action in actions:
        if not isinstance(action, list) or len(action) != 2:
            raise ValueError("reasoning oracle invalid operation")
        name, operand = action
        if name not in OPS or type(operand) is not int:
            raise ValueError("reasoning oracle unsupported operation")
        value = OPS[name](value, operand)
        names.append(name)
    if (type(row.get("answer")) is not int or row["answer"] != value
            or not isinstance(row.get("text"), str)
            or not row["text"].endswith("Answer: " + str(value))):
        raise ValueError("reasoning oracle answer mismatch")
    return tuple(names)


def audit_corpus(corpus_root):
    """Return grounded size and structural-transfer limitations, no plaintext.

    Validates all shard bytes and license notices before examining any records.
    Uses SQLite in an isolated temporary directory for cross-split ID/digest
    uniqueness; does not mutate the corpus or expose eval records to a trainer.
    """
    root = Path(corpus_root).resolve(strict=True)
    raw_manifest = _inside(root, "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "CORPUS_BUILT_NO_MODEL_TRAINING":
        raise ValueError("unsupported manifest")
    licenses = manifest.get("licenses")
    if not isinstance(licenses, list) or not licenses:
        raise ValueError("missing license manifest")
    for notice in licenses:
        path = _inside(root, notice["path"])
        if _digest(path) != notice["sha256"]:
            raise ValueError("license checksum mismatch")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("missing shard manifest")
    seen_paths = set()
    split_paths = {"train": set(), "eval": set()}
    for shard in shards:
        split, relative = shard["split"], shard["path"]
        if split not in split_paths or not relative.startswith(split + "/") or not relative.endswith(".jsonl"):
            raise ValueError("invalid split shard path")
        if relative in seen_paths:
            raise ValueError("duplicate shard path")
        seen_paths.add(relative)
        path = _inside(root, relative)
        if path.stat().st_size != shard["bytes"] or _digest(path) != shard["sha256"]:
            raise ValueError("shard checksum mismatch")
        split_paths[split].add(relative)
    for split in split_paths:
        actual = {p.relative_to(root).as_posix() for p in (root / split).glob("*.jsonl")}
        if not actual or actual != split_paths[split]:
            raise ValueError("missing or undeclared shard")

    count = collections.Counter()
    code_lengths = collections.defaultdict(list)
    train_signatures = set()
    eval_signatures = collections.Counter()
    with tempfile.TemporaryDirectory(prefix="vera-trainingday-audit-") as tmp:
        with closing(sqlite3.connect(Path(tmp) / "ids.sqlite3")) as db:
            db.execute("CREATE TABLE seen_ids (id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE seen_digest (digest TEXT PRIMARY KEY)")
            for shard in sorted(shards, key=lambda r: (r["split"], r["path"])):
                path = _inside(root, shard["path"])
                observed_count = 0
                with path.open("r", encoding="utf-8") as rows:
                    for line in rows:
                        row = json.loads(line)
                        digest = row.get("content_sha256")
                        identifier = row.get("id")
                        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(identifier, str):
                            raise ValueError("invalid content digest or id")
                        try:
                            db.execute("INSERT INTO seen_ids VALUES (?)", (identifier,))
                            db.execute("INSERT INTO seen_digest VALUES (?)", (digest,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError("duplicate row identity or cross-split content") from error
                        text = row.get("text")
                        if not isinstance(text, str):
                            raise ValueError("row text required")
                        computed_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if computed_digest != digest:
                            raise ValueError("row content checksum mismatch")
                        split = shard["split"]
                        kind = row.get("kind")
                        if kind == "code":
                            if row.get("license") != "MIT" or row.get("verification") != "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED":
                                raise ValueError("invalid code provenance")
                            if split == "train" and row.get("source_repo") == manifest["heldout_repo"]:
                                raise ValueError("heldout code leaked into training")
                            code_lengths[split].append(len(text))
                        elif kind == "verified_reasoning":
                            if row.get("license") != "SYNTHETIC_ORIGINAL":
                                raise ValueError("invalid procedural provenance")
                            signature = _oracle(row)
                            if split == "train":
                                train_signatures.add(signature)
                            else:
                                eval_signatures[signature] += 1
                        else:
                            raise ValueError("unrecognized row kind")
                        count[(split, kind)] += 1
                        observed_count += 1
                if observed_count != shard["count"]:
                    raise ValueError("shard row count mismatch")
    if sum(v for (split, _), v in count.items() if split == "train") != manifest["counts"]["train"]:
        raise ValueError("train count mismatch")
    if sum(v for (split, _), v in count.items() if split == "eval") != manifest["counts"]["eval"]:
        raise ValueError("eval count mismatch")
    policy = manifest.get("reasoning_split_policy")
    if policy not in (None, "STRUCTURAL_SIGNATURE_HASH_V1"):
        raise ValueError("unknown reasoning split policy")
    common = train_signatures.intersection(eval_signatures)
    if policy == "STRUCTURAL_SIGNATURE_HASH_V1" and common:
        raise ValueError("structural holdout leakage")
    eval_type = ("DEVELOPMENT_STRUCTURAL_HOLDOUT_NOT_BLIND"
                 if policy == "STRUCTURAL_SIGNATURE_HASH_V1"
                 else "DEVELOPMENT_IN_DISTRIBUTION_NOT_BLIND")
    return {
        "schema": "vera.trainingday.audit.v1",
        "status": "PASS_WITH_LIMITATIONS",
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "counts": manifest["counts"],
        "code": {
            "train_code": count[("train", "code")],
            "eval_code": count[("eval", "code")],
            "train_over_4096_characters": sum(x > 4096 for x in code_lengths["train"]),
            "train_over_8192_characters": sum(x > 8192 for x in code_lengths["train"]),
        },
        "verified_reasoning": {"total": count[("train", "verified_reasoning")] + count[("eval", "verified_reasoning")]},
        "reasoning_structure_overlap": {
            "train_signatures": len(train_signatures),
            "eval_signatures": len(eval_signatures),
            "shared_signatures": len(train_signatures.intersection(eval_signatures)),
            "eval_with_train_signature": sum(n for pattern, n in eval_signatures.items() if pattern in train_signatures),
            "eval_total": sum(eval_signatures.values()),
        },
        "eval_type": eval_type,
        "claim_ceiling": "EXACT_ANSWER_AND_SHARD_INTEGRITY_ONLY_NO_REASONING_TRANSFER",
    }
