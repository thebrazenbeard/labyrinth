"""Create a new, auditable train-only corpus; never inspect evaluation gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labyrinth.program_maze import (
    generate_graduated_train_challenges, generate_program_challenges,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--generator-version", choices=("v1", "v2"), default="v1")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.count <= 256:
        p.error("count must be 1..256")
    if a.generator_version == "v1":
        challenges = generate_program_challenges(a.seed, a.count, "train")
    else:
        challenges = generate_graduated_train_challenges(a.seed, a.count, "train")
    records = [
        {"case_id": c.id, "split": c.split, "prompt": c.prompt, "answer": c.answer}
        for c in challenges
    ]
    data = (json.dumps(records, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    if len(data) > 1024 * 1024:
        p.error("dataset exceeds bounded CPU pilot limit")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with a.output.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        p.error("refusing to overwrite a committed training corpus")
    print(json.dumps({"cases": len(records), "sha256": hashlib.sha256(data).hexdigest(),
                      "generator_version": a.generator_version,
                      "output": str(a.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
