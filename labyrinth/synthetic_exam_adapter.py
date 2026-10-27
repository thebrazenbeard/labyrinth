"""Deliberately trivial subprocess fixture. NOT a neural model or private oracle."""

import pathlib
import re
import sys


def main() -> None:
    # Synthetic fixture only: label deliberately embedded in the public prompt.
    # Real private exams must NOT expose gold this way.
    arm = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    prompt = sys.stdin.read()
    match = re.search(r"answer is ([0-9]+)", prompt)
    if match is None:
        raise SystemExit(2)
    answer = int(match.group(1))
    print(f"ANSWER: {answer if arm == 'candidate' else answer + 1}")


if __name__ == "__main__":
    main()
