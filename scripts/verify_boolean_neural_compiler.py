"""Standalone exhaustive verification and optional manifest export."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

from labyrinth.boolean_neural_specs import build


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--write", type=Path, default=None,
                        help="Optional explicit output file for the weight manifest")
    args = parser.parse_args()
    parity, majority, linked = build()
    print("input  parity  majority  combined  arithmetic")
    for bits in product((0, 1), repeat=3):
        result = linked.evaluate(bits)
        ok = result[0] + 2 * result[1] == sum(bits)
        assert ok and result == parity.evaluate(bits) + majority.evaluate(bits)
        print(f"{bits!s:11} {parity.evaluate(bits)!s:7} {majority.evaluate(bits)!s:9} {result!s:9} {ok}")
    print(f"artifact_sha256={linked.sha256()}")
    print(f"parameters={sum(map(len, linked.w1)) + len(linked.b1) + sum(map(len, linked.w2)) + len(linked.b2)}")
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        linked.write_manifest(args.write)
        print(f"saved={args.write}")


if __name__ == "__main__":
    main()
