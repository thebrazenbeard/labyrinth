"""Exhaustive demonstration, with optional reproducible artifact output."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

from labyrinth.sequential_boolean_specs import build


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--write", type=Path, help="Only write a manifest when explicitly requested")
    args = parser.parse_args()
    xor_net, not_net, xnor = build()
    for bits in product((0, 1), repeat=2):
        separate = not_net.evaluate(xor_net.evaluate(bits))
        fused = xnor.evaluate(bits)
        assert fused == separate == (int(bits[0] == bits[1]),)
        print(f"input={bits} xor={xor_net.evaluate(bits)} xnor={fused} pass=True")
    print("architecture=" + "->".join(map(str, [len(xnor.input_names)] +
                                             [layer.output_width for layer in xnor.layers])))
    print("parameter_count=" + str(sum(layer.output_width * layer.input_width + layer.output_width
                                         for layer in xnor.layers)))
    print("sha256=" + xnor.sha256())
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        xnor.save(args.write)
        print(f"saved={args.write}")


if __name__ == "__main__":
    main()
