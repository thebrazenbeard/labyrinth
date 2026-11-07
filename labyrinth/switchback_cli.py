"""Source-public generated Switchback Circuit evidence CLI."""

from __future__ import annotations

import argparse
import json

from .switchback_circuit import make_case
from .switchback_controls import memorized_control, privileged_symbolic_control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded source-only rule-revision challenge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control", choices=("symbolic", "memorized"), default="symbolic")
    args = parser.parse_args(argv)
    case = make_case(args.seed)
    result = (privileged_symbolic_control(case) if args.control == "symbolic"
              else memorized_control(case))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
