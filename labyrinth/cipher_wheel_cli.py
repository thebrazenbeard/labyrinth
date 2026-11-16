"""Deterministic source-only Cipher Wheel exercise; no models or services."""

from __future__ import annotations

import argparse
import json

from .cipher_wheel_controls import memorized_control, symbolic_explorer
from .cipher_wheel_world import CandidateSession, make_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public source-only Cipher Wheel control test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--control", choices=("symbolic", "memorized"), default="symbolic")
    args = parser.parse_args(argv)
    world = make_world(args.seed, heldout=args.heldout, variant=args.variant)
    session = CandidateSession(world)
    receipt = (symbolic_explorer(session) if args.control == "symbolic"
               else memorized_control(session))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
