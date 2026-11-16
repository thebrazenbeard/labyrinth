"""Source-only JSON receipt generator for out-of-catalog Minotaur discoveries.

Example:
    python3 -m labyrinth.novel_discovery_cli --seed 0

The CLI writes a canonical review receipt to stdout, never accepts rules into
production, never writes credentials/files and never triggers neural training.
"""

from __future__ import annotations

import argparse
import json

from .novel_discovery_world import (CandidateSession, POST_PROBES,
                                    PRE_PROBES, make_world)


DEMO_PROPOSAL = {
    "form": "and_literals",
    "literals": [
        {"field": "x", "equals": 1},
        {"field": "y", "equals": 1},
        {"field": "prev_x", "equals": 0},
        {"field": "prev2_x", "equals": 1},
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minotaur novel-rule receipt demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--author", default="candidate")
    parser.add_argument("--proposal-json", help="JSON rule proposal; default demonstrates two-step rebound")
    args = parser.parse_args(argv)
    try:
        proposal = json.loads(args.proposal_json) if args.proposal_json else DEMO_PROPOSAL
        world = make_world(args.seed, control=args.control)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    session = CandidateSession(world)
    for x, y in PRE_PROBES + POST_PROBES:
        session.intervene(x, y)
    result = session.submit_proposal(proposal, author=args.author)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if result.get("review_status") == "PENDING_INDEPENDENT_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
