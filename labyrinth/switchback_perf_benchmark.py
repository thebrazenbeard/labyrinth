"""Bounded reproducible source-only Switchback PLAN+receipt replay microbenchmark."""

from __future__ import annotations

import cProfile
import json
import pstats
from time import perf_counter

from .switchback_circuit import ProbeSession, make_case
from .switchback_controls import MEMORIZED_SEED0_PATH
from .switchback_oracle import verify_receipt_replay


def measure() -> dict[str, object]:
    case = make_case(0)
    plan = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()
    result = ProbeSession(case).submit(plan)
    replay_valid = verify_receipt_replay(result)
    profiler.disable()
    elapsed = perf_counter() - start
    if result["verdict"] != "VERIFIED_WIN" or not replay_valid:
        raise RuntimeError("benchmark requires genuine source-verified receipt")
    entries = pstats.Stats(profiler).stats
    counts = {}
    for name in ("_valid_case", "oracle_case", "_rule_output", "oracle_step",
                 "independent_transition_parity"):
        counts[name] = sum(entry[1] for (filename, lineno, function), entry
                           in entries.items() if function == name)
    return {"subject": "seed0_PLAN_plus_receipt_replay",
            "verdict": result["verdict"], "replay_valid": replay_valid,
            "cprofile_wall_seconds": round(elapsed, 6),
            "total_profile_calls": sum(entry[1] for entry in entries.values()),
            "function_calls": counts}


if __name__ == "__main__":
    print(json.dumps(measure(), sort_keys=True))
