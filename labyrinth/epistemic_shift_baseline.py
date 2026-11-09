"""Frozen and adaptive-zero-update controls for nonstationary causal inference.

These are hand-written symbolic scripts. Their behavior is a control against
overstating neural adaptation, not evidence that a model learned the rules.
"""

from __future__ import annotations

from typing import Callable


PRE_PROBES = ((0, 0), (1, 0), (0, 1))
POST_PROBES = ((1, 1), (1, 0), (1, 0), (0, 1), (1, 1), (0, 0), (1, 0))
KNOWN_HYPOTHESES = ("copy", "xor", "xnor", "and", "or", "edge", "lag")


def _post_prediction(kind: str, previous: tuple[int, int],
                     sequence: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    prev_x, prev_y = previous
    forecast = []
    for x, y in sequence:
        if kind == "copy":
            z = x
        elif kind == "xor":
            z = (x + y) % 2
        elif kind == "xnor":
            z = int(x == y)
        elif kind == "and":
            z = int(x == 1 and y == 1)
        elif kind == "or":
            z = int(x == 1 or y == 1)
        elif kind == "edge":
            z = int(y == 1 and prev_x != x)
        elif kind == "lag":
            z = int(x == 1 and prev_y == 1)
        else:
            raise ValueError("unsupported frozen candidate mechanism")
        forecast.append(z)
        prev_x, prev_y = x, y
    return tuple(forecast)


def execute_probe_sequence(intervene: Callable[[int, int], dict[str, object]],
                           ) -> tuple[int, ...]:
    seen = []
    for x, y in PRE_PROBES + POST_PROBES:
        current = intervene(x, y)
        output = current["observations"][-1]["output"]
        if type(output) is not int or output not in (0, 1):
            return ()
        seen.append(output)
    return tuple(seen)


def run_public_symbolic_revision(intervene: Callable[[int, int], dict[str, object]],
                                 hypothesize: Callable[[str], dict[str, object]],
                                 submit: Callable[[dict[str, object]], dict[str, object]],
                                 ) -> dict[str, object]:
    hypothesize("copy")  # Precommitted initial belief, before any observation.
    observed = execute_probe_sequence(intervene)
    if len(observed) != 10:
        return submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
    post = observed[3:]
    previous = PRE_PROBES[-1]
    consistent = [kind for kind in KNOWN_HYPOTHESES
                  if _post_prediction(kind, previous, POST_PROBES) == post]
    if len(consistent) != 1:
        return submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
    chosen = consistent[0]
    if chosen == "copy":
        return submit({"kind": "HYPOTHESIS", "kernel": chosen})
    hypothesize(chosen)
    return submit({"kind": "REVISION", "revised_kernel": chosen})


def run_frozen_pre_law(intervene: Callable[[int, int], dict[str, object]],
                       submit: Callable[[dict[str, object]], dict[str, object]],
                       ) -> dict[str, object]:
    execute_probe_sequence(intervene)
    return submit({"kind": "HYPOTHESIS", "kernel": "copy"})


def run_frozen_training_law(intervene: Callable[[int, int], dict[str, object]],
                            submit: Callable[[dict[str, object]], dict[str, object]],
                            ) -> dict[str, object]:
    execute_probe_sequence(intervene)
    return submit({"kind": "HYPOTHESIS", "kernel": "xor"})
