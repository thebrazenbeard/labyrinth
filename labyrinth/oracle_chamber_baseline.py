"""Zero-update observation-only policy for Oracle Chamber.

It parses candidate-visible rules, never accesses hidden secret/lie slots,
and uses a fixed five-query error-correcting code. Perfect performance here
is evidence that this public toy challenge is solvable without model learning.
"""

from __future__ import annotations

import re
from typing import Callable


SCHEDULE = ("alpha", "alpha", "beta", "beta", "parity")


def _bits_for(secret: int) -> tuple[int, ...]:
    a, b = secret & 1, (secret >> 1) & 1
    return a, a, b, b, a ^ b


def decode_single_lie(observed: tuple[int, ...]) -> int | None:
    """Local independent nearest-codeword decoder; exact tie means abstain."""
    if len(observed) != 5 or any(type(bit) is not int or bit not in (0, 1)
                                     for bit in observed):
        return None
    scores = [sum(x != y for x, y in zip(_bits_for(secret), observed))
              for secret in range(4)]
    minimum = min(scores)
    candidates = [secret for secret, score in enumerate(scores) if score == minimum]
    return candidates[0] if len(candidates) == 1 else None


def run_public_parser(initial_view: dict[str, object],
                      ask: Callable[[str], dict[str, object]],
                      submit: Callable[[dict[str, object]], dict[str, object]],
                      *, instance_only: bool = False) -> dict[str, object]:
    prompt = str(initial_view["prompt"])
    match = re.search(r"at most (\d+) answers? may be inverted adversarially", prompt)
    if match is None:
        return submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
    lies_allowed = int(match.group(1))
    if lies_allowed > 1:
        # Conditional counting argument; never count a certificate as a win.
        return submit({"kind": "UNSAT_CERTIFICATE", "declared_rules": {
            "secrets": 4, "binary_answers": 2, "max_queries": 5,
            "lie_budget": lies_allowed}})
    observations: list[int] = []
    for question in SCHEDULE:
        outcome = ask(question)
        observed = outcome["answers"][-1]["answer"]
        if type(observed) is not int:
            return submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        observations.append(observed)
    if instance_only:
        answer = decode_single_lie(tuple(observations))
        if answer is None:
            return submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        return submit({"kind": "COMMIT", "secret": answer})
    return submit({"kind": "POLICY", "policy": {
        "questions": list(SCHEDULE), "decoder": "nearest_codeword"}})
