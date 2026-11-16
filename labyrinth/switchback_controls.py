"""Explicit zero-update and privileged symbolic controls, not neural runs."""

from __future__ import annotations

from functools import lru_cache

from .switchback_circuit import Case, make_case, public_view
from .switchback_oracle import adjudicate, search


# Public, fixed sequence known to solve only the seed-0 instance by design.
# No observation, update or hidden source access occurs in this control.
MEMORIZED_SEED0_PATH = ("0", "0", "WAIT", "0", "COMMIT")


def memorized_control(case: Case) -> dict[str, object]:
    return adjudicate(case, {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})


def privileged_symbolic_control(case: Case) -> dict[str, object]:
    """Upper bound with direct mechanism access; *not* candidate-public-only."""
    report = search(case)
    if report["verdict"] == "VERIFIED_WIN":
        return adjudicate(case, {"kind": "PLAN", "actions": report["witness"]})
    if report["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        return adjudicate(case, {"kind": "CLAIM_UNSAT"})
    return adjudicate(case, {"kind": "ABSTAIN"})


def _public_fingerprint(view: dict[str, object]) -> tuple[object, ...]:
    return (view["initial_register"], view["target_register"],
            view["switch_at"], view["maximum_pulses"])


@lru_cache(maxsize=1)
def _published_train_catalogue() -> dict[tuple[object, ...], tuple[dict[str, object], ...]]:
    """An explicit contamination baseline, NOT an authorized private oracle."""
    indexed: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for seed in range(100):
        case = make_case(seed)
        outcome = search(case)
        proposal = ({"kind": "PLAN", "actions": outcome["witness"]}
                    if outcome["verdict"] == "VERIFIED_WIN" else {"kind": "CLAIM_UNSAT"})
        indexed.setdefault(_public_fingerprint(public_view(case)), []).append(proposal)
    return {key: tuple(proposals) for key, proposals in indexed.items()}


def published_lookup_control(case: Case) -> dict[str, object]:
    """Zero-probe source-aware memorizer, abstains on ambiguous public keys."""
    matches = _published_train_catalogue().get(_public_fingerprint(public_view(case)), ())
    proposal = matches[0] if len(matches) == 1 else {"kind": "ABSTAIN"}
    return adjudicate(case, proposal)
