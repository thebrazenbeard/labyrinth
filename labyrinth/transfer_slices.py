"""Development-only transfer slices for the finite tier-four register language.

Different numbers on familiar templates, different same-length compositions,
and longer compositions are distinct hypotheses. This code supplies a public
development set; it is not a blind or independent final evaluator.
"""

from __future__ import annotations

import hashlib
import json

from .maze import Challenge
from .program_maze import generate_program_challenges


SLICES = (
    ("fresh_numbers", "dev_near", "train"),
    ("new_composition", "dev_structure", "matched_length_ood"),
    ("longer_program", "dev_length", "structural_holdout"),
)


def generate_transfer_slices(seed: int, count: int = 8) -> dict[str, list[Challenge]]:
    """Generate disjoint deterministic development cases for three transfer axes.

    ``count`` is the number of numeric instances per slice, NOT the number of
    independent structures (at most four per slice). Do not train on these
    slices if they are to remain development evaluation cases.
    """
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be a nonnegative signed-31-bit integer")
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("count must be an integer from 1 through 128")

    slices = {
        name: generate_program_challenges(seed, count, split, profile=profile)
        for name, split, profile in SLICES
    }
    all_cases = [case for group in slices.values() for case in group]
    if len({case.id for case in all_cases}) != len(all_cases):
        raise RuntimeError("transfer slice challenge IDs overlap")
    if len({case.prompt for case in all_cases}) != len(all_cases):
        raise RuntimeError("transfer slice prompts overlap")

    fingerprints = {
        name: {dict(case.metadata)["structure_fingerprint"] for case in group}
        for name, group in slices.items()
    }
    if not fingerprints["fresh_numbers"].isdisjoint(fingerprints["new_composition"]):
        raise RuntimeError("matched-length composition holdout is not disjoint")
    if not fingerprints["fresh_numbers"].isdisjoint(fingerprints["longer_program"]):
        raise RuntimeError("long-program holdout is not disjoint")
    if not fingerprints["new_composition"].isdisjoint(fingerprints["longer_program"]):
        raise RuntimeError("transfer slices share structures")

    return slices


def transfer_commitment(slices: dict[str, list[Challenge]]) -> dict:
    """Return public question commitments, without answer keys or prompts.

    A digest does not authenticate a model run and is not a sealed-set boundary.
    """
    if set(slices) != {name for name, _, _ in SLICES}:
        raise ValueError("transfer slice names differ from the expected protocol")
    payload = {}
    for name, _, _ in SLICES:
        group = slices[name]
        public = [
            {"id": case.id, "prompt": case.prompt,
             "structure": dict(case.metadata)["structure_fingerprint"]}
            for case in group
        ]
        digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode("utf-8")).hexdigest()
        payload[name] = {
            "case_count": len(group),
            "distinct_structures": len({row["structure"] for row in public}),
            "question_structure_sha256": digest,
        }
    return {"schema": "labyrinth.transfer-development.v1", "status": "development_only", "slices": payload}
