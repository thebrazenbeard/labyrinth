"""Deterministic *public* paired counterfactuals for source-priority reasoning.

Three variants keep the same source values, so a fixed-number or fixed-row
shortcut cannot get all three right. Public templates are not a blind
Minotaur examination or deployment authority.
"""
from __future__ import annotations

import re
from .evaluation import score_response
from .maze import Challenge, _make_challenge, generate_maze


_PRIORITY = re.compile(
    r"priority is (cobalt|amber|slate) > (cobalt|amber|slate) > "
    r"(cobalt|amber|slate) \(highest first\)\."
)
_ENTRY = re.compile(r"source=(cobalt|amber|slate); measured_value=([0-9]{5})")


def generate_priority_triplets(
    seed: int, count: int, split: str = "priority_pairs"
) -> list[tuple[Challenge, Challenge, Challenge]]:
    """Generate (original, priority-swap, row-reorder) scoring triplets.

    Reject ambiguous generated cases with identical values for the two
    highest priority sources. Their swap must change the correct answer.
    """
    # Let the authoritative public generator validate all seed/count/split
    # constraints before creating any transformed challenge.
    generate_maze(seed=seed, tiers=(3,), count=count, split=split)
    result = []
    index = 0
    # Bounded by the generator's 1000-case maximum plus collision headroom.
    while len(result) < count:
        if index >= count + 100:
            raise ValueError("unable to generate diagnostic non-ambiguous pairs")
        original = _make_challenge(seed, split, 3, index)
        index += 1
        match = _PRIORITY.search(original.prompt)
        if match is None:
            raise ValueError("source-priority prompt grammar has changed")
        priority = match.groups()
        if len(set(priority)) != 3:
            raise ValueError("invalid priority permutation")
        values = dict(_ENTRY.findall(original.prompt))
        if set(values) != set(priority):
            raise ValueError("source-priority rows incomplete")
        if values[priority[0]] != original.answer:
            raise ValueError("source-priority answer disagrees with report")
        if values[priority[0]] == values[priority[1]]:
            continue
        new_order = (priority[1], priority[0], priority[2])
        replacement = "priority is " + " > ".join(new_order) + " (highest first)."
        swapped_text = (
            original.prompt[:match.start()] + replacement + original.prompt[match.end():]
        )
        rows = original.prompt.splitlines()
        entry_positions = [i for i, line in enumerate(rows) if line.startswith("- source=")]
        if len(entry_positions) != 3:
            raise ValueError("source-priority task has unexpected row count")
        source_rows = [rows[i] for i in entry_positions]
        for position, source_row in zip(entry_positions, reversed(source_rows)):
            rows[position] = source_row
        reordered_text = "\n".join(rows)
        if reordered_text == original.prompt:
            raise ValueError("source-row reorder failed to change prompt")
        swapped = Challenge(
            original.id + "-swap", original.kind, original.tier, original.split,
            swapped_text, values[new_order[0]],
            original.metadata + (("pair", "priority_swap"),),
        )
        reordered = Challenge(
            original.id + "-rows", original.kind, original.tier, original.split,
            reordered_text, original.answer,
            original.metadata + (("pair", "row_reorder"),),
        )
        result.append((original, swapped, reordered))
    return result


def score_priority_triplets(
    triplets: list[tuple[Challenge, Challenge, Challenge]],
    responses: dict[str, str],
) -> dict[str, int]:
    """Require every member of a semantic triplet to score correctly."""
    expected = [case for triplet in triplets for case in triplet]
    if not expected or len(set(case.id for case in expected)) != len(expected):
        raise ValueError("nonempty triplets with unique case identities required")
    if not isinstance(responses, dict) or set(responses) != {case.id for case in expected}:
        raise ValueError("exact response IDs required for paired evaluation")
    individual = [
        score_response(case, responses[case.id]).correct for case in expected
    ]
    triplets_all_correct = sum(
        all(individual[i : i + 3]) for i in range(0, len(individual), 3)
    )
    return {
        "triplets_all_correct": triplets_all_correct,
        "triplets_total": len(triplets),
        "individually_correct": sum(individual),
        "total_responses": len(expected),
    }
