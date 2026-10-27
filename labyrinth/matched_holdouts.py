"""Evaluator-only matched-length structural and representation holdouts.

This deliberately has no training-batch API or import path from the train CLI.
An evaluator must choose and seal an unused seed, keep gold private, and pass
*only* ``Challenge.prompt`` to an inference adapter. Public generator code is
not a secret benchmark; a source-bound manifest is a commitment, not secrecy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random
import re

from .maze import Challenge


GENERATOR_VERSION = "labyrinth-matched-final-v1"

# A matched structural-family contrast to program_maze._TRAIN: two 3-step
# patterns (ADD,ADD,MUL and ADD,SUB,MUL), and two 4-step patterns (2 ADD,
# 1 SUB, 1 MUL). None are exact train sequences. Equal length + multiset
# doesn't make the task equally difficult; report per-family results.
_MATCHED_STRUCTURES = (
    (("MUL", "B", "C"), ("ADD", "A", "B"), ("ADD", "A", "C")),
    (("ADD", "B", "A"), ("MUL", "A", "C"), ("SUB", "A", "B")),
    (("MUL", "B", "C"), ("ADD", "A", "B"), ("SUB", "B", "A"),
     ("ADD", "A", "B")),
    (("SUB", "A", "B"), ("ADD", "B", "C"), ("MUL", "A", "B"),
     ("ADD", "A", "C")),
)
_ALIAS = {"ADD": "plus", "SUB": "minus", "MUL": "times"}


def _execute(program: list[tuple[str, str, str]], a: int, b: int) -> int:
    state = {"A": a, "B": b}
    for op, dst, rhs in program:
        value = state[rhs] if rhs in state else int(rhs)
        if op == "ADD":
            state[dst] = (state[dst] + value) % 97
        elif op == "SUB":
            state[dst] = (state[dst] - value) % 97
        elif op == "MUL":
            state[dst] = (state[dst] * value) % 97
        else:
            raise ValueError("invalid operation")
    return state["A"]


def generate_matched_length_pairs(seed: int, count: int) -> list[tuple[Challenge, Challenge]]:
    """Generate canonical/paraphrased pairs on four disjoint train-length patterns.

    Results are evaluator-private objects containing answer keys. Never use
    the generator for train or adaptive development. The frozen seed must be
    excluded from the training process until final model selection is over.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("count must be between 1 and 1000")

    pairs: list[tuple[Challenge, Challenge]] = []
    for index in range(count):
        digest = sha256(f"{GENERATOR_VERSION}\x00{seed}\x00{index}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        structure = _MATCHED_STRUCTURES[index % len(_MATCHED_STRUCTURES)]
        a, b = rng.randrange(97), rng.randrange(97)
        program = [(op, dst, str(rng.randint(2, 9)) if rhs == "C" else rhs)
                   for op, dst, rhs in structure]
        answer = str(_execute(program, a, b))
        signature = ";".join(" ".join(step) for step in structure)
        pair_id = f"matched-t4-{index}-{digest.hex()[:16]}"

        canonical_lines = "\n".join(
            f"{step}. {op} {dst} {rhs}"
            for step, (op, dst, rhs) in enumerate(program, 1)
        )
        canonical = (
            "Execute this register program in order. ADD adds, SUB subtracts, "
            "and MUL multiplies the target's current value by/with its operand. "
            "Operands are literals or current register values. Reduce the target "
            "to its nonnegative remainder modulo 97 after every update.\n"
            f"Initial registers: A={a}, B={b}\n"
            f"{canonical_lines}\n"
            "Give the final value of A as exactly one line: ANSWER: <integer>."
        )
        shifted_lines = "\n".join(
            f"row-{step:02d} :: destination {dst} :: {_ALIAS[op]} :: "
            f"{'register' if rhs in ('A', 'B') else 'literal'} {rhs}"
            for step, (op, dst, rhs) in enumerate(program, 1)
        )
        shifted = (
            "Process the ordered rows as state updates. Each row modifies only "
            "its destination: plus means add, minus means subtract, times means "
            "multiply. Use the current value for each register operand. "
            "After every row apply modulo 97, keeping the result in 0..96.\n"
            f"Initial state (B first): B={b}, A={a}\n"
            f"{shifted_lines}\n"
            "Output the ending A value as exactly ANSWER: <integer>."
        )

        shared = (
            ("generator", GENERATOR_VERSION),
            ("structure", signature),
            ("structure_fingerprint", sha256(signature.encode()).hexdigest()),
            ("pair_id", pair_id),
        )
        pair = tuple(
            Challenge(
                id=f"{pair_id}-{repr_name}", kind="register_program", tier=4,
                split="final_matched", prompt=prompt, answer=answer,
                metadata=shared + (("representation", repr_name),),
            )
            for repr_name, prompt in (("canonical", canonical), ("paraphrased", shifted))
        )
        pairs.append((pair[0], pair[1]))
    return pairs


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    case_id: str
    prompt_sha256: str
    structure_fingerprint: str
    pair_id: str
    representation: str


@dataclass(frozen=True, slots=True)
class FrozenManifest:
    """An immutable evaluator-side case commitment without gold answers."""

    schema_version: str
    generator_version: str
    source_commit: str
    seed: int
    entries: tuple[ManifestEntry, ...]
    commitment: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def freeze_matched_manifest(
    seed: int, pairs: list[tuple[Challenge, Challenge]], *, source_commit: str
) -> FrozenManifest:
    """Bind exact cases to a caller-attested full source commit and seed.

    The source hash is supplied by the caller and must be independently
    verified against its executable checkout; this function can't attest Git.
    Store output in evaluator custody before training/config selection.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid seed")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 1000:
        raise ValueError("invalid pairs")
    if pairs != generate_matched_length_pairs(seed, len(pairs)):
        raise ValueError("pairs do not match declared seed and generator")

    entries = tuple(
        ManifestEntry(
            case_id=challenge.id,
            prompt_sha256=sha256(challenge.prompt.encode()).hexdigest(),
            structure_fingerprint=dict(challenge.metadata)["structure_fingerprint"],
            pair_id=dict(challenge.metadata)["pair_id"],
            representation=dict(challenge.metadata)["representation"],
        )
        for pair in pairs for challenge in pair
    )
    payload = {
        "schema_version": "labyrinth-eval-manifest-v1",
        "generator_version": GENERATOR_VERSION,
        "source_commit": source_commit,
        "seed": seed,
        "entries": [asdict(entry) for entry in entries],
    }
    commitment = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return FrozenManifest(
        schema_version=payload["schema_version"],
        generator_version=GENERATOR_VERSION,
        source_commit=source_commit,
        seed=seed,
        entries=entries,
        commitment=commitment,
    )
