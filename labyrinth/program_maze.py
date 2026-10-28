"""Deterministic register-machine reasoning tasks with structural holdouts.

The public generator is not a secret test set. Reserve seeds and split names
before tuning, and treat scores as evidence about this small instruction set.
"""

from __future__ import annotations

from hashlib import sha256
import random
import re

from labyrinth.maze import Challenge


# A structure describes instruction order, destinations and operand types;
# C represents a literal, whose value is deliberately excluded. Evaluation
# structures have novel lengths, not merely unseen literal values.
_TRAIN = (
    (("ADD", "A", "C"), ("MUL", "B", "C"), ("ADD", "A", "B")),
    (("SUB", "B", "C"), ("ADD", "A", "B"), ("MUL", "A", "C")),
    (("ADD", "B", "A"), ("MUL", "B", "C"), ("SUB", "A", "B"), ("ADD", "A", "C")),
    (("MUL", "A", "C"), ("SUB", "B", "A"), ("ADD", "B", "C"), ("ADD", "A", "B")),
)
_HOLDOUT = (
    (("ADD", "A", "C"), ("SUB", "B", "A"), ("MUL", "B", "C"), ("ADD", "A", "B"), ("SUB", "A", "C")),
    (("MUL", "B", "C"), ("ADD", "A", "B"), ("SUB", "A", "C"), ("ADD", "B", "A"), ("MUL", "A", "C")),
    (("SUB", "A", "B"), ("ADD", "B", "C"), ("MUL", "A", "C"), ("ADD", "A", "B"), ("SUB", "B", "A"), ("ADD", "A", "B")),
    (("ADD", "B", "A"), ("SUB", "A", "C"), ("MUL", "B", "C"), ("ADD", "A", "B"), ("MUL", "A", "C"), ("SUB", "A", "B")),
)

# Different compositions at *the same* 3–4 instruction lengths as training.
# These distinguish composition transfer from the length shift in _HOLDOUT.
_MATCHED_LENGTH_OOD = (
    (("SUB", "A", "C"), ("MUL", "B", "A"), ("ADD", "A", "B")),
    (("MUL", "B", "A"), ("SUB", "A", "C"), ("ADD", "B", "C")),
    (("SUB", "A", "B"), ("MUL", "A", "C"), ("ADD", "B", "A"), ("SUB", "A", "C")),
    (("ADD", "A", "C"), ("MUL", "A", "B"), ("SUB", "B", "C"), ("ADD", "B", "A")),
)

_PROFILES = {
    "train": _TRAIN,
    "structural_holdout": _HOLDOUT,  # legacy: new composition AND greater length
    "matched_length_ood": _MATCHED_LENGTH_OOD,
}


def generate_program_challenges(
    seed: int, count: int, split: str, *, profile: str | None = None
) -> list[Challenge]:
    """Generate tier-four tasks; every non-train split uses held-out structures.

    Numeric literals and all register values are bounded; instructions use
    nonnegative modulo 97 after every assignment. Metadata fingerprints the
    ordered instruction structure, independently of the sampled numbers.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
        raise ValueError("count must be an integer between 1 and 1000")
    if not isinstance(split, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", split):
        raise ValueError("split must be a short lowercase identifier")

    # Preserve legacy IDs/prompts for all existing three-argument calls. An
    # explicit structure profile creates a separate, domain-separated sample
    # population, including for fresh-number development cases on train shapes.
    explicit_profile = profile is not None
    if profile is None:
        profile = "train" if split == "train" else "structural_holdout"
    if type(profile) is not str or profile not in _PROFILES or (split == "train" and profile != "train"):
        raise ValueError("invalid profile for the requested split")
    patterns = _PROFILES[profile]
    challenges = []
    for index in range(count):
        tag = (
            f"labyrinth-program-v2\x00{seed}\x00{split}\x00{profile}\x00{index}"
            if explicit_profile else f"labyrinth-program-v1\x00{seed}\x00{split}\x00{index}"
        ).encode()
        digest = sha256(tag).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        program = rng.choice(patterns)
        structure = ";".join(" ".join(instruction) for instruction in program)
        fingerprint = sha256(structure.encode()).hexdigest()
        registers = {"A": rng.randrange(97), "B": rng.randrange(97)}
        initial_a, initial_b = registers["A"], registers["B"]
        lines = []
        for number, (operation, target, operand) in enumerate(program, 1):
            printed_operand = str(rng.randint(2, 9)) if operand == "C" else operand
            value = int(printed_operand) if operand == "C" else registers[operand]
            if operation == "ADD":
                result = registers[target] + value
            elif operation == "SUB":
                result = registers[target] - value
            else:
                result = registers[target] * value
            registers[target] = result % 97
            lines.append(f"{number}. {operation} {target} {printed_operand}")
        prompt = (
            "Execute this register program in order. ADD, SUB, and MUL update "
            "the named target register using its current value and the operand. "
            "An operand is either a literal integer or the current value of the "
            "named register. After each instruction, replace the updated value "
            "with its nonnegative remainder modulo 97. Other registers do not change.\n"
            f"Initial registers: A={initial_a}, B={initial_b}\n"
            + "\n".join(lines)
            + "\nWhat is the final value of A? Respond with exactly one line: ANSWER: <integer>."
        )
        challenges.append(Challenge(
            id=f"{split}-t4-{index}-{digest.hex()[:16]}", kind="register_program",
            tier=4, split=split, prompt=prompt, answer=str(registers["A"]),
            metadata=(("generator", "labyrinth-program-v1"), ("profile", profile),
                      ("structure", structure), ("structure_fingerprint", fingerprint)),
        ))
    return challenges
