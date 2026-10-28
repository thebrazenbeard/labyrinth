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

# Separate, versioned *training-only* structures.  The four blocks below
# deliberately increase instruction-chain length from one to four.  These are
# public structures, not a sealed evaluation key or a claim of generalization.
_GRADUATED_TRAIN_V2 = (
    ((("ADD", "A", "C"),), (("SUB", "A", "B"),), (("MUL", "A", "C"),)),
    (
        (("ADD", "A", "C"), ("MUL", "A", "C")),
        (("SUB", "B", "C"), ("ADD", "A", "B")),
        (("MUL", "B", "C"), ("SUB", "A", "B")),
    ),
    (
        (("ADD", "B", "C"), ("SUB", "A", "B"), ("MUL", "A", "C")),
        (("MUL", "A", "C"), ("ADD", "B", "A"), ("SUB", "A", "C")),
        (("ADD", "A", "B"), ("MUL", "B", "C"), ("SUB", "A", "B")),
    ),
    (
        (("MUL", "B", "C"), ("ADD", "A", "B"), ("SUB", "B", "C"), ("MUL", "A", "C")),
        (("ADD", "A", "C"), ("SUB", "B", "A"), ("MUL", "A", "B"), ("ADD", "A", "C")),
        (("SUB", "A", "C"), ("ADD", "B", "A"), ("MUL", "B", "C"), ("SUB", "A", "B")),
    ),
)


def generate_graduated_train_challenges(seed: int, count: int, split: str = "train") -> list[Challenge]:
    """Generate deterministic 1→4-step procedural *training* curriculum v2.

    The first half of a 128-case corpus contains only one- and two-step
    programs. A 64-step ordered trainer therefore never sees tiers three/four.
    The two textual preambles exercise rendering diversity, but neither is a
    representation-shift holdout. Never expose evaluation cases through this
    training generator.
    """
    if type(seed) is not int or type(count) is not int or not 1 <= count <= 256:
        raise ValueError("graduated generator requires integer seed and 1..256 count")
    if split != "train":
        raise ValueError("graduated v2 generator is train-only")
    challenges = []
    for index in range(count):
        tier = min(4, 1 + (index * 4) // count)
        digest = sha256(f"labyrinth-program-graduated-v2\x00{seed}\x00{index}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        program = rng.choice(_GRADUATED_TRAIN_V2[tier - 1])
        structure = ";".join(" ".join(instruction) for instruction in program)
        registers = {"A": rng.randrange(97), "B": rng.randrange(97)}
        initial_a, initial_b = registers["A"], registers["B"]
        lines = []
        for number, (operation, target, operand) in enumerate(program, 1):
            shown = str(rng.randint(2, 9)) if operand == "C" else operand
            value = int(shown) if operand == "C" else registers[operand]
            if operation == "ADD":
                registers[target] = (registers[target] + value) % 97
            elif operation == "SUB":
                registers[target] = (registers[target] - value) % 97
            else:
                registers[target] = (registers[target] * value) % 97
            lines.append(f"{number}. {operation} {target} {shown}")
        representation = "expanded" if index % 2 == 0 else "compact"
        if representation == "expanded":
            preamble = (
                "Execute each register instruction in order. ADD adds, SUB subtracts, "
                "and MUL multiplies the target by the stated literal or current "
                "source register. After each instruction, reduce the changed "
                "register modulo 97; leave the other unchanged."
            )
        else:
            preamble = (
                "Two-register machine over nonnegative remainders modulo 97. "
                "Read each line as OP TARGET OPERAND, evaluating the operand "
                "before changing TARGET. ADD means plus, SUB minus, MUL times."
            )
        prompt = (
            f"{preamble}\nInitial registers: A={initial_a}, B={initial_b}\n"
            + "\n".join(lines)
            + "\nWhat is the final value of A? Respond with exactly one line: ANSWER: <integer>."
        )
        challenges.append(Challenge(
            id=f"train-gv2-t{tier}-{index}-{digest.hex()[:16]}",
            kind="register_program", tier=tier, split="train", prompt=prompt,
            answer=str(registers["A"]), metadata=(
                ("generator", "labyrinth-program-graduated-v2"),
                ("profile", "train_only_graduated"),
                ("representation", representation),
                ("structure", structure),
                ("structure_fingerprint", sha256(structure.encode()).hexdigest()),
            ),
        ))
    return challenges


def generate_program_challenges(seed: int, count: int, split: str) -> list[Challenge]:
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

    profile = "train" if split == "train" else "structural_holdout"
    patterns = _TRAIN if profile == "train" else _HOLDOUT
    challenges = []
    for index in range(count):
        tag = f"labyrinth-program-v1\x00{seed}\x00{split}\x00{index}".encode()
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
