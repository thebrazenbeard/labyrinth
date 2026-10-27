"""Reserved register-program holdouts with paired representation shifts.

These tasks deliberately do not extend the training generator. A separate
evaluator should hold their private answers and choose an unseen final seed.
Public source and a fixed seed alone do *not* constitute a blind benchmark.
"""

from __future__ import annotations

from hashlib import sha256
import random

from .maze import Challenge


# Neither training's 3-4-step nor development's 5-6-step structures occur here.
# A source revision adding these structures to training invalidates the holdout.
_RESERVED_STRUCTURES = (
    (("ADD", "B", "C"), ("SUB", "A", "B"), ("MUL", "B", "C"),
     ("ADD", "A", "B"), ("SUB", "B", "C"), ("MUL", "A", "C"),
     ("ADD", "A", "B")),
    (("MUL", "A", "C"), ("ADD", "B", "A"), ("SUB", "A", "C"),
     ("MUL", "B", "C"), ("SUB", "B", "A"), ("ADD", "A", "C"),
     ("MUL", "A", "B")),
    (("SUB", "B", "C"), ("MUL", "A", "B"), ("ADD", "B", "C"),
     ("SUB", "A", "B"), ("ADD", "A", "C"), ("MUL", "B", "C"),
     ("SUB", "B", "A"), ("ADD", "A", "B")),
    (("ADD", "A", "B"), ("MUL", "B", "C"), ("SUB", "A", "C"),
     ("ADD", "B", "A"), ("MUL", "A", "C"), ("SUB", "B", "C"),
     ("ADD", "B", "C"), ("SUB", "A", "B")),
)

_ALIASES = {"ADD": "plus", "SUB": "minus", "MUL": "times"}


def _execute(registers: dict[str, int], program: list[tuple[str, str, str]]) -> int:
    """Evaluator-side oracle; callers must pass only ``prompt`` to a model."""
    for operation, target, operand in program:
        value = registers[operand] if operand in registers else int(operand)
        if operation == "ADD":
            registers[target] = (registers[target] + value) % 97
        elif operation == "SUB":
            registers[target] = (registers[target] - value) % 97
        elif operation == "MUL":
            registers[target] = (registers[target] * value) % 97
        else:
            raise ValueError("unsupported operation")
    return registers["A"]


def generate_holdout_pairs(seed: int, count: int) -> list[tuple[Challenge, Challenge]]:
    """Return (canonical, representation-shift) pairs for final evaluation.

    Each pair has identical initial values and operations. The shifted variant
    changes serialization, operation vocabulary and initial-value ordering.
    The generator must never be used to create training or development labels.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("count must be between 1 and 1000")

    pairs: list[tuple[Challenge, Challenge]] = []
    for index in range(count):
        domain = f"labyrinth-reserved-v1\x00{seed}\x00{index}".encode("utf-8")
        digest = sha256(domain).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        structure = rng.choice(_RESERVED_STRUCTURES)
        initial_a, initial_b = rng.randrange(97), rng.randrange(97)
        program = [(op, target, str(rng.randint(2, 9)) if operand == "C" else operand)
                   for op, target, operand in structure]
        oracle_answer = str(_execute({"A": initial_a, "B": initial_b}, program))

        program_lines = "\n".join(
            f"{number}. {op} {target} {operand}"
            for number, (op, target, operand) in enumerate(program, 1)
        )
        canonical = (
            "Execute this register program in order. ADD, SUB and MUL change "
            "only their destination register by using its current value and "
            "the operand (a literal or a register's current value). "
            "Take the nonnegative remainder modulo 97 immediately after "
            "every update.\n"
            f"Initial registers: A={initial_a}, B={initial_b}\n"
            f"{program_lines}\n"
            "Give the final value in register A. Respond with exactly one line: "
            "ANSWER: <integer>."
        )
        shifted_lines = "\n".join(
            f"step {number:02d} | destination={target} | action={_ALIASES[op]} | "
            f"input={'register:' + operand if operand in ('A', 'B') else 'literal:' + operand}"
            for number, (op, target, operand) in enumerate(program, 1)
        )
        shifted = (
            "Run the following ordered table. Each row modifies exactly the "
            "destination register: plus adds, minus subtracts, times multiplies. "
            "Read register inputs at the moment the row is executed. "
            "Normalize the updated register to the range 0..96 modulo 97 "
            "before the next row.\n"
            f"Starting state (reordered): B={initial_b} | A={initial_a}\n"
            f"{shifted_lines}\n"
            "Report the ending content of A, exactly as ANSWER: <integer>."
        )

        structure_text = ";".join(" ".join(instruction) for instruction in structure)
        shared = (
            ("generator", "labyrinth-reserved-v1"),
            ("structure", structure_text),
            ("structure_fingerprint", sha256(structure_text.encode()).hexdigest()),
            ("pair_id", f"final-t4-{index}-{digest.hex()[:16]}"),
        )
        tasks = tuple(
            Challenge(
                id=f"final-t4-{index}-{digest.hex()[:16]}-{variant}",
                kind="register_program", tier=4, split="final", prompt=prompt,
                answer=oracle_answer, metadata=shared + (("representation", variant),),
            )
            for variant, prompt in (("canonical", canonical), ("shifted", shifted))
        )
        pairs.append((tasks[0], tasks[1]))
    return pairs
