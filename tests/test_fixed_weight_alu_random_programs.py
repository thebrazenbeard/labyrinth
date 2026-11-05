"""Independent seeded oracle challenge for the frozen tiny neural ALU.

This file is deliberately outside the implementation directory. It does not
import the compiler's opcode_reference or ask it to evaluate expected results.
The deterministic corpus and SHA-256 provide a repeatable check against
test-selection bias. This is finite-ISA coverage, not general neural reasoning.

Run from repository root:
    python -m unittest tests.test_fixed_weight_alu_random_programs -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labyrinth import fixed_weight_alu as alu  # noqa: E402


SEED = 80451
N_PROGRAMS = 1000
EXPECTED_INSTRUCTIONS = 16453
EXPECTED_ORACLE_DIGEST = "101a22c775b420adc56fd08dd70461df7db97d69c4e7fc71c68ae45f5b3234ee"
EXPECTED_WEIGHT_SHA = "27dd18f48ccf0593d9688f4bf6f475cf49702f56c09316eea0bf354dba73f80c"


def independent_oracle(initial_register: int, operations: list[tuple[int, int]]) -> tuple[int, int]:
    """Compute supported ISA semantics with ordinary arithmetic; never inspect weights."""
    register = initial_register
    for opcode, argument in operations:
        if opcode == 0:  # NOP
            pass
        elif opcode == 1:  # ADD immediate modulo four
            register = (register + argument) & 3
        elif opcode == 2:  # XOR designated register bit
            register ^= 1 << argument
        else:
            raise AssertionError("unexpected generated opcode")
    return register & 1, register >> 1


class IndependentRandomProgramChallenge(unittest.TestCase):
    def test_thousand_postfreeze_programs_against_separate_oracle(self) -> None:
        # Construct ONE set of weights before creating any of these programs.
        weights = alu.compile_stepper()
        frozen_hash = weights.sha256()
        self.assertEqual(frozen_hash, EXPECTED_WEIGHT_SHA)

        randomizer = random.Random(SEED)
        digest = hashlib.sha256()
        instruction_count = 0

        # If the supervisor cheats by calling the compiler's reference
        # semantics rather than numeric weights, this entire trial fails.
        with patch.object(alu, "opcode_reference", side_effect=AssertionError("source oracle called at runtime")):
            for index in range(N_PROGRAMS):
                operations = [
                    (randomizer.randrange(3), randomizer.randrange(2))
                    for _ in range(randomizer.randrange(1, 33))
                ]
                words = tuple((opcode >> 1, opcode & 1, arg) for opcode, arg in operations)
                initial = randomizer.randrange(4)
                program = alu.EncodedProgram(
                    f"hostile-random-{index}", words, (initial & 1, initial >> 1)
                )

                expected = independent_oracle(initial, operations)
                actual = alu.HostSupervisor.start(weights, program).run()
                self.assertEqual(actual, expected, f"disagreement at generated program {index}")

                instruction_count += len(operations)
                digest.update(f"{index}|{len(operations)}|{initial}|{expected[0]},{expected[1]}\n".encode())

        self.assertEqual(instruction_count, EXPECTED_INSTRUCTIONS)
        self.assertEqual(digest.hexdigest(), EXPECTED_ORACLE_DIGEST)
        self.assertEqual(weights.sha256(), frozen_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
