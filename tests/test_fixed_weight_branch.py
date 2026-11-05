"""Branch pilot: exhaustive numerical checks and hostile custody controls."""

from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.fixed_weight_alu import compile_stepper, UnsupportedInstructionError  # noqa: E402
from labyrinth.fixed_weight_branch import (  # noqa: E402
    CANONICAL_SHA256, BranchProgram, BranchSupervisor, NeuralBranchCircuit,
    branch_reference, compile_branch, load_branch,
)


def independent_step(pc, state, word):
    """Independent test oracle, deliberately not reused by production runtime."""
    skip, op_hi, op_lo, operand = word
    register = state[0] + 2 * state[1]
    next_pc = (pc + (2 if skip and register == 0 else 1)) % 4
    opcode = 2 * op_hi + op_lo
    if opcode == 0:
        pass
    elif opcode == 1:
        register = (register + operand) % 4
    elif opcode == 2:
        register ^= 1 << operand
    else:
        raise ValueError("reserved opcode")
    return next_pc, (register & 1, register >> 1)


class FixedWeightBranchTests(unittest.TestCase):
    def setUp(self):
        self.branch = compile_branch()
        self.alu = compile_stepper()

    def test_exhaustive_branch_truth_table_and_canonical_digest(self):
        self.assertEqual(self.branch.sha256(), CANONICAL_SHA256)
        for bits in product((0, 1), repeat=5):
            with self.subTest(bits=bits):
                self.assertEqual(self.branch.forward(bits), branch_reference(bits))

    def test_host_reuses_frozen_weights_across_branching_programs(self):
        # External PC increments/conditional jumps depend on frozen numerical
        # weights and input register; host never decodes branch instruction.
        words = ((0, 0, 1, 1), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0))
        program = BranchProgram("conditional-ring", words)
        session = BranchSupervisor.start(self.alu, self.branch, program)
        self.assertEqual(session.step(), (1, (1, 0)))
        self.assertEqual(session.step(), (2, (1, 0)))  # nonzero: no skip
        self.assertEqual(session.step(), (3, (0, 0)))
        self.assertEqual(session.step(), (1, (0, 0)))  # zero: skip
        self.assertEqual(session.step(), (3, (0, 0)))
        self.assertEqual(self.branch.sha256(), CANONICAL_SHA256)

    def test_independent_postfreeze_random_programs_and_no_runtime_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            self.branch.save(path)
            sealed_bytes = path.read_bytes()
            loaded = load_branch(path)
            rng = random.Random(20261105)
            with (patch("labyrinth.fixed_weight_branch.branch_reference", side_effect=AssertionError("compiler called at runtime")),
                  patch("labyrinth.fixed_weight_alu.opcode_reference", side_effect=AssertionError("ALU oracle at runtime"))):
                for count in range(100):
                    # Programs are sampled only after compiling/saving/reloading.
                    words = tuple((rng.randrange(2), *rng.choice(((0, 0, 0), (0, 1, 1),
                                                                  (1, 0, 0), (1, 0, 1)))) for _ in range(4))
                    initial = (rng.randrange(2), rng.randrange(2))
                    program = BranchProgram(f"postfreeze-{count}", words, initial)
                    session = BranchSupervisor.start(self.alu, loaded, program)
                    pc, state = 0, initial
                    for _ in range(24):
                        pc, state = independent_step(pc, state, words[pc])
                        self.assertEqual(session.step(), (pc, state))
            self.assertEqual(path.read_bytes(), sealed_bytes)

    def test_numerical_weight_edit_changes_branch_without_host_change(self):
        # At branch=1, reg=0, PC=1 output is binary 11 (PC=3). Modify only
        # one second-layer coefficient for this exact one-hot activation.
        pattern = (1, 0, 0, 1, 0)
        position = list(product((0, 1), repeat=5)).index(pattern)
        rows = [list(row) for row in self.branch.w2]
        self.assertEqual(rows[0][position], 1)
        rows[0][position] = 0
        edited = NeuralBranchCircuit(self.branch.w1, self.branch.b1,
                                     tuple(tuple(row) for row in rows), self.branch.b2)
        self.assertEqual(self.branch.forward(pattern), (1, 1))
        self.assertEqual(edited.forward(pattern), (0, 1))
        program = BranchProgram("edit-control", ((0, 0, 0, 0), (1, 0, 0, 0),
                                                 (0, 0, 0, 0), (0, 0, 0, 0)))
        original = BranchSupervisor.start(self.alu, self.branch, program)
        mutant = BranchSupervisor.start(self.alu, edited, program)
        self.assertEqual(original.run_steps(2), (3, (0, 0)))
        self.assertEqual(mutant.run_steps(2), (2, (0, 0)))

    def test_sealed_reload_and_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.json"
            self.branch.save(path)
            self.assertEqual(load_branch(path).sha256(), CANONICAL_SHA256)
            with self.assertRaises(FileExistsError):
                self.branch.save(path)
            script = ("import json,sys; from pathlib import Path; "
                      "from labyrinth.fixed_weight_branch import load_branch; "
                      "w=load_branch(Path(sys.argv[1])); "
                      "print(json.dumps({'sha':w.sha256(),'out':w.forward((1,0,0,1,0))}))")
            child = subprocess.run((sys.executable, "-c", script, str(path)),
                                   cwd=str(Path(__file__).resolve().parents[1]),
                                   capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(child.stdout), {"sha": CANONICAL_SHA256, "out": [1, 1]})

    def test_self_consistent_tamper_and_wrong_pin_both_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.json"
            self.branch.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["w2"][0][0] ^= 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_branch(path)
            from labyrinth.fixed_weight_alu import _canonical
            from hashlib import sha256
            payload["sha256"] = sha256(_canonical({k: v for k, v in payload.items()
                                                   if k != "sha256"})).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_branch(path)
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_branch(path, expected_sha256="0" * 64)

    def test_type_magnitude_invalid_op_and_budget_guards(self):
        rows = [list(row) for row in self.branch.w1]
        rows[0][0] = 10**100
        with self.assertRaisesRegex(ValueError, "bounded"):
            NeuralBranchCircuit(tuple(tuple(row) for row in rows), self.branch.b1,
                                self.branch.w2, self.branch.b2)
        for malformed in ((False, 0, 0, 0, 0), (0, 2, 0, 0, 0), (1, 0, 0, 1)):
            with self.assertRaises(ValueError):
                self.branch.forward(malformed)
        with self.assertRaises(ValueError):
            BranchProgram("wrong-width", ((0, 0, 0, 0),))
        with self.assertRaises(ValueError):
            BranchProgram("invalid", ((False, 0, 0, 0),) * 4)
        program = BranchProgram("unsupported", ((0, 1, 1, 0),) * 4)
        supervisor = BranchSupervisor.start(self.alu, self.branch, program)
        with self.assertRaises(UnsupportedInstructionError):
            supervisor.step()
        self.assertEqual((supervisor.pc, supervisor.state), (0, (0, 0)))
        for count in (-1, True, 4097):
            with self.assertRaises(ValueError):
                supervisor.run_steps(count)


if __name__ == "__main__":
    unittest.main()
