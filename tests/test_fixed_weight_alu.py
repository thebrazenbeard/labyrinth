"""Exhaustive instructions/state tests and hostile failure controls."""

from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.fixed_weight_alu import (  # noqa: E402
    EncodedProgram, HostSupervisor, NeuralStepper,
    UnsupportedInstructionError, compile_stepper, load_stepper, opcode_reference,
)


class FixedWeightNeuralALUTests(unittest.TestCase):
    def setUp(self):
        self.weights = compile_stepper()

    def test_exhaustive_32_instruction_state_cases(self):
        for bits in product((0, 1), repeat=5):
            with self.subTest(bits=bits):
                self.assertEqual(self.weights.forward(bits), opcode_reference(bits))

    def test_real_weights_and_architecture(self):
        self.assertEqual(len(self.weights.w1), 32)
        self.assertEqual(len(self.weights.w1[0]), 5)
        self.assertEqual(len(self.weights.w2), 3)
        self.assertEqual(len(self.weights.w2[0]), 32)
        self.assertEqual(len(self.weights.b1), 32)
        self.assertEqual(len(self.weights.b2), 3)
        self.assertEqual(sum(map(len, self.weights.w1)) + len(self.weights.b1) +
                         sum(map(len, self.weights.w2)) + len(self.weights.b2), 291)

    def test_three_different_programs_use_same_frozen_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "fixed.json"
            self.weights.save(manifest)  # weight bytes are now frozen
            frozen_bytes = manifest.read_bytes()
            frozen_digest = self.weights.sha256()
            loaded = load_stepper(manifest)
            program1 = EncodedProgram("increment3", ((0, 1, 1),) * 3)
            program2 = EncodedProgram("xor-low-high-add", ((1, 0, 0), (1, 0, 1), (0, 1, 1)))
            # This third program is chosen ONLY AFTER the weights were saved
            # and reloaded. It has no influence on model creation.
            chosen_later = EncodedProgram("postfreeze", ((0, 1, 1), (1, 0, 1),
                                                         (0, 1, 1), (1, 0, 1)))
            self.assertEqual(HostSupervisor.start(loaded, program1).run(), (1, 1))
            self.assertEqual(HostSupervisor.start(loaded, program2).run(), (0, 0))
            self.assertEqual(HostSupervisor.start(loaded, chosen_later).run(), (0, 1))
            self.assertEqual(frozen_digest, loaded.sha256())
            self.assertEqual(frozen_bytes, manifest.read_bytes())

    def test_numeric_weight_mutation_changes_program_output(self):
        # Change exactly one compiled numeric coefficient selected for INC1
        # from input register 0. No host operation implementation changes.
        chosen_input = (0, 1, 1, 0, 0)
        position = list(product((0, 1), repeat=5)).index(chosen_input)
        rows = [list(row) for row in self.weights.w2]
        self.assertEqual(rows[0][position], 1)
        rows[0][position] = 0
        mutant = NeuralStepper(self.weights.w1, self.weights.b1,
                               tuple(tuple(row) for row in rows), self.weights.b2)
        program = EncodedProgram("inc1", ((0, 1, 1),))
        self.assertEqual(HostSupervisor.start(self.weights, program).run(), (1, 0))
        self.assertEqual(HostSupervisor.start(mutant, program).run(), (0, 0))
        self.assertNotEqual(self.weights.sha256(), mutant.sha256())

    def test_supervisor_does_not_call_compiler_opcode_reference(self):
        program = EncodedProgram("afterfreeze", ((0, 1, 1), (1, 0, 1)))
        with patch("labyrinth.fixed_weight_alu.opcode_reference", side_effect=AssertionError("host opcode interpretation")):
            self.assertEqual(HostSupervisor.start(self.weights, program).run(), (1, 1))

    def test_unsupported_opcode_from_neural_valid_bit(self):
        self.assertEqual(self.weights.forward((1, 1, 0, 0, 0)), (0, 0, 0))
        program = EncodedProgram("reserved-op", ((1, 1, 0),))
        session = HostSupervisor.start(self.weights, program)
        with self.assertRaises(UnsupportedInstructionError):
            session.step()
        self.assertEqual(session.pc, 0)
        self.assertEqual(session.state, (0, 0))

    def test_rejects_out_of_domain_binary_values(self):
        for invalid in ((0, 1, 2, 0, 0), (0, 1, 1, 0), (0, 1, 1, 0, 0, 1), (False, 1, 1, 0, 0)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.weights.forward(invalid)
        with self.assertRaises(ValueError):
            EncodedProgram("bad", ((2, 1, 0),))

    def test_guard_mutated_supervisor_pc_and_state(self):
        program = EncodedProgram("inc1", ((0, 1, 1),))
        session = HostSupervisor.start(self.weights, program)
        session.pc = -1
        with self.assertRaises(ValueError):
            session.step()
        with self.assertRaises(ValueError):
            session.run()
        session.pc = True
        with self.assertRaises(ValueError):
            session.step()
        session.pc = 2
        with self.assertRaises(ValueError):
            session.run()
        session.pc = 0
        session.state = (2, 0)
        with self.assertRaises(ValueError):
            session.step()

    def test_snapshot_custody_and_deterministic_recovery(self):
        program = EncodedProgram("count-and-toggle", ((0, 1, 1), (0, 1, 1), (1, 0, 1)))
        original = HostSupervisor.start(self.weights, program)
        original.step()
        original.step()
        snapshot = original.snapshot()
        self.assertEqual(snapshot["pc"], 2)
        self.assertEqual(snapshot["state"], [0, 1])
        restored = HostSupervisor.restore(self.weights, program, snapshot)
        self.assertEqual(restored.run(), original.run())
        self.assertEqual(restored.pc, 3)

    def test_snapshot_rejects_tampered_state_and_wrong_weights(self):
        program = EncodedProgram("inc2", ((0, 1, 1), (0, 1, 1)))
        session = HostSupervisor.start(self.weights, program)
        session.step()
        snap = session.snapshot()
        changed = dict(snap, state=[1, 1])
        with self.assertRaisesRegex(ValueError, "replayed"):
            HostSupervisor.restore(self.weights, program, changed)
        changed = dict(snap, weights_sha256="x" * 64)
        with self.assertRaisesRegex(ValueError, "provenance"):
            HostSupervisor.restore(self.weights, program, changed)
        changed = dict(snap, pc=-1)
        with self.assertRaisesRegex(ValueError, "PC"):
            HostSupervisor.restore(self.weights, program, changed)
        changed = dict(snap, isa="not-the-selected-isa")
        with self.assertRaisesRegex(ValueError, "schema/ISA"):
            HostSupervisor.restore(self.weights, program, changed)

    def test_serialized_weights_reload_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "weights.json"
            self.weights.save(manifest)
            code = ("import json,sys; from pathlib import Path; "
                    "from labyrinth.fixed_weight_alu import load_stepper,HostSupervisor,EncodedProgram; "
                    "w=load_stepper(Path(sys.argv[1])); "
                    "p=EncodedProgram('late-program',((1,0,0),(0,1,1))); "
                    "print(json.dumps({'sha':w.sha256(),'result':HostSupervisor.start(w,p).run()}))")
            run = subprocess.run([sys.executable, "-c", code, str(manifest)],
                                 cwd=str(Path(__file__).resolve().parents[1]),
                                 capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout),
                             {"sha": self.weights.sha256(), "result": [0, 1]})

    def test_weight_file_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            self.weights.save(path)
            data = json.loads(path.read_text())
            data["w2"][0][0] = 1 - data["w2"][0][0]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_stepper(path)


if __name__ == "__main__":
    unittest.main()
