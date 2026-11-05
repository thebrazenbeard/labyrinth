"""Standard-library executable proof obligations, not independent verification."""

import json
import subprocess
import sys
from itertools import product
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labyrinth.boolean_neural_compiler import NeuralArtifact, compile_boolean, link_parallel, load_manifest  # noqa: E402
from labyrinth.boolean_neural_specs import INPUT_ABI, build, parity3, majority3  # noqa: E402


class NeuralCompilationTests(unittest.TestCase):
    def setUp(self):
        self.parity, self.majority, self.linked = build()

    def test_exhaustive_parity_and_majority(self):
        for bits in product((0, 1), repeat=3):
            with self.subTest(bits=bits):
                self.assertEqual(self.parity.evaluate(bits), parity3(bits))
                self.assertEqual(self.majority.evaluate(bits), majority3(bits))

    def test_composed_full_adder_exhaustive(self):
        for bits in product((0, 1), repeat=3):
            with self.subTest(bits=bits):
                actual_sum, actual_carry = self.linked.evaluate(bits)
                self.assertEqual(actual_sum + 2 * actual_carry, sum(bits))
                self.assertEqual(self.linked.evaluate(bits),
                                 self.parity.evaluate(bits) + self.majority.evaluate(bits))

    def test_abi_is_explicit_and_stable(self):
        self.assertEqual(self.linked.input_names, INPUT_ABI)
        self.assertEqual(self.linked.output_names, ("sum_bit", "carry_out"))
        self.assertEqual((len(self.linked.w1), len(self.linked.w1[0])), (16, 3))
        self.assertEqual((len(self.linked.w2), len(self.linked.w2[0])), (2, 16))

    def test_deterministic_weight_generation(self):
        self.assertEqual(self.linked.sha256(), build()[2].sha256())

    def test_immutable_parameters(self):
        with self.assertRaises(TypeError):
            self.linked.w1[0][0] = 777

    def test_negative_control_corrupted_weights_fail(self):
        mutable_weights = [list(row) for row in self.linked.w2]
        mutable_weights[0][0] = 1 - mutable_weights[0][0]
        broken = NeuralArtifact("mutant", INPUT_ABI, self.linked.output_names,
                                self.linked.w1, self.linked.b1,
                                tuple(tuple(row) for row in mutable_weights), self.linked.b2)
        failures = 0
        for bits in product((0, 1), repeat=3):
            try:
                failures += int(broken.evaluate(bits) != self.linked.evaluate(bits))
            except ArithmeticError:
                failures += 1
        self.assertGreater(failures, 0)

    def test_rejects_nonbinary_inputs(self):
        for bad in ((1, 0, 2), (0.1, 0, 1), (0, 1), (0, 0, 1, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.linked.evaluate(bad)

    def test_rejects_incompatible_linked_components(self):
        incompatible = compile_boolean("a", ("x", "y", "z"), ("o",), parity3)
        with self.assertRaises(ValueError):
            link_parallel("bad", (self.parity, incompatible))
        with self.assertRaises(ValueError):
            link_parallel("bad", (self.parity, self.parity))

    def test_rejects_undefined_reference_values(self):
        with self.assertRaises(ValueError):
            compile_boolean("bad", ("x",), ("v",), lambda _: (2,))

    def test_guard_prevents_exponential_truth_table(self):
        with self.assertRaises(ValueError):
            compile_boolean("wide", tuple(f"x{i}" for i in range(11)), ("y",), lambda _: (0,))

    def test_reference_function_is_not_called_during_inference(self):
        calls = []

        def observed_reference(bits):
            calls.append(bits)
            return (sum(bits) % 2,)

        compiled = compile_boolean("observed", INPUT_ABI, ("parity",), observed_reference)
        self.assertEqual(len(calls), 8)  # compilation enumerates the full domain
        for bits in product((0, 1), repeat=3):
            self.assertEqual(compiled.evaluate(bits), parity3(bits))
        self.assertEqual(len(calls), 8)  # numerical weight execution only

    def test_fp64_forward_is_exact_for_this_small_circuit(self):
        for bits in product((0, 1), repeat=3):
            x = tuple(float(v) for v in bits)
            hidden = tuple(max(0.0, sum(float(w) * v for w, v in zip(row, x)) + float(b))
                           for row, b in zip(self.linked.w1, self.linked.b1))
            output = tuple(int(sum(float(w) * v for w, v in zip(row, hidden)) + float(b))
                           for row, b in zip(self.linked.w2, self.linked.b2))
            self.assertEqual(output, self.linked.evaluate(bits))

    def test_manifest_reproducibility(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "one.json", Path(root) / "two.json"
            self.linked.write_manifest(a)
            build()[2].write_manifest(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertIn(self.linked.sha256(), a.read_text())
            restored = load_manifest(a)
            self.assertEqual(restored.sha256(), self.linked.sha256())
            for bits in product((0, 1), repeat=3):
                self.assertEqual(restored.evaluate(bits), self.linked.evaluate(bits))

    def test_manifest_reloaded_in_fresh_python_process(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "compiled.json"
            self.linked.write_manifest(path)
            code = ("import json,sys; from pathlib import Path; "
                    "from labyrinth.boolean_neural_compiler import load_manifest; "
                    "x=load_manifest(Path(sys.argv[1])); "
                    "print(json.dumps({'sha':x.sha256(),'out':x.evaluate((1,1,1))}))")
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    cwd=str(Path(__file__).resolve().parents[1]),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), {"sha": self.linked.sha256(), "out": [1, 1]})

    def test_corrupted_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "compiled.json"
            self.linked.write_manifest(path)
            payload = json.loads(path.read_text())
            payload["w2"][0][0] = 1 - payload["w2"][0][0]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
