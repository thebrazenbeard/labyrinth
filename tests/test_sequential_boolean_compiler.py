"""Proof obligations for algebraic, numerically chained neural layers."""

from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.sequential_boolean_compiler import (  # noqa: E402
    DenseLayer,
    NeuralProgram,
    compile_boolean,
    compose_sequential,
    load,
)
from labyrinth.sequential_boolean_specs import build, xor2, invert1  # noqa: E402


class SequentialNeuralCompilerTests(unittest.TestCase):
    def setUp(self):
        self.xor_net, self.not_net, self.xnor_net = build()

    def test_source_components_exhaustive(self):
        for bits in product((0, 1), repeat=2):
            self.assertEqual(self.xor_net.evaluate(bits), xor2(bits))
        for bits in product((0, 1), repeat=1):
            self.assertEqual(self.not_net.evaluate(bits), invert1(bits))

    def test_sequential_fusion_exhaustive_against_two_independent_networks(self):
        for bits in product((0, 1), repeat=2):
            separately = self.not_net.evaluate(self.xor_net.evaluate(bits))
            fused = self.xnor_net.evaluate(bits)
            self.assertEqual(fused, separately)
            self.assertEqual(fused, (int(bits[0] == bits[1]),))

    def test_actual_architecture_has_two_relu_hidden_layers(self):
        self.assertEqual([len(self.xnor_net.input_names)] +
                         [layer.output_width for layer in self.xnor_net.layers], [2, 4, 2, 1])
        self.assertEqual([layer.activation for layer in self.xnor_net.layers],
                         ["relu", "relu", "linear"])
        self.assertEqual(self.xnor_net.input_names, ("a", "b"))
        self.assertEqual(self.xnor_net.output_names, ("xnor_bit",))

    def test_weight_fusion_exact_matrix_algebra(self):
        left, right = self.xor_net.layers[-1], self.not_net.layers[0]
        folded = self.xnor_net.layers[1]
        for out in range(right.output_width):
            expected_row = tuple(sum(right.weights[out][k] * left.weights[k][col]
                                     for k in range(left.output_width))
                                 for col in range(left.input_width))
            expected_bias = sum(right.weights[out][k] * left.biases[k]
                                for k in range(left.output_width)) + right.biases[out]
            self.assertEqual(folded.weights[out], expected_row)
            self.assertEqual(folded.biases[out], expected_bias)

    def test_no_composite_truth_table_or_reference_callback(self):
        calls = {"xor": 0, "not": 0}

        def traced_xor(bits):
            calls["xor"] += 1
            return xor2(bits)

        def traced_not(bits):
            calls["not"] += 1
            return invert1(bits)

        xor = compile_boolean("xor", ("a", "b"), ("link",), traced_xor)
        invert = compile_boolean("not", ("link",), ("result",), traced_not)
        self.assertEqual(calls, {"xor": 4, "not": 2})
        fused = compose_sequential("fused", xor, invert)
        for bits in product((0, 1), repeat=2):
            self.assertEqual(fused.evaluate(bits), (int(bits[0] == bits[1]),))
        self.assertEqual(calls, {"xor": 4, "not": 2})

    def test_negative_control_corrupt_first_net_changes_composition(self):
        # Change XOR(00) from 0 to 1 in the first network's numeric readout.
        layer = self.xor_net.layers[-1]
        mutated = list(layer.weights[0])
        mutated[0] = 1 - mutated[0]
        corrupt_first = NeuralProgram("xor-mutant", self.xor_net.input_names,
                                      self.xor_net.output_names,
                                      self.xor_net.layers[:-1] +
                                      (DenseLayer((tuple(mutated),), layer.biases, "linear"),))
        fused_mutant = compose_sequential("mutant-a", corrupt_first, self.not_net)
        self.assertNotEqual(fused_mutant.evaluate((0, 0)), self.xnor_net.evaluate((0, 0)))

    def test_reject_out_of_contract_first_stage_before_neural_fusion(self):
        # The surrounding NOT gate could coincidentally map a bad 2-valued
        # intermediate result back into 0/1. The linker must catch the bad
        # first component before algebraic fusion masks that violation.
        layer = self.xor_net.layers[-1]
        mutated = list(layer.weights[0])
        mutated[0] = 2
        out_of_contract = NeuralProgram(
            "xor-bad-range", self.xor_net.input_names, self.xor_net.output_names,
            self.xor_net.layers[:-1] +
            (DenseLayer((tuple(mutated),), layer.biases, "linear"),),
        )
        with self.assertRaisesRegex(ArithmeticError, "Boolean"):
            out_of_contract.evaluate((0, 0))
        with self.assertRaisesRegex(ValueError, "binary output contract"):
            compose_sequential("must-reject", out_of_contract, self.not_net)

    def test_reject_out_of_contract_second_stage_before_neural_fusion(self):
        layer = self.not_net.layers[-1]
        mutant = list(layer.weights[0])
        mutant[0] = 2
        out_of_contract = NeuralProgram(
            "not-bad-range", self.not_net.input_names, self.not_net.output_names,
            self.not_net.layers[:-1] +
            (DenseLayer((tuple(mutant),), layer.biases, "linear"),),
        )
        with self.assertRaisesRegex(ValueError, "binary output contract"):
            compose_sequential("must-reject", self.xor_net, out_of_contract)

    def test_negative_control_corrupt_second_net_changes_composition(self):
        # Change NOT(0) from 1 to 0 by altering the second network's readout.
        layer = self.not_net.layers[-1]
        mutated = list(layer.weights[0])
        mutated[0] = 1 - mutated[0]
        corrupt_second = NeuralProgram("not-mutant", self.not_net.input_names,
                                       self.not_net.output_names,
                                       self.not_net.layers[:-1] +
                                       (DenseLayer((tuple(mutated),), layer.biases, "linear"),))
        fused_mutant = compose_sequential("mutant-b", self.xor_net, corrupt_second)
        differences = [bits for bits in product((0, 1), repeat=2)
                       if fused_mutant.evaluate(bits) != self.xnor_net.evaluate(bits)]
        self.assertGreater(len(differences), 0)

    def test_reject_sequential_abi_mismatch(self):
        bad = compile_boolean("bad", ("wrong-name",), ("result",), invert1)
        with self.assertRaisesRegex(ValueError, "ABI"):
            compose_sequential("invalid", self.xor_net, bad)

    def test_reject_nonbinary_inputs(self):
        for bad in ((0, 2), (1,), (1, 1, 0), (0.25, 1), (True, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.xnor_net.evaluate(bad)

    def test_reject_invalid_reference(self):
        with self.assertRaises(ValueError):
            compile_boolean("bad", ("x",), ("y",), lambda _: (99,))

    def test_reject_mutable_list_of_layers(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            NeuralProgram("mutable", self.xor_net.input_names,
                          self.xor_net.output_names, list(self.xor_net.layers))

    def test_bound_truth_table_growth(self):
        with self.assertRaises(ValueError):
            compile_boolean("wide", tuple(f"i{i}" for i in range(11)), ("result",),
                            lambda _: (0,))

    def test_deterministic_model_and_roundtrip(self):
        self.assertEqual(self.xnor_net.sha256(), build()[2].sha256())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xnor.json"
            self.xnor_net.save(path)
            restored = load(path)
            self.assertEqual(restored.sha256(), self.xnor_net.sha256())
            for bits in product((0, 1), repeat=2):
                self.assertEqual(restored.evaluate(bits), self.xnor_net.evaluate(bits))

    def test_fresh_process_load_with_no_reference_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xnor.json"
            self.xnor_net.save(path)
            code = ("import json,sys; from pathlib import Path; "
                    "from labyrinth.sequential_boolean_compiler import load; "
                    "a=load(Path(sys.argv[1])); "
                    "print(json.dumps({'sha':a.sha256(),'out':a.evaluate((0,0))}))")
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    cwd=str(Path(__file__).resolve().parents[1]),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout),
                             {"sha": self.xnor_net.sha256(), "out": [1]})

    def test_tamper_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xnor.json"
            self.xnor_net.save(path)
            payload = json.loads(path.read_text())
            payload["layers"][1]["weights"][0][0] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load(path)

    def test_rehashed_manifest_with_boolean_architecture_rejected(self):
        from labyrinth.sequential_boolean_compiler import _canonical  # validate schema beyond digest
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xnor.json"
            payload = self.xnor_net.manifest()
            payload["architecture"][0] = True  # bool equals int numerically, but is not int
            payload["sha256"] = sha256(_canonical(payload)).hexdigest()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "architecture"):
                load(path)


if __name__ == "__main__":
    unittest.main()
