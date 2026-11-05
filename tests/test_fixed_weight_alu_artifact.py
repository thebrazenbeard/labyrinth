"""Check the committed fixed-weight ISA artifact against live source compilation.

The fixture is reviewable binary-free evidence of the emitted numerical weights;
a successful test does not establish external authorization or a deployed runtime.
"""
from itertools import product
from pathlib import Path
import subprocess
import sys
import unittest

from labyrinth.fixed_weight_alu import compile_stepper, load_stepper, opcode_reference


class CommittedFixedWeightArtifactTests(unittest.TestCase):
    def test_committed_fixture_exactly_matches_compilation(self) -> None:
        artifact = Path(__file__).resolve().parent / "fixtures" / "fixed_weight_isa.json"
        compiled = compile_stepper()
        loaded = load_stepper(artifact)
        self.assertEqual(loaded.sha256(), compiled.sha256())
        self.assertEqual(
            loaded.sha256(),
            "27dd18f48ccf0593d9688f4bf6f475cf49702f56c09316eea0bf354dba73f80c",
        )
        for bits in product((0, 1), repeat=5):
            with self.subTest(bits=bits):
                self.assertEqual(loaded.forward(bits), opcode_reference(bits))

    def test_optimized_mode_cli_verifier_executes_every_check(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-O", "-m", "scripts.verify_fixed_weight_alu"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        self.assertIn("exhaustive_domain=32/32", result.stdout)
        self.assertIn("parameter_count=291", result.stdout)
        self.assertIn("program=postfreeze-new-code", result.stdout)


if __name__ == "__main__":
    unittest.main()
