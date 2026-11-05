"""CI smoke for the explicitly local, non-general neural timing comparison."""

import unittest

from labyrinth.boolean_neural_specs import build
from scripts.benchmark_boolean_neural_compiler import benchmark


class BooleanNeuralBenchmarkSmoke(unittest.TestCase):
    def test_exhaustive_cpu_measurement_and_artifact_identity(self):
        report = benchmark(repeats=50, samples=3, warmup=2)
        self.assertEqual(report["evidence_class"], "LOCAL_PROCESS_BENCHMARK_ONLY")
        self.assertEqual(report["domain"]["exhaustive_inputs"], 8)
        self.assertTrue(report["domain"]["agreement"])
        self.assertEqual(report["artifact"]["scalar_parameters_total"], 98)
        self.assertEqual(report["artifact"]["manifest_semantic_sha256"], build()[2].sha256())
        for arm in ("direct_python", "compiled_relu_weights"):
            self.assertEqual(len(report["timing"][arm]["samples_ns"]), 3)
            self.assertGreater(report["timing"][arm]["median_ns"], 0)
        self.assertGreater(report["timing"]["median_compiled_over_direct_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
