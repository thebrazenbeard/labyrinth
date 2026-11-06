"""Source-only preflight for opt-in real MiniLM optimizer continuation.

The actual pretrained runner is not executed here: its model cache and host
lease are outside CI. These tests do not claim interrupted-run equivalence.
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.qualify_maze_conditioned_minilm_cpu import (
    OptimizerContinuityRequest, build_training_cycles, default_scope,
    execute, main,
)


class MiniLMRunnerOptimizerOptinTests(unittest.TestCase):
    @staticmethod
    def request() -> OptimizerContinuityRequest:
        return OptimizerContinuityRequest(
            logical_run_id="labyrinth-five-control",
            prior_checkpoint_sha256="a" * 64,
            fence=7,
        )

    def test_bound_request_has_strict_identity_without_torch(self):
        self.request().validate()
        invalid = (
            replace(self.request(), logical_run_id="../escape"),
            replace(self.request(), logical_run_id=""),
            replace(self.request(), prior_checkpoint_sha256="A" * 64),
            replace(self.request(), prior_checkpoint_sha256="short"),
            replace(self.request(), fence=-1),
            replace(self.request(), fence=True),
            replace(self.request(), fence=2**63),
        )
        for item in invalid:
            with self.subTest(item=item), self.assertRaisesRegex(ValueError, "operator pins"):
                item.validate()

    def test_execute_rejects_invalid_optin_before_any_neural_import(self):
        scope = default_scope()
        cycles = build_training_cycles(scope)
        with self.assertRaisesRegex(ValueError, "invalid optimizer continuity request"):
            execute(Path("/unused/source-only-path"), scope=scope,
                    cycles=cycles, steps_per_cycle=2, learning_rate=.002,
                    optimizer_continuity=object())
        with self.assertRaisesRegex(ValueError, "operator pins"):
            execute(Path("/unused/source-only-path"), scope=scope,
                    cycles=cycles, steps_per_cycle=2, learning_rate=.002,
                    optimizer_continuity=replace(self.request(), fence=True))

    def test_cli_requires_all_optin_pins_and_rejects_stray_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ["--output-parent", directory, "--run-name", "candidate"]
            for extra in (
                ["--full-optimizer-cycle-continuity"],
                ["--optimizer-run-id", "unbound"],
                ["--full-optimizer-cycle-continuity", "--optimizer-run-id", "test"],
                ["--full-optimizer-cycle-continuity", "--optimizer-run-id", "test",
                 "--optimizer-prior-sha256", "b" * 64],
            ):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main(common + extra)
                    self.assertEqual(caught.exception.code, 2)
            self.assertFalse((Path(directory) / "candidate").exists())

    def test_optin_dryrun_is_only_a_source_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            args = [
                "--output-parent", directory, "--run-name", "candidate",
                "--preregistered-controls", "--full-optimizer-cycle-continuity",
                "--optimizer-run-id", "labyrinth-five-control",
                "--optimizer-prior-sha256", "a" * 64,
                "--optimizer-fence", "7",
            ]
            with redirect_stdout(output):
                self.assertEqual(main(args), 0)
            report = json.loads(output.getvalue())
            self.assertFalse(report["execute"])
            self.assertFalse(report["graduation_authorized"])
            self.assertEqual(report["full_optimizer_continuity"],
                             "LOCAL_RECEIPT_NO_HOST_FENCE")
            self.assertEqual(len(report["arms"]), 5)
            self.assertFalse((Path(directory) / "candidate").exists())

    def test_default_dryrun_does_not_enable_full_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--output-parent", directory,
                                       "--run-name", "default"]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["full_optimizer_continuity"], "DISABLED")
            self.assertFalse(report["graduation_authorized"])


if __name__ == "__main__":
    unittest.main()
