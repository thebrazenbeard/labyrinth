"""Release-gate regression: signed Minotaur auth must reach pinned neural CI.

The generic Python suite exercises these tests, but it cannot substitute for
the explicit, dependency-present Minotaur CPU gate. Every change confined to
the signed pair seam must independently trigger that gate on PR and main push.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github/workflows/minotaur-neural-cpu.yml"
_MODULE = "tests.test_signed_minotaur_pair_r25"
_SOURCE_PATH = "labyrinth/signed_minotaur_pair_r25.py"
_TEST_PATH = "tests/test_signed_minotaur_pair_r25.py"
_MIN_AUTH_CASES = 10


class SignedPairNeuralCIMembershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = _WORKFLOW.read_text(encoding="utf-8")

    def _event_path_lists(self) -> dict[str, str]:
        workflow = self.workflow
        # Workflow sections are the CI contract, not a Python-side simulation
        # of GitHub's path-filter matching. Both actual event lists must name
        # these exact source/test paths explicitly.
        self.assertEqual(workflow.count("  pull_request:\n    paths:\n"), 1)
        self.assertEqual(workflow.count("  push:\n    branches: [main]\n    paths:\n"), 1)
        self.assertEqual(workflow.count("  workflow_dispatch:\n"), 1)
        return {
            "pull_request": workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
                "  push:\n", 1)[0],
            "push/main": workflow.split("  push:\n    branches: [main]\n    paths:\n", 1)[1].split(
                "  workflow_dispatch:\n", 1)[0],
        }

    def test_signed_pair_source_and_tests_trigger_neural_cpu_for_pr_and_main_push(self):
        for event, manifest in self._event_path_lists().items():
            for path in (_SOURCE_PATH, _TEST_PATH):
                with self.subTest(event=event, changed_file=path):
                    self.assertTrue((_ROOT / path).is_file())
                    self.assertTrue(f'      - "{path}"' in manifest,
                                    f"{path} must trigger dependency-present CPU tests on {event}")

    def test_signed_pair_auth_cases_are_in_required_cpu_test_inventory(self):
        step = self.workflow.split(
            "      - name: Minotaur tests with neural dependencies present\n", 1)[1]
        # Inclusion in the optional/general Python discovery is insufficient;
        # the required CPU list has an explicit test-count and skip guard.
        required = step.split("          new_required = (\n", 1)[1].split(
            "          )\n          for module_name, min_count in new_required:\n", 1)[0]
        self.assertTrue(
            re.search(rf'\(\s*"{re.escape(_MODULE)}"\s*,\s*{_MIN_AUTH_CASES}\s*\)',
                      required) is not None,
            "signed Minotaur authentication regressions must be required CPU tests",
        )

    def test_required_auth_tests_cannot_be_qualified_as_skipped(self):
        workflow = self.workflow
        self.assertIn('"cryptography==50.0.2"', workflow)
        self.assertIn('cryptography.__version__ == "50.0.2"', workflow)
        self.assertTrue('"tests.test_signed_minotaur_pair_r25"' in workflow,
                        "the signer must be among the neural required CPU modules")
        step = workflow.split("          for module_name, min_count in new_required:\n", 1)[1]
        self.assertIn("result.testsRun < min_count", step)
        self.assertIn("not result.wasSuccessful()", step)
        self.assertIn("or result.skipped", step,
                      "required authentication checks must fail closed on any skip")
        self.assertIn("raise SystemExit", step)

    def test_signed_pair_cases_exist_and_do_not_silently_skip_locally(self):
        # An accidentally renamed or skipped test suite cannot satisfy an
        # explicit ten-case required declaration through metadata alone.
        suite = unittest.defaultTestLoader.loadTestsFromName(_MODULE)
        self.assertGreaterEqual(suite.countTestCases(), _MIN_AUTH_CASES)
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), output.getvalue())
        self.assertFalse(result.skipped, output.getvalue())
        self.assertGreaterEqual(result.testsRun, _MIN_AUTH_CASES)


if __name__ == "__main__":
    unittest.main()
