import unittest
from pathlib import Path

from scripts.minotaur_ci import missing_required, test_ids, unexpected_skips


class MembershipGateTests(unittest.TestCase):
    def test_missing_case_is_reported_even_when_other_cases_remain(self):
        self.assertEqual(missing_required(["module.Case.first", "module.Case.second"],
                                          ["module.Case.first", "module.Case.extra"]),
                         ["module.Case.second"])

    def test_empty_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            missing_required([], ["module.Case.first"])

    def test_nested_suite_ids_are_discovered(self):
        suite = unittest.TestSuite([unittest.TestSuite([self])])
        self.assertEqual(list(test_ids(suite)), [self.id()])

class MinotaurPlatformSkipTests(unittest.TestCase):
    def test_only_named_fifo_probe_is_exempted_on_windows(self):
        class Fake:
            def __init__(self, ident): self.ident=ident
            def id(self): return self.ident
        case = Fake("test_minotaur_public_checkpoint.PublicCheckpointTests."
                    "test_fifo_challenge_refuses_without_blocking_open")
        skipped = [(case,"named pipes unsupported")]
        self.assertEqual(unexpected_skips(skipped,platform="nt"),[])
        self.assertEqual(unexpected_skips(skipped,platform="posix"),skipped)
        wrong_reason=[(case,"artificial skip")]
        self.assertEqual(unexpected_skips(wrong_reason,platform="nt"),wrong_reason)
        other=Fake("test_minotaur_public_checkpoint.PublicCheckpointTests."
                   "test_checkpoint_and_public_cases_score_without_graduation")
        self.assertEqual(unexpected_skips([(other,"named pipes unsupported")],
                                          platform="nt"),
                         [(other,"named pipes unsupported")])


class NeuralWorkflowCoverageTests(unittest.TestCase):
    @staticmethod
    def _workflow(name):
        return (Path(__file__).resolve().parents[1] / ".github" / "workflows" / name).read_text()

    def test_pr_jobs_use_pinned_actions_and_read_only_token(self):
        for name in ("ci.yml", "minotaur-neural-cpu.yml"):
            workflow = self._workflow(name)
            with self.subTest(workflow=name):
                self.assertIn("\npermissions:\n  contents: read\n", workflow)
                self.assertIn("uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683", workflow)
                self.assertIn("uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", workflow)
                self.assertIn("persist-credentials: false", workflow)
                self.assertIn("Assert exact event head checkout", workflow)

    def test_neural_cpu_ci_selects_and_runs_kobayashi_source_contracts(self):
        workflow = self._workflow("minotaur-neural-cpu.yml")
        pull = workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  push:\n", 1)[0]
        push = workflow.split("  push:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0]
        for scope in (pull, push):
            with self.subTest(scope=scope[:20]):
                for pattern in ("labyrinth/kobayashi_*.py",
                                "labyrinth/local_model_kobayashi_runner_v1.py",
                                "labyrinth/local_model_world_session_v1.py",
                                "labyrinth/local_model_world_runner_v1.py",
                                "labyrinth/local_model_action_adapter_v1.py",
                                "labyrinth/living_world_v1.py",
                                "labyrinth/open_action_v1.py",
                                "tests/test_kobayashi_*.py",
                                "tests/test_local_model_kobayashi_runner_v1.py",
                                "tests/test_local_model_world_session_v1.py",
                                "tests/test_local_model_world_runner_v1.py",
                                "tests/test_local_model_action_adapter_v1.py",
                                "tests/test_living_world_v1.py"):
                    self.assertIn(pattern, scope)
        self.assertIn("Kobayashi source and local-model boundary tests", workflow)
        self.assertIn("tests.test_kobayashi_minotaur_v1", workflow)
        self.assertIn("tests.test_kobayashi_completion_gate_v1", workflow)
        self.assertIn("tests.test_local_model_world_session_v1", workflow)
        self.assertIn("tests.test_local_model_world_runner_v1", workflow)
        self.assertIn("tests.test_kobayashi_capability_guard_v1", workflow)
