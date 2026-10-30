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


class NeuralCPUWorkflowCoverageTests(unittest.TestCase):
    def test_l12_l14_neural_and_custody_boundaries_are_required_in_cpu_ci(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths_by_trigger = {
            "pull_request": workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
                "  push:\n", 1)[0],
            "push/main": workflow.split("  push:\n", 1)[1].split(
                "  workflow_dispatch:\n", 1)[0],
        }
        boundaries = (
            "labyrinth/monotonic_anchor_protocol.py",
            "labyrinth/retention_family_registry.py",
            "labyrinth/retention_entrypoint_guard.py",
            "labyrinth/measured_weakness_feedback.py",
            "labyrinth/atomic_artifact_publication.py",
            "labyrinth/preregistered_control_matrix.py",
            "labyrinth/arithmetic_semantic_case_key.py",
            "scripts/qualify_maze_conditioned_minilm_cpu.py",
            "scripts/qualify_neural_journey_e2e_cpu.py",
            "scripts/qualify_taskmaster_counterfactual_cpu.py",
            "tests/test_monotonic_anchor_protocol.py",
            "tests/test_retention_family_registry.py",
            "tests/test_retention_entrypoint_guard.py",
            "tests/test_measured_weakness_feedback.py",
            "tests/test_atomic_artifact_publication.py",
            "tests/test_preregistered_control_matrix.py",
            "tests/test_arithmetic_semantic_case_key.py",
            "tests/test_qualify_maze_conditioned_minilm_cpu.py",
            "tests/test_qualify_neural_journey_e2e_cpu.py",
            "tests/test_qualify_taskmaster_counterfactual_cpu.py",
        )
        for path in boundaries:
            with self.subTest(path=path, exists=True):
                self.assertTrue((root / path).is_file())
            for trigger, manifest in paths_by_trigger.items():
                with self.subTest(path=path, trigger=trigger):
                    self.assertIn(f'      - "{path}"', manifest)

        step = workflow.split(
            "      - name: Minotaur tests with neural dependencies present\n", 1)[1]
        # The listed source tests are cheap and deterministic. They do not
        # substitute for a physical pretrained training run or final exam.
        for module in (
            "tests.test_monotonic_anchor_protocol",
            "tests.test_retention_family_registry",
            "tests.test_retention_entrypoint_guard",
            "tests.test_measured_weakness_feedback",
            "tests.test_atomic_artifact_publication",
            "tests.test_preregistered_control_matrix",
            "tests.test_arithmetic_semantic_case_key",
            "tests.test_qualify_maze_conditioned_minilm_cpu",
            "tests.test_qualify_neural_journey_e2e_cpu",
            "tests.test_qualify_taskmaster_counterfactual_cpu",
        ):
            with self.subTest(module=module):
                self.assertIn(f'"{module}"', step)

    def test_neural_journey_transitive_source_changes_trigger_pinned_cpu_ci(self):
        # These modules are imported by the HF pilot, objective-conditioned
        # generator, report auditor or adaptive taskmaster. Editing only one
        # must not bypass the dependency-present CPU checks.
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths_by_trigger = {
            "pull_request": workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
                "  push:\n", 1)[0],
            "push/main": workflow.split("  push:\n", 1)[1].split(
                "  workflow_dispatch:\n", 1)[0],
        }
        for path in (
            "labyrinth/runner.py",
            "labyrinth/training_scope.py",
            "labyrinth/evaluation.py",
            "labyrinth/maze.py",
            "labyrinth/regenerative_maze.py",
            "labyrinth/journey.py",
            "labyrinth/program_maze.py",
            "labyrinth/taskmaster_routing.py",
            "labyrinth/minotaur_model.py",
            "labyrinth/minotaur_policy_learning.py",
        ):
            with self.subTest(path=path, exists=True):
                self.assertTrue((Path(__file__).resolve().parents[1] / path).is_file())
            for trigger, manifest in paths_by_trigger.items():
                with self.subTest(path=path, trigger=trigger):
                    self.assertIn(f'      - "{path}"', manifest)

    def test_adaptive_training_integrations_trigger_and_run_under_cpu_dependencies(self):
        # The generic Python workflow allows optional neural skips. This
        # separate dependency-present job must cover the TRAIN-only
        # Minotaur/Labyrinth integration when its source changes.
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        pull_request_paths = workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
            "  push:\n", 1)[0]
        main_push_paths = workflow.split("  push:\n", 1)[1].split(
            "  workflow_dispatch:\n", 1)[0]
        for path in (
            "labyrinth/adaptive_corridors.py",
            "labyrinth/journey_policy_runner.py",
            "labyrinth/journey_challenges.py",
            "labyrinth/semantic_chambers.py",
            "tests/test_adaptive_corridors.py",
            "tests/test_journey_policy_runner.py",
            "tests/test_journey_challenges.py",
            "tests/test_semantic_chambers.py",
        ):
            for trigger, manifest in (("pull_request", pull_request_paths),
                                      ("push/main", main_push_paths)):
                with self.subTest(path=path, trigger=trigger):
                    self.assertIn(f'      - "{path}"', manifest)
        test_step = workflow.split(
            "      - name: Minotaur tests with neural dependencies present\n", 1)[1]
        for module in (
            "tests.test_adaptive_corridors",
            "tests.test_journey_policy_runner",
            "tests.test_journey_challenges",
            "tests.test_semantic_chambers",
            "tests.test_minotaur_exam",
        ):
            with self.subTest(module=module):
                self.assertIn(module, test_step)

    def test_hf_neural_transition_changes_trigger_dependency_present_checks(self):
        # The signed development transition and opt-in two-cycle harness are
        # new neural boundaries. They must not rely solely on the generic
        # Python workflow, which permits optional neural dependencies to skip.
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        pull_request_paths = workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
            "  push:\n", 1)[0]
        main_push_paths = workflow.split("  push:\n", 1)[1].split(
            "  workflow_dispatch:\n", 1)[0]
        for path in (
            "labyrinth/hf_development_transition.py",
            "labyrinth/hf_resume_bridge.py",
            "scripts/qualify_hf_regenerative.py",
            "tests/test_hf_development_transition.py",
            "tests/test_hf_resume_bridge.py",
            "tests/test_hf_regenerative_harness.py",
        ):
            for trigger, manifest in (("pull_request", pull_request_paths),
                                      ("push/main", main_push_paths)):
                with self.subTest(path=path, trigger=trigger):
                    self.assertIn(f'      - "{path}"', manifest)
        test_step = workflow.split(
            "      - name: Minotaur tests with neural dependencies present\n", 1)[1]
        for module in (
            "tests.test_hf_development_transition",
            "tests.test_hf_resume_bridge",
            "tests.test_hf_regenerative_harness",
        ):
            with self.subTest(module=module):
                self.assertIn(module, test_step)

    def test_scoped_hf_and_objective_changes_trigger_dependency_present_checks(self):
        # Public diagnostic rotation, scoped report custody and objective
        # TRAIN materialization are also part of the HF journey boundary.
        # An edit confined to any of them must trigger the pinned CPU job.
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        pull_request_paths = workflow.split("  pull_request:\n    paths:\n", 1)[1].split(
            "  push:\n", 1)[0]
        main_push_paths = workflow.split("  push:\n", 1)[1].split(
            "  workflow_dispatch:\n", 1)[0]
        for path in (
            "labyrinth/hf_pilot.py",
            "labyrinth/hf_report_audit.py",
            "labyrinth/objective_contract.py",
            "tests/test_hf_pilot.py",
            "tests/test_hf_report_audit.py",
            "tests/test_hf_journey_resume_integration.py",
            "tests/test_hf_scoped_report_audit.py",
            "tests/test_objective_contract.py",
            "tests/test_reserved_structural_transfer.py",
        ):
            for trigger, manifest in (("pull_request", pull_request_paths),
                                      ("push/main", main_push_paths)):
                with self.subTest(path=path, trigger=trigger):
                    self.assertIn(f'      - "{path}"', manifest)
        test_step = workflow.split(
            "      - name: Minotaur tests with neural dependencies present\n", 1)[1]
        for module in (
            "tests.test_hf_pilot",
            "tests.test_hf_report_audit",
            "tests.test_hf_journey_resume_integration",
            "tests.test_hf_scoped_report_audit",
            "tests.test_objective_contract",
            "tests.test_reserved_structural_transfer",
        ):
            with self.subTest(module=module):
                self.assertIn(module, test_step)
