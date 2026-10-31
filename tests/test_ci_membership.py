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
    def test_train_completion_primitive_boundary_is_required_by_cpu_ci(self):
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths = (
            workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  push:\n", 1)[0],
            workflow.split("  push:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0],
        )
        for event_paths in paths:
            self.assertIn('"tests/test_r42_complete_maze_primitive_boundary.py"', event_paths)
        self.assertIn('("tests.test_r42_complete_maze_primitive_boundary", 5)', workflow)
        self.assertIn('or result.skipped', workflow)

    def test_legacy_minotaur_exam_boundary_is_a_required_cpu_suite(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths = (
            workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  push:\n", 1)[0],
            workflow.split("  push:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0],
        )
        for event_paths in paths:
            self.assertIn('"tests/test_r40_face_minotaur_primitive_boundary.py"',
                          event_paths)
        self.assertIn('("tests.test_r40_face_minotaur_primitive_boundary", 6)',
                      workflow)
        self.assertIn('or result.skipped', workflow)

    def test_r36_toy_evidence_is_required_and_archived_as_non_authorizing_artifact(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths = (
            workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  push:\n", 1)[0],
            workflow.split("  push:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0],
        )
        for event_paths in paths:
            self.assertIn('"scripts/qualify_toy_cpu_evidence_r36.py"', event_paths)
            self.assertIn('"tests/test_r36_toy_cpu_evidence_receipt.py"', event_paths)
        self.assertIn('("tests.test_r36_toy_cpu_evidence_receipt", 9)', workflow)
        self.assertIn('python -m scripts.qualify_toy_cpu_evidence_r36 --execute', workflow)
        self.assertIn('checked_head="$(git rev-parse HEAD)"', workflow)
        self.assertIn('checked_tree="$(git rev-parse \'HEAD^{tree}\')"', workflow)
        self.assertIn('uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02', workflow)
        self.assertIn('if-no-files-found: error', workflow)
        self.assertIn('retention-days: 14', workflow)

    def test_hardened_minotaur_exam_intake_is_required_without_optional_skips(self):
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/minotaur-neural-cpu.yml").read_text()
        self.assertIn('("tests.test_minotaur_exam_intake_guard_r36", 10)', workflow)
        self.assertIn('or result.skipped', workflow)

    def test_real_toy_physical_case_identities_cannot_be_diluted_by_counts(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        required = (
            "tests.test_fullstate_physical_gate_r8.PhysicalFullStateContinuityGate.test_interrupted_vs_uninterrupted_adamw_rng_and_exposure_cursor",
            "tests.test_optimizer_runner_v2_r19.OptimizerRunnerV2ToyAdamWTests.test_v2_journal_binds_cycle0_receipt_and_restores_continuous_adamw",
            "tests.test_optimizer_runner_v2_r19.OptimizerRunnerV2ToyAdamWTests.test_v2_foreign_fork_is_rejected_and_pending_is_not_replayed",
            "tests.test_checkpoint_manifest_v2_r19.V2PhysicalCheckpointTests.test_real_adamw_v2_save_and_restore_never_cross_load_as_v1",
            "tests.test_v2_checkpoint_cpu_smoke_r22.ToyV2CpuPhysicalTests.test_real_toy_torch_adamw_two_cycle_resume_and_fresh_reload",
            "tests.test_qualify_v2_weight_updates_cpu_r25.WeightUpdatePhysicalTests.test_real_torch_tensor_updates_frozen_base_and_optimizer_moments",
            "tests.test_qualify_v2_weight_updates_cpu_r25.WeightUpdatePhysicalTests.test_noop_training_cannot_claim_real_learning",
        )
        for exact_case in required:
            self.assertIn('"' + exact_case + '"', workflow)
        self.assertIn("missing_required(required_physical_ids, discovered_ids)", workflow)

    def test_five_arm_source_contract_changes_trigger_required_cpu_tests(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        paths = (
            workflow.split("  pull_request:\n    paths:\n", 1)[1].split("  push:\n", 1)[0],
            workflow.split("  push:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0],
        )
        for event_paths in paths:
            self.assertIn('"labyrinth/five_arm_experiment_contract.py"', event_paths)
            self.assertIn('"tests/test_five_arm_experiment_contract.py"', event_paths)
        self.assertIn('("tests.test_five_arm_experiment_contract", 39)', workflow)
        self.assertIn('or result.skipped', workflow)

    def test_graduation_auth_dependency_is_pinned_in_both_workflows(self):
        root = Path(__file__).resolve().parents[1]
        generic = (root / ".github/workflows/ci.yml").read_text()
        neural = (root / ".github/workflows/minotaur-neural-cpu.yml").read_text()
        self.assertIn("-e '.[auth]' \"cryptography==50.0.2\"", generic)
        self.assertIn('"cryptography==50.0.2"', neural)
        self.assertIn('cryptography.__version__ == "50.0.2"', neural)

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
            "labyrinth/measured_corridor_handoff.py",
            "labyrinth/measured_taskmaster_transition.py",
            "labyrinth/minotaur_development_feedback.py",
            "labyrinth/authority_publication_service.py",
            "labyrinth/arithmetic_retention_admission.py",
            "labyrinth/development_event_reservation_service.py",
            "labyrinth/measured_sentiment_corridors.py",
            "labyrinth/minilm_optimizer_state_resume.py",
            "labyrinth/sealed_semantic_exam_protocol.py",
            "labyrinth/external_monotonic_authority_contract.py",
            "labyrinth/measured_neural_batch_binding.py",
            "labyrinth/train_exposure_attestation.py",
            "labyrinth/maze_crash_resume_journal.py",
            "labyrinth/pinned_model_inference_receipt.py",
            "scripts/qualify_maze_conditioned_minilm_cpu.py",
            "scripts/qualify_maze_multiseed_protocol_cpu.py",
            "scripts/qualify_neural_journey_e2e_cpu.py",
            "scripts/qualify_taskmaster_counterfactual_cpu.py",
            "tests/test_monotonic_anchor_protocol.py",
            "tests/test_retention_family_registry.py",
            "tests/test_retention_entrypoint_guard.py",
            "tests/test_measured_weakness_feedback.py",
            "tests/test_atomic_artifact_publication.py",
            "tests/test_preregistered_control_matrix.py",
            "tests/test_arithmetic_semantic_case_key.py",
            "tests/test_measured_corridor_handoff.py",
            "tests/test_measured_taskmaster_transition.py",
            "tests/test_minotaur_development_feedback.py",
            "tests/test_authority_publication_service.py",
            "tests/test_arithmetic_retention_admission.py",
            "tests/test_development_event_reservation_service.py",
            "tests/test_measured_sentiment_corridors.py",
            "tests/test_minilm_optimizer_state_resume.py",
            "tests/test_sealed_semantic_exam_protocol.py",
            "tests/test_external_monotonic_authority_contract.py",
            "tests/test_measured_neural_batch_binding.py",
            "tests/test_minilm_runner_optimizer_optin.py",
            "tests/test_train_exposure_attestation.py",
            "tests/test_maze_crash_resume_journal.py",
            "tests/test_minilm_measured_batches_optin.py",
            "tests/test_signed_minotaur_tactic_neural_selection.py",
            "tests/test_pinned_model_inference_receipt.py",
            "tests/test_qualify_maze_measured_tactics_cpu.py",
            "tests/test_qualify_maze_conditioned_minilm_cpu.py",
            "tests/test_qualify_maze_prereg_controls_cpu.py",
            "tests/test_qualify_maze_multiseed_protocol_cpu.py",
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
            "tests.test_measured_corridor_handoff",
            "tests.test_measured_taskmaster_transition",
            "tests.test_minotaur_development_feedback",
            "tests.test_authority_publication_service",
            "tests.test_arithmetic_retention_admission",
            "tests.test_development_event_reservation_service",
            "tests.test_measured_sentiment_corridors",
            "tests.test_minilm_optimizer_state_resume",
            "tests.test_sealed_semantic_exam_protocol",
            "tests.test_external_monotonic_authority_contract",
            "tests.test_measured_neural_batch_binding",
            "tests.test_minilm_runner_optimizer_optin",
            "tests.test_train_exposure_attestation",
            "tests.test_maze_crash_resume_journal",
            "tests.test_minilm_measured_batches_optin",
            "tests.test_signed_minotaur_tactic_neural_selection",
            "tests.test_pinned_model_inference_receipt",
            "tests.test_qualify_maze_measured_tactics_cpu",
            "tests.test_qualify_maze_multiseed_protocol_cpu",
            "tests.test_qualify_maze_conditioned_minilm_cpu",
            "tests.test_qualify_maze_prereg_controls_cpu",
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
