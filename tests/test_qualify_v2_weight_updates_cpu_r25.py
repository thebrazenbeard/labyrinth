"""Non-vacuous physical-learning evidence gate for the opt-in toy R25 probe."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from unittest import TestCase, mock, skipUnless

from scripts import qualify_v2_weight_updates_cpu_r25 as probe


class TensorDouble:
    """Tensor-like source fixture with independently specified finiteness."""

    dtype = "torch.float32"
    shape = (1,)

    def __init__(self, finite: bool, payload: bytes = b"actual-tensor-bytes"):
        self.finite = finite
        self.payload = payload

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def isfinite(self):
        finite = self.finite

        class FiniteMask:
            def all(self):
                class Scalar:
                    def item(self):
                        return finite
                return Scalar()

        return FiniteMask()

    def numpy(self):
        payload = self.payload

        class Array:
            def tobytes(self):
                return payload

        return Array()


class WeightUpdateEvidenceTests(TestCase):
    def setUp(self):
        self.valid = {
            "frozen": ("1" * 64, "1" * 64, "1" * 64),
            "trainable": ("a" * 64, "b" * 64, "c" * 64),
            "moments": ("d" * 64, "e" * 64),
            "steps": (4, 8),
            "steps_per_cycle": 4,
        }

    def test_fresh_two_cycle_evidence_admitted(self):
        probe.validate_observations(**self.valid)

    def test_noop_both_trajectories_cannot_pass(self):
        evidence = {**self.valid, "trainable": ("a" * 64,) * 3}
        with self.assertRaisesRegex(ValueError, "cycle zero made no"):
            probe.validate_observations(**evidence)

    def test_regeneration_noop_cannot_pass(self):
        evidence = {**self.valid,
                    "trainable": ("a" * 64, "b" * 64, "b" * 64)}
        with self.assertRaisesRegex(ValueError, "regenerated cycle made no"):
            probe.validate_observations(**evidence)

    def test_frozen_base_parameter_update_rejected(self):
        evidence = {**self.valid, "frozen": ("1" * 64, "2" * 64, "2" * 64)}
        with self.assertRaisesRegex(ValueError, "frozen base weights changed"):
            probe.validate_observations(**evidence)

    def test_optimizer_moments_must_evolve(self):
        evidence = {**self.valid, "moments": ("d" * 64, "d" * 64)}
        with self.assertRaisesRegex(ValueError, "optimizer moments did not advance"):
            probe.validate_observations(**evidence)

    def test_advanced_weights_but_missing_optimizer_step_rejected(self):
        evidence = {**self.valid, "steps": (0, 0)}
        with self.assertRaisesRegex(ValueError, "step cursor"):
            probe.validate_observations(**evidence)

    def test_malformed_receipt_digest_rejected(self):
        evidence = {**self.valid, "trainable": ("a" * 64, "X" * 64, "c" * 64)}
        with self.assertRaisesRegex(ValueError, "invalid tensor evidence"):
            probe.validate_observations(**evidence)

    def test_fractional_optimizer_cursor_cannot_round_into_valid_step(self):
        parameter = object()

        class OptimizerDouble:
            param_groups = ({"params": (parameter,)},)
            state = {parameter: {"step": 4.5, "exp_avg": object(),
                                 "exp_avg_sq": object()}}

        with self.assertRaisesRegex(ValueError, "not an exact integer"):
            probe._optimizer_moment_digest(OptimizerDouble(), 4)

    def test_optimizer_state_without_moments_fails_before_hashing(self):
        parameter = object()

        class OptimizerDouble:
            param_groups = ({"params": (parameter,)},)
            state = {parameter: {"step": 4}}

        with self.assertRaisesRegex(ValueError, "moments were not initialized"):
            probe._optimizer_moment_digest(OptimizerDouble(), 4)

    def test_nonfinite_weight_bytes_cannot_form_valid_update_digest(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe._digest_tensors([("trainable", TensorDouble(False, b"nan"))])

    def test_nonfinite_adamw_moment_rejected(self):
        parameter = object()

        class OptimizerDouble:
            param_groups = ({"params": (parameter,)},)
            state = {parameter: {"step": 4,
                                 "exp_avg": TensorDouble(False, b"inf"),
                                 "exp_avg_sq": TensorDouble(True)}}

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe._optimizer_moment_digest(OptimizerDouble(), 4)

    def test_nonfinite_gradient_rejected_before_cycle_receipt(self):
        class Parameter:
            requires_grad = True
            grad = TensorDouble(False, b"nonfinite-gradient")

        class Module:
            def named_parameters(self):
                return (("p", Parameter()),)

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe._require_finite_gradients(Module(), Module())

    def test_valid_gradients_and_finite_tensors_remain_allowed(self):
        class Parameter:
            requires_grad = True
            grad = TensorDouble(True)

        class Module:
            def named_parameters(self):
                return (("p", Parameter()),)

        self.assertEqual(len(probe._digest_tensors([("weight", TensorDouble(True))])), 64)
        probe._require_finite_gradients(Module(), Module())

    def test_missing_trainable_gradient_fails_closed(self):
        class Parameter:
            requires_grad = True
            grad = None

        class Module:
            def named_parameters(self):
                return (("p", Parameter()),)

        with self.assertRaisesRegex(ValueError, "missing trainable gradient"):
            probe._require_finite_gradients(Module())

    def test_default_preflight_never_executes_torch(self):
        output = io.StringIO()
        with mock.patch.object(probe, "execute_weight_update_probe",
                               side_effect=AssertionError("unexpected execution")):
            with redirect_stdout(output):
                code = probe.main([])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "PREFLIGHT_ONLY_NO_EFFECT")
        self.assertFalse(report["physical_cpu_toy_executed"])
        self.assertFalse(report["graduation_authorized"])

    def test_execute_without_torch_returns_skip_not_pass(self):
        output = io.StringIO()
        with mock.patch.object(probe, "missing_dependencies",
                               return_value=("torch",)):
            with mock.patch.object(probe, "execute_weight_update_probe",
                                   side_effect=AssertionError("unexpected execution")):
                with redirect_stdout(output):
                    code = probe.main(["--execute"])
        self.assertEqual(code, 3)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "SKIPPED_NO_PHYSICAL_EXECUTION")
        self.assertFalse(report["physical_cpu_toy_executed"])
        self.assertFalse(report["graduation_authorized"])


@skipUnless(importlib.util.find_spec("torch"),
            "physical toy CPU PyTorch runtime unavailable")
class WeightUpdatePhysicalTests(TestCase):
    def test_real_torch_tensor_updates_frozen_base_and_optimizer_moments(self):
        report = probe.execute_weight_update_probe()
        self.assertEqual(report["status"], probe.STATUS)
        self.assertTrue(report["actual_trainable_tensor_updates_both_cycles"])
        self.assertTrue(report["frozen_base_weights_unchanged"])
        self.assertTrue(report["adamw_moments_advanced_both_cycles"])
        self.assertEqual(report["optimizer_steps"], [4, 8])
        self.assertFalse(report["pretrained_model_loaded"])
        self.assertFalse(report["host_custody_verified"])
        self.assertFalse(report["neural_generalization_verified"])
        self.assertFalse(report["graduation_authorized"])

    def test_noop_training_cannot_claim_real_learning(self):
        with mock.patch("scripts.qualify_v2_checkpoint_cpu_r22._train_step",
                        return_value=None):
            with self.assertRaisesRegex(ValueError, "missing trainable gradient"):
                probe.execute_weight_update_probe()
