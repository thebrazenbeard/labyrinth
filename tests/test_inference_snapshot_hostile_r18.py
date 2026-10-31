"""Adversarial R18 tests of R17 inference-input snapshots; SOURCE_ONLY."""

from __future__ import annotations

from dataclasses import replace
import os
import unittest

import test_pinned_model_inference_receipt as _fixture_module

from labyrinth.pinned_model_inference_receipt import (
    InferenceSourceError, PinnedComponent,
    candidate_component_binding_sha256, model_component_set_sha256,
)


class InferenceSnapshotHostileR18(unittest.TestCase):
    def setUp(self) -> None:
        f = _fixture_module.PinnedModelInferenceReceiptTests(
            "test_logits_are_derived_inside_evaluator_source_and_signed")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f

    def test_factory_mutating_original_roster_is_denied(self) -> None:
        f = self.f

        class Driver:
            def predict_logits(self, _prompt):
                return (0.0, 1.0)

        def factory(_snapshot):
            object.__setattr__(f.cases[1], "prompt", "provider-altered prompt")
            return Driver()

        with self.assertRaisesRegex(InferenceSourceError, "roster|changed"):
            f.perform(evaluator_selected_driver_factory=factory)

    def test_later_prompt_is_fed_from_snapshot_and_original_mutation_denied(self) -> None:
        f = self.f
        seen: list[str] = []

        class Driver:
            def predict_logits(self, prompt):
                seen.append(prompt)
                if len(seen) == 1:
                    object.__setattr__(f.cases[1], "prompt", "swapped prompt")
                return (1.0, 0.0)

        with self.assertRaisesRegex(InferenceSourceError, "roster|changed"):
            f.perform(evaluator_selected_driver_factory=lambda _: Driver())
        self.assertEqual(seen, ["positive synthetic prompt", "negative synthetic prompt"])

    def test_original_intent_changed_during_factory_is_denied(self) -> None:
        f = self.f

        class Driver:
            def predict_logits(self, _prompt):
                return (0.0, 1.0)

        def factory(_snapshot):
            object.__setattr__(f.intent, "run_id", "different-run")
            return Driver()

        with self.assertRaisesRegex(InferenceSourceError, "intent|changed"):
            f.perform(evaluator_selected_driver_factory=factory)

    def test_original_model_pin_replaced_after_precheck_is_denied(self) -> None:
        f = self.f

        class Driver:
            def predict_logits(self, _prompt):
                return (0.0, 1.0)

        def factory(_snapshot):
            original = f.model.pretrained
            object.__setattr__(f.model, "pretrained", PinnedComponent(
                original.path, "a" * 64))
            return Driver()

        with self.assertRaisesRegex(InferenceSourceError, "pins|changed"):
            f.perform(evaluator_selected_driver_factory=factory)

    def test_component_bytes_changed_during_logits_are_denied(self) -> None:
        f = self.f
        path = f.model.adapter.path
        original = path.read_bytes()

        class Driver:
            def predict_logits(self, _prompt):
                path.write_bytes(b"x" * len(original))
                return (0.0, 1.0)

        with self.assertRaisesRegex(InferenceSourceError, "component"):
            f.perform(evaluator_selected_driver_factory=lambda _: Driver())

    @unittest.skipUnless(hasattr(os, "link"), "hardlink unavailable")
    def test_hardlinked_component_identity_is_denied_even_with_pins(self) -> None:
        f = self.f
        alias = f.root / "alias-adapter.safetensors"
        os.link(f.model.pretrained.path, alias)
        model = replace(f.model, adapter=PinnedComponent(
            alias, f.model.pretrained.sha256))
        components = model_component_set_sha256(model)
        intent = replace(f.intent, component_set_sha256=components)
        binding = candidate_component_binding_sha256(
            intent.candidate_artifact_sha256, components)
        with self.assertRaisesRegex(InferenceSourceError, "distinct file identities"):
            f.perform(model=model, intent=intent,
                      externally_pinned_component_set_sha256=components,
                      externally_pinned_candidate_component_binding_sha256=binding)

    def test_opaque_driver_can_ignore_prompt_without_host_attestation(self) -> None:
        f = self.f

        class IgnoringDriver:
            def predict_logits(self, _prompt):
                return (0.0, 1.0)

        receipt, _ = f.perform(evaluator_selected_driver_factory=lambda _: IgnoringDriver())
        self.assertEqual(f.verify(receipt), (("case01", 1), ("case02", 1)))
        self.assertFalse(receipt.physical_execution_authenticated)
        self.assertFalse(receipt.independent_evaluator_host_verified)
        self.assertFalse(receipt.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
