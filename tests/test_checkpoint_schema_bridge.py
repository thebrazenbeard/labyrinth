"""Incompatible HF pilot receipts must never be relabelled as resume proof."""

from copy import deepcopy
from dataclasses import asdict, replace
import unittest

from labyrinth.checkpoint_schema_bridge import (
    classify_checkpoint_report, require_v3_adapter_preflight,
)
from labyrinth.hf_pilot import SCHEMA as EXPOSURE_SCHEMA


SHA = "a" * 64


def _v3_report():
    return {
        "schema": "labyrinth.hf-lora-pilot.v3",
        "source": {"source_tree_sha256": SHA},
        "config": {"model_path": "/isolated/model", "output_dir": "/isolated/run",
                   "lora_rank": 4, "lora_alpha": 8, "answer_only": True},
        "base_weights_sha256": {"model.safetensors": SHA},
        "model_metadata_sha256": {"config.json": SHA},
        "candidate_adapter_sha256": {"adapter_model.safetensors": SHA,
                                     "adapter_config.json": SHA},
        "candidate_lora_parameter_sha256": SHA,
        "candidate_lora_tensor_value_sha256": SHA,
        "initial_frozen_parameter_sha256": SHA,
        "initial_persistent_buffers_sha256": SHA,
    }


class CheckpointSchemaBridgeTests(unittest.TestCase):
    def test_current_exposure_schema_routes_only_to_strict_adapter_preflight(self):
        report = _v3_report()
        report["schema"] = EXPOSURE_SCHEMA
        plan = classify_checkpoint_report(report)
        self.assertEqual(plan.source_schema, EXPOSURE_SCHEMA)
        self.assertEqual(plan.source_optimizer, "adamw")
        self.assertEqual(plan.status, "V3_ADAPTER_PREFLIGHT_REQUIRED")
        self.assertTrue(plan.may_call_v3_preflight)
        self.assertFalse(plan.bytes_authenticated)
        self.assertFalse(plan.optimizer_restart_supported)
        self.assertFalse(plan.graduation_authorized)
        self.assertEqual(require_v3_adapter_preflight(report), plan)
        self.assertEqual(classify_checkpoint_report(report, continuity="mid_cycle").status,
                         "MID_CYCLE_STATE_UNAVAILABLE")
        self.assertEqual(classify_checkpoint_report(report, target_optimizer="mezo").status,
                         "CROSS_OPTIMIZER_MIGRATION_REQUIRED")

    def test_current_exposure_schema_cannot_disguise_mezo_v4_metadata(self):
        for schema in ("labyrinth.hf-lora-pilot.v3", EXPOSURE_SCHEMA):
            disguised = _v3_report()
            disguised["schema"] = schema
            disguised["config"].update(optimizer="mezo", zo_epsilon=0.001)
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "AdamW.*MeZO"):
                classify_checkpoint_report(disguised)

    def test_classifier_cannot_forge_physical_optimizer_or_graduation_flags(self):
        plan = classify_checkpoint_report(_v3_report())
        for field in ("optimizer_restart_supported", "bytes_authenticated",
                      "graduation_authorized"):
            with self.subTest(field=field):
                self.assertIs(getattr(plan, field), False)
                with self.assertRaises(TypeError):
                    replace(plan, **{field: True})
                with self.assertRaises(TypeError):
                    type(plan)(**{**asdict(plan), field: True})
                with self.assertRaises(AttributeError):
                    object.__setattr__(plan, field, True)
                self.assertIs(getattr(plan, field), False)

    def test_preflight_routing_hint_cannot_be_set_independently_of_plan_status(self):
        plan = classify_checkpoint_report(_v3_report())
        self.assertTrue(plan.may_call_v3_preflight)
        with self.assertRaises(TypeError):
            replace(plan, may_call_v3_preflight=False)
        with self.assertRaises(TypeError):
            type(plan)(**{**asdict(plan), "may_call_v3_preflight": False})
        with self.assertRaises(AttributeError):
            object.__setattr__(plan, "may_call_v3_preflight", False)
        incompatible = replace(plan, status="CROSS_OPTIMIZER_MIGRATION_REQUIRED")
        self.assertFalse(incompatible.may_call_v3_preflight)

    def test_v3_is_only_eligible_for_full_preflight_not_attested_resume(self):
        result = classify_checkpoint_report(_v3_report())
        self.assertEqual(result.status, "V3_ADAPTER_PREFLIGHT_REQUIRED")
        self.assertTrue(result.may_call_v3_preflight)
        self.assertFalse(result.optimizer_restart_supported)
        self.assertFalse(result.bytes_authenticated)
        self.assertEqual(require_v3_adapter_preflight(_v3_report()), result)

    def test_v4_mezo_requires_native_migration_and_cannot_masquerade_as_v3(self):
        v4 = _v3_report()
        v4["schema"] = "labyrinth.hf-lora-pilot.v4"
        v4["config"].update(optimizer="mezo", zo_epsilon=0.001)
        result = classify_checkpoint_report(v4, target_optimizer="mezo")
        self.assertEqual(result.status, "V4_MEZO_NATIVE_PREFLIGHT_REQUIRED")
        self.assertFalse(result.may_call_v3_preflight)
        with self.assertRaisesRegex(ValueError, "v4 MeZO"):
            require_v3_adapter_preflight(v4)

        disguised = deepcopy(v4)
        disguised["schema"] = "labyrinth.hf-lora-pilot.v3"
        with self.assertRaisesRegex(ValueError, "v3.*MeZO"):
            classify_checkpoint_report(disguised)

        v4["config"]["optimizer"] = "adamw"
        with self.assertRaisesRegex(ValueError, "v4.*MeZO"):
            classify_checkpoint_report(v4)

    def test_no_optimizer_or_rng_restart_is_invented(self):
        for report in (_v3_report(),):
            # An unverified receipt field is not equivalent to a durable,
            # replayable optimizer/RNG/cursor checkpoint.
            report["optimizer_state_sha256"] = SHA
            report["rng_state_sha256"] = SHA
            report["training_cursor"] = 5
            result = classify_checkpoint_report(report, continuity="mid_cycle")
            self.assertEqual(result.status, "MID_CYCLE_STATE_UNAVAILABLE")
            self.assertFalse(result.may_call_v3_preflight)
            self.assertFalse(result.optimizer_restart_supported)
        v4 = _v3_report()
        v4["schema"] = "labyrinth.hf-lora-pilot.v4"
        v4["config"].update(optimizer="mezo", zo_epsilon=0.001)
        self.assertEqual(classify_checkpoint_report(v4, continuity="mid_cycle").status,
                         "MID_CYCLE_STATE_UNAVAILABLE")

    def test_cross_optimizer_and_malformed_report_fail_closed(self):
        v3 = _v3_report()
        self.assertEqual(classify_checkpoint_report(v3, target_optimizer="mezo").status,
                         "CROSS_OPTIMIZER_MIGRATION_REQUIRED")
        with self.assertRaises(ValueError):
            require_v3_adapter_preflight(v3, target_optimizer="mezo")
        for field in ("source", "candidate_adapter_sha256", "candidate_lora_parameter_sha256"):
            broken = deepcopy(v3)
            broken.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                classify_checkpoint_report(broken)
        bad_hash = deepcopy(v3)
        bad_hash["candidate_adapter_sha256"]["adapter_model.safetensors"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "digest"):
            classify_checkpoint_report(bad_hash)
        bad_label = deepcopy(v3)
        bad_label["candidate_adapter_sha256"] = {"../outside.safetensors": SHA}
        with self.assertRaisesRegex(ValueError, "file label"):
            classify_checkpoint_report(bad_label)
        malformed = deepcopy(v3)
        malformed["config"]["optimizer"] = "mezo"
        with self.assertRaises(ValueError):
            classify_checkpoint_report(malformed)
        with self.assertRaises(ValueError):
            classify_checkpoint_report(v3, continuity="unknown")


if __name__ == "__main__":
    unittest.main()
