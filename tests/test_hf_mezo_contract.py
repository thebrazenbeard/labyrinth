"""No-ML-dependency checks for the optional MeZO HF/PEFT pilot contract."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import (
    MEZO_SCHEMA, PilotConfig, _report_config, _train_steps, validate_config,
)
from labyrinth.hf_report_audit import audit_hf_report
from labyrinth.hf_pilot import _training_visit_indices, training_cases
from tests.test_hf_report_audit import pilot_fixture


class MezoPilotContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        model = Path(self.temp.name) / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"small offline fixture")
        self.config = PilotConfig(
            model_path=str(model), output_dir=str(Path(self.temp.name) / "new_output"),
            optimizer="mezo", zo_epsilon=0.001,
        )

    def test_preflight_validates_method_and_epsilon_without_importing_torch(self):
        validate_config(self.config)
        for changes in (
            {"optimizer": "unsafe"}, {"optimizer": 0},
            {"zo_epsilon": 0.0}, {"zo_epsilon": -0.1},
            {"zo_epsilon": float("nan")}, {"zo_epsilon": 1.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(replace(self.config, **changes))

    def test_default_adamw_uses_the_original_v3_receipt_keys(self):
        config = replace(self.config, optimizer="adamw")
        persisted = _report_config(config, Path(config.model_path), Path(config.output_dir))
        self.assertNotIn("optimizer", persisted)
        self.assertNotIn("zo_epsilon", persisted)
        self.assertIn("answer_only", persisted)
        mezo = _report_config(self.config, Path(self.config.model_path), Path(self.config.output_dir))
        self.assertEqual(mezo["optimizer"], "mezo")
        self.assertEqual(mezo["zo_epsilon"], 0.001)

    def test_mezo_train_step_uses_matched_seeded_forward_only_contract(self):
        config = replace(self.config, seed=9, steps=2, train_count=2, eval_count=2)
        train, _ = training_cases(config)

        class FakeTorch:
            long = "long"

            @staticmethod
            def manual_seed(value):
                pass

            @staticmethod
            def tensor(value, **kwargs):
                return value

            @staticmethod
            def isfinite(value):
                return True

        class FakeModel:
            eval_called = False

            def eval(self):
                self.eval_called = True

            def train(self):
                raise AssertionError("MeZO must not enable dropout")

            def parameters(self):
                return []

        model = FakeModel()
        outcomes = [
            {"loss": 1.5, "loss_plus": 1.0, "loss_minus": 2.0, "forward_evaluations": 2},
            {"loss": 2.5, "loss_plus": 2.0, "loss_minus": 3.0, "forward_evaluations": 2},
        ]
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([4, 5], [-100, 5])):
            with patch("labyrinth.mezo.mezo_step", side_effect=outcomes) as step:
                record = _train_steps(model, object(), FakeTorch(), train,
                                      [c.answer for c in train], config, learning_rate=0.0)
        self.assertTrue(model.eval_called)
        self.assertEqual(step.call_count, 2)
        self.assertEqual([c.kwargs["seed"] for c in step.call_args_list], [9, 10])
        self.assertEqual([c.kwargs["learning_rate"] for c in step.call_args_list], [0.0, 0.0])
        self.assertEqual(record["forward_evaluations"], 4)
        self.assertEqual(record["positive_losses"], [1.0, 2.0])
        self.assertEqual(record["negative_losses"], [2.0, 3.0])
        self.assertEqual(record["losses"], [1.5, 2.5])
        indices = _training_visit_indices(train, seed=9, steps=2)
        self.assertEqual(record["visited_case_ids"], [train[i].id for i in indices])

    def test_v4_audit_checks_optimizer_forward_budget_and_loss_pairs(self):
        report = pilot_fixture()
        report["schema"] = MEZO_SCHEMA
        report["initial_persistent_buffers_sha256"] = "a" * 64
        report["config"].update({"answer_only": False, "optimizer": "mezo", "zo_epsilon": 0.001})
        report["control_data"].update({
            "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
            "sham_evaluation_before_updates": False, "sham_is_compute_matched": True,
            "answer_scaffold_supplied": False,
            "mezo_loss_forward_evaluations_per_arm": 4,
            "mezo_snapshot_restoration": True,
        })
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        expected_ids = [train[i].id for i in _training_visit_indices(train, seed=13, steps=2)]
        for arm in ("sham", "candidate", "corrupted"):
            report["training"][arm] = {
                "steps": 2, "example_visits": 2, "visited_case_ids": expected_ids,
                "input_tokens_seen": 200, "losses": [1.5, 2.5],
                "positive_losses": [1.0, 2.0], "negative_losses": [2.0, 3.0],
                "forward_evaluations": 4, "direction_seeds": [13, 14],
            }
        self.assertEqual(audit_hf_report(report)["status"], "PASS")
        for field, corrupted in (("forward_evaluations", 3),
                                 ("direction_seeds", [14, 13]),
                                 ("positive_losses", [1.0, 400.0])):
            test = {**report, "training": {**report["training"], "sham": {
                **report["training"]["sham"], field: corrupted,
            }}}
            with self.subTest(field=field):
                self.assertEqual(audit_hf_report(test)["status"], "FAIL")
        report["config"].pop("zo_epsilon")
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
