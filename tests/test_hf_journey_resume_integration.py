"""Dependency-free integration contracts for opt-in HF journey continuation.

The fake model in the last test proves pilot control-flow and digest invariants,
not real pretrained neural learning. A separate host with Torch/PEFT and a
verified local checkpoint must run and independently reload any actual update.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from labyrinth.hf_pilot import (
    PilotConfig, _resume_training_cases, _scoped_training_cases,
    run_pilot, training_cases,
)
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.training_scope import TrainingScopeSpec


class ResumeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"architectures":["TinyTest"]}\n', encoding="utf-8")
        (self.base / "model.safetensors").write_bytes(b"not a real model")
        self.prior = self.root / "prior"
        self.prior.mkdir()
        self.config = PilotConfig(
            model_path=str(self.base), output_dir=str(self.root / "next"),
            seed=33, train_count=4, eval_count=4, tiers=(1,), steps=4,
        )
        self.prior_config = PilotConfig(**{**asdict(self.config), "output_dir": str(self.prior)})
        self.scope = TrainingScopeSpec(
            "reasoning-arithmetic", "reasoning", ("arithmetic",), 1, 3, 4, 2,
        )

    def _prior_report(self, *, scoped_cycle: int | None = None) -> dict:
        if scoped_cycle is None:
            train, heldout = training_cases(self.prior_config)
        else:
            train, heldout, _ = _scoped_training_cases(self.prior_config, self.scope, scoped_cycle, ())
        report = {
            "config": asdict(self.prior_config),
            "training_case_ids": [case.id for case in train],
            "heldout_case_ids": [case.id for case in heldout],
        }
        if scoped_cycle is not None:
            cases = train + heldout
            report["resumed_journey"] = {
                "scope": self.scope.to_dict(),
                "scope_fingerprint": self.scope.fingerprint,
                "cycle_index": scoped_cycle,
                "weak_skills": [],
                "historical_case_ids": sorted({case.id for case in cases}),
                "historical_prompt_sha256": sorted({hashlib.sha256(case.prompt.encode()).hexdigest() for case in cases}),
            }
        return report

    def test_predeclared_scope_generates_different_train_only_rooms(self):
        first, heldout, first_plan = _scoped_training_cases(self.config, self.scope, 0, ())
        second, next_heldout, second_plan = _scoped_training_cases(self.config, self.scope, 1, ())
        self.assertEqual({case.split for case in first + second}, {"train"})
        self.assertEqual(first_plan.scope_fingerprint, self.scope.fingerprint)
        self.assertEqual(second_plan.cycle_index, 1)
        self.assertTrue({case.id for case in first}.isdisjoint({case.id for case in second}))
        self.assertTrue({case.prompt for case in second}.isdisjoint({case.prompt for case in heldout}))
        self.assertEqual([case.id for case in heldout], [case.id for case in next_heldout])

    def test_unsupported_skill_is_not_misrepresented_by_tier_four_diagnostic(self):
        unsupported = TrainingScopeSpec(
            "learn-programs", "reasoning", ("program_reasoning",), 1, 3, 4, 2,
        )
        with self.assertRaisesRegex(ValueError, "matching heldout diagnostic"):
            _scoped_training_cases(self.config, unsupported, 0, ())

    def test_unscoped_legacy_bootstrap_requires_explicit_transfer_authority(self):
        prior = self._prior_report()
        with self.assertRaisesRegex(ValueError, "legacy bootstrap authorization"):
            _resume_training_cases(self.config, self.scope, 1, (), prior)
        train, _, _ = _resume_training_cases(
            self.config, self.scope, 1, (), prior, allow_unscoped_predecessor=True,
        )
        self.assertTrue(train)

    def test_legacy_authority_cannot_disagree_between_context_and_run(self):
        context = ResumeContext(
            self.scope.fingerprint, 1, "f" * 64, "a" * 64,
            "patrick-explicit-resume", allow_legacy_unscoped=True,
        )
        with mock.patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("no heavy imports")):
            with self.assertRaisesRegex(ValueError, "legacy bootstrap authorization differs"):
                run_pilot(
                    self.config, scope=self.scope, cycle_index=1,
                    predecessor_run_dir=self.prior, resume_context=context,
                    allow_unscoped_predecessor=False,
                )
        self.assertFalse(Path(self.config.output_dir).exists())

    def test_scoped_predecessor_and_historical_seal_are_enforced(self):
        prior = self._prior_report(scoped_cycle=1)
        train, _, _ = _resume_training_cases(self.config, self.scope, 2, (), prior)
        self.assertEqual({case.split for case in train}, {"train"})
        # A previously seen cycle-zero prompt may not be emitted again, even
        # if absent from the immediate predecessor's own training cases.
        new_case = train[0]
        prior["resumed_journey"]["historical_prompt_sha256"].append(
            hashlib.sha256(new_case.prompt.encode("utf-8")).hexdigest()
        )
        with self.assertRaisesRegex(ValueError, "overlaps prior"):
            _resume_training_cases(self.config, self.scope, 2, (), prior)
        prior["resumed_journey"]["historical_prompt_sha256"] = []
        with self.assertRaisesRegex(ValueError, "history does not seal"):
            _resume_training_cases(self.config, self.scope, 2, (), prior)

    def test_run_resume_starts_sham_candidate_and_corrupt_from_prior_values(self):
        prior = self._prior_report()
        prior_file = self.prior / "pilot_report.json"
        prior_file.write_text(json.dumps(prior), encoding="utf-8")
        initial = "a" * 64
        trained = "b" * 64
        corrupt = "c" * 64
        frozen = "d" * 64
        buffers = "e" * 64
        context = ResumeContext(
            self.scope.fingerprint, 1, hashlib.sha256(prior_file.read_bytes()).hexdigest(),
            initial, "patrick-explicit-resume",
            allow_legacy_unscoped=True,
        )

        class Parameter:
            requires_grad = True
            def __init__(self, model, snapshot=None):
                self.model = model
                self.snapshot = snapshot
            def detach(self):
                return self
            def clone(self):
                return Parameter(self.model, self.model.value)
            def copy_(self, other):
                self.model.value = other.snapshot

        class Model:
            def __init__(self):
                self.value = initial
                self.parameter = Parameter(self)
            def named_parameters(self):
                return [("layer.lora_A.default.weight", self.parameter)]
            def save_pretrained(self, directory, *, safe_serialization):
                self.saved = self.value
                path = Path(directory)
                path.mkdir()
                (path / "adapter_model.safetensors").write_bytes(b"fake-weights")

        model = Model()
        visits = []
        def fake_train(current, _tokenizer, _torch, train, labels, config, *, learning_rate=None):
            visits.append(("sham" if learning_rate == 0 else "correct" if labels == [c.answer for c in train] else "corrupt", current.value))
            if learning_rate != 0:
                current.value = trained if labels == [c.answer for c in train] else corrupt
            return {"steps": config.steps, "visited_case_ids": [c.id for c in train]}

        torch = SimpleNamespace(
            get_num_threads=lambda: 4, set_num_threads=lambda _value: None,
            manual_seed=lambda _seed: None,
            use_deterministic_algorithms=lambda _bool: None,
            no_grad=lambda: nullcontext(),
        )
        mock_deps = (torch, object(), object(), (object(), object(), mock.Mock(side_effect=AssertionError("fresh LoRA forbidden"))), "cpu")
        receipt = {
            "previous_cycle_final_lora_sha256": initial,
            "this_cycle_initial_lora_sha256": initial,
            "frozen_parameter_sha256": frozen,
            "persistent_buffers_sha256": buffers,
        }
        with (
            mock.patch("labyrinth.hf_resume_bridge.preflight_resume", return_value=SimpleNamespace(predecessor_run=self.prior)),
            mock.patch("labyrinth.hf_resume_bridge.load_trainable_resume", return_value=SimpleNamespace(model=model, tokenizer=object(), receipt=receipt)),
            mock.patch("labyrinth.hf_pilot._deps", return_value=mock_deps),
            mock.patch("labyrinth.hf_pilot._device_headroom", return_value=None),
            mock.patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])),
            mock.patch("labyrinth.hf_pilot._score_model", return_value={"trials": [], "correct": 0, "total": 0}),
            mock.patch("labyrinth.hf_pilot._train_steps", side_effect=fake_train),
            mock.patch("labyrinth.hf_pilot._parameter_digest", side_effect=lambda obj, _torch: obj.value),
            mock.patch("labyrinth.hf_pilot._lora_value_digest", side_effect=lambda obj, _torch: obj.value),
            mock.patch("labyrinth.hf_pilot._frozen_parameter_digest", return_value=frozen),
            mock.patch("labyrinth.hf_pilot._buffer_digest", return_value=buffers),
            mock.patch("labyrinth.hf_pilot.package_version", return_value="synthetic-test"),
        ):
            result = run_pilot(
                self.config, scope=self.scope, cycle_index=1,
                predecessor_run_dir=self.prior, resume_context=context,
                allow_unscoped_predecessor=True,
            )
        self.assertEqual(visits, [("sham", initial), ("correct", initial), ("corrupt", initial)])
        self.assertEqual(model.saved, trained)
        self.assertEqual(result["initial_lora_parameter_sha256"], initial)
        self.assertEqual(result["candidate_lora_parameter_sha256"], trained)
        self.assertEqual(result["corrupted_lora_parameter_sha256"], corrupt)
        self.assertEqual(result["resumed_journey"]["cycle_index"], 1)
        self.assertEqual(result["resumed_journey"]["predecessor_scope_status"],
                         "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF")
        self.assertEqual(result["resumed_journey"]["qualification"], "NOT_ESTABLISHED")
        self.assertFalse(result["resumed_journey"]["minotaur_private_graduation"])
        self.assertTrue((Path(self.config.output_dir) / "candidate_adapter" / "adapter_model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
