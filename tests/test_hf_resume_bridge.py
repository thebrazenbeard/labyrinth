"""Source-level guards for loading a previous LoRA adapter as trainable.

No Torch, Transformers, PEFT, model weights, network access, or host mutation
is required. A synthetic adapter file tests the provenance/loader boundary;
it is not a neural training or reload qualification result.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from labyrinth.hf_pilot import (
    PilotConfig, SCHEMA, _adapter_manifest, _checkpoint_manifest,
    _sha_file, _source_state,
)
from labyrinth.hf_resume_bridge import (
    ResumeContext, SCHEMA_RESUME, load_trainable_resume, preflight_resume,
)
from labyrinth.training_scope import TrainingScopeSpec


def _json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")


class ResumeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        _json(self.base / "config.json", {"architectures": ["TinyTestCausalLM"]})
        (self.base / "model.safetensors").write_bytes(b"synthetic-base-bytes-not-real-weights")
        self.prior = self.root / "previous_cycle"
        self.prior.mkdir()
        self.adapter = self.prior / "candidate_adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"synthetic-lora-data")
        _json(self.adapter / "adapter_config.json", {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
            "r": 4, "lora_alpha": 8,
        })
        self.previous_config = PilotConfig(model_path=str(self.base), output_dir=str(self.prior))
        self.config = PilotConfig(model_path=str(self.base), output_dir=str(self.root / "next_cycle"))
        self.initial_lora_sha = "a" * 64
        self.frozen_sha = "b" * 64
        self.buffer_sha = "c" * 64
        self.prior_report = {
            "schema": SCHEMA,
            "source": _source_state(),
            "config": asdict(self.previous_config),
            "base_weights_sha256": _checkpoint_manifest(self.base, [self.base / "model.safetensors"])["weights"],
            "model_metadata_sha256": _checkpoint_manifest(self.base, [self.base / "model.safetensors"])["metadata"],
            "candidate_adapter_sha256": _adapter_manifest(self.adapter),
            "candidate_lora_parameter_sha256": self.initial_lora_sha,
            "candidate_lora_tensor_value_sha256": self.initial_lora_sha,
            "initial_lora_parameter_sha256": "e" * 64,
            "initial_frozen_parameter_sha256": self.frozen_sha,
            "initial_persistent_buffers_sha256": self.buffer_sha,
        }
        self.report_path = self.prior / "pilot_report.json"
        self._rewrite_report()

    def _rewrite_report(self) -> None:
        _json(self.report_path, self.prior_report)
        scoped = self.prior_report.get("resumed_journey")
        self.context = ResumeContext(
            scope_fingerprint=(scoped["scope_fingerprint"] if scoped else "d" * 64),
            cycle_index=(scoped["cycle_index"] + 1 if scoped else 1),
            predecessor_report_sha256=_sha_file(self.report_path),
            predecessor_lora_value_sha256=self.initial_lora_sha,
            authority_ref="user-declared-labyrinth-resume-v1",
            allow_legacy_unscoped=scoped is None,
        )

    def _scoped_report(self, completed_cycle: int = 0) -> TrainingScopeSpec:
        scope = TrainingScopeSpec(
            objective_id="perspective-empathy", domain="empathy",
            skills=("cognitive_empathy",), specificity=1, scale=1,
            max_cycles=4, seed=17,
        )
        self.prior_report["resumed_journey"] = {
            "scope": scope.to_dict(), "scope_fingerprint": scope.fingerprint,
            "cycle_index": completed_cycle,
            "predecessor_scope_status": (
                "NEW_SCOPED_JOURNEY_INITIALIZATION" if completed_cycle == 0
                else "SAME_DECLARED_SCOPE_RECORDED"
            ),
            "predecessor_resume_receipt": None if completed_cycle == 0 else {
                "schema": SCHEMA_RESUME,
                "state_family": "LOCAL_WEIGHTS_ONLY",
                "objective_scope_fingerprint": scope.fingerprint,
                "cycle_index": completed_cycle,
                "this_cycle_initial_lora_sha256": "e" * 64,
                "same_objective_predecessor": True,
            },
        }
        self._rewrite_report()
        return scope

    def test_valid_preflight_is_read_only_and_binds_prior_files(self):
        source = preflight_resume(self.config, self.prior, self.context)
        self.assertEqual(source.predecessor_lora_value_sha256, self.initial_lora_sha)
        self.assertEqual(source.predecessor_adapter_manifest, _adapter_manifest(self.adapter))
        self.assertFalse(source.same_objective_predecessor)
        self.assertEqual(source.predecessor_scope_status,
                         "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF")
        self.assertFalse(Path(self.config.output_dir).exists())

    def test_legacy_v3_requires_explicit_bootstrap_authorization(self):
        unauthorized = replace(self.context, allow_legacy_unscoped=False)
        with self.assertRaisesRegex(ValueError, "explicit cycle-1 legacy bootstrap"):
            preflight_resume(self.config, self.prior, unauthorized)
        with self.assertRaisesRegex(ValueError, "only at cycle 1"):
            replace(self.context, cycle_index=2)

    def test_scoped_cycle_zero_can_resume_at_cycle_one(self):
        self._scoped_report(completed_cycle=0)
        source = preflight_resume(self.config, self.prior, self.context)
        self.assertTrue(source.same_objective_predecessor)
        self.assertEqual(source.predecessor_cycle_index, 0)
        self.assertEqual(source.predecessor_scope_status,
                         "MATCHED_SCOPED_PREDECESSOR_REPORT_NOT_INDEPENDENTLY_ATTESTED")

    def test_scoped_cycle_one_can_resume_at_cycle_two(self):
        self._scoped_report(completed_cycle=1)
        source = preflight_resume(self.config, self.prior, self.context)
        self.assertTrue(source.same_objective_predecessor)
        self.assertEqual(source.predecessor_cycle_index, 1)

    def test_scoped_cycle_one_legacy_bootstrap_can_continue_without_upstream_scope_claim(self):
        self._scoped_report(completed_cycle=1)
        self.prior_report["resumed_journey"]["predecessor_scope_status"] = (
            "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF"
        )
        self.prior_report["resumed_journey"]["predecessor_resume_receipt"]["same_objective_predecessor"] = False
        self._rewrite_report()
        source = preflight_resume(self.config, self.prior, self.context)
        self.assertTrue(source.same_objective_predecessor)
        self.assertEqual(source.predecessor_cycle_index, 1)

    def test_scoped_cycle_one_refuses_missing_or_inconsistent_nested_resume_receipt(self):
        self._scoped_report(completed_cycle=1)
        self.prior_report["resumed_journey"]["predecessor_resume_receipt"] = None
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "receipt does not bind"):
            preflight_resume(self.config, self.prior, self.context)
        self.prior_report["resumed_journey"]["predecessor_resume_receipt"] = {
            "schema": SCHEMA_RESUME, "state_family": "LOCAL_WEIGHTS_ONLY",
            "objective_scope_fingerprint": self.context.scope_fingerprint,
            "cycle_index": 1, "this_cycle_initial_lora_sha256": "e" * 64,
            "same_objective_predecessor": False,
        }
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "contradicts recorded lineage"):
            preflight_resume(self.config, self.prior, self.context)

    def test_misattributed_scope_is_refused_even_when_report_hash_is_pinned(self):
        self._scoped_report(completed_cycle=1)
        dishonest = replace(self.context, scope_fingerprint="d" * 64)
        with self.assertRaisesRegex(ValueError, "predecessor scoped objective differs"):
            preflight_resume(self.config, self.prior, dishonest)
        self.prior_report["resumed_journey"]["scope_fingerprint"] = "d" * 64
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "predecessor scoped objective differs"):
            preflight_resume(self.config, self.prior, self.context)

    def test_scoped_cycle_skip_replay_and_status_contradiction_are_refused(self):
        self._scoped_report(completed_cycle=1)
        for requested in (1, 3):
            with self.subTest(requested_cycle=requested):
                with self.assertRaisesRegex(ValueError, "exactly one before"):
                    preflight_resume(self.config, self.prior,
                                     replace(self.context, cycle_index=requested))
        self.prior_report["resumed_journey"]["predecessor_scope_status"] = (
            "NEW_SCOPED_JOURNEY_INITIALIZATION"
        )
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "contradicts cycle"):
            preflight_resume(self.config, self.prior, self.context)

    def test_scoped_predecessor_rejects_legacy_flag_and_invalid_scope_receipt(self):
        self._scoped_report(completed_cycle=0)
        with self.assertRaisesRegex(ValueError, "cannot be used for scoped predecessor"):
            preflight_resume(self.config, self.prior,
                             replace(self.context, allow_legacy_unscoped=True))
        self.prior_report["resumed_journey"]["scope"]["seed"] = -1
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "declaration is invalid"):
            preflight_resume(self.config, self.prior, self.context)

    def test_tampered_report_refused_before_dependency_import(self):
        with self.report_path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "prior report bytes differ"):
            preflight_resume(self.config, self.prior, self.context)

    def test_missing_provenance_or_source_change_is_refused(self):
        self.prior_report["source"]["source_tree_sha256"] = "f" * 64
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "predecessor source"):
            preflight_resume(self.config, self.prior, self.context)

    def test_unmigrated_mezo_v4_report_is_not_silently_accepted(self):
        self.prior_report["schema"] = "labyrinth.hf-lora-pilot.v4"
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "only current v3"):
            preflight_resume(self.config, self.prior, self.context)

    def test_incompatible_lora_rank_is_refused(self):
        changed = replace(self.config, lora_rank=8)
        with self.assertRaisesRegex(ValueError, "LoRA setup"):
            preflight_resume(changed, self.prior, self.context)

    def test_changed_base_and_adapter_bytes_are_refused(self):
        (self.base / "model.safetensors").write_bytes(b"tampered-base")
        with self.assertRaisesRegex(ValueError, "base weight/tokenizer"):
            preflight_resume(self.config, self.prior, self.context)
        (self.base / "model.safetensors").write_bytes(b"synthetic-base-bytes-not-real-weights")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"tampered-adapter")
        with self.assertRaisesRegex(ValueError, "adapter files changed"):
            preflight_resume(self.config, self.prior, self.context)

    def test_existing_successor_root_is_refused_and_not_modified(self):
        successor = Path(self.config.output_dir)
        successor.mkdir()
        marker = successor / "existing.txt"
        marker.write_text("owned", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            preflight_resume(self.config, self.prior, self.context)
        self.assertEqual(marker.read_text(encoding="utf-8"), "owned")

    def test_successor_cannot_nest_within_predecessor_run(self):
        nested = replace(self.config, output_dir=str(self.prior / "next"))
        with self.assertRaisesRegex(ValueError, "distinct from predecessor"):
            preflight_resume(nested, self.prior, self.context)

    def test_adapter_symlink_refused_before_any_weight_loading(self):
        linked = self.adapter / "unexpected_link.txt"
        try:
            linked.symlink_to(self.report_path)
        except (OSError, NotImplementedError):
            self.skipTest("platform does not permit test symlinks")
        with self.assertRaisesRegex(ValueError, "symlink"):
            preflight_resume(self.config, self.prior, self.context)

    def test_bad_adapter_config_is_refused_even_if_manifest_edited_together(self):
        _json(self.adapter / "adapter_config.json", {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
            "r": 8, "lora_alpha": 8,
        })
        self.prior_report["candidate_adapter_sha256"] = _adapter_manifest(self.adapter)
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "PEFT adapter config"):
            preflight_resume(self.config, self.prior, self.context)

    def test_incompatible_state_family_or_context_refused(self):
        with self.assertRaisesRegex(ValueError, "LOCAL_WEIGHTS_ONLY"):
            ResumeContext("d" * 64, 1, self.context.predecessor_report_sha256,
                          self.initial_lora_sha, "valid-authority", "IDENTITY_REWRITE")
        with self.assertRaisesRegex(ValueError, "cycle_index"):
            replace(self.context, cycle_index=0)
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            replace(self.context, allow_legacy_unscoped=1)

    def test_inconsistent_prior_candidate_digest_receipt_is_refused(self):
        self.prior_report["candidate_lora_parameter_sha256"] = "0" * 64
        self._rewrite_report()
        with self.assertRaisesRegex(ValueError, "digest fields disagree"):
            preflight_resume(self.config, self.prior, self.context)

    def test_actual_loader_uses_trainable_peft_and_attests_initial_hash(self):
        fake_model = object()
        fake_tokenizer = object()
        calls = []

        class FakePeft:
            @staticmethod
            def from_pretrained(base, path, *, is_trainable, local_files_only):
                calls.append((base, path, is_trainable, local_files_only))
                return fake_model

        patches = (
            mock.patch("labyrinth.hf_resume_bridge._deps", return_value=(
                object(), object(), object(), (object(), FakePeft, object()), "cpu")),
            mock.patch("labyrinth.hf_resume_bridge._load_base", return_value=(object(), fake_tokenizer)),
            mock.patch("labyrinth.hf_resume_bridge._device_headroom", return_value=None),
            mock.patch("labyrinth.hf_resume_bridge._lora_value_digest", return_value=self.initial_lora_sha),
            mock.patch("labyrinth.hf_resume_bridge._parameter_digest", return_value=self.initial_lora_sha),
            mock.patch("labyrinth.hf_resume_bridge._frozen_parameter_digest", return_value=self.frozen_sha),
            mock.patch("labyrinth.hf_resume_bridge._buffer_digest", return_value=self.buffer_sha),
        )
        from contextlib import ExitStack

        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            result = load_trainable_resume(self.config, self.prior, self.context)
        self.assertIs(result.model, fake_model)
        self.assertIs(result.tokenizer, fake_tokenizer)
        self.assertEqual(calls[0][1:], (str(self.adapter), True, True))
        self.assertEqual(result.receipt["schema"], SCHEMA_RESUME)
        self.assertEqual(result.receipt["previous_cycle_final_lora_sha256"],
                         result.receipt["this_cycle_initial_lora_sha256"])
        self.assertEqual(result.receipt["phase"], "TRAINABLE_RESUME_READY_ONLY")
        self.assertFalse(result.receipt["same_objective_predecessor"])
        self.assertTrue(result.receipt["allow_legacy_unscoped"])
        self.assertFalse(result.receipt["private_graduation"])
        self.assertEqual(result.receipt["optimizer_steps"], 0)
        self.assertFalse(Path(self.config.output_dir).exists())

    def test_reloaded_tensor_digest_mismatch_is_not_accepted(self):
        class FakePeft:
            @staticmethod
            def from_pretrained(_base, _path, **_kwargs):
                return object()

        with (
            mock.patch("labyrinth.hf_resume_bridge._deps", return_value=(
                object(), object(), object(), (object(), FakePeft, object()), "cpu")),
            mock.patch("labyrinth.hf_resume_bridge._load_base", return_value=(object(), object())),
            mock.patch("labyrinth.hf_resume_bridge._lora_value_digest", return_value="e" * 64),
            mock.patch("labyrinth.hf_resume_bridge._parameter_digest", return_value="e" * 64),
        ):
            with self.assertRaisesRegex(RuntimeError, "LoRA values differ"):
                load_trainable_resume(self.config, self.prior, self.context)


if __name__ == "__main__":
    unittest.main()
