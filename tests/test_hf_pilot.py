"""Dependency-free contract tests; these do not claim that a HF model trained."""

import json
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import (
    PilotConfig,
    SCHEMA,
    _metadata_digest,
    _weights_digest,
    _device_headroom,
    _encode_pair,
    _lora_value_digest,
    _score_model,
    _trial_record,
    _summarize_trials,
    _train_steps,
    derange_answers,
    reload_evaluate,
    run_pilot,
    training_cases,
    validate_config,
    validate_tokenized_answer_discrimination,
)
from labyrinth.maze import Challenge


class DummyTokenizer:
    eos_token = "~"

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [1] + [ord(c) for c in text] if add_special_tokens
                else [ord(c) for c in text]}


class HfPilotContractTests(unittest.TestCase):
    def test_zero_lr_sham_executes_full_gradient_and_optimizer_steps(self):
        events = []

        class Finite:
            def __bool__(self):
                return True

            def all(self):
                return self

        class Loss:
            def backward(self):
                events.append("backward")

            def detach(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return 1.0

        class Parameter:
            requires_grad = True

        class Model:
            def parameters(self):
                return [Parameter()]

            def train(self):
                events.append("train")

            def __call__(self, *, input_ids, labels):
                events.append("forward")
                return SimpleNamespace(loss=Loss())

        class AdamW:
            def __init__(self, params, *, lr):
                list(params)
                events.append(("optimizer_lr", lr))

            def zero_grad(self, *, set_to_none):
                events.append("zero_grad")

            def step(self):
                events.append("optimizer_step")

        fake_torch = SimpleNamespace(
            long="long", manual_seed=lambda seed: events.append(("seed", seed)),
            tensor=lambda value, **kwargs: value,
            isfinite=lambda value: Finite(),
            optim=SimpleNamespace(AdamW=AdamW),
            nn=SimpleNamespace(utils=SimpleNamespace(
                clip_grad_norm_=lambda params, max_norm: events.append("clip"))),
        )
        cases = [Challenge("train-one", "arithmetic", 1, "train", "1 + 2", "3")]
        config = self.spec(steps=3)
        sham = _train_steps(Model(), DummyTokenizer(), fake_torch, cases, ["3"],
                            config, zero_learning_rate_sham=True)
        self.assertEqual(sham["steps"], 3)
        self.assertEqual(sham["optimizer_learning_rate"], 0.0)
        self.assertEqual(events.count("forward"), 3)
        self.assertEqual(events.count("backward"), 3)
        self.assertEqual(events.count("optimizer_step"), 3)
        self.assertIn(("optimizer_lr", 0.0), events)
        events.clear()
        candidate = _train_steps(Model(), DummyTokenizer(), fake_torch, cases, ["3"], config)
        self.assertEqual(candidate["optimizer_learning_rate"], config.learning_rate)
        self.assertEqual(sham["visited_case_ids"], candidate["visited_case_ids"])
        self.assertEqual(sham["input_tokens_seen"], candidate["input_tokens_seen"])
        self.assertIn(("optimizer_lr", config.learning_rate), events)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "local_model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"small-fixture-not-real-weights")
        self.output = root / "new_run"

    def spec(self, **kwargs):
        return PilotConfig(model_path=str(self.model), output_dir=str(self.output), **kwargs)

    def test_preflight_does_not_create_output_or_download(self):
        model, output, weights = validate_config(self.spec())
        self.assertEqual(model, self.model)
        self.assertEqual(output, self.output)
        self.assertEqual([p.name for p in weights], ["model.safetensors"])
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output_and_model_directory(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            validate_config(self.spec())
        with self.assertRaises(ValueError):
            validate_config(PilotConfig(model_path=str(self.model), output_dir=str(self.model)))

    def test_refuses_oversized_weights_and_bin_only(self):
        (self.model / "model.safetensors").unlink()
        (self.model / "pytorch_model.bin").write_bytes(b"pickle")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            validate_config(self.spec())

    def test_refuses_invalid_resource_bounds_before_heavy_import(self):
        for field, value in [("steps", 0), ("steps", 33), ("max_tokens", 10),
                             ("train_count", 100), ("seed", True), ("lora_rank", 99),
                             ("learning_rate", float("nan")), ("device", "auto")]:
            with self.subTest(field=field):
                payload = {**self.spec().__dict__, field: value}
                with self.assertRaises(ValueError):
                    validate_config(PilotConfig(**payload))

    def test_train_split_is_disjoint_and_pinned(self):
        config = self.spec(seed=13, train_count=5, eval_count=5, tiers=(1, 2))
        train, heldout = training_cases(config)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(heldout), 10)
        self.assertEqual({c.split for c in train}, {"train"})
        self.assertEqual({c.split for c in heldout}, {"eval"})
        self.assertFalse({c.id for c in train} & {c.id for c in heldout})
        self.assertEqual(train, training_cases(config)[0])

    def test_derangement_preserves_multiset_without_collisions(self):
        cases = [Challenge(str(i), "test", 1, "train", str(i), answer)
                 for i, answer in enumerate(["7", "7", "8", "8", "9", "10"])]
        changed = derange_answers(cases)
        self.assertEqual(sorted(changed), sorted(c.answer for c in cases))
        self.assertTrue(all(a.answer != b for a, b in zip(cases, changed)))
        with self.assertRaisesRegex(ValueError, "derange"):
            derange_answers(cases[:3])

    def test_supervision_masks_entire_prompt(self):
        tokenizer = DummyTokenizer()
        ids, labels = _encode_pair(tokenizer, "Compute 1+1", "2", 256)
        visible = len(ids) - len([c for c in "ANSWER: 2~"])
        self.assertEqual(labels[:visible], [-100] * visible)
        self.assertEqual(labels[visible:], ids[visible:])
        with self.assertRaisesRegex(ValueError, "token budget"):
            _encode_pair(tokenizer, "X" * 200, "2", 64)

    def test_semantic_preflight_refuses_collapsed_numeric_labels(self):
        cases = [Challenge("case1", "Compute 1+2", 1, "train", "add", "3"),
                 Challenge("case2", "Compute 2+2", 1, "train", "add", "4")]
        wrong = ["4", "3"]
        tokenizer = DummyTokenizer()
        self.assertEqual(validate_tokenized_answer_discrimination(
            tokenizer, cases, wrong, 256), 2)

        class NumericCollapse(DummyTokenizer):
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": ([1] if add_special_tokens else [])
                        + [ord("0") if c.isascii() and c.isdigit() else ord(c)
                           for c in text]}

        with self.assertRaisesRegex(ValueError, "tokenized.*collide"):
            validate_tokenized_answer_discrimination(NumericCollapse(), cases, wrong, 256)

        with self.assertRaisesRegex(ValueError, "mismatched"):
            validate_tokenized_answer_discrimination(tokenizer, cases, wrong[:1], 256)

    def test_no_torch_import_neither_creates_nor_changes_run_dir(self):
        with patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError("optional deps unavailable")):
            with self.assertRaisesRegex(RuntimeError, "optional deps"):
                run_pilot(self.spec())
        self.assertFalse(self.output.exists())

    def test_pilot_rejects_base_weight_replacement_during_initial_model_load(self):
        class FakeTorch:
            @staticmethod
            def set_num_threads(value):
                pass

            @staticmethod
            def get_num_threads():
                return 1

            @staticmethod
            def manual_seed(value):
                pass

            @staticmethod
            def use_deterministic_algorithms(value):
                pass

        for change in ("replace-existing", "add-new-shard"):
            with self.subTest(change=change):
                (self.model / "model.safetensors").write_bytes(b"small-fixture-not-real-weights")
                extra = self.model / "extra.safetensors"
                if extra.exists():
                    extra.unlink()

                def swapped_load(*args, **kwargs):
                    if change == "replace-existing":
                        (self.model / "model.safetensors").write_bytes(b"tampered-after-initial-snapshot")
                    else:
                        extra.write_bytes(b"new-unpinned-weight")
                    return object(), object()

                with patch("labyrinth.hf_pilot._deps", return_value=(FakeTorch, None, None, (None, None, None), "cpu")):
                    with patch("labyrinth.hf_pilot._load_base", side_effect=swapped_load):
                        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
                            with patch("labyrinth.hf_pilot._score_model", side_effect=AssertionError("model inference should not run")):
                                with self.assertRaisesRegex(RuntimeError, "base model.*changed"):
                                    run_pilot(self.spec())
                self.assertFalse(self.output.exists())

    def test_tokenizer_metadata_symlink_rejected(self):
        external = Path(self.temp.name) / "other"
        external.write_text("read me?", encoding="utf-8")
        try:
            (self.model / "tokenizer.json").symlink_to(external)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            validate_config(self.spec())

    def test_sharded_weight_index_rejects_parent_traversal_before_heavy_import(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 123},
            "weight_map": {"model.layer": "../outside.safetensors"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index"):
            validate_config(self.spec())

    def test_valid_local_sharded_index_is_included_in_model_metadata_hash(self):
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": 123},
            "weight_map": {"model.layer": "model.safetensors"},
        }), encoding="utf-8")
        validate_config(self.spec())
        before = _metadata_digest(self.model)
        self.assertIn(index.name, before)
        index.write_text(json.dumps({
            "metadata": {"total_size": 124},
            "weight_map": {"model.layer": "model.safetensors"},
        }), encoding="utf-8")
        self.assertNotEqual(before[index.name], _metadata_digest(self.model)[index.name])

    def test_reload_refuses_symlink_adapter_root_before_loading_model(self):
        external = Path(self.temp.name) / "outside_adapter"
        external.mkdir()
        (external / "adapter_model.safetensors").write_bytes(b"fake-adapter")
        self.output.mkdir()
        try:
            (self.output / "candidate_adapter").symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        from labyrinth.hf_pilot import _prompt_commitment, training_cases
        from labyrinth.runner import _source_state

        config = self.spec()
        _, heldout = training_cases(config)
        report = {
            "schema": SCHEMA,
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": {
                "adapter_model.safetensors": hashlib.sha256(b"fake-adapter").hexdigest()
            },
            "heldout_case_ids": [case.id for case in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        }
        (self.output / "pilot_report.json").write_text(json.dumps(report), encoding="utf-8")
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("loaded heavy dependencies")):
            with self.assertRaisesRegex(ValueError, "symlink"):
                reload_evaluate(self.output, self.output / "verify.json")

    def test_reload_detects_base_weight_changes_while_loading(self):
        from labyrinth.hf_pilot import _prompt_commitment, training_cases
        from labyrinth.runner import _source_state

        self.output.mkdir()
        candidate = self.output / "candidate_adapter"
        candidate.mkdir()
        (candidate / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
        config = self.spec()
        _, heldout = training_cases(config)
        (self.output / "pilot_report.json").write_text(json.dumps({
            "schema": SCHEMA,
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": {
                "adapter_model.safetensors": hashlib.sha256(b"fixture-adapter").hexdigest()},
            "heldout_case_ids": [c.id for c in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        }), encoding="utf-8")

        def swapped_load(*args, **kwargs):
            (self.model / "model.safetensors").write_bytes(b"swapped-during-reloader")
            return object(), object()

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, None, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", side_effect=swapped_load):
                with self.assertRaisesRegex(RuntimeError, "base model.*changed"):
                    reload_evaluate(self.output, self.output / "verify.json")
        self.assertFalse((self.output / "verify.json").exists())

    def _write_reload_adapter_fixture(self):
        from labyrinth.hf_pilot import _prompt_commitment
        from labyrinth.runner import _source_state

        self.output.mkdir()
        candidate = self.output / "candidate_adapter"
        candidate.mkdir()
        (candidate / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
        config_file = candidate / "adapter_config.json"
        config_file.write_text('{"lora_alpha":8}', encoding="utf-8")
        config = self.spec()
        _, heldout = training_cases(config)
        adapter_sha256 = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in candidate.iterdir()}
        (self.output / "pilot_report.json").write_text(json.dumps({
            "schema": SCHEMA,
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": adapter_sha256,
            "candidate_lora_tensor_value_sha256": "pretend-lora-digest",
            "heldout_case_ids": [case.id for case in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
            "evaluations": {"candidate": {"trials": []}},
        }), encoding="utf-8")
        return candidate, config_file

    def test_reload_refuses_adapter_config_swap_during_peft_load(self):
        candidate, config_file = self._write_reload_adapter_fixture()

        class ReplacingPeft:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                config_file.write_text('{"lora_alpha":64}', encoding="utf-8")
                return object()

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, ReplacingPeft, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", return_value=(object(), object())):
                with patch("labyrinth.hf_pilot._lora_value_digest", return_value="pretend-lora-digest"):
                    with patch("labyrinth.hf_pilot._score_model", side_effect=AssertionError("tampered adapter must not be evaluated")):
                        with self.assertRaisesRegex(RuntimeError, "adapter.*changed"):
                            reload_evaluate(self.output, self.output / "verify.json")
        self.assertFalse((self.output / "verify.json").exists())

    def test_reload_refuses_new_adapter_file_during_evaluation(self):
        candidate, config_file = self._write_reload_adapter_fixture()

        class UnchangedPeft:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        def changed_during_score(*args, **kwargs):
            (candidate / "unexpected.safetensors").write_bytes(b"injected")
            return {"trials": []}

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, UnchangedPeft, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", return_value=(object(), object())):
                with patch("labyrinth.hf_pilot._lora_value_digest", return_value="pretend-lora-digest"):
                    with patch("labyrinth.hf_pilot._score_model", side_effect=changed_during_score):
                        with self.assertRaisesRegex(RuntimeError, "adapter.*changed"):
                            reload_evaluate(self.output, self.output / "verify.json")
        self.assertFalse((self.output / "verify.json").exists())

    def test_reload_refuses_nested_adapter_directory_symlink_before_dependencies(self):
        candidate, _ = self._write_reload_adapter_fixture()
        outside = Path(self.temp.name) / "outside_directory"
        outside.mkdir()
        try:
            (candidate / "nested_link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("heavy deps must not load")):
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                reload_evaluate(self.output, self.output / "verify.json")
        self.assertFalse((self.output / "verify.json").exists())

    def test_unknown_report_fails_before_dependency_loading(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(json.dumps({"schema": "invalid"}))
        with self.assertRaisesRegex(ValueError, "schema"):
            reload_evaluate(self.output, self.output / "verify.json")

    def test_oversized_report_fails_before_dependency_loading(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text('{"padding":"' + "x" * 128 + '"}', encoding="utf-8")
        with patch("labyrinth.runner._MAX_JSON_BYTES", 64):
            with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("loaded heavy dependencies")):
                with self.assertRaisesRegex(ValueError, "JSON input exceeds"):
                    reload_evaluate(self.output, self.output / "verify.json")

    def test_module_help_does_not_import_optional_packages(self):
        result = subprocess.run([sys.executable, "-m", "labyrinth.hf_pilot", "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("train", result.stdout)
        self.assertIn("eval", result.stdout)

    def test_top_level_hf_cli_delegates_without_optional_packages(self):
        result = subprocess.run([sys.executable, "-m", "labyrinth", "train-hf", "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("--model-path", result.stdout)

    def test_tiny_fixture_help_does_not_import_optional_packages(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "make_tiny_hf_fixture.py"
        result = subprocess.run([sys.executable, str(script), "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("--output-dir", result.stdout)

    def test_cuda_preflight_fails_closed_for_small_free_vram(self):
        class TorchWithMemory:
            class cuda:
                @staticmethod
                def mem_get_info():
                    return 128 * 1024 * 1024, 4 * 1024 * 1024 * 1024

        weights = [self.model / "model.safetensors"]
        self.assertIsNone(_device_headroom(TorchWithMemory, "cpu", weights))
        with self.assertRaisesRegex(RuntimeError, "CUDA memory"):
            _device_headroom(TorchWithMemory, "cuda", weights)

    def test_frozen_reload_lora_values_are_still_hashed(self):
        class Tensor:
            requires_grad = False
            shape = (1,)
            dtype = "float32"

            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def flatten(self):
                return self

            def view(self, dtype):
                return self

            def numpy(self):
                return self

            def tobytes(self):
                return self.value

        class Model:
            def __init__(self, value):
                self.value = value

            def named_parameters(self):
                return iter([("base_model.layer.lora_A.default.weight", Tensor(self.value))])

        class FakeTorch:
            uint8 = object()

        self.assertNotEqual(_lora_value_digest(Model(b"before"), FakeTorch),
                            _lora_value_digest(Model(b"after"), FakeTorch))

    def test_format_rate_is_not_conflated_with_accuracy(self):
        trials = [
            {"correct": False, "score_reason": "invalid_format"},
            {"correct": False, "score_reason": "incorrect"},
        ]
        summary = _summarize_trials(trials)
        self.assertEqual((summary["correct"], summary["format_valid"]), (0, 1))
        self.assertEqual((summary["accuracy"], summary["format_rate"]), (0.0, 0.5))
        with self.assertRaises(ValueError):
            _summarize_trials([])

    def test_training_probe_separates_oracle_and_corrupted_label_memorization(self):
        case = Challenge("train-example", "arithmetic", 1, "train", "Calculate 3 + 4.", "7")
        response = "ANSWER: 2"
        trial = _trial_record(case, response, training_target="2")
        self.assertFalse(trial["correct"])
        self.assertTrue(trial["training_target_correct"])
        self.assertEqual(trial["score_reason"], "incorrect")
        summary = _summarize_trials([trial])
        self.assertEqual(summary["correct"], 0)
        self.assertEqual(summary["training_target_hits"], 1)
        self.assertEqual(summary["format_valid"], 1)

    def test_training_target_probe_refuses_mixed_or_malformed_provenance(self):
        case = Challenge("train-example", "arithmetic", 1, "train", "Calculate 3 + 4.", "7")
        with self.assertRaisesRegex(ValueError, "training target"):
            _trial_record(case, "ANSWER: 2", training_target=7)
        trial = _trial_record(case, "ANSWER: 7", training_target="2")
        del trial["training_target_correct"]
        with self.assertRaisesRegex(ValueError, "mixed training target"):
            _summarize_trials([trial, _trial_record(case, "ANSWER: 2", training_target="2")])

    def test_training_probe_uses_actual_generated_text_once_for_both_targets(self):
        class FakeTensor:
            shape = (1, 2)

            def to(self, device):
                return self

            def __getitem__(self, index):
                return self

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 3

            def __call__(self, prompt, *, add_special_tokens, return_tensors):
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

            def decode(self, generated, *, skip_special_tokens):
                return "ANSWER: 2"

        class FakeModel:
            generated = 0

            def eval(self):
                pass

            def generate(self, **kwargs):
                self.generated += 1
                return FakeTensor()

        class FakeTorch:
            no_grad = staticmethod(nullcontext)

        case = Challenge("seen", "arithmetic", 1, "train", "Calculate 3 + 4.", "7")
        model = FakeModel()
        outcome = _score_model(model, FakeTokenizer(), FakeTorch(), [case],
                               device="cpu", max_tokens=128, max_new_tokens=8,
                               training_targets=["2"])
        self.assertEqual(model.generated, 1)
        self.assertEqual((outcome["correct"], outcome["training_target_hits"]), (0, 1))
        with self.assertRaisesRegex(ValueError, "training target count"):
            _score_model(model, FakeTokenizer(), FakeTorch(), [case],
                         device="cpu", max_tokens=128, max_new_tokens=8,
                         training_targets=[])


if __name__ == "__main__":
    unittest.main()
