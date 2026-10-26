"""Dependency-free contract tests; these do not claim that a HF model trained."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import (
    PilotConfig,
    _device_headroom,
    _encode_pair,
    derange_answers,
    reload_evaluate,
    run_pilot,
    training_cases,
    validate_config,
)
from labyrinth.maze import Challenge


class DummyTokenizer:
    eos_token = "~"

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [1] + [ord(c) for c in text] if add_special_tokens
                else [ord(c) for c in text]}


class HfPilotContractTests(unittest.TestCase):
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

    def test_no_torch_import_neither_creates_nor_changes_run_dir(self):
        with patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError("optional deps unavailable")):
            with self.assertRaisesRegex(RuntimeError, "optional deps"):
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

    def test_unknown_report_fails_before_dependency_loading(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(json.dumps({"schema": "invalid"}))
        with self.assertRaisesRegex(ValueError, "schema"):
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


if __name__ == "__main__":
    unittest.main()
