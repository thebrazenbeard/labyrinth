"""Real-byte and fresh-process checks for source-only EntranceState W0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.entrance_state_v1 import (
    build_entrance_state, compare_adapter_to_w0, save_entrance_state_once,
    verify_entrance_state,
)


class EntranceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base = root / "base"
        self.base.mkdir()
        self.adapter = root / "adapter"
        self.adapter.mkdir()
        (self.base / "model.safetensors").write_bytes(b"test tensor bytes W0")
        (self.base / "config.json").write_text('{"model_type":"tiny"}')
        (self.base / "tokenizer.json").write_text('{"version":"test"}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"tiny adapter W0")
        (self.adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        self.receipt = root / "source-receipt.json"
        self.receipt.write_bytes(b'{"source":"local-dev"}')
        self.output = root / "entrance.json"
        self.prompt = "Calculate 3 + 4."
        self.observations = [{
            "case_id": "devbaseline-0", "skill": "arithmetic",
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "score": 0, "max_score": 1, "scorer": "local-arithmetic-oracle",
        }]

    def _capture(self):
        return build_entrance_state(
            base_dir=self.base, adapter_dir=self.adapter,
            scope_sha256="a" * 64, source_tree_sha256="b" * 64,
            source_receipt_path=self.receipt,
            source_receipt_sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
            baseline_observations=self.observations,
        )

    def test_w0_bytes_and_train_safe_behavior_are_separate_and_readable_fresh_process(self):
        entrance = self._capture()
        self.assertEqual(entrance["training_safe_baseline"]["observations"][0]["score"], 0)
        self.assertFalse(entrance["neural_improvement_proven"])
        self.assertEqual(entrance["state_gaps"], [
            "OPTIMIZER_NOT_CAPTURED", "SCHEDULER_NOT_CAPTURED", "RNG_NOT_CAPTURED",
        ])
        save_entrance_state_once(entrance, self.output)
        with self.assertRaises(FileExistsError):
            save_entrance_state_once(entrance, self.output)
        process = subprocess.run(
            [sys.executable, "-m", "labyrinth.entrance_state_v1", "verify",
             "--manifest", str(self.output)], capture_output=True, text=True,
            check=True,
        )
        observed = json.loads(process.stdout)
        self.assertEqual(observed["status"], "LOCAL_BYTES_MATCH_ENTRANCE")
        self.assertFalse(observed["full_resume_state_available"])
        self.assertFalse(observed["graduation_authorized"])

    def test_mutation_and_symlink_are_rejected_without_overwriting_state(self):
        entrance = self._capture()
        save_entrance_state_once(entrance, self.output)
        (self.adapter / "adapter_model.safetensors").write_bytes(b"mutated model")
        with self.assertRaisesRegex(ValueError, "adapter bytes changed"):
            verify_entrance_state(self.output)
        (self.adapter / "adapter_model.safetensors").unlink()
        (self.adapter / "adapter_model.safetensors").symlink_to(self.base / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "symlink"):
            build_entrance_state(
                base_dir=self.base, adapter_dir=self.adapter,
                scope_sha256="a" * 64, source_tree_sha256="b" * 64,
                source_receipt_path=self.receipt,
                source_receipt_sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
                baseline_observations=self.observations,
            )

    def test_adapter_delta_is_not_mistaken_for_learning_or_capability_improvement(self):
        entrance = self._capture()
        successor = self.adapter.parent / "successor"
        successor.mkdir()
        (successor / "adapter_model.safetensors").write_bytes(b"changed successor")
        (successor / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        observed = compare_adapter_to_w0(entrance, successor)
        self.assertTrue(observed["bytes_changed"])
        self.assertFalse(observed["learning_demonstrated"])

    def test_declared_auxiliary_bytes_do_not_prove_restorable_state(self):
        # Arbitrary files can be hashed successfully. That does NOT establish
        # a loadable optimizer/scheduler/RNG state or an exact trainer resume.
        root = self.base.parent
        kwargs = {}
        for name in ("optimizer", "scheduler", "rng"):
            directory = root / name
            directory.mkdir()
            (directory / "junk.bin").write_bytes(b"x")
            kwargs[name + "_dir"] = directory
        entrance = build_entrance_state(
            base_dir=self.base, adapter_dir=self.adapter,
            scope_sha256="a" * 64, source_tree_sha256="b" * 64,
            source_receipt_path=self.receipt,
            source_receipt_sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
            baseline_observations=self.observations, **kwargs,
        )
        self.assertEqual(entrance["state_gaps"], [])
        self.assertNotEqual(entrance["w0_state_sha256"],
                            entrance["w0_captured_training_state_sha256"])
        save_entrance_state_once(entrance, self.output)
        result = verify_entrance_state(self.output)
        self.assertTrue(result["all_declared_auxiliary_files_captured"])
        self.assertFalse(result["full_resume_state_available"])
        (kwargs["optimizer_dir"] / "junk.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "optimizer state changed"):
            verify_entrance_state(self.output)

    def test_private_exam_cases_and_fake_scores_cannot_enter_train_safe_baseline(self):
        original = self.observations
        for field, changed in (
            ("case_id", "private-final-0"), ("score", True),
            ("prompt_sha256", "f" * 64), ("max_score", 0),
        ):
            self.observations = [{**original[0], field: changed}]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._capture()
        self.observations = original

    def test_rehashed_manifest_cannot_swap_canonical_receipt_path_for_symlinked_parent(self):
        from labyrinth.entrance_state_v1 import _manifest_hash
        entrance = self._capture()
        save_entrance_state_once(entrance, self.output)
        alias = self.base.parent / "alias-parent"
        alias.symlink_to(self.base.parent, target_is_directory=True)
        forged = json.loads(self.output.read_text())
        forged["source_receipt_path"] = str(alias / self.receipt.name)
        forged["manifest_sha256"] = _manifest_hash(forged)
        self.output.write_text(json.dumps(forged))
        with self.assertRaisesRegex(ValueError, "not canonical"):
            verify_entrance_state(self.output)


if __name__ == "__main__":
    unittest.main()
