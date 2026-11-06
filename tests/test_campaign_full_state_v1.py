"""Full physical campaign ancestry is distinct from development exam success."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch


class CampaignFullStateTests(unittest.TestCase):
    @staticmethod
    def _tensor(name: str, value: bytes) -> bytes:
        header = json.dumps({name: {
            "dtype": "U8", "shape": [len(value)],
            "data_offsets": [0, len(value)],
        }}, separators=(",", ":")).encode()
        return struct.pack("<Q", len(header)) + header + value

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {name: root / name for name in (
            "base", "adapter", "optimizer", "scheduler", "rng",
        )}
        for directory in paths.values():
            directory.mkdir(parents=True)
        (paths["base"] / "config.json").write_text('{"model_type":"qwen3"}')
        (paths["base"] / "tokenizer.json").write_text('{"tokens":["A"]}')
        (paths["base"] / "model-00001-of-00002.safetensors").write_bytes(
            self._tensor("model.embed_tokens.weight", b"shard1"))
        (paths["base"] / "model-00002-of-00002.safetensors").write_bytes(
            self._tensor("lm_head.weight", b"shard2"))
        (paths["base"] / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }))
        (paths["adapter"] / "adapter_config.json").write_text('{"r":8}')
        (paths["adapter"] / "adapter_model.safetensors").write_bytes(
            self._tensor("lora_A.weight", b"adapter W0"))
        (paths["optimizer"] / "optimizer.pt").write_bytes(b"Adam state W0")
        (paths["scheduler"] / "scheduler.pt").write_bytes(b"scheduler state W0")
        (paths["rng"] / "rng_state.pt").write_bytes(b"random state W0")
        paths["data_cursor"] = root / "cursor.json"
        paths["data_cursor"].write_text('{"epoch":0,"step":0}')
        return paths

    def _capture(self, paths: dict[str, Path], **extra):
        from labyrinth.campaign_full_state_v1 import capture_campaign_state

        return capture_campaign_state(
            base_dir=paths["base"], adapter_dir=paths["adapter"],
            optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
            rng_dir=paths["rng"], data_cursor_path=paths["data_cursor"],
            family="hf_safetensors_lora", mutation_kind="lora_adapter", **extra,
        )

    def test_capture_full_sharded_state_and_fresh_process_verify_bytes(self):
        from labyrinth.campaign_full_state_v1 import (
            save_campaign_state_once, verify_campaign_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._fixture(root)
            manifest = self._capture(paths)
            self.assertEqual(manifest["schema"], "labyrinth.campaign-full-state.v1")
            self.assertEqual(manifest["generation_index"], 0)
            self.assertIsNone(manifest["parent_state_sha256"])
            self.assertEqual(len(manifest["base"]["files"]), 5)
            self.assertFalse(manifest["restorable_training_state_proven"])
            destination = root / "W0-manifest.json"
            save_campaign_state_once(manifest, destination)
            report = verify_campaign_state(destination, fresh_process=True)
            self.assertEqual(report["status"], "LOCAL_FILE_READBACK_ONLY")
            self.assertEqual(report["state_sha256"], manifest["state_sha256"])
            self.assertNotEqual(report["pid"], os.getpid())
            self.assertFalse(report["restorable_training_state_proven"])

    @unittest.skipUnless(os.name == "posix", "POSIX directory fsync contract")
    def test_manifest_publication_syncs_parent_after_staging_removed(self):
        from labyrinth.campaign_full_state_v1 import (
            save_campaign_state_once, verify_campaign_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._capture(self._fixture(root))
            destination = root / "W0-manifest.json"
            original_fsync = os.fsync
            seen = []

            def capture_sync(fd):
                is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
                seen.append("directory" if is_directory else "file")
                if is_directory:
                    self.assertEqual(list(root.glob(".campaign-staged-*")), [])
                    self.assertTrue(destination.is_file())
                return original_fsync(fd)

            with patch("labyrinth.campaign_full_state_v1.os.fsync", side_effect=capture_sync):
                receipt = save_campaign_state_once(manifest, destination)
            self.assertEqual(seen, ["file", "directory"])
            self.assertEqual(receipt, manifest["manifest_sha256"])
            self.assertEqual(verify_campaign_state(destination, fresh_process=False)["status"],
                             "LOCAL_FILE_READBACK_ONLY")

            # A write-once path must reject a repeat without modifying content.
            original_bytes = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "new canonical"):
                save_campaign_state_once(manifest, destination)
            self.assertEqual(destination.read_bytes(), original_bytes)

    @unittest.skipUnless(os.name == "posix", "POSIX directory fsync contract")
    def test_directory_fsync_error_withholds_receipt_and_attempts_rollback(self):
        from labyrinth.campaign_full_state_v1 import save_campaign_state_once

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._capture(self._fixture(root))
            destination = root / "unacknowledged.json"
            original_fsync = os.fsync
            seen = []

            def fail_first_directory_sync(fd):
                is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
                seen.append("directory" if is_directory else "file")
                if is_directory and seen.count("directory") == 1:
                    raise OSError("injected parent directory fsync failure")
                return original_fsync(fd)

            with patch("labyrinth.campaign_full_state_v1.os.fsync",
                       side_effect=fail_first_directory_sync):
                with self.assertRaisesRegex(OSError, "injected parent"):
                    save_campaign_state_once(manifest, destination)
            self.assertEqual(seen, ["file", "directory", "directory"])
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".campaign-staged-*")), [])

    def test_two_generation_branching_and_selected_entry_require_exact_full_state(self):
        from labyrinth.campaign_full_state_v1 import (
            save_campaign_state_once, verify_generation_ancestry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w0 = self._fixture(root / "w0")
            parent = self._capture(w0)
            parent_path = root / "w0.json"
            save_campaign_state_once(parent, parent_path)
            children = []
            for name, delta in (("left", b"left-trained"), ("right", b"right-trained")):
                paths = self._fixture(root / name)
                (paths["adapter"] / "adapter_model.safetensors").write_bytes(
                    self._tensor("lora_A.weight", delta))
                (paths["optimizer"] / "optimizer.pt").write_bytes(b"optimizer" + delta)
                (paths["data_cursor"]).write_text('{"epoch":0,"step":1}')
                manifest = self._capture(paths, parent_manifest=parent)
                path = root / f"{name}.json"
                save_campaign_state_once(manifest, path)
                children.append((path, manifest))

            winning_path, winning = children[1]
            successor_paths = self._fixture(root / "gen2_entry")
            for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
                shutil.rmtree(successor_paths[role])
                shutil.copytree(root / "right" / role, successor_paths[role])
            shutil.copyfile(root / "right" / "cursor.json", successor_paths["data_cursor"])
            next_entry = self._capture(successor_paths, parent_manifest=winning)
            next_path = root / "gen2-entry.json"
            save_campaign_state_once(next_entry, next_path)

            verdict = verify_generation_ancestry(
                parent_path, tuple(path for path, _ in children), winning_path,
                next_path,
            )
            self.assertEqual(verdict["status"], "SOURCE_VERIFIED_LINEAGE_NOT_CAUSAL_TRAINING")
            self.assertEqual(verdict["champion_state_sha256"], winning["state_sha256"])
            self.assertEqual(verdict["next_entry_state_sha256"], winning["state_sha256"])
            self.assertFalse(verdict["neural_update_proven"])

            # A separately captured but valid 'next entry' must not smuggle
            # a different optimizer under identical model+adapter weights.
            false_entry_paths = self._fixture(root / "false-entry")
            for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
                shutil.rmtree(false_entry_paths[role])
                shutil.copytree(root / "right" / role, false_entry_paths[role])
            shutil.copyfile(root / "right" / "cursor.json", false_entry_paths["data_cursor"])
            (false_entry_paths["optimizer"] / "optimizer.pt").write_bytes(
                b"different optimizer with same model weights")
            false_entry = self._capture(false_entry_paths, parent_manifest=winning)
            false_path = root / "false-entry.json"
            save_campaign_state_once(false_entry, false_path)
            with self.assertRaisesRegex(ValueError, "exact selected champion"):
                verify_generation_ancestry(
                    parent_path, tuple(path for path, _ in children), winning_path,
                    false_path,
                )

            # Fresh-process readback must catch one-byte optimizer tampering,
            # even when all weight files and candidate ancestry still match.
            (successor_paths["optimizer"] / "optimizer.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "readback|physical"):
                verify_generation_ancestry(
                    parent_path, tuple(path for path, _ in children), winning_path,
                    next_path,
                )

    def test_rejects_missing_state_unsupported_families_and_broken_shard_index(self):
        from labyrinth.campaign_full_state_v1 import capture_campaign_state

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            arguments = dict(
                base_dir=paths["base"], adapter_dir=paths["adapter"],
                optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
                rng_dir=paths["rng"], data_cursor_path=paths["data_cursor"],
                family="hf_safetensors_lora", mutation_kind="lora_adapter",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                capture_campaign_state(**{**arguments, "family": "gguf-unknown"})
            with self.assertRaisesRegex(ValueError, "required"):
                capture_campaign_state(**{**arguments, "rng_dir": None})
            with self.assertRaisesRegex(ValueError, "data cursor"):
                capture_campaign_state(**{**arguments, "data_cursor_path": None})
            second_shard = paths["base"] / "model-00002-of-00002.safetensors"
            second_shard.unlink()
            with self.assertRaisesRegex(ValueError, "missing|unindexed"):
                capture_campaign_state(**arguments)

    def test_full_weight_edit_is_typed_not_a_fake_lora(self):
        from labyrinth.campaign_full_state_v1 import capture_campaign_state

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            manifest = capture_campaign_state(
                base_dir=paths["base"], adapter_dir=None,
                optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
                rng_dir=paths["rng"], data_cursor_path=paths["data_cursor"],
                family="hf_safetensors_full", mutation_kind="full_weight_edit",
            )
            self.assertIsNone(manifest["adapter"])
            self.assertEqual(manifest["mutation_kind"], "full_weight_edit")
            with self.assertRaisesRegex(ValueError, "omit"):
                capture_campaign_state(
                    base_dir=paths["base"], adapter_dir=paths["adapter"],
                    optimizer_dir=paths["optimizer"], scheduler_dir=paths["scheduler"],
                    rng_dir=paths["rng"], data_cursor_path=paths["data_cursor"],
                    family="hf_safetensors_full", mutation_kind="full_weight_edit",
                )

    def test_real_optimizer_state_can_exceed_small_entrance_auxiliary_budget(self):
        """A 4B-scale optimizer is not a 512 MiB 'small auxiliary' file."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            with (paths["optimizer"] / "optimizer.pt").open("wb") as output:
                output.truncate(512 * 1024 * 1024 + 1)
            manifest = self._capture(paths)
            self.assertGreater(
                manifest["optimizer"]["files"]["optimizer.pt"]["bytes"],
                512 * 1024 * 1024,
            )

    def test_random_bytes_with_safetensors_extension_are_not_model_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            (paths["adapter"] / "adapter_model.safetensors").write_bytes(b"not a tensor")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                self._capture(paths)


if __name__ == "__main__":
    unittest.main()
