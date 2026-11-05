"""Local-byte provenance and hostile-path checks for cached pretrained bases."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path
import tempfile
import unittest

from labyrinth.pretrained_model_provenance import (
    ModelProvenanceError,
    capture_pretrained_model_manifest,
    verify_pretrained_model_manifest,
)


def _fixture(root: Path) -> None:
    (root / "config.json").write_bytes(b'{"model_type":"bert"}\n')
    (root / "tokenizer.json").write_bytes(b'{"tokenizer":"fixture"}\n')
    (root / "model.safetensors").write_bytes(b"fixture-pretrained-weights")
    (root / "1_Pooling").mkdir()
    (root / "1_Pooling" / "config.json").write_bytes(b'{"pooling":true}\n')
    (root / "README.md").write_text("Untrusted descriptive metadata\n")


class PretrainedModelProvenanceTests(unittest.TestCase):
    def test_deterministic_canonical_relative_byte_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            first = capture_pretrained_model_manifest(
                root, model_id_claim="sentence-transformers/all-MiniLM-L6-v2",
                revision_claim="main",
            )
            second = capture_pretrained_model_manifest(
                root, model_id_claim="sentence-transformers/all-MiniLM-L6-v2",
                revision_claim="main",
            )
            self.assertEqual(first, second)
            self.assertEqual(first["evidence_class"], "LOCAL_BYTES_ONLY")
            self.assertFalse(first["origin_attested"])
            self.assertFalse(first["graduation_authorized"])
            self.assertEqual(first["cache_revision_hint"], None)
            names = [entry["path"] for entry in first["files"]]
            self.assertEqual(names, [
                "1_Pooling/config.json", "config.json", "model.safetensors",
                "tokenizer.json",
            ])
            self.assertEqual(
                first["files"][2]["sha256"],
                hashlib.sha256(b"fixture-pretrained-weights").hexdigest(),
            )
            self.assertNotIn(str(root), str(first))
            self.assertEqual(
                verify_pretrained_model_manifest(
                    root, first,
                    trusted_manifest_sha256=first["manifest_sha256"],
                ), first,
            )

    def test_changed_or_missing_bytes_and_recomputed_self_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            original = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            trusted = original["manifest_sha256"]
            (root / "model.safetensors").write_bytes(b"different-weights")
            with self.assertRaisesRegex(ModelProvenanceError, "bytes|inventory"):
                verify_pretrained_model_manifest(
                    root, original, trusted_manifest_sha256=trusted,
                )
            changed = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            with self.assertRaisesRegex(ModelProvenanceError, "trusted"):
                verify_pretrained_model_manifest(
                    root, changed, trusted_manifest_sha256=trusted,
                )
            (root / "tokenizer.json").unlink()
            with self.assertRaisesRegex(ModelProvenanceError, "tokenizer"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")

    def test_rejects_symlinked_file_directory_and_model_root(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "model"
            root.mkdir()
            _fixture(root)
            external = base / "external.safetensors"
            external.write_bytes(b"outside")
            linked = root / "model-00001-of-00001.safetensors"
            try:
                linked.symlink_to(external)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable on this platform")
            with self.assertRaisesRegex(ModelProvenanceError, "symlink|junction"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            linked.unlink()
            (root / "linked_dir").symlink_to(base, target_is_directory=True)
            with self.assertRaisesRegex(ModelProvenanceError, "symlink|junction"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            (root / "linked_dir").unlink()
            alias = base / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ModelProvenanceError, "symlink|junction"):
                capture_pretrained_model_manifest(alias, model_id_claim="org/model")

    def test_rejects_missing_weight_and_config_or_unbounded_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            (root / "model.safetensors").unlink()
            with self.assertRaisesRegex(ModelProvenanceError, "safetensors"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            (root / "model.safetensors").write_bytes(b"model")
            (root / "config.json").unlink()
            with self.assertRaisesRegex(ModelProvenanceError, "config"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            for bad in ("", "../model", "a\\b", "a/../b", "bad id", "a//b"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    capture_pretrained_model_manifest(root, model_id_claim=bad)

    def test_cache_snapshot_revision_is_path_hint_not_attestation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models--org--model" / "snapshots" / ("a" * 40)
            root.mkdir(parents=True)
            _fixture(root)
            result = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            self.assertEqual(result["cache_revision_hint"], "a" * 40)
            self.assertFalse(result["origin_attested"])

    def test_incomplete_and_ambiguous_weight_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            (root / "model.safetensors").unlink()
            (root / "model-00001-of-00002.safetensors").write_bytes(b"shard-a")
            with self.assertRaisesRegex(ModelProvenanceError, "incomplete"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            (root / "model-00002-of-00002.safetensors").write_bytes(b"shard-b")
            complete = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            self.assertEqual(len([f for f in complete["files"] if f["kind"] == "weights"]), 2)
            (root / "model.safetensors").write_bytes(b"ambiguous-single-file")
            with self.assertRaisesRegex(ModelProvenanceError, "ambiguous"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")

    def test_malformed_untrusted_manifest_never_grants_origin_or_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            expected = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            trusted = expected["manifest_sha256"]
            for key, bad_value in (
                ("origin_attested", True),
                ("graduation_authorized", True),
                ("file_count", True),
                ("files", "not-a-file-list"),
                ("manifest_sha256", "not-a-sha"),
            ):
                with self.subTest(key=key):
                    changed = copy.deepcopy(expected)
                    changed[key] = bad_value
                    with self.assertRaises(ModelProvenanceError):
                        verify_pretrained_model_manifest(root, changed,
                                                         trusted_manifest_sha256=trusted)
            for path in ("../config.json", "sub\\model.safetensors", "/absolute/config.json"):
                with self.subTest(path=path):
                    changed = copy.deepcopy(expected)
                    changed["files"][0]["path"] = path
                    with self.assertRaises(ModelProvenanceError):
                        verify_pretrained_model_manifest(root, changed,
                                                         trusted_manifest_sha256=trusted)
            changed = copy.deepcopy(expected)
            changed["files"].append(copy.deepcopy(changed["files"][0]))
            changed["file_count"] += 1
            changed["total_bytes"] += changed["files"][-1]["size"]
            with self.assertRaises(ModelProvenanceError):
                verify_pretrained_model_manifest(root, changed,
                                                 trusted_manifest_sha256=trusted)

    def test_decoy_nested_safetensors_cannot_hide_loadable_root_bin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_bytes(b'{}')
            (root / "vocab.txt").write_bytes(b'hello\\n')
            (root / "pytorch_model.bin").write_bytes(b'actual-unmeasured-weights')
            (root / "decoy").mkdir()
            (root / "decoy" / "model.safetensors").write_bytes(b'decoy')
            with self.assertRaisesRegex(ModelProvenanceError, "unsupported|weight|model"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            (root / "pytorch_model.bin").unlink()
            with self.assertRaisesRegex(ModelProvenanceError, "weight|folder|location"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")

    def test_unselected_loadable_weight_candidates_are_not_silently_ignored(self):
        for name in (
            "pytorch_model.bin", "pytorch_model.bin.index.json", "model.gguf",
            "adapter_model.safetensors", "adapter_model.bin", "tf_model.h5",
            "flax_model.msgpack", "model.onnx", "model.pt", "model.pth",
            "model.ckpt", "model.pdparams", "rust_model.ot",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _fixture(root)
                (root / name).write_bytes(b'other potentially loadable weights')
                with self.assertRaisesRegex(ModelProvenanceError, "unsupported|weight"):
                    capture_pretrained_model_manifest(root, model_id_claim="org/model")

    def test_only_one_supported_canonical_weight_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root)
            (root / "0_Transformer").mkdir()
            (root / "0_Transformer" / "model.safetensors").write_bytes(b'nested')
            with self.assertRaisesRegex(ModelProvenanceError, "ambiguous|multiple"):
                capture_pretrained_model_manifest(root, model_id_claim="org/model")
            (root / "model.safetensors").unlink()
            manifest = capture_pretrained_model_manifest(root, model_id_claim="org/model")
            self.assertIn("0_Transformer/model.safetensors",
                          [entry["path"] for entry in manifest["files"]])


if __name__ == "__main__":
    unittest.main()
