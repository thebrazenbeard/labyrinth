"""GGUF full-byte read-only provenance, v3 metadata and LoRA pairing claims."""

from hashlib import sha256
from pathlib import Path
import struct
import tempfile
import unittest

from labyrinth.gguf_entrance_fingerprint_v1 import (
    GGUFEntranceError, capture_gguf_entrance, fingerprint_gguf_file,
    verify_gguf_entrance,
)


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(metadata: list[tuple[str, int, object]], *,
          tensor_name: str = "blk.0.attn_q.weight", version: int = 3) -> bytes:
    chunks = [b"GGUF", struct.pack("<IQQ", version, 1, len(metadata))]
    for key, kind, value in metadata:
        chunks.append(_string(key) + struct.pack("<I", kind))
        if kind == 8:
            chunks.append(_string(value))
        elif kind == 4:
            chunks.append(struct.pack("<I", value))
        elif kind == 9:
            # ARRAY of GGUF STRING values, including token metadata.
            chunks.append(struct.pack("<IQ", 8, len(value)))
            chunks.extend(_string(word) for word in value)
        else:
            raise AssertionError("test fixture does not implement metadata kind")
    chunks.append(_string(tensor_name) + struct.pack("<IQQIQ", 2, 2, 2, 0, 0))
    header = b"".join(chunks)
    return header + b"\x00" * ((-len(header)) % 32) + b"0123456789ABCDEF"


class GGUFEntranceFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root / "model_q5_k_s.gguf"
        self.adapter = self.root / "vera_lora.gguf"
        self.base.write_bytes(_gguf([
            ("general.architecture", 8, "qwen3"),
            ("general.name", 8, "Qwen-like fixture, not authentic Qwen"),
            ("general.file_type", 4, 16),
            ("tokenizer.ggml.tokens", 9, ["hello", "world", "not trained"]),
            ("general.base_model.0.name", 8, "claimed origin"),
        ]))
        self.adapter.write_bytes(_gguf([
            ("general.type", 8, "adapter"),
            ("general.architecture", 8, "qwen3"),
            ("adapter.type", 8, "lora"),
        ], tensor_name="blk.0.attn_q.lora_a"))

    def test_base_full_byte_hash_and_optional_adapter_pair_are_only_metadata_claims(self):
        before = self.base.stat().st_mtime_ns
        base_bytes = self.base.read_bytes()
        result = capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.adapter)
        self.assertEqual(result["base"]["bytes"], len(base_bytes))
        self.assertEqual(result["base"]["sha256"], sha256(base_bytes).hexdigest())
        self.assertEqual(result["base"]["format_version"], 3)
        self.assertEqual(result["base"]["tensor_count"], 1)
        self.assertEqual(result["base"]["metadata_count"], 5)
        self.assertEqual(result["base"]["tensor_ggml_types"], {"0": 1})
        self.assertEqual(result["base"]["metadata"]["general.file_type"]["value"], 16)
        self.assertEqual(result["lora_adapter"]["metadata"]["adapter.type"]["value"], "lora")
        self.assertEqual(result["pairing_status"], "DECLARED_ARCHITECTURE_MATCH_ONLY")
        self.assertFalse(result["hf_trainable_genotype_proven"])
        self.assertFalse(result["gguf_model_loaded"])
        self.assertFalse(result["adapter_applied_proven"])
        self.assertFalse(result["model_identity_proven"])
        self.assertFalse(result["neural_improvement_proven"])
        self.assertFalse(result["graduation_authorized"])
        self.assertEqual(self.base.read_bytes(), base_bytes)
        self.assertEqual(self.base.stat().st_mtime_ns, before)
        self.assertEqual(verify_gguf_entrance(result,
                         trusted_manifest_sha256=result["manifest_sha256"]), result)

    def test_separate_origin_commitment_required_and_byte_tamper_rejected(self):
        expected = capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.adapter)
        pinned = expected["manifest_sha256"]
        with self.assertRaisesRegex(GGUFEntranceError, "trusted commitment"):
            verify_gguf_entrance(expected, trusted_manifest_sha256="0"*64)
        self.base.write_bytes(self.base.read_bytes()[:-1] + b"x")
        with self.assertRaisesRegex(GGUFEntranceError, "differs"):
            verify_gguf_entrance(expected, trusted_manifest_sha256=pinned)
        forged = capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.adapter)
        with self.assertRaisesRegex(GGUFEntranceError, "trusted commitment"):
            verify_gguf_entrance(forged, trusted_manifest_sha256=pinned)

    def test_metadata_mismatch_is_not_cryptographic_incompatibility_proof(self):
        with self.assertRaisesRegex(GGUFEntranceError, "same physical file"):
            capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.base)
        self.adapter.write_bytes(_gguf([
            ("general.type", 8, "adapter"),
            ("general.architecture", 8, "llama"),
            ("adapter.type", 8, "lora"),
        ]))
        out = capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.adapter)
        self.assertEqual(out["pairing_status"], "DECLARED_ARCHITECTURE_MISMATCH")
        self.adapter.write_bytes(_gguf([
            ("general.type", 8, "model"),
            ("general.architecture", 8, "qwen3"),
            ("adapter.type", 8, "adapters-not-lora"),
        ]))
        out = capture_gguf_entrance(base_path=self.base, lora_adapter_path=self.adapter)
        self.assertEqual(out["pairing_status"], "UNVERIFIED_ADAPTER_FORMAT")
        self.assertEqual(capture_gguf_entrance(base_path=self.base)["pairing_status"],
                         "NO_ADAPTER")
        self.adapter.write_bytes(_gguf([
            ("general.type", 8, "adapter"),
            ("adapter.type", 8, "lora"),
        ]))
        with self.assertRaisesRegex(GGUFEntranceError, "base role"):
            capture_gguf_entrance(base_path=self.adapter)

    def test_invalid_huge_header_unknown_version_and_symlink_are_refused(self):
        self.base.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 10_000) + b"x"*32)
        with self.assertRaisesRegex(GGUFEntranceError, "bounded"):
            fingerprint_gguf_file(self.base)
        self.base.write_bytes(_gguf([("general.architecture", 8, "qwen3")], version=2))
        with self.assertRaisesRegex(GGUFEntranceError, "GGUF v3"):
            fingerprint_gguf_file(self.base)
        self.base.write_bytes(_gguf([("general.architecture", 8, "qwen3")]))
        link = self.root / "link.gguf"
        try:
            link.symlink_to(self.base)
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaisesRegex(GGUFEntranceError, "symlink"):
                fingerprint_gguf_file(link)

    def test_metadata_nested_array_count_and_offset_are_not_trusted(self):
        malicious = self.root / "evil.gguf"
        metadata = (_string("tokenizer.ggml.tokens") + struct.pack("<I", 9)
                    + struct.pack("<IQ", 8, 1_000_001))
        malicious.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + metadata)
        with self.assertRaisesRegex(GGUFEntranceError, "array subtype|element count"):
            fingerprint_gguf_file(malicious)
        invalid_offsets = bytearray(_gguf([("general.architecture", 8, "qwen3")]))
        # Last 8-byte zero is tensor offset; force it past the file payload.
        # Locate the offset by reconstructing expected header size.
        h = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
        h += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen3")
        h += _string("blk.0.attn_q.weight") + struct.pack("<IQQI", 2, 2, 2, 0)
        struct.pack_into("<Q", invalid_offsets, len(h), 8192)
        malicious.write_bytes(invalid_offsets)
        with self.assertRaisesRegex(GGUFEntranceError, "beyond aligned"):
            fingerprint_gguf_file(malicious)


if __name__ == "__main__":
    unittest.main()
