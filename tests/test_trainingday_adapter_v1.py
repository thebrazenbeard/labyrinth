"""Mock artifact custody never becomes permission to train or graduate Qwen."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.trainingday_adapter_v1 import (
    OFFICIAL_BASE_ID, OFFICIAL_BASE_REVISION, SCHEMA, TRAINER_ENTRYPOINT,
    TRAINER_HEAD, TrainingDayHandoffError, TrainingDayPhysicalExecutionHold,
    _stream_file, inspect_trainingday_handoff, invoke_trainingday_trainer,
)


class TrainingDayAdapterTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        roles = {
            "base": ["base/model.safetensors-00001-of-00002.safetensors",
                     "base/model.safetensors-00002-of-00002.safetensors",
                     "base/config.json", "base/model.safetensors.index.json"],
            "tokenizer": ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"],
            "adapter": ["adapter/adapter_model.safetensors", "adapter/adapter_config.json"],
            "optimizer": ["optimizer/state.pt"],
            "scheduler": ["scheduler/state.json"],
            "rng": ["rng/python.json", "rng/numpy.json", "rng/torch_cpu.json",
                    "rng/torch_cuda.json"],
            "data": ["data/train_cursor.json"],
        }
        bodies = {}
        for names in roles.values():
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                content = ("synthetic physical bytes: " + name).encode()
                path.write_bytes(content)
                bodies[name] = content
        sample_sha = {name: sha256(content).hexdigest() for name, content in bodies.items()}
        for label, name in (("LOCAL_MOCK_WORKER", "receipts/worker.json"),
                            ("LOCAL_MOCK_RELOAD", "receipts/reload.json"),
                            ("LOCAL_MOCK_RESULT", "receipts/result.json")):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {"kind": label, "checkpoint_id": "qwen4b-test-0", "step": 0,
                     "artifact_sha256": sample_sha,
                     "neural_update_verified": False}
            if label == "LOCAL_MOCK_RELOAD":
                value["fresh_process"] = True
            path.write_text(json.dumps(value, sort_keys=True))
            bodies[name] = path.read_bytes()
        manifest = {
            "schema": SCHEMA,
            "source": {"repository": "thebrazenbeard/model-training",
                       "head": TRAINER_HEAD, "entrypoint": TRAINER_ENTRYPOINT,
                       "status": "SOURCE_ONLY_HARD_HOLD"},
            "model": {"id": OFFICIAL_BASE_ID, "revision": OFFICIAL_BASE_REVISION,
                      "loader_class": "Qwen3_5ForCausalLM"},
            "authority": {"physical_training_authorized": False,
                          "network": "disabled", "data_scope": "PUBLIC_TRAIN_ONLY",
                          "budget_usd": 0},
            "checkpoint": {"id": "qwen4b-test-0", "step": 0,
                           "neural_update_verified": False,
                           "full_resume_verified": False,
                           "graduation_authorized": False},
            "components": roles,
            "files": {name: {"sha256": sha256(content).hexdigest(),
                             "bytes": len(content)} for name, content in bodies.items()},
            "worker_receipt": "receipts/worker.json",
            "reload_receipt": "receipts/reload.json",
            "result_receipt": "receipts/result.json",
        }
        (root / "handoff.json").write_text(json.dumps(manifest, sort_keys=True))
        expected = {name: sha256(content).hexdigest() for name, content in bodies.items()}
        expected["handoff.json"] = sha256((root / "handoff.json").read_bytes()).hexdigest()
        return manifest, expected

    def test_source_bound_mock_inspection_never_claims_neural_step(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, trusted = self.make_fixture(root)
            receipt = inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)
            self.assertEqual(receipt.file_count, len(manifest["files"]))
            self.assertEqual(receipt.checkpoint_id, "qwen4b-test-0")
            self.assertEqual(receipt.classification, "SOURCE_ONLY_LOCAL_MOCK_FILE_INTEGRITY")
            self.assertFalse(receipt.neural_update_verified)
            self.assertFalse(receipt.full_resume_verified)
            self.assertFalse(receipt.graduation_authorized)

    def test_outside_trusted_digest_required_not_copied_from_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trusted = self.make_fixture(root)
            with self.assertRaisesRegex(TrainingDayHandoffError, "independent SHA"):
                inspect_trainingday_handoff(root, "handoff.json", trusted_sha256={})
            trusted["base/config.json"] = "0" * 64
            with self.assertRaisesRegex(TrainingDayHandoffError, "owner-trusted hash"):
                inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)

    def test_tampered_artifact_fails_streamed_readback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trusted = self.make_fixture(root)
            (root / "optimizer/state.pt").write_bytes(b"not the original optimizer")
            with self.assertRaisesRegex(TrainingDayHandoffError, "manifest file digest"):
                inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)

    def test_overclaim_physical_training_or_graduation_fails(self):
        for key in ("physical_training_authorized", "graduation_authorized"):
            with self.subTest(key=key), TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, trusted = self.make_fixture(root)
                subject = manifest["authority"] if key == "physical_training_authorized" else manifest["checkpoint"]
                subject[key] = True
                (root / "handoff.json").write_text(json.dumps(manifest))
                trusted["handoff.json"] = sha256((root / "handoff.json").read_bytes()).hexdigest()
                with self.assertRaisesRegex(TrainingDayHandoffError, "authority|checkpoint"):
                    inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)

    def test_symlink_and_path_traversal_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trusted = self.make_fixture(root)
            root_alias = root / "alias"
            root_alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(TrainingDayHandoffError, "root cannot be a symlink"):
                inspect_trainingday_handoff(root_alias, "handoff.json", trusted_sha256=trusted)
            (root / "optimizer/state.pt").unlink()
            (root / "optimizer/state.pt").symlink_to(root / "base/config.json")
            with self.assertRaisesRegex(TrainingDayHandoffError, "symlink"):
                inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)
            with self.assertRaisesRegex(TrainingDayHandoffError, "relative|traverses"):
                _stream_file(root, "../outside.json")

    def test_worker_cannot_forge_real_neural_quality(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trusted = self.make_fixture(root)
            source = root / "receipts/worker.json"
            payload = json.loads(source.read_text())
            payload["kind"] = "LOCAL_AUTHENTIC_NEURAL_UPDATE"
            source.write_text(json.dumps(payload))
            digest = sha256(source.read_bytes()).hexdigest()
            trusted["receipts/worker.json"] = digest
            manifest = json.loads((root / "handoff.json").read_text())
            manifest["files"]["receipts/worker.json"] = {
                "sha256": digest, "bytes": source.stat().st_size}
            (root / "handoff.json").write_text(json.dumps(manifest))
            trusted["handoff.json"] = sha256((root / "handoff.json").read_bytes()).hexdigest()
            with self.assertRaisesRegex(TrainingDayHandoffError, "mock receipt"):
                inspect_trainingday_handoff(root, "handoff.json", trusted_sha256=trusted)

    def test_large_artifact_digest_is_streamed_not_buffered(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.bin").write_bytes(b"0")
            with (root / "large.bin").open("ab") as stream:
                stream.truncate(4 * 1024 * 1024 + 3)
            digest, count = _stream_file(root, "large.bin")
            self.assertEqual(count, 4 * 1024 * 1024 + 3)
            self.assertRegex(digest, r"^[a-f0-9]{64}$")

    def test_direct_source_execution_is_explicit_hard_hold(self):
        with self.assertRaisesRegex(TrainingDayPhysicalExecutionHold, "SOURCE_ONLY_HARD_HOLD"):
            invoke_trainingday_trainer("untrusted-broker-override")


if __name__ == "__main__":
    unittest.main()
