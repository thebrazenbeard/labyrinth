"""Offline contract tests: VN1 coordinator, Lappy mock GPU worker, no network."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.private_worker_dispatch import (
    ApprovedTrainWork, ArtifactPart, IdleGpuLease, LocalDispatchWAL,
    ModelSnapshot, PrivateTrainJob, WorkerOutputReceipt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _snapshot() -> ModelSnapshot:
    records = (
        ("adapter_config.json", "adapter_config"),
        ("adapter_model.safetensors", "adapter_weight"),
        ("config.json", "model_config"),
        ("model-00001-of-00002.safetensors", "weight_shard"),
        ("model-00002-of-00002.safetensors", "weight_shard"),
        ("model.safetensors.index.json", "weight_index"),
        ("tokenizer.json", "tokenizer"),
    )
    return ModelSnapshot(
        entrance_manifest_sha256=_sha("verified entrance manifest"),
        w0_state_sha256=_sha("w0 full model and adapter"),
        artifacts=tuple(ArtifactPart(name, role, _sha(name), 1024 + i)
                        for i, (name, role) in enumerate(records)),
        adapter_status="ADAPTER_PRESENT",
    )


def _job(*, nonce: str = "a" * 32, lease_nonce: str = "b" * 32,
         cycle: int = 0, train_manifest: str = "approved-r18", predecessor: str | None = None) -> PrivateTrainJob:
    snapshot = _snapshot()
    return PrivateTrainJob(
        snapshot=snapshot,
        work=ApprovedTrainWork(
            objective_id="labyrinth-test", scope_sha256=_sha("approved scope"),
            approved_train_manifest_sha256=_sha(train_manifest),
            train_source_receipt_sha256=_sha("rights-approved-train-source"),
            predecessor_checkpoint_sha256=predecessor or snapshot.w0_state_sha256,
            predecessor_worker_receipt_sha256=_sha("prior legit worker receipt") if cycle else None,
            cycle_index=cycle, chamber_count=4,
        ),
        lease=IdleGpuLease(
            lease_nonce=lease_nonce, worker_host="Lappy",
            issued_unix_s=1_000_000, expires_unix_s=1_000_180,
            min_vram_mib=4096, min_ram_mib=16384, min_disk_free_mib=8192,
        ),
        attempt_nonce=nonce,
    )


class MockWorker:
    """Cannot touch a GPU or claim trained weights; emits unsigned fake output."""

    def run(self, job: PrivateTrainJob) -> WorkerOutputReceipt:
        return WorkerOutputReceipt(
            job_id=job.job_id, idempotency_key=job.idempotency_key,
            lease_nonce=job.lease.lease_nonce, worker_host="Lappy",
            predecessor_state_sha256=job.work.predecessor_checkpoint_sha256,
            output_state_sha256=_sha("mock-output-" + job.job_id),
            output_manifest_sha256=_sha("mock-files-manifest-" + job.job_id),
            completed_train_chambers=job.work.chamber_count,
        )


class PrivateWorkerDispatchTests(unittest.TestCase):
    def test_entire_w0_snapshot_closure_has_every_expected_file_role(self):
        snapshot = _snapshot()
        self.assertEqual(len(snapshot.artifacts), 7)
        self.assertIn("weight_index", [part.role for part in snapshot.artifacts])
        self.assertIn("tokenizer", [part.role for part in snapshot.artifacts])
        self.assertIn("adapter_weight", [part.role for part in snapshot.artifacts])
        self.assertEqual(len(snapshot.closure_sha256), 64)

        for bad_path in ("../secret", "C:\\VERA\\models\\weights.safetensors", "/etc/passwd",
                         "https://example.com/weights", "a/../../b", "a\\b.safetensors",
                         "CON.safetensors", "NUL/config.json", "COM1"):
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(ValueError):
                    ArtifactPart(bad_path, "weight_shard", _sha("test"), 10)

    def test_rejects_missing_index_tokenizer_adapter_pair_case_collisions_and_oversized(self):
        good = _snapshot()
        with self.assertRaisesRegex(ValueError, "single weight index"):
            replace(good, artifacts=tuple(part for part in good.artifacts if part.role != "weight_index"))
        with self.assertRaisesRegex(ValueError, "weights, config and tokenizer"):
            replace(good, artifacts=tuple(part for part in good.artifacts if part.role != "tokenizer"))
        with self.assertRaisesRegex(ValueError, "adapter files/status disagree"):
            replace(good, artifacts=tuple(part for part in good.artifacts if part.role != "adapter_config"))
        with self.assertRaisesRegex(ValueError, "canonical path order"):
            replace(good, artifacts=tuple(reversed(good.artifacts)))
        with self.assertRaisesRegex(ValueError, "artifact closure"):
            replace(good, artifacts=good.artifacts * 30)
        with self.assertRaises(ValueError):
            ArtifactPart("weights.safetensors", "weight_shard", _sha("a"), (1 << 40) + 1)

    def test_train_only_predecessor_and_idempotency_survive_nonce_rotation(self):
        first, retry = _job(), _job(nonce="c" * 32, lease_nonce="d" * 32)
        self.assertNotEqual(first.job_id, retry.job_id)
        self.assertEqual(first.idempotency_key, retry.idempotency_key)
        self.assertNotEqual(first.idempotency_key, _job(train_manifest="different-data").idempotency_key)
        self.assertEqual(first.coordinator_host, "VN1")
        self.assertEqual(first.worker_host, "Lappy")
        self.assertEqual(first.route, "PRIVATE_LAN_OR_TAILSCALE_FUTURE")

        with self.assertRaisesRegex(ValueError, "TRAIN"):
            replace(first.work, train_split="eval")
        with self.assertRaisesRegex(ValueError, "bounded, approved TRAIN"):
            replace(first.work, predecessor_worker_receipt_sha256=_sha("spurious"))
        successor = _job(cycle=1, predecessor=_sha("w1"))
        self.assertNotEqual(successor.work.predecessor_checkpoint_sha256,
                            successor.snapshot.w0_state_sha256)
        with self.assertRaisesRegex(ValueError, "predecessor"):
            replace(first, work=replace(first.work, predecessor_checkpoint_sha256=_sha("not-w0")))

    def test_serialized_manifest_excludes_private_exam_gold_prompts_and_secrets(self):
        job = _job()
        serialized = json.dumps(job.to_dict(), sort_keys=True)
        self.assertIn("approved_train_manifest_sha256", serialized)
        for forbidden in ("private_exam", "heldout", "gold_answer", "case_id", "prompt_text",
                          "secret", "bearer", "api_key", "ssh_key"):
            self.assertNotIn(forbidden, serialized.lower())
        with self.assertRaisesRegex(ValueError, "invalid objective"):
            replace(job.work, objective_id="x" * 100_000)

    def test_lease_enforces_bounded_gpu_idle_claim_without_host_probing(self):
        lease = _job().lease
        self.assertTrue(lease.valid_at(1_000_050))
        self.assertFalse(lease.valid_at(1_000_180))
        with self.assertRaisesRegex(ValueError, "600"):
            replace(lease, expires_unix_s=1_001_000)
        with self.assertRaisesRegex(ValueError, "wrong-host"):
            replace(lease, worker_host="VN1")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            replace(lease, cuda_required=False)

    def test_write_ahead_states_are_durable_replay_checked_and_no_resubmit_after_intent(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            job = _job()
            wal = LocalDispatchWAL(job, directory)
            self.assertIsNone(wal.state)
            wal.prepare()
            self.assertTrue(wal.retry_authorized())
            with self.assertRaisesRegex(ValueError, "expired"):
                wal.record_send_intent(unix_s=1_000_200)
            self.assertEqual(wal.state, "PREPARED")
            wal.record_send_intent(unix_s=1_000_050)
            self.assertFalse(wal.retry_authorized())
            self.assertEqual(LocalDispatchWAL(job, directory).state, "DISPATCH_INTENT_DURABLE")
            with self.assertRaises(ValueError):
                wal.prepare()
            with self.assertRaisesRegex(ValueError, "cross-job"):
                wal.acknowledge(job_id=_sha("wrong"), idempotency_key=job.idempotency_key,
                                lease_nonce=job.lease.lease_nonce)
            wal.acknowledge(job_id=job.job_id, idempotency_key=job.idempotency_key,
                            lease_nonce=job.lease.lease_nonce)
            receipt = MockWorker().run(job)
            receipt.validate_for(job)
            wal.record_output(receipt)
            wal.complete()
            self.assertEqual(LocalDispatchWAL(job, directory).state, "COMPLETED")
            self.assertEqual([e["index"] for e in wal.events], list(range(5)))
            self.assertEqual(wal.events[3]["worker_receipt"], receipt.to_dict())

    def test_unknown_effect_forbids_retry_even_with_new_job_nonce(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            job = _job()
            wal = LocalDispatchWAL(job, directory)
            wal.prepare()
            wal.record_send_intent(unix_s=1_000_050)
            wal.mark_unknown_effect()
            self.assertFalse(wal.retry_authorized())
            with self.assertRaisesRegex(ValueError, "illegal"):
                wal.record_send_intent(unix_s=1_000_052)
            with self.assertRaisesRegex(ValueError, "cross-job"):
                LocalDispatchWAL(_job(nonce="c" * 32), directory)

    def test_cross_job_or_unmodified_worker_receipts_refuse(self):
        job = _job()
        good = MockWorker().run(job)
        good.validate_for(job)
        with self.assertRaisesRegex(ValueError, "replayed"):
            replace(good, job_id=_sha("different-job")).validate_for(job)
        with self.assertRaisesRegex(ValueError, "replayed"):
            good.validate_for(_job(train_manifest="other-train-set"))
        with self.assertRaisesRegex(ValueError, "replayed"):
            replace(good, output_state_sha256=job.snapshot.w0_state_sha256).validate_for(job)
        with self.assertRaisesRegex(ValueError, "replayed"):
            replace(good, completed_train_chambers=3).validate_for(job)
        self.assertEqual(good.classification, "WORKER_SELF_REPORTED_UNVERIFIED")

    def test_tampered_wal_and_missing_sequence_fail_on_readback(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            job = _job()
            wal = LocalDispatchWAL(job, directory)
            wal.prepare()
            wal.record_send_intent(unix_s=1_000_050)
            second = directory / f"dispatch-{job.idempotency_key}-01.json"
            original = second.read_text()
            second.write_text(original.replace("DISPATCH_INTENT_DURABLE", "PREPARED"))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                LocalDispatchWAL(job, directory)

    def test_tampered_embedded_worker_result_fails_readback(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp).resolve()
            job = _job()
            wal = LocalDispatchWAL(job, directory)
            wal.prepare()
            wal.record_send_intent(unix_s=1_000_050)
            wal.acknowledge(job_id=job.job_id, idempotency_key=job.idempotency_key,
                            lease_nonce=job.lease.lease_nonce)
            wal.record_output(MockWorker().run(job))
            result_file = directory / f"dispatch-{job.idempotency_key}-03.json"
            data = json.loads(result_file.read_text())
            data["worker_receipt"]["output_state_sha256"] = _sha("replayed other output")
            result_file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                LocalDispatchWAL(job, directory)


if __name__ == "__main__":
    unittest.main()
