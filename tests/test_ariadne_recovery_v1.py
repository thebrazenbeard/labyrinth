"""Ariadne's physical ancestry and write-ahead effect crash tests."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import labyrinth.ariadne_recovery_v1 as recovery
from labyrinth.ariadne_recovery_v1 import (
    AriadneJournal, IndeterminateTrainerEffect, LineageIntegrityError,
    UnsupportedJournalPlatform,
    _digest, trainer_request,
)
from labyrinth.campaign_full_state_v1 import (
    capture_campaign_state, save_campaign_state_once,
)


def _tensor(value: bytes) -> bytes:
    header = json.dumps({"adapter.weight": {
        "dtype": "U8", "shape": [len(value)],
        "data_offsets": [0, len(value)],
    }}, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + value


def _physical_state(root: Path, marker: str, parent: dict | None = None):
    root.mkdir()
    parts = {role: root / role for role in ("base", "adapter", "optimizer", "scheduler", "rng")}
    for path in parts.values():
        path.mkdir()
    (parts["base"] / "config.json").write_text('{"model_type":"test"}')
    (parts["base"] / "tokenizer.json").write_text('{"tokens":["A"]}')
    (parts["base"] / "model.safetensors").write_bytes(_tensor(b"base physical weights"))
    (parts["adapter"] / "adapter_config.json").write_text('{"r":1}')
    (parts["adapter"] / "adapter_model.safetensors").write_bytes(_tensor(marker.encode()))
    (parts["optimizer"] / "adam.bin").write_bytes(("adam:" + marker).encode())
    (parts["scheduler"] / "scheduler.bin").write_bytes(("scheduler:" + marker).encode())
    (parts["rng"] / "rng.bin").write_bytes(("rng:" + marker).encode())
    cursor = root / "cursor.json"
    cursor.write_text(json.dumps({"cursor": marker}))
    manifest = capture_campaign_state(
        base_dir=parts["base"], adapter_dir=parts["adapter"],
        optimizer_dir=parts["optimizer"], scheduler_dir=parts["scheduler"],
        rng_dir=parts["rng"], data_cursor_path=cursor,
        family="hf_safetensors_lora", mutation_kind="lora_adapter",
        parent_manifest=parent,
    )
    destination = root.parent / (root.name + ".json")
    save_campaign_state_once(manifest, destination)
    return manifest, destination, parts


def _experience(marker: str):
    return {"observations": [{"scene": marker}], "actions": ["east", "train"],
            "outcomes": [{"resource": "changed"}]}


def _eval(marker: str):
    return {"metric_id": "public_development", "score": 1.0 if marker == "right" else 0.0,
            "direction": "higher", "dataset_ref": "local_public_fixture",
            "evidence": {"receipt": marker, "private_gold_included": False}}


def _intent_worker(root: str):
    journal = AriadneJournal.open(Path(root))
    journal.prepare_training(candidate_id="uncertain", branch_id="branch_a",
                             trainer_request=trainer_request("fixture", parameters={}),
                             idempotency_key="prepared_crash")
    # A trainer can modify physical bytes after PREPARED and die before COMMITTED.
    (Path(root).parent / "side_effect.txt").write_text("optimizer possibly stepped")
    os._exit(23)


def _fork_select_worker(root: str, candidate_id: str, result):
    try:
        AriadneJournal.open(Path(root)).select_candidate(
            candidate_id=candidate_id, selection_reason="public-score",
            idempotency_key="select_" + candidate_id)
        result.put("selected:" + candidate_id)
    except LineageIntegrityError:
        result.put("refused:" + candidate_id)


class AriadneRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.w0, self.w0_path, self.w0_parts = _physical_state(self.root / "w0", "W0")
        self.journal = AriadneJournal.create(self.root / "ariadne")
        self.journal.initialize(w0_manifest_path=self.w0_path, world_id="world_1",
                                version=1, curriculum={"objective": "counterfactual"},
                                world_state={"rooms": ["entrance", "threshold"]},
                                idempotency_key="genesis")

    def _world_branch(self, name: str):
        return self.journal.record_branch(
            branch_id="branch_" + name, world_id="world_1",
            experience=_experience(name), selected_path=["entrance", "threshold", name],
            idempotency_key="branch_" + name)

    def _candidate(self, name: str, parent: dict):
        branch_id = "branch_" + name
        self._world_branch(name)
        permit = self.journal.prepare_training(
            candidate_id=name, branch_id=branch_id,
            trainer_request=trainer_request("fixture", parameters={"seed": name}),
            idempotency_key="intent_" + name)
        manifest, path, _ = _physical_state(self.root / ("child_" + name), name, parent)
        self.journal.commit_training(
            permit, child_manifest_path=path,
            trainer_receipt={"trainer_id": "fixture", "effect_status": "completed",
                             "effect_token": permit.effect_token,
                             "details": {"source": "test_callback_not_real_sgd"}},
            idempotency_key="commit_" + name)
        self.journal.record_evaluation(
            candidate_id=name, evaluation=_eval(name), idempotency_key="eval_" + name)
        return manifest, path

    def test_restart_after_two_generation_selections_reconstructs_full_descriptor(self):
        self.journal.record_world(world_id="world_2", parent_world_id="world_1",
                                  version=2, curriculum={"objective": "new curriculum"},
                                  world_state={"rooms": ["threshold", "branch"]},
                                  idempotency_key="world_2")
        left, _ = self._candidate("left", self.w0)
        right, _ = self._candidate("right", self.w0)
        self.journal.reject_candidate(candidate_id="left", reason="less fit",
                                      idempotency_key="reject_left")
        self.journal.select_candidate(candidate_id="right", selection_reason="public score",
                                      idempotency_key="select_right")
        # The next world's two siblings explicitly inherit the chosen full
        # physical state (not merely its adapter/weight hash).
        a, _ = self._candidate("g2_a", right)
        b, bpath = self._candidate("g2_b", right)
        self.journal.reject_candidate(candidate_id="g2_a", reason="less fit",
                                      idempotency_key="reject_g2_a")
        self.journal.select_candidate(candidate_id="g2_b", selection_reason="public score",
                                      idempotency_key="select_g2_b")
        expected_head = self.journal.continuation_descriptor()["event_head_sha256"]
        # A new Python interpreter independently reopens all manifests, and
        # validates every optimizer, RNG, scheduler, model and cursor file.
        script = ("import json,sys; from pathlib import Path; "
                  "from labyrinth.ariadne_recovery_v1 import AriadneJournal; "
                  "print(json.dumps(AriadneJournal.open(Path(sys.argv[1]))"
                  ".continuation_descriptor(), sort_keys=True))")
        process = subprocess.run([sys.executable, "-c", script, str(self.journal.root)],
                                 capture_output=True, text=True, timeout=60, check=True)
        recovered = json.loads(process.stdout)
        self.assertEqual(recovered["event_head_sha256"], expected_head)
        self.assertEqual(recovered["next_generation_index"], 2)
        self.assertEqual(recovered["selected_parent_state_sha256"], b["state_sha256"])
        self.assertEqual(recovered["selected_parent_manifest_path"], str(bpath))
        self.assertEqual(recovered["origin_state_sha256"], self.w0["state_sha256"])
        self.assertEqual(len(recovered["selected_lineage"]), 2)
        self.assertEqual({c["fate"] for c in recovered["candidates"]},
                         {"SELECTED", "REJECTED"})
        self.assertEqual(recovered["branches"][-1]["selected_path"][-1], "g2_b")
        self.assertEqual(recovered["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertFalse(recovered["model_loaded"])
        self.assertFalse(recovered["optimizer_loaded"])
        self.assertFalse(recovered["restorable_training_state_proven"])
        self.assertFalse(recovered["neural_update_proven"])
        self.assertNotEqual(left["state_sha256"], right["state_sha256"])
        self.assertNotEqual(a["state_sha256"], b["state_sha256"])

    def test_crash_after_fsynced_prepared_effect_blocks_new_process_and_retry(self):
        self._world_branch("a")
        process = multiprocessing.Process(target=_intent_worker, args=(str(self.journal.root),))
        process.start()
        process.join(20)
        self.assertEqual(process.exitcode, 23)
        self.assertTrue((self.root / "side_effect.txt").is_file())
        reopened = AriadneJournal.open(self.journal.root)
        descriptor = reopened.continuation_descriptor()
        self.assertEqual(descriptor["status"], "INDETERMINATE_TRAINER_EFFECT")
        self.assertEqual(descriptor["pending_training"]["candidate_id"], "uncertain")
        self.assertEqual(len(reopened.events()), 3)
        with self.assertRaises(IndeterminateTrainerEffect):
            reopened.prepare_training(candidate_id="uncertain", branch_id="branch_a",
                                      trainer_request=trainer_request("fixture", parameters={}),
                                      idempotency_key="prepared_crash")
        with self.assertRaises(IndeterminateTrainerEffect):
            reopened.record_world(world_id="new", parent_world_id="world_1", version=2,
                                  curriculum={}, world_state={}, idempotency_key="new_world")
        with self.assertRaises(IndeterminateTrainerEffect):
            reopened.commit_training(permit=None, child_manifest_path=self.w0_path,
                                     trainer_receipt={}, idempotency_key="fake_commit")

    def test_damaged_truncated_deleted_middle_replayed_and_orphan_files_fail(self):
        self._world_branch("a")
        witnessed_head = self.journal.continuation_descriptor()["event_head_sha256"]
        event_file = self.journal.root / "events" / "000000000001.json"
        original = event_file.read_bytes()
        event_file.write_bytes(original[:-3])
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor()
        event_file.write_bytes(original)
        altered = json.loads(original)
        altered["payload"]["world_state"] = {"rooms": ["malicious"]}
        event_file.write_text(json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor()
        event_file.write_bytes(original)
        event_file.unlink()
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor(
                expected_head_sha256=witnessed_head)
        event_file.write_bytes(original)
        # A copied event under a later sequence number is a replay, not lineage.
        (self.journal.root / "events" / "000000000002.json").write_bytes(original)
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor()
        (self.journal.root / "events" / "000000000002.json").unlink()
        (self.journal.root / "events" / ".stage-abandoned").write_bytes(b"crash")
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor()

    def test_tampered_full_state_even_if_model_and_manifest_hash_strings_unchanged(self):
        self._world_branch("a")
        (self.w0_parts["optimizer"] / "adam.bin").write_bytes(b"altered optimizer")
        with self.assertRaisesRegex(LineageIntegrityError, "physical bytes|readback"):
            AriadneJournal.open(self.journal.root).continuation_descriptor()

    def test_unknown_event_version_or_wrong_wt_fails_even_after_hash_recomputed(self):
        genesis_path = self.journal.root / "events" / "000000000000.json"
        original = genesis_path.read_bytes()
        changed = json.loads(original)
        changed["event_version"] = 2
        changed["event_sha256"] = _digest({k: v for k, v in changed.items()
                                            if k != "event_sha256"})
        genesis_path.write_text(json.dumps(changed, sort_keys=True,
                                           separators=(",", ":")) + "\n")
        with self.assertRaises(LineageIntegrityError):
            AriadneJournal.open(self.journal.root).continuation_descriptor()
        genesis_path.write_bytes(original)
        self._world_branch("a")
        branch_path = self.journal.root / "events" / "000000000001.json"
        branch_original = branch_path.read_bytes()
        wrong_parent = json.loads(branch_original)
        wrong_parent["payload"]["parent_state_sha256"] = "d" * 64
        wrong_parent["event_sha256"] = _digest({k: v for k, v in wrong_parent.items()
                                                 if k != "event_sha256"})
        branch_path.write_text(json.dumps(wrong_parent, sort_keys=True,
                                          separators=(",", ":")) + "\n")
        with self.assertRaisesRegex(LineageIntegrityError, "diverges"):
            AriadneJournal.open(self.journal.root).continuation_descriptor()
        branch_path.write_bytes(branch_original)

    def test_hash_only_cannot_initialize_full_model_ancestry(self):
        fake = self.root / "hash-only.json"
        fake.write_text(json.dumps({"weights_sha256": "a" * 64}))
        second = AriadneJournal.create(self.root / "not_full_state")
        with self.assertRaises(LineageIntegrityError):
            second.initialize(w0_manifest_path=fake, world_id="fake", version=1,
                              curriculum={}, world_state={}, idempotency_key="fake")
        self.assertEqual(list((second.root / "events").iterdir()), [])

    def test_windows_unsupported_backend_is_importable_and_refuses_all_journal_io(self):
        script = (
            "import sys,tempfile; sys.modules['fcntl']=None; "
            "from pathlib import Path; "
            "from labyrinth.ariadne_recovery_v1 import (AriadneJournal,"
            "UnsupportedJournalPlatform); "
            "p=Path(tempfile.gettempdir())/'should-never-exist-ariadne-native'; "
            "assert not p.exists(); "
            "\ntry: AriadneJournal.create(p)\n"
            "except UnsupportedJournalPlatform: pass\n"
            "else: raise AssertionError('missing lock backend accepted')\n"
            "assert not p.exists()\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(recovery, "_SUPPORTED_PLATFORM", False):
            missing = self.root / "rejected_backend"
            with self.assertRaises(UnsupportedJournalPlatform):
                AriadneJournal.create(missing)
            with self.assertRaises(UnsupportedJournalPlatform):
                AriadneJournal.open(self.journal.root)
            self.assertFalse(missing.exists())

    def test_config_only_lora_child_does_not_count_as_changed_tensor_weights(self):
        self._world_branch("a")
        permit = self.journal.prepare_training(
            candidate_id="a", branch_id="branch_a",
            trainer_request=trainer_request("fixture", parameters={}),
            idempotency_key="intent_a")
        _, child_path, parts = _physical_state(self.root / "config_only", "child", self.w0)
        (parts["adapter"] / "adapter_model.safetensors").write_bytes(_tensor(b"W0"))
        (parts["adapter"] / "adapter_config.json").write_text('{"r":2}')
        manifest = capture_campaign_state(
            base_dir=parts["base"], adapter_dir=parts["adapter"],
            optimizer_dir=parts["optimizer"], scheduler_dir=parts["scheduler"],
            rng_dir=parts["rng"], data_cursor_path=self.root / "config_only" / "cursor.json",
            family="hf_safetensors_lora", mutation_kind="lora_adapter",
            parent_manifest=self.w0,
        )
        valid_path = self.root / "config_only_v2.json"
        save_campaign_state_once(manifest, valid_path)
        with self.assertRaisesRegex(LineageIntegrityError, "change adapter tensor"):
            self.journal.commit_training(
                permit, child_manifest_path=valid_path,
                trainer_receipt={"trainer_id": "fixture", "effect_status": "completed",
                                 "effect_token": permit.effect_token, "details": {}},
                idempotency_key="commit_config_only")
        self.assertEqual(self.journal.continuation_descriptor()["status"],
                         "INDETERMINATE_TRAINER_EFFECT")

    def test_concurrent_select_cannot_create_two_parent_forks(self):
        self._candidate("left", self.w0)
        self._candidate("right", self.w0)
        # Both evaluations must exist, but competing selection cannot succeed
        # without first explicitly recording loser fate.
        self.journal.reject_candidate(candidate_id="left", reason="lower measured score",
                                      idempotency_key="reject_left")
        queue = multiprocessing.Queue()
        children = [multiprocessing.Process(target=_fork_select_worker,
                                            args=(str(self.journal.root), name, queue))
                    for name in ("left", "right")]
        for child in children:
            child.start()
        for child in children:
            child.join(20)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(sorted(queue.get(timeout=2) for _ in children),
                         ["refused:left", "selected:right"])
        reopened = AriadneJournal.open(self.journal.root).continuation_descriptor()
        self.assertEqual(len(reopened["selected_lineage"]), 1)
        self.assertEqual(reopened["next_generation_index"], 1)

    def test_idempotency_is_exact_and_prepared_never_issues_second_permit(self):
        self._world_branch("a")
        branch = self._world_branch("a")
        self.assertEqual(branch["idempotency_key"], "branch_a")
        with self.assertRaises(LineageIntegrityError):
            self.journal.record_branch(branch_id="different", world_id="world_1",
                                       experience=_experience("a"), selected_path=["a"],
                                       idempotency_key="branch_a")
        self.journal.prepare_training(candidate_id="a", branch_id="branch_a",
                                      trainer_request=trainer_request("fixture", parameters={}),
                                      idempotency_key="prepare_a")
        with self.assertRaises(IndeterminateTrainerEffect):
            self.journal.prepare_training(candidate_id="a", branch_id="branch_a",
                                          trainer_request=trainer_request("fixture", parameters={}),
                                          idempotency_key="prepare_a")


if __name__ == "__main__":
    unittest.main()
