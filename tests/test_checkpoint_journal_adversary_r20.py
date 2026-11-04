"""Independent hostile probes of the opt-in v2 Maze checkpoint journal.

The fixtures contain byte-consistent manifests, not physically restored neural
states.  Success establishes local source contracts only.  In particular,
neither an internally computed digest nor a complete local filesystem tree is
an independently owned monotonic trust anchor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from labyrinth.maze_crash_resume_journal import (
    LEGACY_UNQUALIFIED, V2_SOURCE_ONLY, MazeJournalError, MazeJournalUnknown,
    commit_maze_cycle, initialize_maze_journal, inspect_maze_journal,
    precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import (
    SCHEMA as LEGACY_CHECKPOINT_SCHEMA, SCHEMA_V2, _json_bytes,
)
from test_checkpoint_manifest_v2_r19 import versioned_context, versioned_fixture
from test_maze_crash_resume_journal import binding


@unittest.skipIf(os.name == "nt", "Windows requires a separately qualified durable directory flush")
class CheckpointJournalAdversaryR20(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.area = Path(temp.name)
        self.run_parent = self.area / "run-a"
        self.run_parent.mkdir()
        self.journal = self.run_parent / "journal"
        self.conf = binding()
        initialize_maze_journal(self.journal, self.conf, schema_version="v2")

    def _cycle(self, cycle: int, predecessor: str, tag: bytes):
        ctx = versioned_context(cycle, predecessor)
        path = self.run_parent / ("snapshot-" + tag.decode("ascii"))
        digest = versioned_fixture(path, ctx, tensor_salt=tag)
        return ctx, path, digest

    def _commit(self, cycle: int, predecessor: str, tag: bytes):
        ctx, path, digest = self._cycle(cycle, predecessor, tag)
        precommit_maze_cycle(self.journal, self.conf, cycle, ctx,
                             schema_version="v2")
        self.assertEqual(commit_maze_cycle(self.journal, self.conf, cycle, ctx,
                                            path, digest, schema_version="v2"), digest)
        return ctx, path, digest

    def test_duplicate_precommit_race_has_exactly_one_local_winner(self) -> None:
        ctx = versioned_context(0, self.conf.prior_checkpoint_sha256)
        start = threading.Barrier(2)

        def compete() -> str:
            start.wait(timeout=5)
            try:
                precommit_maze_cycle(self.journal, self.conf, 0, ctx,
                                     schema_version="v2")
                return "committed_intent"
            except (MazeJournalError, MazeJournalUnknown):
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(compete)
            right = pool.submit(compete)
            self.assertEqual(sorted((left.result(), right.result())),
                             ["committed_intent", "rejected"])
        observed = inspect_maze_journal(self.journal, self.conf,
                                        schema_version="v2")
        self.assertEqual((observed.committed_cycles, observed.pending_cycle), (0, 0))

    def test_pending_intent_after_process_restart_blocks_unrelated_or_duplicate_work(self) -> None:
        ctx = versioned_context(0, self.conf.prior_checkpoint_sha256)
        precommit_maze_cycle(self.journal, self.conf, 0, ctx,
                             schema_version="v2")
        # A new reader reconstructs state solely from persisted records.
        observed = inspect_maze_journal(Path(str(self.journal)), self.conf,
                                        schema_version="v2")
        self.assertEqual(observed.pending_cycle, 0)
        self.assertFalse(observed.safe_to_begin_next_cycle)
        for next_cycle, predecessor in ((0, self.conf.prior_checkpoint_sha256),
                                        (1, "f" * 64)):
            with self.subTest(cycle=next_cycle), self.assertRaises(MazeJournalUnknown):
                precommit_maze_cycle(self.journal, self.conf, next_cycle,
                                     versioned_context(next_cycle, predecessor),
                                     schema_version="v2")

    def test_commit_uses_manifest_schema_not_just_attacker_chosen_sha_domain(self) -> None:
        ctx, path, _ = self._cycle(0, self.conf.prior_checkpoint_sha256, b"downgrade")
        precommit_maze_cycle(self.journal, self.conf, 0, ctx, schema_version="v2")
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["schema"] = LEGACY_CHECKPOINT_SCHEMA
        raw = _json_bytes(manifest)
        (path / "manifest.json").write_bytes(raw)
        # Even the right *v2* domain cannot promote a v1 schema.
        fake_trusted = sha256(SCHEMA_V2.encode("ascii") + b"\0" + raw).hexdigest()
        with self.assertRaisesRegex(MazeJournalError, "checkpoint context"):
            commit_maze_cycle(self.journal, self.conf, 0, ctx,
                              path, fake_trusted, schema_version="v2")
        self.assertEqual(inspect_maze_journal(self.journal, self.conf,
                                              schema_version="v2").pending_cycle, 0)

    def test_post_snapshot_precommit_byte_mutation_keeps_forensic_hold(self) -> None:
        ctx, path, digest = self._cycle(0, self.conf.prior_checkpoint_sha256,
                                        b"torn")
        precommit_maze_cycle(self.journal, self.conf, 0, ctx, schema_version="v2")
        import labyrinth.maze_crash_resume_journal as journal_module
        original = journal_module._write_once

        def corrupt_before_persist(target: Path, value: dict) -> None:
            if target.name.endswith(".commit.json"):
                (path / "state.safetensors").write_bytes(b"changed-after-first-hash")
            original(target, value)

        with patch.object(journal_module, "_write_once",
                          side_effect=corrupt_before_persist):
            with self.assertRaises(MazeJournalError):
                commit_maze_cycle(self.journal, self.conf, 0, ctx, path, digest,
                                  schema_version="v2")
        self.assertTrue((self.journal / "cycle-0000.commit.json").exists())
        self.assertTrue((self.journal / "cycle-0000.intent.json").exists())
        with self.assertRaises(MazeJournalError):
            inspect_maze_journal(self.journal, self.conf, schema_version="v2")

    def test_whole_valid_local_tree_rollback_remains_undetectable_without_external_anchor(self) -> None:
        _, _, first_receipt = self._commit(0, self.conf.prior_checkpoint_sha256,
                                          b"rollback0")
        old_tree = self.area / "old-valid-tree"
        shutil.copytree(self.run_parent, old_tree)
        self._commit(1, first_receipt, b"rollback1")
        live = inspect_maze_journal(self.journal, self.conf, schema_version="v2")
        self.assertEqual(live.committed_cycles, 2)
        rolled_back = inspect_maze_journal(old_tree / "journal", self.conf,
                                           schema_version="v2")
        self.assertEqual(rolled_back.committed_cycles, 1)
        self.assertEqual(rolled_back.last_checkpoint_sha256, first_receipt)
        self.assertEqual(rolled_back.qualification, V2_SOURCE_ONLY)
        self.assertNotEqual(rolled_back.lineage_qualification, LEGACY_UNQUALIFIED)

    def test_locally_rehashed_foreign_tensor_is_not_proof_of_neural_derivation(self) -> None:
        _, _, first_receipt = self._commit(0, self.conf.prior_checkpoint_sha256,
                                          b"parent")
        # A writer with control of both tensor bytes and v2 manifest can
        # honestly describe an unrelated tensor as the next local checkpoint.
        # This is an expected *limit*, not an implementation defect; the
        # journal only attests consistency of caller-controlled source bytes.
        ctx, path, digest = self._cycle(1, first_receipt, b"unrelated")
        precommit_maze_cycle(self.journal, self.conf, 1, ctx, schema_version="v2")
        commit_maze_cycle(self.journal, self.conf, 1, ctx, path, digest,
                          schema_version="v2")
        observed = inspect_maze_journal(self.journal, self.conf,
                                        schema_version="v2")
        self.assertEqual(observed.committed_cycles, 2)
        self.assertEqual(observed.qualification, V2_SOURCE_ONLY)
        self.assertNotEqual(observed.lineage_qualification, "NEURAL_DERIVATION_VERIFIED")


if __name__ == "__main__":
    unittest.main()
