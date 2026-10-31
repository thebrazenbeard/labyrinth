"""R19 v2 immediate-predecessor checkpoint/journal source qualification.

These byte-level fixtures do not claim Torch execution. The separate optional
Torch test exercises actual AdamW tensors, restoration, and v1/v2 separation.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from labyrinth.maze_crash_resume_journal import (
    LEGACY_UNQUALIFIED, SCHEMA_V2 as JOURNAL_SCHEMA_V2,
    MazeJournalError, MazeJournalUnknown, commit_maze_cycle,
    initialize_maze_journal, inspect_maze_journal, precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import (
    SCHEMA, SCHEMA_V2, FullCheckpointContext, FullCheckpointContextV2,
    FullCheckpointError, _context_json, _json_bytes, _manifest_digest,
    load_full_checkpoint, save_full_checkpoint,
)
from tests.test_maze_crash_resume_journal import binding, context, fake_bytes_snapshot


def versioned_context(cycle: int, predecessor: str) -> FullCheckpointContextV2:
    legacy = context(cycle)
    return FullCheckpointContextV2(
        **vars(legacy), cycle_index=cycle,
        immediate_predecessor_sha256=predecessor)


def versioned_fixture(path: Path, ctx: FullCheckpointContextV2,
                      *, tensor_salt: bytes) -> str:
    """Create a hash-valid v2 snapshot; not physically loadable safetensors."""
    fake_bytes_snapshot(path, ctx)
    tensor = b"source-only-adamw-bytes-" + tensor_salt
    (path / "state.safetensors").write_bytes(tensor)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["schema"] = SCHEMA_V2
    manifest["context"] = _context_json(ctx)
    manifest["tensor_sha256"] = sha256(tensor).hexdigest()
    raw = _json_bytes(manifest)
    (path / "manifest.json").write_bytes(raw)
    return sha256(SCHEMA_V2.encode("ascii") + b"\0" + raw).hexdigest()


@unittest.skipIf(os.name == "nt", "Windows requires qualified journal directory fsync")
class ManifestV2JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name) / "arm-correct"
        self.parent.mkdir()
        self.root = self.parent / "journal"
        self.binding = binding()
        initialize_maze_journal(self.root, self.binding, schema_version="v2")

    def _cycle(self, cycle: int, predecessor: str, salt: bytes):
        ctx = versioned_context(cycle, predecessor)
        path = self.parent / f"cycle-{cycle}-{salt.decode('ascii')}"
        digest = versioned_fixture(path, ctx, tensor_salt=salt)
        return ctx, path, digest

    def test_two_cycle_v2_commits_bind_immediate_predecessor_and_inspect(self):
        c0, p0, sha0 = self._cycle(0, self.binding.prior_checkpoint_sha256, b"a0")
        precommit_maze_cycle(self.root, self.binding, 0, c0, schema_version="v2")
        commit_maze_cycle(self.root, self.binding, 0, c0, p0, sha0,
                          schema_version="v2")
        c1, p1, sha1 = self._cycle(1, sha0, b"a1")
        precommit_maze_cycle(self.root, self.binding, 1, c1, schema_version="v2")
        commit_maze_cycle(self.root, self.binding, 1, c1, p1, sha1,
                          schema_version="v2")
        observed = inspect_maze_journal(self.root, self.binding, schema_version="v2")
        self.assertEqual(observed.committed_cycles, 2)
        self.assertEqual(observed.last_checkpoint_sha256, sha1)
        self.assertNotEqual(observed.lineage_qualification, LEGACY_UNQUALIFIED)
        self.assertEqual(json.loads((self.root / "run.json").read_bytes())["schema"],
                         JOURNAL_SCHEMA_V2)

    def test_foreign_fork_cycle_one_cannot_attach_to_other_fork(self):
        # Both forks have the same baseline, run/scope/fence and exposures.
        # Their cycle-0 tensor bytes differ. A1's intended predecessor A0 is
        # therefore distinct from B1's independently committed B0.
        c0, p0, sha_a0 = self._cycle(0, self.binding.prior_checkpoint_sha256, b"a0")
        precommit_maze_cycle(self.root, self.binding, 0, c0, schema_version="v2")
        commit_maze_cycle(self.root, self.binding, 0, c0, p0, sha_a0,
                          schema_version="v2")
        fork_parent = Path(self.tmp.name) / "foreign-fork"
        fork_parent.mkdir()
        fork = fork_parent / "journal"
        initialize_maze_journal(fork, self.binding, schema_version="v2")
        b0 = versioned_context(0, self.binding.prior_checkpoint_sha256)
        bp0 = fork_parent / "b0"
        sha_b0 = versioned_fixture(bp0, b0, tensor_salt=b"b0")
        precommit_maze_cycle(fork, self.binding, 0, b0, schema_version="v2")
        commit_maze_cycle(fork, self.binding, 0, b0, bp0, sha_b0,
                          schema_version="v2")
        self.assertNotEqual(sha_a0, sha_b0)
        foreign_b1 = versioned_context(1, sha_b0)
        bp1 = fork_parent / "b1"
        sha_b1 = versioned_fixture(bp1, foreign_b1, tensor_salt=b"b1")
        with self.assertRaisesRegex(MazeJournalError, "immediate predecessor"):
            precommit_maze_cycle(self.root, self.binding, 1, foreign_b1,
                                 schema_version="v2")
        local_a1 = versioned_context(1, sha_a0)
        precommit_maze_cycle(self.root, self.binding, 1, local_a1,
                             schema_version="v2")
        # Copy the hostile valid bytes *inside* A's run parent, so the
        # location gate cannot mask the actual immediate-lineage check.
        injected = self.parent / "foreign-b1-same-root"
        shutil.copytree(bp1, injected)
        with self.assertRaisesRegex(MazeJournalError, "context"):
            commit_maze_cycle(self.root, self.binding, 1, local_a1, injected,
                              sha_b1, schema_version="v2")
        observed = inspect_maze_journal(self.root, self.binding, schema_version="v2")
        self.assertEqual((observed.committed_cycles, observed.pending_cycle), (1, 1))

    def test_uncommitted_v2_intent_blocks_replay_and_future_cycle(self):
        c0 = versioned_context(0, self.binding.prior_checkpoint_sha256)
        precommit_maze_cycle(self.root, self.binding, 0, c0, schema_version="v2")
        with self.assertRaises(MazeJournalUnknown):
            precommit_maze_cycle(self.root, self.binding, 0, c0,
                                 schema_version="v2")
        with self.assertRaises(MazeJournalUnknown):
            precommit_maze_cycle(self.root, self.binding, 1,
                                 versioned_context(1, "a" * 64), schema_version="v2")
        self.assertFalse(inspect_maze_journal(self.root, self.binding,
                                              schema_version="v2").safe_to_begin_next_cycle)

    def test_v1_routes_are_legacy_only_and_cannot_be_promoted_by_argument(self):
        old_context = context(0)
        with self.assertRaises(ValueError):
            precommit_maze_cycle(self.root, self.binding, 0, old_context,
                                 schema_version="v2")
        with self.assertRaises(MazeJournalError):
            inspect_maze_journal(self.root, self.binding)
        old_root = self.parent / "legacy-journal"
        initialize_maze_journal(old_root, self.binding)
        self.assertEqual(inspect_maze_journal(old_root, self.binding).lineage_qualification,
                         LEGACY_UNQUALIFIED)
        with self.assertRaises(MazeJournalError):
            inspect_maze_journal(old_root, self.binding, schema_version="v2")
        with self.assertRaises(ValueError):
            precommit_maze_cycle(old_root, self.binding, 0,
                                 versioned_context(0, self.binding.prior_checkpoint_sha256))

    def test_v2_manifest_context_and_digest_are_domain_separated(self):
        cp = versioned_context(0, self.binding.prior_checkpoint_sha256)
        path = self.parent / "v2"
        digest = versioned_fixture(path, cp, tensor_salt=b"source")
        raw = (path / "manifest.json").read_bytes()
        self.assertEqual(_manifest_digest(raw, schema=SCHEMA_V2), digest)
        self.assertNotEqual(_manifest_digest(raw, schema=SCHEMA), digest)
        self.assertEqual(json.loads(raw)["context"]["immediate_predecessor_sha256"],
                         cp.prior_checkpoint_sha256)
        with self.assertRaises(ValueError):
            replace(cp, immediate_predecessor_sha256="f" * 64).validate()


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "optional torch+safetensors physical CPU dependencies unavailable",
)
class V2PhysicalCheckpointTests(unittest.TestCase):
    def test_real_adamw_v2_save_and_restore_never_cross_load_as_v1(self):
        import torch

        torch.manual_seed(812)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        x = torch.tensor([[0.1, 0.3, -0.2]])
        optimizer.zero_grad(set_to_none=True)
        model(x).sum().backward()
        optimizer.step()
        saved = {name: parameter.detach().clone()
                 for name, parameter in model.named_parameters()}
        states = {name: {key: value.detach().clone()
                         for key, value in optimizer.state[parameter].items()}
                  for name, parameter in model.named_parameters()}
        ctx = FullCheckpointContextV2(**vars(context(0)), cycle_index=0,
                                      immediate_predecessor_sha256="d" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-v2"
            receipt = save_full_checkpoint(path, model=model, optimizer=optimizer,
                                           context=ctx)
            self.assertEqual(json.loads((path / "manifest.json").read_bytes())["schema"],
                             SCHEMA_V2)
            optimizer.zero_grad(set_to_none=True)
            model(x).sum().backward()
            optimizer.step()
            with self.assertRaises(FullCheckpointError):
                load_full_checkpoint(path, model=model, optimizer=optimizer,
                                     expected_sha256=receipt.sha256,
                                     expected_context=FullCheckpointContext(**vars(context(0))))
            restored = load_full_checkpoint(path, model=model, optimizer=optimizer,
                                            expected_sha256=receipt.sha256,
                                            expected_context=ctx)
            self.assertEqual(restored.sha256, receipt.sha256)
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.equal(saved[name], parameter))
                for key, value in optimizer.state[parameter].items():
                    self.assertTrue(torch.equal(states[name][key], value))


if __name__ == "__main__":
    unittest.main()
