"""Adversarial source-only checkpoint-journal custody tests.

Fixtures have valid byte-level manifests, not physically loadable Torch state;
the separate neural checkpoint tests prove pickle-free tensor restoration.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.maze_crash_resume_journal import (
    SOURCE_ONLY, MazeJournalBinding, MazeJournalError, MazeJournalUnknown,
    commit_maze_cycle, initialize_maze_journal, inspect_maze_journal,
    precommit_maze_cycle,
)
from labyrinth.neural_resume_checkpoint import (
    SCHEMA as FULL_SCHEMA, FullCheckpointContext, _context_json,
    _json_bytes, _manifest_digest,
)


def binding(**changes) -> MazeJournalBinding:
    options = {
        "run_id": "bt2.maze-run-01",
        "objective_id": "learn.sentiment",
        "arm": "correct",
        "candidate_artifact_sha256": "a" * 64,
        "scope_fingerprint": "b" * 64,
        "base_checkpoint_sha256": "c" * 64,
        "prior_checkpoint_sha256": "d" * 64,
        "checkpoint_run_id": "model.maze-run-01",
        "source_sha256": "e" * 64,
        "fence": 12,
        "cycle_count": 2,
    }
    options.update(changes)
    return MazeJournalBinding(**options)


def context(cycle: int, conf: MazeJournalBinding | None = None,
            **changes) -> FullCheckpointContext:
    conf = conf or binding()
    cases = tuple((f"cycle0.step{i:02d}.visit", str(i) * 64)
                  for i in range(4))
    values = {
        "run_id": conf.checkpoint_run_id,
        "objective_id": conf.objective_id,
        "base_checkpoint_sha256": conf.base_checkpoint_sha256,
        "prior_checkpoint_sha256": conf.prior_checkpoint_sha256,
        "fence": conf.fence,
        "cursor": (cycle + 1) * 2,
        "exposures": cases,
    }
    values.update(changes)
    return FullCheckpointContext(**values)


def fake_bytes_snapshot(directory: Path, expected: FullCheckpointContext,
                        *, manifest_changes: dict | None = None,
                        bad_tensor: bool = False) -> str:
    """Minimal byte-custody fixture, deliberately not a Torch model snapshot."""
    directory.mkdir(parents=True)
    tensor = b"byte-custody-only-not-physical-safetensors"
    manifest = {
        "schema": FULL_SCHEMA,
        "context": _context_json(expected),
        "optimizer": "AdamW",
        "optimizer_groups": [["head.weight"]],
        "trainable_names": ["head.weight"],
        "model_tree": {"type": "dict", "items": []},
        "optimizer_tree": {"type": "dict", "items": [
            [{"type": "scalar", "value": "state"},
             {"type": "dict", "items": [
                 [{"type": "scalar", "value": 0},
                  {"type": "dict", "items": [
                      [{"type": "scalar", "value": "step"},
                       {"type": "tensor", "key": "tensor_000000"}],
                      [{"type": "scalar", "value": "exp_avg"},
                       {"type": "tensor", "key": "tensor_000001"}],
                      [{"type": "scalar", "value": "exp_avg_sq"},
                       {"type": "tensor", "key": "tensor_000002"}],
                 ]}],
             ]}],
            [{"type": "scalar", "value": "param_groups"},
             {"type": "list", "items": [{"type": "dict", "items": []}]}],
        ]},
        "rng_tree": {"type": "dict", "items": []},
        "numpy_rng": None,
        "tensor_sha256": sha256(tensor).hexdigest(),
    }
    if manifest_changes:
        manifest.update(manifest_changes)
    raw = _json_bytes(manifest)
    (directory / "manifest.json").write_bytes(raw)
    (directory / "state.safetensors").write_bytes(
        b"tampered-physical-tensor" if bad_tensor else tensor)
    return _manifest_digest(raw)


@unittest.skipIf(os.name == "nt", "Windows directory fsync needs a qualified host provider")
class MazeCrashResumeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / "arm-correct"
        self.parent.mkdir()
        self.root = self.parent / "journal"
        self.conf = binding()
        initialize_maze_journal(self.root, self.conf)

    def _snap(self, cycle, *, conf=None, context_changes=None, **kwargs):
        expected = context(cycle, conf or self.conf, **(context_changes or {}))
        path = self.parent / "snapshots" / f"cycle-{cycle}"
        digest = fake_bytes_snapshot(path, expected, **kwargs)
        return expected, path, digest

    def _commit(self, cycle):
        expected, path, digest = self._snap(cycle)
        precommit_maze_cycle(self.root, self.conf, cycle, expected)
        self.assertEqual(commit_maze_cycle(
            self.root, self.conf, cycle, expected, path, digest), digest)
        return path, digest

    def test_full_two_cycle_chain_resumes_only_verified_complete_bytes(self):
        start = inspect_maze_journal(self.root, self.conf)
        self.assertEqual((start.committed_cycles, start.pending_cycle), (0, None))
        self.assertEqual(start.last_checkpoint_sha256, "d" * 64)
        first, digest0 = self._commit(0)
        self.assertEqual(inspect_maze_journal(self.root, self.conf).last_checkpoint_path,
                         first)
        second, digest1 = self._commit(1)
        final = inspect_maze_journal(self.root, self.conf)
        self.assertEqual((final.committed_cycles, final.pending_cycle), (2, None))
        self.assertEqual(final.last_checkpoint_sha256, digest1)
        self.assertEqual(final.last_checkpoint_path, second)
        self.assertEqual(final.qualification, SOURCE_ONLY)
        self.assertNotEqual(digest0, digest1)

    def test_unresolved_intent_stays_unknown_and_forbids_fresh_cycle(self):
        expected = context(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        status = inspect_maze_journal(self.root, self.conf)
        self.assertEqual(status.pending_cycle, 0)
        self.assertFalse(status.safe_to_begin_next_cycle)
        with self.assertRaises(MazeJournalUnknown):
            precommit_maze_cycle(self.root, self.conf, 0, expected)
        with self.assertRaises(MazeJournalUnknown):
            precommit_maze_cycle(self.root, self.conf, 1, context(1))
        # A trusted operator may reconcile an exact already-published snapshot,
        # but this is never an automatic retry of the optimizer step.
        path = self.parent / "snapshots" / "cycle-0"
        sha = fake_bytes_snapshot(path, expected)
        commit_maze_cycle(self.root, self.conf, 0, expected, path, sha)
        self.assertEqual(inspect_maze_journal(self.root, self.conf).pending_cycle, None)

    def test_commit_write_fault_keeps_intent_and_forensic_evidence(self):
        expected, path, digest = self._snap(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        original = __import__(
            "labyrinth.maze_crash_resume_journal", fromlist=["_write_once"]
        )._write_once

        def fail_only_commit(target, payload):
            if target.name.endswith(".commit.json"):
                raise MazeJournalUnknown("injected interrupted commit")
            return original(target, payload)

        with patch("labyrinth.maze_crash_resume_journal._write_once",
                   side_effect=fail_only_commit):
            with self.assertRaises(MazeJournalUnknown):
                commit_maze_cycle(self.root, self.conf, 0, expected, path, digest)
        status = inspect_maze_journal(self.root, self.conf)
        self.assertEqual(status.pending_cycle, 0)
        self.assertTrue((self.root / "cycle-0000.intent.json").is_file())
        self.assertFalse((self.root / "cycle-0000.commit.json").exists())
        self.assertTrue(path.is_dir())

    def test_post_write_fsync_fault_needs_readback_not_blind_retry(self):
        expected, path, digest = self._snap(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        with patch("labyrinth.maze_crash_resume_journal._fsync_directory",
                   side_effect=OSError("simulated ambiguous fsync")):
            with self.assertRaises(MazeJournalUnknown):
                commit_maze_cycle(self.root, self.conf, 0, expected, path, digest)
        # After an acknowledged fsync failure the caller has UNKNOWN. The
        # observed file is not silently replaced or the optimizer re-run.
        self.assertTrue((self.root / "cycle-0000.commit.json").exists())
        self.assertEqual(inspect_maze_journal(self.root, self.conf).committed_cycles, 1)
        with self.assertRaises(MazeJournalError):
            commit_maze_cycle(self.root, self.conf, 0, expected, path, digest)

    def test_context_reordering_and_fence_or_predecessor_forgery_refused(self):
        first = context(0)
        for invalid in (
            replace(first, fence=13),
            replace(first, objective_id="foreign.objective"),
            replace(first, base_checkpoint_sha256="f" * 64),
            replace(first, prior_checkpoint_sha256="f" * 64),
            replace(first, exposures=first.exposures[::-1]),
            replace(first, cursor=0),
        ):
            if invalid.exposures != first.exposures or invalid.cursor == 0:
                # Reordering/zero cursor can be locally valid but cannot reuse
                # an already-committed source intent (tested after precommit).
                continue
            with self.subTest(invalid=invalid), self.assertRaises(MazeJournalError):
                precommit_maze_cycle(self.root, self.conf, 0, invalid)
        precommit_maze_cycle(self.root, self.conf, 0, first)
        for changed in (replace(first, exposures=first.exposures[::-1]),
                        replace(first, cursor=1)):
            cp = self.parent / "snapshots" / f"bad-{changed.cursor}-{changed.exposures[0][0]}"
            digest = fake_bytes_snapshot(cp, changed)
            with self.assertRaises(MazeJournalError):
                commit_maze_cycle(self.root, self.conf, 0, changed, cp, digest)

    def test_wrong_run_candidate_scope_fence_source_never_takeover(self):
        for changed in (
            replace(self.conf, run_id="foreign"),
            replace(self.conf, candidate_artifact_sha256="f" * 64),
            replace(self.conf, scope_fingerprint="f" * 64),
            replace(self.conf, source_sha256="f" * 64),
            replace(self.conf, arm="inverted"),
            replace(self.conf, fence=13),
        ):
            with self.subTest(changed=changed), self.assertRaises(MazeJournalError):
                inspect_maze_journal(self.root, changed)
        self.assertEqual(inspect_maze_journal(self.root, self.conf).committed_cycles, 0)

    def test_partial_commit_record_or_unexpected_entry_fail_closed(self):
        precommit_maze_cycle(self.root, self.conf, 0, context(0))
        partial = self.root / "cycle-0000.commit.json"
        partial.write_bytes(b'{"schema":')
        with self.assertRaisesRegex(MazeJournalError, "partial|malformed"):
            inspect_maze_journal(self.root, self.conf)
        partial.unlink()
        (self.root / "cycle-0002.intent.json").write_bytes(b"{}\n")
        with self.assertRaises(MazeJournalError):
            inspect_maze_journal(self.root, self.conf)

    def test_checkpoint_digest_context_tensor_and_optimizer_are_not_optional(self):
        expected = context(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        cases = (
            ("wrong-manifest", {}, False, "f" * 64),
            ("wrong-context", {"context": _context_json(context(1))}, False, None),
            ("wrong-tensor", {}, True, None),
            ("wrong-optimizer", {"optimizer": "SGD"}, False, None),
            ("missing-optimizer", {"optimizer_tree": None}, False, None),
        )
        for name, changes, damaged, pinned in cases:
            with self.subTest(name=name):
                path = self.parent / "snapshots" / name
                sha = fake_bytes_snapshot(path, expected,
                                          manifest_changes=changes,
                                          bad_tensor=damaged)
                if name == "missing-optimizer":
                    # Explicit no-cold/absent moment requirement cannot be
                    # fully verified without a Torch optimizer load. A null
                    # optimizer tree must at least fail source readback.
                    pass
                with self.assertRaises(MazeJournalError):
                    commit_maze_cycle(self.root, self.conf, 0, expected, path,
                                      pinned if pinned else sha)

    def test_cold_optimizer_moments_refused_except_named_zero_update_control(self):
        empty_state = {"type": "dict", "items": [
            [{"type": "scalar", "value": "state"},
             {"type": "dict", "items": []}],
            [{"type": "scalar", "value": "param_groups"},
             {"type": "list", "items": [{"type": "dict", "items": []}]}],
        ]}
        expected = context(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        path = self.parent / "snapshots" / "cold-moments"
        digest = fake_bytes_snapshot(
            path, expected, manifest_changes={"optimizer_tree": empty_state})
        with self.assertRaises(MazeJournalError):
            commit_maze_cycle(self.root, self.conf, 0, expected, path, digest)

        # A preregistered zero-update control has NO optimizer.step and no
        # moments by construction; accepting its empty state is intentional.
        control = replace(self.conf, arm="zero_update", run_id="control-zero")
        separate = Path(self.temp.name) / "control-zero"
        separate.mkdir()
        journal = separate / "journal"
        initialize_maze_journal(journal, control)
        control_context = context(0, control)
        precommit_maze_cycle(journal, control, 0, control_context)
        control_path = separate / "cycle-0"
        control_digest = fake_bytes_snapshot(
            control_path, control_context,
            manifest_changes={"optimizer_tree": empty_state})
        self.assertEqual(commit_maze_cycle(
            journal, control, 0, control_context,
            control_path, control_digest), control_digest)

    def test_incomplete_checkpoint_dir_refused_without_replacing_intent(self):
        expected = context(0)
        precommit_maze_cycle(self.root, self.conf, 0, expected)
        path = self.parent / "snapshots" / "partial"
        digest = fake_bytes_snapshot(path, expected)
        (path / "state.safetensors").unlink()
        with self.assertRaises(MazeJournalError):
            commit_maze_cycle(self.root, self.conf, 0, expected, path, digest)
        self.assertEqual(inspect_maze_journal(self.root, self.conf).pending_cycle, 0)

    def test_lost_checkpoint_bytes_block_even_previously_committed_receipt(self):
        path, _ = self._commit(0)
        (path / "state.safetensors").write_bytes(b"tampered-after-commit")
        with self.assertRaises(MazeJournalError):
            inspect_maze_journal(self.root, self.conf)

    def test_second_commit_or_reverse_cycle_never_replays(self):
        first_path, first_sha = self._commit(0)
        with self.assertRaises(MazeJournalError):
            commit_maze_cycle(self.root, self.conf, 0, context(0),
                              first_path, first_sha)
        with self.assertRaises(MazeJournalError):
            precommit_maze_cycle(self.root, self.conf, 0, context(0))
        with self.assertRaises(ValueError):
            precommit_maze_cycle(self.root, self.conf, 2, context(1))
        with self.assertRaises(MazeJournalError):
            initialize_maze_journal(self.root, self.conf)

    def test_valid_older_journal_snapshot_rollback_remains_possible(self):
        # Hostile qualification falsifier: local filesystem copies are NOT a
        # monotonic host anchor, even when all bytes and digests match.
        self._commit(0)
        clone_parent = Path(self.temp.name) / "cloned-arm"
        shutil.copytree(self.parent, clone_parent)
        self._commit(1)
        cloned = clone_parent / "journal"
        self.assertEqual(inspect_maze_journal(self.root, self.conf).committed_cycles, 2)
        self.assertEqual(inspect_maze_journal(cloned, self.conf).committed_cycles, 1)
        # On a cloned local authority, the second transition can be published
        # again. This is the reason Q8 stays FAIL until a separate host anchor.
        expected = context(1)
        precommit_maze_cycle(cloned, self.conf, 1, expected)
        snap = clone_parent / "snapshots" / "cycle-1"
        digest = fake_bytes_snapshot(snap, expected)
        commit_maze_cycle(cloned, self.conf, 1, expected, snap, digest)
        self.assertEqual(inspect_maze_journal(cloned, self.conf).committed_cycles, 2)


if __name__ == "__main__":
    unittest.main()
