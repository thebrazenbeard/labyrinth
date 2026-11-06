"""Byte-level offline evaluator boundary; no model or host is loaded."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.verify_minilm_fresh_process_r57 import (
    ArtifactVerificationError,
    ROLE_PATHS,
    verify_staged_bundle,
)


HEAD = "a" * 40
TREE = "b" * 40
RUN = "minilm-source-r57"


class IndependentByteReadbackTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name) / "bundle"
        self.root.mkdir()
        self.contents = {}
        for role, relative in ROLE_PATHS.items():
            contents = ("fixture:" + role + "\n").encode("ascii")
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            self.contents[role] = contents
        self.manifest = {
            "schema": "labyrinth.minilm-fresh-process-byte-readback.v1",
            "source_head": HEAD,
            "source_tree": TREE,
            "run_id": RUN,
            "trainer_pid": os.getpid(),
            "files": {
                role: {"path": relative, "sha256": hashlib.sha256(self.contents[role]).hexdigest(),
                       "size_bytes": len(self.contents[role])}
                for role, relative in ROLE_PATHS.items()
            },
        }
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self):
        data = (json.dumps(self.manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        self.manifest_path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def verify(self, **overrides):
        args = dict(bundle_root=self.root, manifest_path=self.manifest_path,
                    expected_manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                    expected_source_head=HEAD, expected_source_tree=TREE,
                    expected_run_id=RUN)
        args.update(overrides)
        return verify_staged_bundle(**args)

    def test_exact_staged_bytes_pass_but_authorize_nothing(self):
        report = self.verify()
        self.assertEqual(report["status"], "OFFLINE_STAGED_ARTIFACT_BYTES_MATCH_SOURCE_ONLY_UNATTESTED")
        self.assertEqual(set(report["files"]), set(ROLE_PATHS))
        for flag in ("model_loaded", "optimizer_executed", "independent_evaluator_custody_verified",
                     "private_evaluation_verified", "can_authorize_training",
                     "can_authorize_graduation", "can_authorize_deployment"):
            self.assertIs(report[flag], False)

    def test_all_six_role_bytes_must_match_after_manifest_is_frozen(self):
        for role, relative in ROLE_PATHS.items():
            with self.subTest(role=role):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaises(ArtifactVerificationError):
                    self.verify()
                path.write_bytes(original)

    def test_source_run_and_manifest_pins_cannot_be_self_substituted(self):
        for option, bad in (("expected_source_head", "c" * 40),
                            ("expected_source_tree", "c" * 40),
                            ("expected_run_id", "other-run"),
                            ("expected_manifest_sha256", "c" * 64)):
            with self.subTest(pin=option), self.assertRaises(ArtifactVerificationError):
                self.verify(**{option: bad})
        for option in ("expected_source_head", "expected_source_tree",
                       "expected_run_id", "expected_manifest_sha256"):
            with self.subTest(missing=option), self.assertRaises(ArtifactVerificationError):
                self.verify(**{option: ""})

    def test_duplicate_json_keys_and_noncanonical_encoding_fail_closed(self):
        data = self.manifest_path.read_bytes()
        self.manifest_path.write_bytes(data.replace(b'"run_id":', b'"run_id":"forged","run_id":', 1))
        with self.assertRaises(ArtifactVerificationError):
            self.verify()
        self.write_manifest()
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b" ")
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_unknown_authority_fields_and_partial_file_roster_fail_closed(self):
        self.manifest["graduation_authorized"] = True
        self.write_manifest()
        with self.assertRaises(ArtifactVerificationError):
            self.verify()
        del self.manifest["graduation_authorized"]
        del self.manifest["files"]["frozen_base"]
        self.write_manifest()
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_path_traversal_and_duplicate_role_aliases_rejected(self):
        self.manifest["files"]["adapter"]["path"] = "../outside"
        self.write_manifest()
        with self.assertRaises(ArtifactVerificationError):
            self.verify()
        self.manifest["files"]["adapter"]["path"] = ROLE_PATHS["classifier_head"]
        self.write_manifest()
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_symlinks_are_rejected_even_if_bytes_match(self):
        target = self.root / ROLE_PATHS["adapter"]
        alias = target.with_suffix(".original")
        target.rename(alias)
        target.symlink_to(alias)
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_bundle_and_manifest_ancestor_symlinks_are_rejected(self):
        # A symlink above the final `bundle` and `manifest.json` components
        # previously bypassed both Path.is_symlink() guards.
        parent_alias = Path(self._temp.name) / "alias-parent"
        parent_alias.symlink_to(Path(self._temp.name), target_is_directory=True)
        alias_root = parent_alias / "bundle"
        alias_manifest = alias_root / "manifest.json"
        with self.assertRaises(ArtifactVerificationError):
            self.verify(bundle_root=alias_root, manifest_path=alias_manifest)
        with self.assertRaises(ArtifactVerificationError):
            self.verify(manifest_path=alias_manifest)
        with self.assertRaises(ArtifactVerificationError):
            self.verify(bundle_root=alias_root)

    def test_nested_artifact_directory_symlink_is_rejected(self):
        source = self.root / "optimizer-state"
        moved = self.root / "optimizer-real"
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_wrong_bytes_kind_and_oversized_manifest_fail_closed(self):
        path = self.root / ROLE_PATHS["optimizer_state"]
        path.unlink()
        path.mkdir()
        with self.assertRaises(ArtifactVerificationError):
            self.verify()
        self.manifest_path.write_bytes(b"x" * (65536 + 1))
        with self.assertRaises(ArtifactVerificationError):
            self.verify()

    def test_cli_is_an_actual_separate_process_with_no_model_import(self):
        cli = Path(__file__).resolve().parents[1] / "scripts/verify_minilm_fresh_process_r57.py"
        result = subprocess.run([
            sys.executable, str(cli),
            "--bundle-root", str(self.root),
            "--manifest-file", str(self.manifest_path),
            "--expected-manifest-sha256", hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "--expected-source-head", HEAD,
            "--expected-source-tree", TREE,
            "--expected-run-id", RUN,
        ], text=True, capture_output=True, check=True)
        report = json.loads(result.stdout)
        self.assertNotEqual(report["evaluator_pid"], os.getpid())
        self.assertIs(report["model_loaded"], False)
        self.assertIs(report["independent_evaluator_custody_verified"], False)

    def test_cli_invalid_expected_pin_refuses_without_report(self):
        cli = Path(__file__).resolve().parents[1] / "scripts/verify_minilm_fresh_process_r57.py"
        result = subprocess.run([
            sys.executable, str(cli),
            "--bundle-root", str(self.root),
            "--manifest-file", str(self.manifest_path),
            "--expected-manifest-sha256", hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "--expected-source-head", "c" * 40,
            "--expected-source-tree", TREE,
            "--expected-run-id", RUN,
        ], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("refused", result.stderr)


if __name__ == "__main__":
    unittest.main()
