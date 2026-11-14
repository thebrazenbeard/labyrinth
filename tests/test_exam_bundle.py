"""Source-only staged exam bundle: fail closed until directory publication."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labyrinth import exam_bundle


class StagedExamBundleTests(unittest.TestCase):
    def setUp(self):
        t = TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.bundle = self.root / "ready-exam"

    def create(self):
        return exam_bundle.prepare_exam_bundle(
            self.bundle, count=4, rng=random.Random(50011))

    def test_ready_directory_contains_complete_manifest_and_valid_original_exam(self):
        prepared = self.create()
        self.assertTrue(self.bundle.is_dir())
        self.assertEqual({p.name for p in self.bundle.iterdir()},
                         {"private.json", "public.json", "precommit.json", "manifest.json"})
        self.assertEqual(prepared["status"], "SOURCE_ONLY_READY_NOT_INDEPENDENT")
        self.assertEqual(prepared["case_count"], 4)
        verified = exam_bundle.verify_exam_bundle(
            self.bundle, expected_manifest_sha256=prepared["manifest_sha256"])
        self.assertEqual(verified["status"], "SOURCE_ONLY_VERIFIED_NOT_INDEPENDENT")
        self.assertEqual(verified["case_count"], 4)
        self.assertEqual(verified["precommit_receipt_sha256"],
                         json.loads((self.bundle / "precommit.json").read_text())["receipt_sha256"])
        self.assertFalse(verified["independent_custody_verified"])
        self.assertFalse(verified["neural_learning_verified"])
        self.assertFalse(any(p.name.startswith(".ready-exam.stage-")
                             for p in self.root.iterdir()))

    def test_missing_bundle_directory_is_not_ready(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256="0" * 64)

    def test_missing_parent_fails_without_visible_bundle(self):
        dest = self.root / "missing-parent" / "ready-exam"
        with self.assertRaises((FileNotFoundError, ValueError)):
            exam_bundle.prepare_exam_bundle(
                dest, count=2, rng=random.Random(50012))
        self.assertFalse(dest.exists())
        self.assertFalse(dest.parent.exists())

    def test_manifest_write_failure_leaves_no_ready_directory_or_stage(self):
        original = exam_bundle._write_once

        def reject_manifest(path, obj):
            if Path(path).name == "manifest.json":
                raise OSError("injected manifest write failure")
            return original(path, obj)

        with patch.object(exam_bundle, "_write_once", side_effect=reject_manifest):
            with self.assertRaisesRegex(OSError, "manifest write failure"):
                self.create()
        self.assertFalse(self.bundle.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_corrupted_staged_public_source_cannot_be_published(self):
        original = exam_bundle._write_once

        def tamper_at_manifest_write(path, obj):
            result = original(path, obj)
            if Path(path).name == "manifest.json":
                published_public = Path(path).parent / "public.json"
                published_public.write_bytes(published_public.read_bytes() + b" ")
            return result

        with patch.object(exam_bundle, "_write_once", side_effect=tamper_at_manifest_write):
            with self.assertRaises(ValueError):
                self.create()
        self.assertFalse(self.bundle.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_final_rename_failure_does_not_publish_partial_bundle(self):
        with patch.object(exam_bundle, "_publish_ready",
                          side_effect=OSError("injected final rename failure")):
            with self.assertRaisesRegex(OSError, "rename failure"):
                self.create()
        self.assertFalse(self.bundle.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unix_raced_empty_destination_is_not_silently_replaced(self):
        if os.name != "posix":
            self.skipTest("POSIX directory rename can replace an empty destination")
        real_exists = Path.exists
        checks = 0

        def race_between_check_and_publish(path, *args, **kwargs):
            nonlocal checks
            was_present = real_exists(path, *args, **kwargs)
            if path == self.bundle and not was_present:
                checks += 1
                if checks == 2:
                    # A separate actor creates an empty directory AFTER the
                    # pre-rename freshness check but BEFORE the actual rename.
                    self.bundle.mkdir()
            return was_present

        with patch.object(Path, "exists", new=race_between_check_and_publish):
            with self.assertRaises(FileExistsError):
                self.create()
        self.assertEqual(checks, 2)
        self.assertTrue(self.bundle.is_dir())
        self.assertEqual(list(self.bundle.iterdir()), [],
                         "concurrent empty directory must not be overwritten")
        self.assertFalse(any(".stage-" in p.name for p in self.root.iterdir()),
                         "rejected bundle stage must be cleaned")

    def test_linux_missing_no_replace_syscall_refuses_publication(self):
        if not os.sys.platform.startswith("linux"):
            self.skipTest("Linux-specific fail-closed feature detection")
        with patch.object(exam_bundle.ctypes, "CDLL", return_value=object()):
            with self.assertRaisesRegex(OSError, "renameat2"):
                self.create()
        self.assertFalse(self.bundle.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_final_bundle_is_never_modified(self):
        self.bundle.mkdir()
        witness = self.bundle / "operator-data.txt"
        witness.write_text("preserve this", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            self.create()
        self.assertEqual(witness.read_text(), "preserve this")
        self.assertEqual([p.name for p in self.bundle.iterdir()], ["operator-data.txt"])

    def test_dangling_final_symlink_never_receives_bundle(self):
        other = self.root / "dangling-target"
        try:
            os.symlink(other, self.bundle, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"filesystem symlink creation unavailable: {exc}")
        self.assertFalse(self.bundle.exists())
        with self.assertRaises((ValueError, FileExistsError)):
            self.create()
        self.assertFalse(other.exists())
        self.assertTrue(self.bundle.is_symlink())

    def test_parent_symlink_never_redirects_ready_bundle(self):
        target = self.root / "real-parent"
        target.mkdir()
        alias = self.root / "alias-parent"
        try:
            os.symlink(target, alias, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlink creation unavailable: {exc}")
        with self.assertRaises(ValueError):
            exam_bundle.prepare_exam_bundle(
                alias / "ready-exam", count=2, rng=random.Random(50014))
        self.assertFalse((target / "ready-exam").exists())

    def test_unpublished_staging_name_is_not_accepted_as_ready(self):
        prepared = self.create()
        staged = self.root / ".ready-exam.stage-interrupted"
        self.bundle.rename(staged)
        self.assertTrue((staged / "manifest.json").is_file())
        with self.assertRaisesRegex(ValueError, "ready|publish|name|staging"):
            exam_bundle.verify_exam_bundle(
                staged, expected_manifest_sha256=prepared["manifest_sha256"])

    def test_windows_junction_alias_cannot_masquerade_as_ready_bundle(self):
        if os.name != "nt":
            self.skipTest("Windows NTFS junction-only regression")
        prepared = self.create()
        alias = self.root / "redirected-ready"
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(alias), str(self.bundle)],
            capture_output=True, text=True)
        if result.returncode != 0:
            self.skipTest("isolated Windows junction creation unavailable")
        self.assertTrue(alias.is_dir())
        self.assertFalse(alias.is_symlink())
        with self.assertRaisesRegex(ValueError, "junction|reparse|ready|redirect"):
            exam_bundle.verify_exam_bundle(
                alias, expected_manifest_sha256=prepared["manifest_sha256"])

    def test_windows_junction_with_original_ready_name_must_still_fail(self):
        if os.name != "nt":
            self.skipTest("Windows NTFS junction-only regression")
        prepared = self.create()
        relocated_parent = self.root / "other-parent"
        relocated_parent.mkdir()
        original = relocated_parent / self.bundle.name
        self.bundle.rename(original)
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(self.bundle), str(original)],
            capture_output=True, text=True)
        if result.returncode != 0:
            self.skipTest("isolated Windows junction creation unavailable")
        self.assertEqual(self.bundle.name, original.name)
        self.assertTrue(self.bundle.is_dir())
        self.assertFalse(self.bundle.is_symlink())
        with self.assertRaisesRegex(ValueError, "junction|reparse|ready|redirect"):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=prepared["manifest_sha256"])

    def test_hardlinked_bundle_file_rejected_despite_valid_manifest(self):
        # Same bytes and original manifest SHA do not establish exclusive custody.
        for name in ("public.json", "private.json", "precommit.json", "manifest.json"):
            with self.subTest(name=name):
                ready = self.root / ("ready-" + name.replace(".", "-"))
                receipt = exam_bundle.prepare_exam_bundle(
                    ready, count=2, rng=random.Random(50011))
                linked = ready / name
                outside = self.root / ("external-" + name)
                os.link(linked, outside)
                self.assertGreater(linked.stat().st_nlink, 1)
                with self.assertRaisesRegex(ValueError, "hardlink|link count|multiple links"):
                    exam_bundle.verify_exam_bundle(
                        ready, expected_manifest_sha256=receipt["manifest_sha256"])

    def test_hardlink_created_during_manifest_read_invalidates_bundle(self):
        prepared = self.create()
        external = self.root / "raced-external-manifest.json"
        original = exam_bundle._file_bytes

        def link_during_read(path, maximum):
            if Path(path) == self.bundle / "manifest.json" and not external.exists():
                os.link(path, external)
            return original(path, maximum)

        with patch.object(exam_bundle, "_file_bytes", side_effect=link_during_read):
            with self.assertRaisesRegex(ValueError, "hardlink|link count"):
                exam_bundle.verify_exam_bundle(
                    self.bundle, expected_manifest_sha256=prepared["manifest_sha256"])
        self.assertEqual((self.bundle / "manifest.json").stat().st_nlink, 2)

    def test_symlink_swap_during_manifest_read_invalidates_bundle(self):
        if os.name != "posix":
            self.skipTest("repro uses POSIX symlink replacement")
        prepared = self.create()
        source = self.bundle / "manifest.json"
        external = self.root / "raced-external-copy.json"
        original = exam_bundle._file_bytes

        def redirect_during_read(path, maximum):
            if Path(path) == source and not external.exists():
                external.write_bytes(original(path, maximum))
                source.unlink()
                source.symlink_to(external)
            return original(path, maximum)

        with patch.object(exam_bundle, "_file_bytes", side_effect=redirect_during_read):
            with self.assertRaisesRegex(ValueError, "redirect|symlink|regular"):
                exam_bundle.verify_exam_bundle(
                    self.bundle, expected_manifest_sha256=prepared["manifest_sha256"])
        self.assertTrue(source.is_symlink())

    def test_wrong_held_manifest_pin_rejected(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "manifest|pin|SHA"):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256="0" * 64)

    def test_tampered_private_answers_rejected_with_original_pin(self):
        result = self.create()
        private = self.bundle / "private.json"
        private.write_bytes(private.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=result["manifest_sha256"])

    def test_tampered_public_case_rejected_with_original_pin(self):
        result = self.create()
        public = self.bundle / "public.json"
        public.write_bytes(public.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=result["manifest_sha256"])

    def test_tampered_manifest_rejected_even_if_caller_does_not_read_private(self):
        result = self.create()
        manifest = self.bundle / "manifest.json"
        doc = json.loads(manifest.read_bytes())
        doc["case_count"] = 99
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ValueError):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=result["manifest_sha256"])

    def test_extra_or_missing_files_make_bundle_invalid(self):
        result = self.create()
        (self.bundle / "bonus.json").write_text("untrusted")
        with self.assertRaises(ValueError):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=result["manifest_sha256"])
        (self.bundle / "bonus.json").unlink()
        (self.bundle / "private.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            exam_bundle.verify_exam_bundle(
                self.bundle, expected_manifest_sha256=result["manifest_sha256"])

    def test_source_only_cli_prepare_verify(self):
        from contextlib import redirect_stdout
        from io import StringIO
        out = StringIO()
        with redirect_stdout(out):
            exam_bundle.main(["prepare", "--out-bundle", str(self.bundle),
                              "--count", "2"])
        prepared = json.loads(out.getvalue())
        self.assertEqual(prepared["case_count"], 2)
        out = StringIO()
        with redirect_stdout(out):
            exam_bundle.main(["verify", "--bundle", str(self.bundle),
                              "--expected-manifest-sha256", prepared["manifest_sha256"]])
        verified = json.loads(out.getvalue())
        self.assertEqual(verified["status"], "SOURCE_ONLY_VERIFIED_NOT_INDEPENDENT")


if __name__ == "__main__":
    unittest.main()
