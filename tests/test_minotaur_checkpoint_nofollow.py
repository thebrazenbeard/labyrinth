"""Fail closed at Windows checkpoint handles and before costly neural replay."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labyrinth.minotaur_public_checkpoint import _read_regular, _safe_path


class PublicCheckpointFilesystemTests(unittest.TestCase):
    def test_windows_regular_reads_do_not_use_link_following_os_open(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            path.write_bytes(b'{"case":"safe"}')
            if os.name == "nt":
                with patch("labyrinth.minotaur_public_checkpoint.os.open",
                           side_effect=AssertionError("os.open follows links on Windows")):
                    self.assertEqual(_read_regular(path, 200), b'{"case":"safe"}')
            else:
                self.assertEqual(_read_regular(path, 200), b'{"case":"safe"}')

    def test_junction_directory_path_refuses_before_read(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "junction" / "case.json"
            original = type(path).is_junction
            def is_junction(instance):
                if instance.name == "junction":
                    return True
                return original(instance)
            with patch.object(type(path), "is_junction", is_junction):
                with self.assertRaisesRegex(ValueError, "junction"):
                    _safe_path(path, must_exist=True)

    def test_replacing_regular_file_during_open_refuses(self):
        from labyrinth import minotaur_public_checkpoint as public
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            candidate = Path(folder) / "replacement.json"
            path.write_bytes(b'{"legitimate":1}')
            candidate.write_bytes(b'{"malicious":2}')
            original = public._open_checkpoint_readonly
            def replace_before_open(target):
                os.replace(candidate, path)
                return original(target)
            with patch.object(public, "_open_checkpoint_readonly",
                              side_effect=replace_before_open):
                with self.assertRaisesRegex(ValueError, "file_changed_during_open"):
                    _read_regular(path, 200)

    def test_swapped_final_symlink_during_open_refuses(self):
        from labyrinth import minotaur_public_checkpoint as public
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_bytes(b'{"trusted":1}')
            destination = Path(folder) / "other.json"
            destination.write_bytes(b'{"attacker":2}')
            race_link = Path(folder) / "replacement-link.json"
            try:
                race_link.symlink_to(destination)
            except OSError:
                # Some Windows installations cannot create symlinks. The
                # regular-file replacement test still verifies fstat identity.
                return
            original = public._open_checkpoint_readonly
            def replace_with_link(target):
                os.replace(race_link, path)
                return original(target)
            with patch.object(public, "_open_checkpoint_readonly",
                              side_effect=replace_with_link):
                with self.assertRaises((ValueError, OSError)):
                    _read_regular(path, 200)



if __name__ == "__main__":
    unittest.main()
