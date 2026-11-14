"""Guard against editor-open command execution hidden in source assets.

These are source checks only. They never invoke an editor, Node, or an asset.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
FONT_HEADERS = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1")
# Fail closed if repository source traversal grows unexpectedly large.
MAX_SCAN_DIRECTORIES = 20000
MAX_SCAN_FILES = 100000


def source_files_under(root):
    """Walk source directories without executing files or following symlink dirs."""
    if not root.exists():
        return
    directories = 0
    files = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        directories += 1
        if directories > MAX_SCAN_DIRECTORIES:
            raise AssertionError("source traversal exceeded directory bound")
        # Git internals are not project source and cannot be editor workspaces.
        names[:] = [name for name in names if name != ".git"]
        for name in names:
            if (Path(directory) / name).is_symlink():
                raise AssertionError(f"source directory symlink hides content: {directory}/{name}")
        for name in filenames:
            files += 1
            if files > MAX_SCAN_FILES:
                raise AssertionError("source traversal exceeded file bound")
            yield Path(directory) / name


def task_entries(document):
    """Handle both VS Code tasks.json and code-workspace task layouts."""
    if not isinstance(document, dict):
        raise AssertionError("editor task document must be an object")
    tasks = document.get("tasks", [])
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])
    if not isinstance(tasks, list):
        raise AssertionError("editor tasks must be a list")
    for task in tasks:
        if not isinstance(task, dict):
            raise AssertionError("editor task entry must be an object")
        options = task.get("runOptions", {})
        if not isinstance(options, dict):
            raise AssertionError("editor task runOptions must be an object")
        yield options.get("runOn")


class EditorAutostartGuardTests(unittest.TestCase):
    def _check_fixture(self, root, test_name, *, should_reject):
        isolated_test = EditorAutostartGuardTests(test_name)
        # Same module globals even when unittest discovery imports by bare name.
        with patch.dict(getattr(isolated_test, test_name).__globals__, {"ROOT": root}):
            if should_reject:
                with self.assertRaises(AssertionError):
                    getattr(isolated_test, test_name)()
            else:
                getattr(isolated_test, test_name)()

    def test_rejects_workspace_file_folder_open_in_both_task_layouts(self):
        # This is an inert JSON fixture, not an executed editor command.
        for tasks in ([{"label": "inert-fixture", "type": "shell",
                        "command": "never-executed", "runOptions": {"runOn": "folderOpen"}}],
                      {"tasks": [{"label": "inert-fixture", "type": "shell",
                                  "command": "never-executed",
                                  "runOptions": {"runOn": "folderOpen"}}]}):
            with self.subTest(tasks_type=type(tasks).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "Attack.code-workspace").write_text(
                        json.dumps({"folders": [{"path": "."}], "tasks": tasks}),
                        encoding="utf-8")
                    self._check_fixture(root, "test_no_editor_folder_open_commands",
                                        should_reject=True)

    def test_rejects_uppercase_ttf_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            font = root / "public" / "fonts" / "FAKEFONT.TTF"
            font.parent.mkdir(parents=True)
            font.write_bytes(b"not a font")
            self._check_fixture(root, "test_ttf_assets_have_font_magic_not_executable_source",
                                should_reject=True)

    def test_rejects_nested_ttf_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            font = root / "public" / "fonts" / "nested" / "fake.ttf"
            font.parent.mkdir(parents=True)
            font.write_bytes(b"not a font")
            self._check_fixture(root, "test_ttf_assets_have_font_magic_not_executable_source",
                                should_reject=True)

    def test_rejects_symlink_directory_bypass(self):
        # A link into another directory must not make unseen assets invisible.
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            root = container / "source"
            font_parent = root / "public" / "fonts"
            font_parent.mkdir(parents=True)
            external = container / "external"
            external.mkdir()
            (external / "fake.ttf").write_bytes(b"not a font")
            try:
                (font_parent / "nested").symlink_to(external, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlink fixture unsupported")
            self._check_fixture(root, "test_ttf_assets_have_font_magic_not_executable_source",
                                should_reject=True)

    def test_benign_workspace_and_nested_font_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "Safe.code-workspace"
            workspace.write_text(json.dumps({"tasks": {"tasks": [
                {"label": "manual", "type": "shell", "command": "not-executed",
                 "runOptions": {"runOn": "manual"}}]}}), encoding="utf-8")
            font = root / "public" / "fonts" / "nested" / "real.TTF"
            font.parent.mkdir(parents=True)
            font.write_bytes(b"OTTO" + b"fixture-only")
            self._check_fixture(root, "test_no_editor_folder_open_commands",
                                should_reject=False)
            self._check_fixture(root, "test_ttf_assets_have_font_magic_not_executable_source",
                                should_reject=False)

    def test_no_editor_folder_open_commands(self):
        tasks = ROOT / ".vscode" / "tasks.json"
        if tasks.exists():
            document = json.loads(tasks.read_text(encoding="utf-8-sig"))
            for run_on in task_entries(document):
                self.assertNotEqual(run_on, "folderOpen",
                                    f"editor-open command is forbidden: {tasks}")

        for workspace in source_files_under(ROOT):
            if workspace.name.lower().endswith(".code-workspace"):
                self.assertFalse(workspace.is_symlink(),
                                 f"editor workspace cannot be a symlink: {workspace}")
                self.assertLessEqual(workspace.stat().st_size, 1_048_576,
                                     f"workspace config exceeds scan bound: {workspace}")
                document = json.loads(workspace.read_text(encoding="utf-8-sig"))
                for run_on in task_entries(document):
                    self.assertNotEqual(run_on, "folderOpen",
                                        f"editor-open command is forbidden: {workspace}")

        settings = ROOT / ".vscode" / "settings.json"
        if settings.exists():
            document = json.loads(settings.read_text(encoding="utf-8-sig"))
            self.assertIsNot(document.get("task.allowAutomaticTasks"), True)
            self.assertNotEqual(document.get("tasks", {}).get("runOn"),
                                "folderOpen", "settings cannot trigger a shell on open")

    def test_ttf_assets_have_font_magic_not_executable_source(self):
        for font in source_files_under(ROOT / "public" / "fonts"):
            if font.suffix.lower() != ".ttf":
                continue
            with self.subTest(path=font.relative_to(ROOT)):
                self.assertFalse(font.is_symlink(),
                                 f"font asset cannot be a symlink: {font.name}")
                with font.open("rb") as stream:
                    header = stream.read(4)
                self.assertIn(header, FONT_HEADERS,
                              f"non-font content masquerades as {font.name}")


if __name__ == "__main__":
    unittest.main()
