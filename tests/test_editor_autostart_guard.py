"""Guard against editor-open command execution hidden in source assets.

These are source checks only. They never invoke an editor, Node, or an asset.
"""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class EditorAutostartGuardTests(unittest.TestCase):
    def test_no_editor_folder_open_commands(self):
        tasks = ROOT / ".vscode" / "tasks.json"
        if tasks.exists():
            document = json.loads(tasks.read_text(encoding="utf-8-sig"))
            for task in document.get("tasks", ()):
                self.assertNotEqual(task.get("runOptions", {}).get("runOn"),
                                    "folderOpen", "editor-open command is forbidden")

        settings = ROOT / ".vscode" / "settings.json"
        if settings.exists():
            document = json.loads(settings.read_text(encoding="utf-8-sig"))
            self.assertIsNot(document.get("task.allowAutomaticTasks"), True)
            self.assertNotEqual(document.get("tasks", {}).get("runOn"),
                                "folderOpen", "settings cannot trigger a shell on open")

    def test_ttf_assets_have_font_magic_not_executable_source(self):
        for font in (ROOT / "public" / "fonts").glob("*.ttf"):
            with self.subTest(path=font.relative_to(ROOT)):
                with font.open("rb") as stream:
                    header = stream.read(4)
                self.assertIn(header, (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"),
                              f"non-font content masquerades as {font.name}")


if __name__ == "__main__":
    unittest.main()
