"""Real offline-wheel regression for the package-private MiniLM source seam.

A source checkout happens to contain ``scripts``; an installed wheel does not.
The public Labyrinth contracts must import in an isolated interpreter without
shipping a generic ``scripts`` package or loading a neural backend.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (
    "labyrinth.five_arm_experiment_contract",
    "labyrinth.five_arm_statistical_screen",
    "labyrinth.minilm_optimizer_state_resume",
    "labyrinth.measured_neural_batch_binding",
    "labyrinth.measured_sentiment_corridors",
    "labyrinth.fivearm_campaign_admission_r13",
    "labyrinth.fivearm_capture_boundary_r15",
)


class PrivateMiniLMWheelTests(unittest.TestCase):
    def test_real_wheel_imports_without_any_top_level_scripts(self):
        with tempfile.TemporaryDirectory(prefix="labyrinth-wheel-private-r60-") as temp:
            root = Path(temp)
            project = root / "source"
            project.mkdir()
            for filename in ("pyproject.toml", "README.md"):
                shutil.copy2(ROOT / filename, project / filename)
            # Deliberately omit source-tree-only `scripts` from the build input.
            shutil.copytree(ROOT / "labyrinth", project / "labyrinth",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

            wheels_dir = root / "wheel"
            wheels_dir.mkdir()
            build = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                 "--no-build-isolation", "--wheel-dir", str(wheels_dir), str(project)],
                cwd=root, capture_output=True, text=True, timeout=90, check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            wheels = tuple(wheels_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)

            installed = root / "site"
            installed.mkdir()
            with zipfile.ZipFile(wheels[0]) as archive:
                names = frozenset(archive.namelist())
                self.assertFalse(any(name.startswith("scripts/") for name in names))
                self.assertIn("labyrinth/maze_conditioned_minilm_cpu_experiment.py", names)
                self.assertIn("labyrinth/minilm_cpu_experiment.py", names)
                self.assertFalse(any(name.endswith((".pt", ".safetensors", ".gguf"))
                                     for name in names))
                entry_paths = [name for name in names
                               if name.endswith(".dist-info/entry_points.txt")]
                self.assertEqual(len(entry_paths), 1)
                entry = archive.read(entry_paths[0]).decode("utf-8")
                self.assertIn("labyrinth = labyrinth.cli:main", entry)
                self.assertNotIn("qualify_", entry)
                archive.extractall(installed)

            # A foreign distribution's same-name top-level source must never
            # redirect the installed multi-seed subprocess to that module.
            foreign_scripts = installed / "scripts"
            foreign_scripts.mkdir()
            (foreign_scripts / "qualify_maze_conditioned_minilm_cpu.py").write_text(
                "raise RuntimeError('foreign module must not execute')\n",
                encoding="utf-8",
            )

            code = (
                "import importlib,pathlib,sys;"
                f"sys.path.insert(0,{str(installed)!r});"
                f"mods={CONSUMERS!r};"
                "[importlib.import_module(m) for m in mods];"
                "from labyrinth.maze_multiseed_protocol_cpu_experiment import "
                "ProtocolSpec, build_plan, child_command;"
                "spec=ProtocolSpec(); job=build_plan(spec)['manifest']['jobs'][0];"
                "cmd=child_command(job,pathlib.Path('/tmp'),spec);"
                "assert cmd[2]=='labyrinth.maze_conditioned_minilm_cpu_experiment';"
                "assert 'scripts' not in sys.modules;"
                "assert 'torch' not in sys.modules;"
                "assert 'transformers' not in sys.modules"
            )
            before = frozenset(root.iterdir())
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=root, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(frozenset(root.iterdir()), before,
                             "isolated import unexpectedly produced output artifacts")


if __name__ == "__main__":
    unittest.main()
