"""Offline installed-wheel CLI/restore continuity, not pretrained learning.

The public console entry must work with no source checkout, a foreign
``scripts`` package, and a fresh interpreter restoring a persisted toy model.
This is deliberately *only* a four-parameter toy persistence check.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, check=False)


class InstalledWheelCLIContinuityTests(unittest.TestCase):
    def test_fresh_process_checkpoint_readback_with_foreign_scripts(self):
        with tempfile.TemporaryDirectory(prefix="labyrinth-wheel-cli-r70-") as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            for filename in ("pyproject.toml", "README.md"):
                shutil.copy2(ROOT / filename, source / filename)
            shutil.copytree(ROOT / "labyrinth", source / "labyrinth",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

            wheelhouse = root / "wheels"
            wheelhouse.mkdir()
            build = _run([sys.executable, "-m", "pip", "wheel", "--no-index",
                          "--no-deps", "--no-build-isolation", "--wheel-dir",
                          str(wheelhouse), str(source)], cwd=root)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            wheels = list(wheelhouse.glob("*.whl"))
            self.assertEqual(len(wheels), 1)

            site = root / "site"
            site.mkdir()
            with zipfile.ZipFile(wheels[0]) as archive:
                self.assertFalse(any(name.startswith("scripts/")
                                     for name in archive.namelist()))
                archive.extractall(site)
            foreign = site / "scripts"
            foreign.mkdir()
            (foreign / "__init__.py").write_text(
                "raise RuntimeError('foreign scripts package was imported')\n",
                encoding="utf-8",
            )
            (foreign / "qualify_maze_conditioned_minilm_cpu.py").write_text(
                "raise RuntimeError('foreign experiment executed')\n",
                encoding="utf-8",
            )

            artifacts = root / "artifacts"
            artifacts.mkdir()
            checkpoint = artifacts / "trained.json"
            training_report = artifacts / "training.json"
            readback_report = artifacts / "readback.json"
            # -I excludes this source checkout and PYTHONPATH. The explicit
            # wheel target is the only project import root; CLI callable is
            # exactly the wheel's registered console entry target.
            entry = (
                "import sys,pathlib;"
                "sys.path.insert(0, sys.argv[1]);"
                "import labyrinth;"
                "assert pathlib.Path(labyrinth.__file__).resolve().is_relative_to("
                "pathlib.Path(sys.argv[1]).resolve());"
                "from labyrinth.cli import main;"
                "assert 'scripts' not in sys.modules;"
                "assert 'torch' not in sys.modules;"
                "result=main(sys.argv[2:]);"
                "assert 'scripts' not in sys.modules;"
                "assert 'torch' not in sys.modules;"
                "raise SystemExit(result)"
            )

            def cli(*args: str) -> subprocess.CompletedProcess[str]:
                return _run([sys.executable, "-I", "-B", "-c", entry,
                             str(site), *args], cwd=root)

            trained = cli("train-toy", "--seed", "17", "--train-count", "12",
                          "--eval-count", "12", "--epochs", "8",
                          "--learning-rate", "0.5", "--checkpoint-out",
                          str(checkpoint), "--report", str(training_report))
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            self.assertTrue(checkpoint.is_file())
            original_checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            training = json.loads(training_report.read_text(encoding="utf-8"))
            self.assertNotEqual(training["arms"]["trained"]["model_hash"],
                                training["arms"]["baseline"]["model_hash"])
            self.assertGreater(training["arms"]["trained"]["training"]["parameter_updates"], 0)

            recovered = cli("toy-eval", "--checkpoint", str(checkpoint),
                            "--seed", "17", "--eval-count", "12", "--report",
                            str(readback_report))
            self.assertEqual(recovered.returncode, 0,
                             recovered.stdout + recovered.stderr)
            readback = json.loads(readback_report.read_text(encoding="utf-8"))
            self.assertEqual(readback["model_hash"],
                             training["arms"]["trained"]["model_hash"])
            self.assertEqual(readback["accuracy"],
                             training["arms"]["trained"]["accuracy"])
            self.assertEqual(readback["trials"],
                             training["arms"]["trained"]["trials"])

            # A second command must not overwrite the persisted checkpoint.
            duplicate = cli("train-toy", "--seed", "17", "--train-count", "12",
                            "--eval-count", "12", "--epochs", "8",
                            "--learning-rate", "0.5", "--checkpoint-out",
                            str(checkpoint))
            self.assertEqual(duplicate.returncode, 2,
                             duplicate.stdout + duplicate.stderr)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             original_checkpoint_sha)
            self.assertEqual(json.loads(readback_report.read_text(encoding="utf-8")),
                             readback)

            # A wheel may be installed with --target into an unrelated Git
            # checkout. Structural coincidence must not turn that checkout's
            # foreign `scripts` package into our neural child execution path.
            (site / ".git").mkdir()
            shutil.copy2(ROOT / "pyproject.toml", site / "pyproject.toml")
            check = (
                "import pathlib,sys;"
                "sys.path.insert(0,sys.argv[1]);"
                "from labyrinth.maze_multiseed_protocol_cpu_experiment import "
                "ProtocolSpec,build_plan,child_command;"
                "spec=ProtocolSpec();"
                "job=build_plan(spec)['manifest']['jobs'][0];"
                "command=child_command(job,pathlib.Path(sys.argv[2]),spec);"
                "assert command[2]=='labyrinth.maze_conditioned_minilm_cpu_experiment',command"
            )
            path_guard = _run([sys.executable, "-I", "-B", "-c", check,
                               str(site), str(artifacts)], cwd=root)
            self.assertEqual(path_guard.returncode, 0,
                             path_guard.stdout + path_guard.stderr)


if __name__ == "__main__":
    unittest.main()
