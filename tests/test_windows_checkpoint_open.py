"""Windows must open checkpoints by a non-following native handle."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.toy_backend import ToyLinearAdapter

@unittest.skipUnless(os.name == "nt", "Windows native handle requirement")
class WindowsCheckpointOpenTests(unittest.TestCase):
    def test_valid_regular_checkpoint_uses_native_nonfollowing_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            ToyLinearAdapter(seed=4).save_checkpoint(checkpoint)
            with patch("labyrinth.toy_backend.os.open",
                       side_effect=AssertionError("os.open can follow symlinks on Windows")):
                loaded = ToyLinearAdapter.load_checkpoint(checkpoint)
            self.assertEqual(loaded.seed, 4)
