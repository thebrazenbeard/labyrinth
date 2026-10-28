"""Bounded semantic research needs repeated training visits without relaxing
checkpoint/size guarantees or accepting unbounded optimizer loops."""
import tempfile
import unittest
from pathlib import Path

from labyrinth.hf_pilot import PilotConfig, validate_config, _training_visit_indices
from labyrinth.maze import generate_maze


class ExtendedBudgetTests(unittest.TestCase):
    def test_repeated_epoch_budget_96_is_admitted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf8")
            (model / "model.safetensors").write_bytes(b"fixture")
            cfg = PilotConfig(model_path=str(model), output_dir=str(root / "run"),
                              seed=17, train_count=32, eval_count=16,
                              tiers=(3,), steps=96, answer_only=True)
            validate_config(cfg)
            train = generate_maze(seed=17, count=32, tiers=(3,), split="train")
            visits = _training_visit_indices(train, seed=17, steps=96)
            self.assertEqual(len(visits), 96)
            self.assertEqual(set(visits[:32]), set(range(32)))
            self.assertEqual(set(visits[32:64]), set(range(32)))
            self.assertEqual(set(visits[64:96]), set(range(32)))

    def test_hard_limit_rejects_129_steps(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf8")
            (model / "model.safetensors").write_bytes(b"fixture")
            cfg = PilotConfig(model_path=str(model), output_dir=str(root / "run"),
                              steps=129)
            with self.assertRaisesRegex(ValueError, "steps"):
                validate_config(cfg)
