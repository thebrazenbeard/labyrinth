import inspect

import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from labyrinth.minotaur_v6_latent import (
        MODULUS, LatentConfig, build_latent_model, grouped_pair_split, train_and_evaluate,
    )
@unittest.skipIf(torch is None, 'Optional PyTorch is exercised in neural CI')
class MinotaurV6LatentTests(unittest.TestCase):
    def test_model_is_compact_seeded_and_has_no_pair_table_or_target_input(self):
        first = build_latent_model(LatentConfig(seed=11, width=32, harmonics=8))
        second = build_latent_model(LatentConfig(seed=11, width=32, harmonics=8))
        assert sum(p.numel() for p in first.parameters()) < 100_000
        assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
        assert "target" not in inspect.signature(first.forward).parameters
        assert all(MODULUS * MODULUS not in p.shape for p in first.parameters())


    def test_forward_shape_gradients_and_residue_representatives(self):
        model = build_latent_model(LatentConfig(seed=3, width=32, harmonics=8))
        x = torch.tensor([0, 96, 101], dtype=torch.long)
        y = torch.tensor([96, 2, 193], dtype=torch.long)
        assert torch.equal(model(x, y), model(x + 97 * 100, y + 97 * 7))
        logits = model(x, y)
        assert logits.shape == (3, MODULUS)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())


    def test_grouped_holdout_has_no_pair_overlap_and_no_unseen_residues(self):
        split = grouped_pair_split()
        train_pairs = set(zip(split["train_x"].tolist(), split["train_y"].tolist()))
        test_pairs = set(zip(split["test_x"].tolist(), split["test_y"].tolist()))
        assert not train_pairs & test_pairs
        assert len(train_pairs) + len(test_pairs) == MODULUS * MODULUS
        assert set(split["train_x"].tolist()) == set(range(MODULUS))
        assert set(split["train_y"].tolist()) == set(range(MODULUS))
        assert set(split["test_x"].tolist()) == set(range(MODULUS))
        assert set(split["test_y"].tolist()) == set(range(MODULUS))
        with self.assertRaises(ValueError):
            grouped_pair_split(heldout_cell=4)


    def test_bounded_training_reports_real_withheld_score(self):
        report = train_and_evaluate(LatentConfig(seed=5, width=24, harmonics=6),
                                    steps=4, batch_size=64, lr=0.003)
        assert report["train"]["total"] > 0
        assert report["withheld_groups"]["total"] > 0
        assert 0.0 <= report["withheld_groups"]["accuracy"] <= 1.0
