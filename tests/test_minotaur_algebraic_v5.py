import inspect

import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from labyrinth.minotaur_algebraic_v5 import AlgebraicConfig, build_algebraic_model


@unittest.skipIf(torch is None, 'optional torch belongs to neural CI')
class AlgebraicNeuralV5Tests(unittest.TestCase):
    def test_config_validation_and_seeded_construction(self):
        config = AlgebraicConfig(seed=19, width=32, rank=12)
        assert config.validate() is config

        first = build_algebraic_model(config)
        second = build_algebraic_model(config)
        for left, right in zip(first.parameters(), second.parameters()):
            assert torch.equal(left, right)

        for invalid in (
            AlgebraicConfig(width=0),
            AlgebraicConfig(rank=0),
            AlgebraicConfig(seed=-1),
            AlgebraicConfig(width=512, rank=512),
        ):
            with self.assertRaises((TypeError, ValueError)):
                invalid.validate()


    def test_output_shape_gradient_and_parameter_update(self):
        model = build_algebraic_model(AlgebraicConfig(seed=7, width=40, rank=16))
        op = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)
        x = torch.tensor([0, 96, 31, 1_000_000, 17], dtype=torch.long)
        y = torch.tensor([96, 0, 65, 97, 83], dtype=torch.long)
        target = torch.tensor([96, 96, 75, 26, 53], dtype=torch.long)

        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        logits = model(op, x, y)
        assert logits.shape == (5, 97)
        assert torch.isfinite(logits).all()

        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert trainable
        assert all(parameter.grad is not None for parameter in trainable)
        assert any(torch.count_nonzero(parameter.grad).item() for parameter in trainable)
        optimizer.step()
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


    def test_compact_shared_operator_has_no_atomic_pair_table_or_target_input(self):
        model = build_algebraic_model(AlgebraicConfig(seed=3, width=48, rank=20))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count < 150_000
        assert not hasattr(model, "atomic_logits")
        assert "target" not in inspect.signature(model.forward).parameters

        forbidden_rows = 3 * 97 * 97
        for name, parameter in model.named_parameters():
            assert forbidden_rows not in parameter.shape, name

        with self.assertRaises(TypeError):
            model(
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([2]),
                target=torch.tensor([3]),
            )


    def test_residue_representatives_and_input_checks(self):
        model = build_algebraic_model(AlgebraicConfig(seed=11, width=24, rank=8))
        op = torch.tensor([0, 1, 2], dtype=torch.long)
        canonical = model(op, torch.tensor([0, 96, 42]), torch.tensor([96, 0, 17]))
        shifted = model(
            op,
            torch.tensor([97 * 1000, 96 + 97 * 123, 42 + 97 * 999]),
            torch.tensor([96 + 97 * 5, 97 * 55, 17 + 97 * 314]),
        )
        assert torch.equal(canonical, shifted)

        with self.assertRaises((TypeError, ValueError)):
            model(op.float(), torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        with self.assertRaises((TypeError, ValueError)):
            model(torch.tensor([0, 1, 3]), torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        with self.assertRaises((TypeError, ValueError)):
            model(torch.tensor([0]), torch.tensor([1, 2]), torch.tensor([3]))


    def test_only_mathematically_commutative_operations_are_swap_invariant(self):
        model = build_algebraic_model(AlgebraicConfig(seed=23, width=32, rank=12))
        op = torch.tensor([0, 0, 2, 2], dtype=torch.long)
        x = torch.tensor([0, 96, 13, 81], dtype=torch.long)
        y = torch.tensor([72, 4, 65, 0], dtype=torch.long)
        assert torch.equal(model(op, x, y), model(op, y, x))

        # Subtraction is directional.  There is no assertion that swapped SUB
        # logits are negatives: modular negation is a permutation of 97 classes.
        sub = torch.ones(4, dtype=torch.long)
        assert not torch.equal(model(sub, x, y), model(sub, y, x))
