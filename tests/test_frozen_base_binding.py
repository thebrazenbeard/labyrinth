"""Opt-in frozen pretrained-model identity checks; Torch tests run where available."""

from __future__ import annotations

import unittest

from labyrinth import frozen_base_binding as binding


class FrozenBindingFormatTests(unittest.TestCase):
    def test_rejects_untrusted_non_digest_claim(self):
        with self.assertRaises(ValueError):
            binding.FrozenBaseIdentity("not a digest")

    def test_rejects_uppercase_digest(self):
        with self.assertRaises(ValueError):
            binding.FrozenBaseIdentity("A" * 64)

    def test_wrapper_rejects_context_with_unbound_base_digest_before_torch(self):
        from labyrinth.neural_resume_checkpoint import FullCheckpointContext

        identity = binding.FrozenBaseIdentity("a" * 64)
        context = FullCheckpointContext(
            run_id="ctx-test", objective_id="base-binding",
            base_checkpoint_sha256="b" * 64,
            prior_checkpoint_sha256="c" * 64,
            fence=0, cursor=0, exposures=(),
        )
        with self.assertRaises(binding.FrozenBaseError):
            binding.load_checkpoint_with_frozen_binding(
                "unused", model=object(), optimizer=object(),
                expected_sha256="d" * 64,
                expected_context=context, frozen_identity=identity,
            )


try:
    import torch
    from safetensors import torch as safetensors_torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "requires real CPU Torch and safetensors")
class FrozenBaseTorchTests(unittest.TestCase):
    def make_model(self):
        class Learner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 2)
                self.register_buffer("centering", torch.tensor([0.5, -0.25]))
                self.encoder.requires_grad_(False)

            def forward(self, x):
                return self.head(self.encoder(x) - self.centering)

        torch.manual_seed(1503)
        return Learner()

    def test_hash_is_repeatable_without_trainable_head_dependency(self):
        model = self.make_model()
        first = binding.digest_frozen_model_tensors(model)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        with torch.no_grad():
            model.head.weight.add_(7)
        self.assertEqual(first, binding.digest_frozen_model_tensors(model))

    def test_frozen_encoder_mutation_rejected_against_pinned_digest(self):
        model = self.make_model()
        pinned = binding.FrozenBaseIdentity(binding.digest_frozen_model_tensors(model))
        pinned.verify(model)
        with torch.no_grad():
            model.encoder.weight[0, 0] += 0.5
        with self.assertRaises(binding.FrozenBaseError):
            pinned.verify(model)

    def test_buffer_mutation_rejected(self):
        model = self.make_model()
        pinned = binding.FrozenBaseIdentity(binding.digest_frozen_model_tensors(model))
        with torch.no_grad():
            model.centering[0] += 0.25
        with self.assertRaises(binding.FrozenBaseError):
            pinned.verify(model)

    def test_rejects_empty_frozen_inventory(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaises(binding.FrozenBaseError):
            binding.digest_frozen_model_tensors(model)

    def test_rejects_non_cpu_frozen_tensor(self):
        # Meta device is available on CPU-only Torch builds, so this negative
        # guard has no conditional CUDA hardware skip.
        model = self.make_model().to(device="meta")
        with self.assertRaises(binding.FrozenBaseError):
            binding.digest_frozen_model_tensors(model)

    def test_checkpoint_wrapper_rejects_tampered_base_before_restoration(self):
        import tempfile
        from pathlib import Path

        from labyrinth.neural_resume_checkpoint import (
            FullCheckpointContext,
            FullCheckpointError,
            save_full_checkpoint,
        )

        model = self.make_model()
        trusted = binding.FrozenBaseIdentity(binding.digest_frozen_model_tensors(model))
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        context = FullCheckpointContext(
            run_id="binding-test", objective_id="frozen-base-test",
            base_checkpoint_sha256=trusted.digest,
            prior_checkpoint_sha256="a" * 64,
            fence=1, cursor=0, exposures=(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "snapshot"
            receipt = save_full_checkpoint(root, model=model,
                                           optimizer=optimizer, context=context)
            trusted.verify(model)
            loaded = binding.load_checkpoint_with_frozen_binding(
                root, model=model, optimizer=optimizer,
                expected_sha256=receipt.sha256,
                expected_context=context, frozen_identity=trusted,
            )
            self.assertEqual(receipt.sha256, loaded.sha256)
            original_encoder_bias = model.encoder.bias.detach().clone()
            with torch.no_grad():
                model.encoder.bias.add_(1)
            with self.assertRaises(binding.FrozenBaseError):
                binding.load_checkpoint_with_frozen_binding(
                    root, model=model, optimizer=optimizer,
                    expected_sha256=receipt.sha256,
                    expected_context=context, frozen_identity=trusted,
                )
            with self.assertRaises(FullCheckpointError):
                # An independently pinned false snapshot digest cannot load,
                # even when the frozen-model identity is correct.
                with torch.no_grad():
                    model.encoder.bias.copy_(original_encoder_bias)
                binding.load_checkpoint_with_frozen_binding(
                    root, model=model, optimizer=optimizer,
                    expected_sha256="f" * 64,
                    expected_context=context, frozen_identity=trusted,
                )


if __name__ == "__main__":
    unittest.main()
