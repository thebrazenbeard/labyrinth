"""R92 source/CPU negative and crash tests for cycle-boundary custody.

Real Torch+safetensors tests skip explicitly when optional CPU dependencies are
not installed. Source-only tests do not pretend to execute tensor training.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.neural_resume_checkpoint import FullCheckpointContextV2
from labyrinth import neural_cycle_checkpoint_r92 as cycle

try:
    import torch
    from safetensors.torch import load_file  # noqa: F401
except ImportError:
    torch = None


_BASE = "a" * 64
_INITIAL = "b" * 64
_EXPOSURES = (("visit0", "c" * 64), ("visit1", "d" * 64))


def _context(index: int = 0, cursor: int = 1,
             predecessor: str = _INITIAL) -> FullCheckpointContextV2:
    return FullCheckpointContextV2(
        run_id="r92-adapter", objective_id="maze-sentiment",
        base_checkpoint_sha256=_BASE, prior_checkpoint_sha256=_INITIAL,
        fence=3, cursor=cursor, exposures=_EXPOSURES[:cursor],
        cycle_index=index, immediate_predecessor_sha256=predecessor,
    )


def _prior() -> cycle.CycleBoundaryReceipt:
    return cycle.CycleBoundaryReceipt(
        manifest_sha256="1" * 64, run_id="r92-adapter", cycle_index=0,
        stage="trained", base_artifact_sha256=_BASE,
        frozen_base_sha256="2" * 64, adapter_head_sha256="3" * 64,
        full_state_sha256="4" * 64, predecessor_manifest_sha256=cycle._ZERO,
        completed_optimizer_steps=1, cursor=1,
        optimizer_moments_sha256="5" * 64,
    )


class SourceBoundaryTests(unittest.TestCase):
    def test_no_replace_atomic_directory_publication_including_empty_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first, second, target = (parent / s for s in ("one", "two", "target"))
            first.mkdir()
            second.mkdir()
            target.mkdir()  # A naive POSIX os.rename would replace this!
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._publish_noreplace(first, target)
            self.assertTrue(first.is_dir())
            self.assertTrue(target.is_dir())
            target.rmdir()
            cycle._publish_noreplace(first, target)
            self.assertFalse(first.exists())
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._publish_noreplace(second, target)
            self.assertTrue(second.is_dir())

    def test_initial_stage_rejects_skip_and_regeneration_first(self):
        for idx, stage in ((1, "trained"), (0, "regeneration_ready"),
                           (0, "unknown")):
            with self.subTest(idx=idx, stage=stage):
                with self.assertRaises(cycle.CycleCheckpointError):
                    cycle._validate_transition(None, context=_context(idx),
                                               stage=stage, steps=1,
                                               base=_BASE, frozen="2" * 64,
                                               adapter="3" * 64, moments="5" * 64)

    def test_stage_and_step_monotonicity_rejects_skips_replays_and_wrong_base(self):
        prev = _prior()
        for index, stage, steps, base, frozen in (
            (0, "trained", 1, _BASE, "2" * 64),  # replay
            (1, "trained", 2, _BASE, "2" * 64),  # skipped regeneration
            (0, "regeneration_ready", 2, _BASE, "2" * 64),  # optimizer in regen
            (0, "regeneration_ready", 1, "f" * 64, "2" * 64),
            (0, "regeneration_ready", 1, _BASE, "e" * 64),
        ):
            with self.subTest(index=index, stage=stage, steps=steps):
                ctx = _context(index, cursor=2, predecessor=(
                    _INITIAL if index == 0 else prev.full_state_sha256))
                with self.assertRaises(cycle.CycleCheckpointError):
                    cycle._validate_transition(prev, context=ctx, stage=stage,
                                               steps=steps, base=base, frozen=frozen,
                                               adapter="3" * 64, moments="5" * 64)
        self.assertEqual(
            cycle._validate_transition(
                prev, context=_context(0),
                stage="regeneration_ready", steps=1, base=_BASE,
                frozen="2" * 64, adapter="3" * 64,
                moments="5" * 64), prev.manifest_sha256,
        )

    def test_context_predecessor_mismatch_and_bool_step_rejected(self):
        prev = _prior()
        for predecessor, steps in ((prev.full_state_sha256, 1), (_INITIAL, True)):
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._validate_transition(
                    prev, context=_context(0, predecessor=predecessor),
                    stage="regeneration_ready", steps=steps,
                    base=_BASE, frozen="2" * 64,
                    adapter="3" * 64, moments="5" * 64)

    def test_regeneration_at_same_optimizer_step_cannot_advance_cursor(self):
        # A new exposure cannot be consumed during the no-training stage.
        with self.assertRaises(cycle.CycleCheckpointError):
            cycle._validate_transition(
                _prior(), context=_context(0, cursor=2),
                stage="regeneration_ready", steps=1,
                base=_BASE, frozen="2" * 64,
                adapter="3" * 64, moments="5" * 64)

    def test_regeneration_rejects_digest_drift_with_unchanged_step_and_cursor(self):
        for adapter, moments in (("e" * 64, "5" * 64),
                                 ("3" * 64, "f" * 64)):
            with self.subTest(adapter=adapter, moments=moments):
                with self.assertRaises(cycle.CycleCheckpointError):
                    cycle._validate_transition(
                        _prior(), context=_context(),
                        stage="regeneration_ready", steps=1,
                        base=_BASE, frozen="2" * 64,
                        adapter=adapter, moments=moments)

    def test_named_adamw_moment_digest_detects_byte_drift_without_rng(self):
        # Tiny tensor protocol exercises digest construction when real Torch is
        # unavailable; CPU Torch roundtrips have their own gated tests below.
        class Tensor:
            def __init__(self, payload, shape=(2,), value=None):
                self.payload, self.shape, self.value = payload, shape, value
                self.dtype = "torch.float32"
                self.device = SimpleNamespace(type="cpu")
                self.is_sparse = self.is_quantized = False

            def numel(self):
                return 1 if self.shape == () else 2

            def item(self):
                return self.value

            def is_floating_point(self):
                return True

            def is_complex(self):
                return False

            def detach(self):
                return self

            def contiguous(self):
                return self

            def reshape(self, *_):
                return self

            def view(self, *_):
                return self

            def numpy(self):
                return SimpleNamespace(tobytes=lambda: self.payload)

        torch_stub = SimpleNamespace(
            Tensor=Tensor, uint8="uint8",
            isfinite=lambda _value: SimpleNamespace(all=lambda: True))
        state = {
            "state": {0: {
                "step": Tensor(b"\x00\x00\x80?", shape=(), value=1),
                "exp_avg": Tensor(b"first-state"),
                "exp_avg_sq": Tensor(b"squared-state"),
            }},
            "param_groups": [{"params": [0], "amsgrad": False}],
        }
        optimizer = SimpleNamespace(state_dict=lambda: state)
        with patch.object(cycle, "_trainable_and_groups",
                          return_value=({"head.weight": Tensor(b"parameter")}, [["head.weight"]])):
            before = cycle._optimizer_moments_digest(
                object(), optimizer, 1, torch_stub)
            state["state"][0]["exp_avg"] = Tensor(b"other-state")
            after = cycle._optimizer_moments_digest(
                object(), optimizer, 1, torch_stub)
        self.assertNotEqual(before, after)

    def test_serialized_readback_mismatch_prevents_boundary_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "not-published"
            def known_tensor_digest(_model, _torch, *, frozen):
                return "2" * 64 if frozen else "3" * 64

            with (patch.object(cycle, "_dependencies", return_value=(None, None, None)),
                  patch.object(cycle, "_tensor_digest", side_effect=known_tensor_digest),
                  patch.object(cycle, "_optimizer_moments_digest", return_value="5" * 64),
                  patch.object(cycle, "save_full_checkpoint",
                               return_value=SimpleNamespace(sha256="4" * 64)),
                  patch.object(cycle, "_readback_digests",
                               return_value=("f" * 64, "5" * 64))):
                with self.assertRaisesRegex(cycle.CycleCheckpointError,
                                            "readback differs"):
                    cycle.save_cycle_checkpoint(
                        destination, model=object(), optimizer=object(),
                        context=_context(), stage="trained",
                        base_artifact_sha256=_BASE,
                        expected_frozen_base_sha256="2" * 64,
                        completed_optimizer_steps=1)
            self.assertFalse(destination.exists())

    def test_caller_context_cannot_change_predecessor_after_transition_validation(self):
        # Mutation of a frozen caller-owned context at the serializer seam
        # must not replace the predecessor whose lineage was just checked.
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "published"
            previous = replace(_prior(), stage="regeneration_ready")
            context = _context(1, cursor=2,
                               predecessor=previous.full_state_sha256)
            observed = []

            def save_state(path, *, model, optimizer, context):
                observed.append(context)
                object.__setattr__(
                    mutable_caller, "immediate_predecessor_sha256", "e" * 64)
                path.mkdir()
                (path / "manifest.json").write_text("mock-state")
                (path / "state.safetensors").write_bytes(b"mock-tensors")
                return SimpleNamespace(sha256="4" * 64)

            mutable_caller = context
            with (patch.object(cycle, "_dependencies", return_value=(None, None, None)),
                  patch.object(cycle, "_tensor_digest",
                               side_effect=lambda _m, _t, *, frozen: "2" * 64 if frozen else "3" * 64),
                  patch.object(cycle, "_optimizer_moments_digest", return_value="5" * 64),
                  patch.object(cycle, "save_full_checkpoint", side_effect=save_state),
                  patch.object(cycle, "_readback_digests", return_value=("3" * 64, "5" * 64))):
                receipt = cycle.save_cycle_checkpoint(
                    destination, model=object(), optimizer=object(),
                    context=context, stage="trained", previous=previous,
                    base_artifact_sha256=_BASE,
                    expected_frozen_base_sha256="2" * 64,
                    completed_optimizer_steps=2)
            self.assertEqual(receipt.predecessor_manifest_sha256, previous.manifest_sha256)
            self.assertEqual(observed[0].immediate_predecessor_sha256,
                             previous.full_state_sha256)
            self.assertIsNot(observed[0], mutable_caller)

    def test_restore_uses_stable_context_after_signed_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "full_state").mkdir()
            context = _context()
            obj = {
                "schema": cycle.SCHEMA, "run_id": context.run_id,
                "cycle_index": 0, "stage": "trained",
                "base_artifact_sha256": _BASE,
                "frozen_base_sha256": "2" * 64,
                "adapter_head_sha256": "3" * 64,
                "optimizer_moments_sha256": "5" * 64,
                "full_state_sha256": "4" * 64,
                "predecessor_manifest_sha256": cycle._ZERO,
                "completed_optimizer_steps": 1, "cursor": 1,
                "optimizer": "AdamW", "optimizer_state_required": True,
                "graduation_authorized": False,
                "change_class": "model_component_update",
                "state_scope": "local_adapter_head_and_optimizer_only",
                "memory_or_identity_write_authorized": False,
            }
            manifest = cycle._canonical(obj)
            (source / "cycle_manifest.json").write_bytes(manifest)
            observed = []

            def readback(*_args, **_kwargs):
                object.__setattr__(context, "immediate_predecessor_sha256", "e" * 64)
                return "3" * 64, "5" * 64

            def restore_state(*_args, **kwargs):
                observed.append(kwargs["expected_context"])

            with (patch.object(cycle, "_dependencies", return_value=(None, None, None)),
                  patch.object(cycle, "_tensor_digest",
                               side_effect=lambda _m, _t, *, frozen: "2" * 64 if frozen else "3" * 64),
                  patch.object(cycle, "_optimizer_moments_digest", return_value="5" * 64),
                  patch.object(cycle, "_readback_digests", side_effect=readback),
                  patch.object(cycle, "load_full_checkpoint", side_effect=restore_state),
                  patch.object(cycle, "_assert_complete_adamw")):
                receipt = cycle.load_cycle_checkpoint(
                    source, model=object(), optimizer=object(),
                    expected_context=context,
                    independently_pinned_manifest_sha256=cycle._receipt_digest(manifest),
                    expected_base_artifact_sha256=_BASE,
                    expected_frozen_base_sha256="2" * 64,
                    expected_predecessor_manifest_sha256=cycle._ZERO,
                    expected_stage="trained", completed_optimizer_steps=1)
            self.assertEqual(receipt.stage, "trained")
            self.assertEqual(observed[0].immediate_predecessor_sha256, _INITIAL)
            self.assertIsNot(observed[0], context)

    def test_publish_crash_seams_distinguish_unsynced_and_durable_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            sync_calls = []
            with patch.object(cycle, "_sync_directory",
                              side_effect=lambda path: sync_calls.append(path)):
                staging = parent / "unsynced-staging"
                staging.mkdir()
                with self.assertRaises(cycle.InjectedCycleCrash):
                    cycle._publish_and_sync(staging, parent / "unsynced",
                                            "before_sync")
                self.assertTrue((parent / "unsynced").is_dir())
                self.assertEqual(sync_calls, [])

                staging = parent / "durable-staging"
                staging.mkdir()
                with self.assertRaises(cycle.InjectedCycleCrash):
                    cycle._publish_and_sync(staging, parent / "durable",
                                            "after_publish")
                self.assertTrue((parent / "durable").is_dir())
                self.assertEqual(sync_calls, [parent])

    def test_manifest_receipt_is_non_authorizing_and_domain_separated(self):
        prior = _prior()
        self.assertFalse(prior.graduation_authorized)
        self.assertNotEqual(cycle._receipt_digest(b"{}\n"),
                            __import__("hashlib").sha256(b"{}\n").hexdigest())
        with self.assertRaises(ValueError):
            cycle.save_cycle_checkpoint(
                Path("would-not-exist"), model=None, optimizer=None,
                context=_context(), stage="trained", base_artifact_sha256=_BASE,
                expected_frozen_base_sha256="2" * 64,
                completed_optimizer_steps=1, failure_at="arbitrary",)

    def test_existing_directory_rejected_without_overwrite_or_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "commit"
            root.mkdir()
            sentinel = root / "user-file"
            sentinel.write_text("must remain")
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._check_root(root, existing=False)
            self.assertEqual(sentinel.read_text(), "must remain")

    def test_symlink_and_partial_snapshot_are_not_recovery_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "partially-written"
            root.mkdir()
            (root / "cycle_manifest.json").write_bytes(b"{}\n")
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._check_root(root, existing=True)
            link = parent / "spoof"
            try:
                link.symlink_to(root, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("OS does not allow local symlinks")
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle._check_root(link, existing=True)


@unittest.skipUnless(torch is not None, "optional torch+safetensors unavailable: no real CPU checkpoint test")
class RealTorchCycleCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(101)

        class AdapterHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(3, 3)
                self.head = torch.nn.Linear(3, 2)
                for p in self.base.parameters():
                    p.requires_grad_(False)

            def forward(self, x):
                return self.head(self.base(x))

        self.model = AdapterHead()
        self.optimizer = torch.optim.AdamW(self.model.head.parameters(), lr=0.01)
        self.frozen = cycle._tensor_digest(self.model, torch, frozen=True)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.model(torch.ones((2, 3))).square().sum()
        loss.backward()
        self.optimizer.step()

    def save(self, name="cycle0", *, context=None, stage="trained", steps=1,
             previous=None, failure_at=None):
        return cycle.save_cycle_checkpoint(
            self.root / name, model=self.model, optimizer=self.optimizer,
            context=context or _context(), stage=stage,
            base_artifact_sha256=_BASE,
            expected_frozen_base_sha256=self.frozen,
            completed_optimizer_steps=steps, previous=previous,
            failure_at=failure_at)

    def restore(self, name, receipt, context, steps):
        return cycle.load_cycle_checkpoint(
            self.root / name, model=self.model, optimizer=self.optimizer,
            expected_context=context,
            independently_pinned_manifest_sha256=receipt.manifest_sha256,
            expected_base_artifact_sha256=_BASE,
            expected_frozen_base_sha256=self.frozen,
            expected_predecessor_manifest_sha256=receipt.predecessor_manifest_sha256,
            expected_stage=receipt.stage, completed_optimizer_steps=steps)

    def test_real_train_save_restart_and_full_adamw_moment_restore(self):
        self.step()
        context = _context()
        trained = self.save()
        original = {n: p.detach().clone() for n, p in self.model.head.named_parameters()}
        moments = {k: v.detach().clone() for k, v in
                   self.optimizer.state[next(iter(self.optimizer.state))].items()}
        self.step()
        with torch.no_grad():
            for p in self.model.head.parameters():
                p.add_(3)
        restored = self.restore("cycle0", trained, context, 1)
        self.assertEqual(restored, trained)
        self.assertEqual(restored.adapter_head_sha256,
                         cycle._tensor_digest(self.model, torch, frozen=False))
        for n, p in self.model.head.named_parameters():
            torch.testing.assert_close(p, original[n])
        for k, v in self.optimizer.state[next(iter(self.optimizer.state))].items():
            torch.testing.assert_close(v, moments[k])

    def test_missing_optimizer_moments_and_frozen_base_drift_fail_closed(self):
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("pre-step")
        self.step()
        param = next(iter(self.optimizer.state))
        self.optimizer.state[param].pop("exp_avg")
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("missing-moment")
        self.optimizer.state[param]["exp_avg"] = torch.zeros_like(param)
        self.model.base.weight.data.add_(1)
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("wrong-base")

    def test_tamper_fails_before_model_mutation(self):
        self.step()
        receipt = self.save()
        path = self.root / "cycle0" / "full_state" / "state.safetensors"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        before = cycle._tensor_digest(self.model, torch, frozen=False)
        with self.assertRaises(cycle.CycleCheckpointError):
            self.restore("cycle0", receipt, _context(), 1)
        self.assertEqual(cycle._tensor_digest(self.model, torch, frozen=False), before)

    def test_fault_injection_before_and_after_atomic_publish(self):
        self.step()
        for point in ("after_state", "after_manifest"):
            with self.assertRaises(cycle.InjectedCycleCrash):
                self.save(point, failure_at=point)
            self.assertFalse((self.root / point).exists())
        with self.assertRaises(cycle.InjectedCycleCrash):
            self.save("after-publish", failure_at="after_publish")
        # A lost acknowledgment does not imply the transaction was rolled back.
        self.assertTrue((self.root / "after-publish" / "cycle_manifest.json").exists())
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("after-publish")

    def test_monotonic_cycle_zero_regen_then_cycle_one_train(self):
        self.step()
        trained = self.save()
        regen_context = _context(0)
        regenerated = self.save("regen", context=regen_context,
                                stage="regeneration_ready", previous=trained)
        self.step()
        next_context = _context(1, cursor=2,
                                predecessor=regenerated.full_state_sha256)
        next_cycle = self.save("cycle1", context=next_context, stage="trained",
                               steps=2, previous=regenerated)
        self.assertEqual(next_cycle.predecessor_manifest_sha256,
                         regenerated.manifest_sha256)
        self.assertNotEqual(next_cycle.adapter_head_sha256,
                            trained.adapter_head_sha256)

    def test_regeneration_can_change_rng_but_not_moment_or_model_digest(self):
        self.step()
        trained = self.save()
        torch.manual_seed(2026)
        regenerated = self.save("regen-rng", stage="regeneration_ready",
                                previous=trained)
        self.assertNotEqual(trained.full_state_sha256,
                            regenerated.full_state_sha256)
        self.assertEqual(trained.adapter_head_sha256,
                         regenerated.adapter_head_sha256)
        self.assertEqual(trained.optimizer_moments_sha256,
                         regenerated.optimizer_moments_sha256)

    def test_loader_binds_manifest_moments_to_saved_bytes_before_restore(self):
        self.step()
        trained = self.save()
        manifest_path = self.root / "cycle0" / "cycle_manifest.json"
        obj = json.loads(manifest_path.read_bytes())
        obj["optimizer_moments_sha256"] = "f" * 64
        tampered = cycle._canonical(obj)
        manifest_path.write_bytes(tampered)
        newly_pinned = replace(
            trained, manifest_sha256=cycle._receipt_digest(tampered))
        before = cycle._tensor_digest(self.model, torch, frozen=False)
        with self.assertRaises(cycle.CycleCheckpointError):
            self.restore("cycle0", newly_pinned, _context(), 1)
        self.assertEqual(before, cycle._tensor_digest(self.model, torch, frozen=False))

    def test_regeneration_rejects_same_step_adapter_and_optimizer_moment_drift(self):
        self.step()
        trained = self.save()
        head_parameter = next(self.model.head.parameters())
        with torch.no_grad():
            head_parameter.add_(0.125)
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("regen-head-drift", stage="regeneration_ready",
                      previous=trained)
        with torch.no_grad():
            head_parameter.sub_(0.125)
        first_state = self.optimizer.state[next(iter(self.optimizer.state))]
        first_state["exp_avg"].add_(0.125)
        with self.assertRaises(cycle.CycleCheckpointError):
            self.save("regen-moment-drift", stage="regeneration_ready",
                      previous=trained)

    def test_restore_rejects_wrong_pins_without_changing_model(self):
        self.step()
        trained = self.save()
        before = cycle._tensor_digest(self.model, torch, frozen=False)
        for kwargs in ({"expected_base_artifact_sha256": "f" * 64},
                       {"expected_predecessor_manifest_sha256": "f" * 64},
                       {"expected_stage": "regeneration_ready"}):
            args = dict(source=self.root / "cycle0", model=self.model,
                        optimizer=self.optimizer, expected_context=_context(),
                        independently_pinned_manifest_sha256=trained.manifest_sha256,
                        expected_base_artifact_sha256=_BASE,
                        expected_frozen_base_sha256=self.frozen,
                        expected_predecessor_manifest_sha256=cycle._ZERO,
                        expected_stage="trained", completed_optimizer_steps=1)
            args.update(kwargs)
            with self.assertRaises(cycle.CycleCheckpointError):
                cycle.load_cycle_checkpoint(**args)
            self.assertEqual(cycle._tensor_digest(self.model, torch, frozen=False), before)


if __name__ == "__main__":
    unittest.main()
