"""Bounded Maze-specific full-state AdamW continuation contracts.

Pure source checks still run without Torch. The optional physical test uses
genuine CPU AdamW, but is not a pretrained MiniLM execution or a host fence.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.minilm_optimizer_state_resume import (
    MazeOptimizerResumePlan, load_maze_optimizer_snapshot,
    save_maze_optimizer_snapshot, _require_complete_adamw_state,
)
from labyrinth.neural_resume_checkpoint import FullCheckpointError, save_full_checkpoint
from scripts.qualify_maze_conditioned_minilm_cpu import (
    PREREG_ARMS, build_training_cycles, default_scope,
)


def plan(**changes) -> MazeOptimizerResumePlan:
    scope = default_scope()
    weak = ("negative_polarity",)
    options = {
        "scope": scope,
        "cycles": build_training_cycles(scope, weak_cycle_one=weak),
        "weak_cycle_one": weak,
        "arm": "correct", "steps_per_cycle": 4,
        "learning_rate": 0.002,
        "logical_run_id": "maze-cpu-one",
        "base_checkpoint_sha256": "a" * 64,
        "prior_checkpoint_sha256": "b" * 64,
        "fence": 4,
    }
    options.update(changes)
    return MazeOptimizerResumePlan(**options)


class MazeFullStateSourceTests(unittest.TestCase):
    def test_actual_regenerated_maze_binds_complete_ordered_exposure_stream(self):
        current = plan()
        current.validate()
        self.assertEqual(current.total_steps, 8)
        visits = current.exposures()
        self.assertEqual(len(visits), 8 * 4)
        self.assertEqual(len(set(item[0] for item in visits)), len(visits))
        # Repeated practice visits retain the same assigned label commitment;
        # the unique occurrence ID disambiguates optimizer steps.
        self.assertLess(len(set(digest for _, digest in visits)), len(visits))
        self.assertTrue(all(len(digest) == 64 for _, digest in visits))
        self.assertEqual(current.context(0).cursor, 0)
        self.assertEqual(current.context(4).cursor, 16)
        self.assertEqual(current.context(8).cursor, 32)
        self.assertEqual(current.context(0).exposures, current.context(8).exposures)
        self.assertEqual(current.context(4).base_checkpoint_sha256, "a" * 64)
        self.assertEqual(current.context(4).prior_checkpoint_sha256, "b" * 64)
        self.assertEqual(len(current._spec()["trainer_source_sha256"]), 64)
        self.assertEqual(len(current._spec()["resume_source_sha256"]), 64)

    def test_arm_label_permutation_and_hyperparameters_cannot_share_identity(self):
        correct = plan()
        original = correct.exposures()
        inverted = plan(arm="inverted")
        randomized = plan(arm="randomized_labels")
        self.assertNotEqual(original, inverted.exposures())
        self.assertNotEqual(original, randomized.exposures())
        self.assertNotEqual(original[:4], randomized.exposures()[:4])
        for arm in PREREG_ARMS:
            other = plan(arm=arm)
            other.validate()
            if arm != "correct":
                self.assertNotEqual(correct.checkpoint_run_id, other.checkpoint_run_id)
        # zero_lr and zero_update share label assignments, but are separate
        # precommitted optimizer/control arms with distinct checkpoint IDs.
        self.assertEqual(original, plan(arm="zero_lr").exposures())
        self.assertNotEqual(correct.checkpoint_run_id,
                            plan(arm="zero_lr").checkpoint_run_id)
        self.assertNotEqual(correct.checkpoint_run_id,
                            plan(learning_rate=.001).checkpoint_run_id)
        self.assertNotEqual(correct.checkpoint_run_id,
                            plan(steps_per_cycle=8).checkpoint_run_id)

    def test_regeneration_and_scope_forgery_fails_closed(self):
        current = plan()
        failures = (
            replace(current, weak_cycle_one=()),
            replace(current, cycles=current.cycles[::-1]),
            replace(current, cycles=(current.cycles[0],)),
            replace(current, arm="operator_secret"),
            replace(current, steps_per_cycle=True),
            replace(current, learning_rate=float("nan")),
            replace(current, learning_rate=0),
            replace(current, logical_run_id="../out"),
            replace(current, base_checkpoint_sha256="not-a-checkpoint"),
            replace(current, prior_checkpoint_sha256="f"),
            replace(current, fence=2**63),
        )
        for invalid in failures:
            with self.subTest(invalid=repr(invalid)[:90]), self.assertRaises(ValueError):
                invalid.validate()

    def test_rejected_cursor_and_stale_lineage_before_loading_model(self):
        current = plan()
        for cursor in (-1, True, 9, 1.25):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                current.context(cursor)
        self.assertNotEqual(
            current.context(4), replace(current, fence=5).context(4))
        self.assertNotEqual(
            current.context(4), replace(current, prior_checkpoint_sha256="f"*64).context(4))

    def test_no_side_effects_if_optional_cpu_backend_missing(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("Torch is installed; optional physical suite exercises backend")
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "checkpoint"
            with self.assertRaisesRegex(FullCheckpointError, "torch"):
                save_maze_optimizer_snapshot(
                    destination, encoder=object(), head=object(), optimizer=object(),
                    plan=plan(), completed_steps=2)
            self.assertFalse(destination.exists())

    def test_duplicate_saved_adamw_group_ids_rejected_without_torch(self):
        # Hostile regression: group IDs [0,0] previously mapped two distinct
        # trainable names to a single state entry and passed a zero-update
        # checkpoint. This pure test must run even without torch installed.
        saved = {"state": {}, "param_groups": [
            {"params": [0, 0], "lr": .002, "amsgrad": False},
        ]}
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            with patch(
                "labyrinth.minilm_optimizer_state_resume._trainable_and_groups",
                return_value=({"encoder.a": object(), "head.b": object()},
                              [["encoder.a", "head.b"]]),
            ):
                with self.assertRaisesRegex(FullCheckpointError, "group IDs"):
                    _require_complete_adamw_state(
                        saved, combined=object(), optimizer=object(),
                        plan=plan(arm="zero_update"), completed_steps=0)

    @staticmethod
    def _canonical_group():
        # Explicit trainer AdamW defaults; never derive the test oracle from
        # the function being tested.
        return {
            "params": [0], "lr": .002, "betas": (0.9, 0.999),
            "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False,
            "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None,
        }

    def test_nondefault_adamw_optimizer_hyperparameters_fail_closed(self):
        baseline = self._canonical_group()
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            with patch(
                "labyrinth.minilm_optimizer_state_resume._trainable_and_groups",
                return_value=({"encoder.a": object()}, [["encoder.a"]]),
            ):
                # Control: the canonical zero-update group is structurally
                # valid with no moments or optimizer steps.
                _require_complete_adamw_state(
                    {"state": {}, "param_groups": [baseline]},
                    combined=object(), optimizer=object(),
                    plan=plan(arm="zero_update"), completed_steps=0)
                for field, wrong in (
                    ("betas", (.01, .02)), ("eps", 99.0),
                    ("weight_decay", 5.0), ("foreach", False),
                    ("maximize", True), ("fused", True),
                    ("lr", 0.001),
                ):
                    with self.subTest(field=field):
                        changed = dict(baseline, **{field: wrong})
                        with self.assertRaisesRegex(
                            FullCheckpointError, "hyperparameter"):
                            _require_complete_adamw_state(
                                {"state": {}, "param_groups": [changed]},
                                combined=object(), optimizer=object(),
                                plan=plan(arm="zero_update"), completed_steps=0)
                changed = dict(baseline, initial_lr=.002)
                with self.assertRaisesRegex(FullCheckpointError, "hyperparameter"):
                    _require_complete_adamw_state(
                        {"state": {}, "param_groups": [changed]},
                        combined=object(), optimizer=object(),
                        plan=plan(arm="zero_update"), completed_steps=0)

    def test_plain_integer_adamw_step_counter_rejected_without_torch(self):
        class FakeTensor:
            pass
        # The malicious state has valid-looking moment keys and correct cursor,
        # but a Python integer is not the scalar tensor emitted by torch AdamW.
        fake_torch = SimpleNamespace(Tensor=FakeTensor)
        saved = {
            "state": {0: {"step": 1, "exp_avg": object(), "exp_avg_sq": object()}},
            "param_groups": [self._canonical_group()],
        }
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch(
                "labyrinth.minilm_optimizer_state_resume._trainable_and_groups",
                return_value=({"encoder.a": object()}, [["encoder.a"]]),
            ):
                with self.assertRaisesRegex(FullCheckpointError, "scalar step tensor"):
                    _require_complete_adamw_state(
                        saved, combined=object(), optimizer=object(),
                        plan=plan(), completed_steps=1)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "optional torch+safetensors physical CPU dependencies unavailable",
)
class PhysicalMazeAdamWResumeTests(unittest.TestCase):
    @staticmethod
    def _make():
        import torch
        torch.manual_seed(203)
        encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                      torch.nn.Linear(4, 4))
        for param in encoder[0].parameters():
            param.requires_grad_(False)
        head = torch.nn.Linear(4, 2)
        params = [param for param in encoder.parameters() if param.requires_grad]
        params += list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=0.002)
        return encoder, head, optimizer

    @staticmethod
    def _step(encoder, head, optimizer):
        import torch
        optimizer.zero_grad(set_to_none=True)
        x = torch.tensor([[.6, -.4, .8]], dtype=torch.float32)
        y = torch.tensor([1])
        loss = torch.nn.functional.cross_entropy(head(encoder(x)), y)
        loss.backward()
        optimizer.step()

    def test_real_cpu_adamw_moments_and_parameters_match_continuous_training(self):
        import torch
        conf = plan()
        encoder, head, optimizer = self._make()
        self._step(encoder, head, optimizer)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "maze-step1"
            receipt = save_maze_optimizer_snapshot(
                target, encoder=encoder, head=head, optimizer=optimizer,
                plan=conf, completed_steps=1)
            self.assertEqual(receipt.context, conf.context(1))
            self.assertTrue((target / "state.safetensors").is_file())
            self.assertTrue((target / "manifest.json").is_file())
            self._step(encoder, head, optimizer)
            expected_params = [param.detach().clone() for module in (encoder, head)
                               for param in module.parameters() if param.requires_grad]
            expected_moments = [(state["exp_avg"].detach().clone(),
                                 state["exp_avg_sq"].detach().clone(),
                                 state["step"].detach().clone())
                                for state in optimizer.state.values()]
            resumed_encoder, resumed_head, resumed_optimizer = self._make()
            load_maze_optimizer_snapshot(
                target, encoder=resumed_encoder, head=resumed_head,
                optimizer=resumed_optimizer, plan=conf, completed_steps=1,
                independently_pinned_checkpoint_sha256=receipt.sha256)
            self._step(resumed_encoder, resumed_head, resumed_optimizer)
            actual_params = [param.detach() for module in (resumed_encoder, resumed_head)
                             for param in module.parameters() if param.requires_grad]
            self.assertTrue(all(torch.equal(a, b)
                                for a, b in zip(expected_params, actual_params)))
            actual_moments = [(state["exp_avg"], state["exp_avg_sq"], state["step"])
                              for state in resumed_optimizer.state.values()]
            self.assertTrue(all(torch.equal(a, b) for x, y in
                                zip(expected_moments, actual_moments) for a, b in zip(x, y)))

    def test_corrupt_digest_stale_fence_and_optimizer_lr_rejected(self):
        import torch
        encoder, head, optimizer = self._make()
        self._step(encoder, head, optimizer)
        conf = plan()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "one"
            receipt = save_maze_optimizer_snapshot(target, encoder=encoder,
                head=head, optimizer=optimizer, plan=conf, completed_steps=1)
            before = [param.detach().clone() for p in (encoder, head)
                      for param in p.parameters() if param.requires_grad]
            for digest, changed_conf in (("f" * 64, conf),
                                         (receipt.sha256, replace(conf, fence=99))):
                with self.assertRaises(FullCheckpointError):
                    load_maze_optimizer_snapshot(target, encoder=encoder,
                        head=head, optimizer=optimizer, plan=changed_conf,
                        completed_steps=1,
                        independently_pinned_checkpoint_sha256=digest)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                before, (param.detach() for p in (encoder, head)
                         for param in p.parameters() if param.requires_grad))))
            wrong_encoder, wrong_head, wrong_optimizer = self._make()
            wrong_optimizer.param_groups[0]["lr"] = .01
            with self.assertRaisesRegex(FullCheckpointError, "learning rate"):
                load_maze_optimizer_snapshot(target, encoder=wrong_encoder,
                    head=wrong_head, optimizer=wrong_optimizer, plan=conf,
                    completed_steps=1,
                    independently_pinned_checkpoint_sha256=receipt.sha256)

    def test_incomplete_adamw_moments_cannot_be_published_at_nonzero_cursor(self):
        encoder, head, optimizer = self._make()
        self._step(encoder, head, optimizer)
        key = next(iter(optimizer.state))
        del optimizer.state[key]["exp_avg_sq"]
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "incomplete"
            with self.assertRaisesRegex(FullCheckpointError, "missing AdamW"):
                save_maze_optimizer_snapshot(
                    destination, encoder=encoder, head=head,
                    optimizer=optimizer, plan=plan(), completed_steps=1)
            self.assertFalse(destination.exists())

    def test_digest_valid_but_cold_optimizer_refused_before_model_mutation(self):
        import torch
        encoder, head, optimizer = self._make()
        self._step(encoder, head, optimizer)
        optimizer.state.clear()  # Existing generic substrate permits this.
        conf = plan()
        combined = torch.nn.Module()
        combined.add_module("encoder", encoder)
        combined.add_module("head", head)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "digest-valid-incomplete"
            receipt = save_full_checkpoint(
                destination, model=combined, optimizer=optimizer,
                context=conf.context(1))
            resumed_encoder, resumed_head, resumed_optimizer = self._make()
            before = [param.detach().clone()
                      for module in (resumed_encoder, resumed_head)
                      for param in module.parameters() if param.requires_grad]
            with self.assertRaisesRegex(FullCheckpointError, "missing AdamW"):
                load_maze_optimizer_snapshot(
                    destination, encoder=resumed_encoder, head=resumed_head,
                    optimizer=resumed_optimizer, plan=conf, completed_steps=1,
                    independently_pinned_checkpoint_sha256=receipt.sha256)
            after = [param.detach()
                     for module in (resumed_encoder, resumed_head)
                     for param in module.parameters() if param.requires_grad]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
            self.assertFalse(resumed_optimizer.state)


if __name__ == "__main__":
    unittest.main()
