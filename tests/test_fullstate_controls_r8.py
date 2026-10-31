"""Source-level control-fork contract tests; no neural runtime is simulated."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from labyrinth.neural_resume_checkpoint import FullCheckpointContext, FullCheckpointReceipt
from labyrinth import fullstate_controls as controls


ARMS = ("baseline", "zero_update", "correct_labels", "corrupted_labels")


def origin():
    context = FullCheckpointContext(
        run_id="candidate-run-1", objective_id="program-reasoning",
        base_checkpoint_sha256="a" * 64, prior_checkpoint_sha256="b" * 64,
        fence=5, cursor=1, exposures=(("case-1", "c" * 64),),
    )
    return FullCheckpointReceipt("d" * 64, context, ("weight",))


def ids():
    return {arm: f"child-{arm}" for arm in ARMS}


class ForkPlanTests(unittest.TestCase):
    def test_plan_is_immutable_and_uses_one_pinned_origin_without_authority(self):
        with patch.object(controls.os, "getpid", return_value=100):
            plan = controls.build_control_plan(origin(), Path("/trusted/controls/origin"), ids())
        self.assertEqual(tuple(spec.arm for spec in plan.forks), ARMS)
        self.assertEqual({spec.origin_sha256 for spec in plan.forks}, {"d" * 64})
        self.assertEqual({spec.origin_context for spec in plan.forks}, {origin().context})
        self.assertEqual(len({spec.fork_nonce for spec in plan.forks}), 4)
        self.assertEqual(plan.parent_pid, 100)
        self.assertFalse(plan.graduation_authorized)
        self.assertFalse(plan.process_isolation_verified)
        self.assertFalse(plan.training_authoritative)
        self.assertFalse(hasattr(plan, "candidate_checkpoint_destination"))
        with self.assertRaises((TypeError, AttributeError)):
            plan.forks[0].arm = "correct_labels"

    def test_rejects_duplicate_workers_wrong_origin_and_missing_arms(self):
        with self.assertRaisesRegex(ValueError, "all four"):
            controls.build_control_plan(origin(), Path("/tmp/one"), {"baseline": "worker-1"})
        collision = ids()
        collision["zero_update"] = collision["baseline"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            controls.build_control_plan(origin(), Path("/tmp/one"), collision)
        with self.assertRaisesRegex(ValueError, "checkpoint receipt"):
            controls.build_control_plan(replace(origin(), graduation_authorized=True), Path("/tmp/one"), ids())
        with self.assertRaisesRegex(ValueError, "absolute"):
            controls.build_control_plan(origin(), Path("relative"), ids())


class ChildStartTests(unittest.TestCase):
    def setUp(self):
        with patch.object(controls.os, "getpid", return_value=100):
            self.plan = controls.build_control_plan(origin(), Path("/trusted/controls/origin"), ids())

    def test_parent_process_cannot_claim_control_fork(self):
        with patch.object(controls.os, "getpid", return_value=100):
            with self.assertRaisesRegex(ValueError, "separate process"):
                controls.load_control_child(self.plan, "baseline", lambda: object(), lambda _: object())

    def test_child_loads_pinned_full_state_before_returning_runtime(self):
        created = []
        def model_factory():
            obj = object()
            created.append("model")
            return obj
        def optimizer_factory(model):
            self.assertIsNotNone(model)
            created.append("optimizer")
            return object()
        def verified_loader(path, *, model, optimizer, expected_sha256, expected_context):
            self.assertEqual(path, Path("/trusted/controls/origin"))
            self.assertEqual(expected_sha256, "d" * 64)
            self.assertEqual(expected_context, origin().context)
            created.append("load")
            return origin()
        with (patch.object(controls.os, "getpid", return_value=200),
              patch.object(controls, "load_full_checkpoint", side_effect=verified_loader)):
            runtime = controls.load_control_child(self.plan, "correct_labels",
                                                 model_factory, optimizer_factory)
        self.assertEqual(created, ["model", "optimizer", "load"])
        self.assertEqual(runtime.start.pid, 200)
        self.assertEqual(runtime.start.arm, "correct_labels")
        self.assertFalse(runtime.start.graduation_authorized)
        self.assertFalse(runtime.start.process_isolation_verified)

    def test_mismatched_load_receipt_fails_closed(self):
        with (patch.object(controls.os, "getpid", return_value=200),
              patch.object(controls, "load_full_checkpoint",
                           return_value=replace(origin(), sha256="0" * 64))):
            with self.assertRaisesRegex(ValueError, "loaded origin"):
                controls.load_control_child(self.plan, "baseline", object, lambda _: object())

    def test_all_four_children_restore_same_origin_without_publication(self):
        with (patch.object(controls.os, "getpid", side_effect=(201, 202, 203, 204)),
              patch.object(controls, "load_full_checkpoint", return_value=origin()) as loader,
              patch.object(controls, "save_full_checkpoint", create=True) as prohibited_save):
            children = tuple(controls.load_control_child(
                self.plan, arm, object, lambda _: object()) for arm in ARMS)
        self.assertEqual(loader.call_count, 4)
        self.assertFalse(prohibited_save.called)
        self.assertEqual({c.start.origin_sha256 for c in children}, {origin().sha256})
        self.assertEqual({c.start.origin_context for c in children}, {origin().context})
        self.assertEqual(controls.verify_control_starts(
            self.plan, tuple(c.start for c in children)).arm_count, 4)
        self.assertEqual(len({id(c.model) for c in children}), 4)
        self.assertEqual(len({id(c.optimizer) for c in children}), 4)

    def test_tampered_plan_rejected_before_factory_or_load(self):
        forged = replace(self.plan,
                         forks=(replace(self.plan.forks[0], fork_nonce="a" * 64),)
                                + self.plan.forks[1:])
        with (patch.object(controls.os, "getpid", return_value=200),
              patch.object(controls, "load_full_checkpoint") as loader):
            with self.assertRaisesRegex(ValueError, "pinned origin"):
                controls.load_control_child(forged, "baseline", object, lambda _: object())
        loader.assert_not_called()

    def test_aggregate_requires_four_distinct_process_ids(self):
        def receipt(arm, pid):
            spec = next(f for f in self.plan.forks if f.arm == arm)
            return controls.ControlStartReceipt(arm, spec.worker_id, spec.fork_nonce,
                                                pid, spec.origin_sha256,
                                                spec.origin_context)
        starts = tuple(receipt(arm, 200 + idx) for idx, arm in enumerate(ARMS))
        summary = controls.verify_control_starts(self.plan, starts)
        self.assertEqual(summary.arm_count, 4)
        self.assertFalse(summary.process_isolation_verified)
        self.assertFalse(summary.graduation_authorized)
        with self.assertRaisesRegex(ValueError, "four"):
            controls.verify_control_starts(self.plan, starts[:-1])
        with self.assertRaisesRegex(ValueError, "distinct"):
            controls.verify_control_starts(self.plan, starts[:-1] + (replace(starts[-1], pid=200),))
        with self.assertRaisesRegex(ValueError, "does not match"):
            controls.verify_control_starts(self.plan, starts[:-1] +
                                           (replace(starts[-1], origin_sha256="e" * 64),))


if __name__ == "__main__":
    unittest.main()
