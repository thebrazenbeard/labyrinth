"""Approved dynamic scene must survive a known duplicate-ID preflight denial.

All physical artifacts here are toy bytes, not a real model update.
"""

import unittest

from tests.test_dynamic_scene_authorization_v1 import DynamicTrainTests


class DynamicSceneNoEffectRetryHostileTests(unittest.TestCase):
    def test_examined_cohort_cannot_gain_late_sibling_or_consume_scene(self):
        fixture = DynamicTrainTests(
            "test_original_model_scene_drives_physical_fixture_train_not_exact_beat_whitelist"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._run("dyn_examined")
        fixture.fixture.journal.record_evaluation(
            candidate_id="dyn_examined",
            evaluation={
                "metric_id": "source_dev_only", "score": .5,
                "direction": "higher", "dataset_ref": "fictional_dev",
                "evidence": {"source_only": True},
            },
            idempotency_key="source_examined_dyn",
        )
        late_scene = fixture._approve()
        late_bridge, late_trainer = fixture._bridge("dyn_too_late")
        previous_events = len(fixture.fixture.journal.events())
        with self.assertRaisesRegex(ValueError, "examination already began"):
            fixture._run(
                "dyn_too_late", bridge=(late_bridge, late_trainer), scene=late_scene
            )
        self.assertEqual(late_trainer.calls, 0)
        self.assertEqual(len(fixture.fixture.journal.events()), previous_events)
        # A denied late candidate must not expend its one-use operator approval.
        fixture.authority.preflight_scene(
            receipt=late_scene, world=fixture.world, bridge=late_bridge
        )

    def test_duplicate_candidate_id_does_not_burn_new_approval_before_train(self):
        fixture = DynamicTrainTests(
            "test_original_model_scene_drives_physical_fixture_train_not_exact_beat_whitelist"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        first_scene = fixture._approve()
        fixture._run("dyn_once", scene=first_scene)
        self.assertEqual(len(fixture.fixture.journal.continuation_descriptor()["candidates"]), 1)

        second_scene = fixture._approve()
        bad_bridge, bad_trainer = fixture._bridge("dyn_bad_retry")
        with self.assertRaises(ValueError):
            fixture._run("dyn_once", bridge=(bad_bridge, bad_trainer),
                         scene=second_scene)
        self.assertEqual(bad_trainer.calls, 0)
        # There was no second TRAIN effect. A distinct approved sibling may
        # still use its untouched one-use scene permit once the wrong
        # candidate ID is corrected to a fresh ID.
        good_bridge, good_trainer = fixture._bridge("dyn_fresh")
        _, _, used = fixture._run("dyn_fresh", bridge=(good_bridge, good_trainer),
                                  scene=second_scene)
        self.assertEqual(used.approval_id, second_scene.approval_id)
        self.assertEqual(good_trainer.calls, 1)


if __name__ == "__main__":
    unittest.main()
