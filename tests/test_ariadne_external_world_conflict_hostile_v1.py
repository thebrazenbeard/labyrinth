"""A known pre-existing successor world cannot consume physical TRAIN first."""

import unittest

from tests.test_campaign_orchestrator_v1 import OrchestratorTests


class AriadneExternalWorldConflictHostileTests(unittest.TestCase):
    def test_existing_next_world_id_rejected_before_any_physical_train(self):
        fixture = OrchestratorTests(
            "test_two_generations_carry_selected_social_world_and_full_model_parent"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.journal.record_world(
            world_id="world_2", parent_world_id="world_1", version=2,
            curriculum={"operator_provenance": "external_parallel_world"},
            world_state={"characters": [
                {"character_id": "miren", "trust": 0,
                 "interactions": 0, "promises": []},
            ]},
            idempotency_key="external_parallel_world_2",
        )
        launches, trainers, _ = fixture._launches(1)
        with self.assertRaises(ValueError):
            fixture.orchestrator.run_generation(
                launches=launches,
                control_manifest_path=fixture._control(fixture.w0_root, 1),
                evaluation_plan=fixture.plan,
            )
        self.assertEqual([trainer.calls for trainer in trainers], [0, 0],
                         "pre-existing successor-world collision was found "
                         "only after costly physical siblings trained")
        self.assertFalse(fixture.orchestrator._blocked,
                         "known preflight collision must remain recoverable")


if __name__ == "__main__":
    unittest.main()
