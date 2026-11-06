"""The durable continuation must address the selected *successor* social world.

Only source fixtures and mock physical TRAIN bytes are involved here.
"""

import unittest

from tests.test_campaign_orchestrator_v1 import OrchestratorTests


class AriadneSelectedWorldHostileTests(unittest.TestCase):
    def test_after_champion_continuation_points_to_new_social_world(self):
        fixture = OrchestratorTests(
            "test_two_generations_carry_selected_social_world_and_full_model_parent"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        launches, _, _ = fixture._launches(1)
        pending = fixture.orchestrator.run_generation(
            launches=launches,
            control_manifest_path=fixture._control(fixture.w0_root, 1),
            evaluation_plan=fixture.plan,
        )
        selected = fixture.orchestrator.finalize_generation(
            pending, fixture._external_fixture_exams(pending.population, 1)
        )
        self.assertEqual(selected.decision.winner_id, "g1_right")
        descriptor = fixture.journal.continuation_descriptor(
            expected_head_sha256=selected.journal_head_sha256
        )
        self.assertEqual(descriptor["selected_world_id"], "world_2",
                         "Ariadne restored precursor world_1 despite a witnessed "
                         "WORLD_VERSION world_2 after champion selection")
        recorded = next(world for world in descriptor["worlds"]
                        if world["world_id"] == descriptor["selected_world_id"])
        self.assertEqual(recorded["world_state"]["characters"][0]["trust"], 1)


if __name__ == "__main__":
    unittest.main()
