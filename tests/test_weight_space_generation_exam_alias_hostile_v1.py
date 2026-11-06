"""A trusted proposal callback must not rewrite preregistered Minotaur cohort.

This is a source-only P1 provenance test; no independent exam or neural SGD.
"""

import unittest

from labyrinth.weight_space_campaign_bridge_v1 import run_weight_space_generation
from tests.test_weight_space_campaign_bridge_v1 import WeightSpaceCampaignBridgeTests
from tests.test_weight_space_spark_v1 import _read_weights


class WeightSpaceExamAliasHostileTests(unittest.TestCase):
    def test_proposal_callback_cannot_rewrite_exam_plan_after_preflight(self):
        fixture = WeightSpaceCampaignBridgeTests(
            "test_real_journal_full_control_and_distinct_physical_sibling_population"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original_commitment = fixture.plan.sealed_case_commitment

        def mutating_proposer(ctx):
            if ctx.candidate_id == "right":
                object.__setattr__(fixture.plan, "sealed_case_commitment", "e" * 64)
            return fixture.propose(ctx)

        result = run_weight_space_generation(
            w0_manifest_path=fixture.w0_path,
            campaign_root=fixture.root / "generation_1",
            candidate_ids=("left", "right"),
            public_train_source_sha256="f" * 64,
            propose=mutating_proposer,
            policy=fixture.policy,
            evaluation_plan=fixture.plan,
        )
        self.assertEqual(result.population.plan.sealed_case_commitment,
                         original_commitment,
                         "untrusted proposal callback changed original sealed "
                         "Minotaur exam cohort after preflight")

    def test_later_callback_cannot_rewrite_earlier_approved_edit(self):
        fixture = WeightSpaceCampaignBridgeTests(
            "test_real_journal_full_control_and_distinct_physical_sibling_population"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        retained = []

        def mutating_proposer(ctx):
            if ctx.candidate_id == "left":
                proposal = fixture.propose(ctx)
                retained.append(proposal)
                return proposal
            object.__setattr__(retained[0], "gain", 1.)
            return fixture.propose(ctx)

        result = run_weight_space_generation(
            w0_manifest_path=fixture.w0_path,
            campaign_root=fixture.root / "generation_1",
            candidate_ids=("left", "right"),
            public_train_source_sha256="f" * 64,
            propose=mutating_proposer,
            policy=fixture.policy,
            evaluation_plan=fixture.plan,
        )
        left_weights = fixture.root / "generation_1" / "candidate_left" / "base" / "model.safetensors"
        right_weights = fixture.root / "generation_1" / "candidate_right" / "base" / "model.safetensors"
        self.assertEqual(_read_weights(left_weights)[0], .5,
                         "earlier approved edit gain .5 changed after proposal callback")
        self.assertEqual(_read_weights(right_weights)[0], 0.)
        self.assertNotEqual(result.population.candidate_states[0][1],
                            result.population.candidate_states[1][1])


if __name__ == "__main__":
    unittest.main()
