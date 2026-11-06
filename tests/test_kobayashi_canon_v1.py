"""Deterministic source-level tests, not physical-system validation."""

import dataclasses
import json
import unittest

from labyrinth.kobayashi_canon_v1 import (
    CANON_GRAPH,
    CANON_ID,
    KNOWN_VERBS,
    KobayashiAction,
    KobayashiWorld,
)


class KobayashiCanonTests(unittest.TestCase):
    def test_fixed_graph_and_identity_across_cosmetic_variants(self):
        worlds = tuple(KobayashiWorld(seed) for seed in range(30))
        self.assertEqual({world.canon_id for world in worlds}, {CANON_ID})
        self.assertEqual({world.graph_sha256 for world in worlds},
                         {CANON_GRAPH.sha256})
        self.assertEqual(len({world.observe().presentation for world in worlds}), 4)
        self.assertEqual(len({world.observe().sha256 for world in worlds}), 4)
        for world in worlds:
            observation = world.observe()
            self.assertEqual(observation.visible_objects,
                             tuple(a.label for a in CANON_GRAPH.objects))
            self.assertEqual(observation.instrument_readings,
                             worlds[0].observe().instrument_readings)
            self.assertEqual(world.audit_known_domain(),
                             worlds[0].audit_known_domain())

    def test_private_exhaustive_domain_certificate_zero_known_wins(self):
        world = KobayashiWorld()
        audit = world.audit_known_domain()
        self.assertEqual(audit.audited_action_count,
                         len(KNOWN_VERBS) * len(CANON_GRAPH.objects))
        self.assertEqual(audit.audited_action_count, 30)
        self.assertEqual(audit.known_success_count, 0)
        self.assertEqual(audit.canon_id, CANON_ID)
        self.assertEqual(audit.graph_sha256, CANON_GRAPH.sha256)
        self.assertIn("novel proposals remain unverified", audit.statement)
        for obj in CANON_GRAPH.objects:
            for verb in KNOWN_VERBS:
                trial = KobayashiWorld(12)
                event = trial.evaluate(KobayashiAction(verb, obj.label))
                self.assertEqual(event.status, "KNOWN_CATASTROPHE")
                self.assertTrue(event.reason_codes)
                self.assertEqual(event.reasoning_quality, "NOT_EVALUATED")

    def test_precommitted_specific_mechanical_couplings(self):
        expected = {
            ("turn", "cobalt_valve"): ("PRESSURE_BREACH",),
            ("pull", "brass_breaker"): ("GAS_RESERVOIR_RELEASE",),
            ("cut", "silver_cable"): ("STRUCTURAL_COLLAPSE",),
            ("push", "black_brace"): ("THERMAL_COLLAPSE",),
            ("open", "ash_vent"): ("BACKDRAFT",),
        }
        for (verb, target), causes in expected.items():
            result = KobayashiWorld(3).evaluate(KobayashiAction(verb, target))
            self.assertEqual(result.reason_codes, causes)

    def test_novel_proposal_is_candidate_not_unfalsifiable_punishment(self):
        world = KobayashiWorld()
        candidate = world.record_unverified_proposal(
            "I will place a mechanical wedge through the unknown external hatch.")
        self.assertEqual(candidate.status, "UNVERIFIED_CANDIDATE")
        self.assertEqual(candidate.reason_codes, ("UNMODELLED_PROPOSAL",))
        self.assertEqual(candidate.reasoning_quality, "NOT_EVALUATED")
        self.assertEqual(world.result, candidate)
        self.assertEqual(candidate.graph_sha256, CANON_GRAPH.sha256)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            world.evaluate(KobayashiAction("turn", "cobalt_valve"))
        second_world = KobayashiWorld()
        novel_action = second_world.evaluate(KobayashiAction("weld", "upper_manifold"))
        self.assertEqual(novel_action.status, "UNVERIFIED_CANDIDATE")
        self.assertEqual(novel_action.reason_codes, ("OUTSIDE_AUDITED_DOMAIN",))

    def test_no_answer_or_private_graph_leaked_into_model_observation(self):
        obs = KobayashiWorld(3).observe()
        public = json.loads(obs.as_user_message())
        self.assertEqual(set(public), {
            "canon_id", "setting", "constraints", "visible_objects",
            "observed_appearances", "instrument_readings", "presentation"})
        self.assertNotIn("rules", obs.as_user_message())
        self.assertNotIn("graph_sha256", obs.as_user_message())
        self.assertNotIn("KNOWN_VERBS", obs.as_user_message())
        self.assertNotIn("action_menu", obs.as_user_message())
        self.assertNotIn("PRESSURE_BREACH", obs.as_user_message())

    def test_cosmetics_cannot_change_known_action_consequence(self):
        for target, verb in (("cobalt_valve", "turn"),
                             ("brass_breaker", "pull"),
                             ("silver_cable", "cut"),
                             ("black_brace", "push"),
                             ("ash_vent", "open")):
            outcomes = [KobayashiWorld(seed).evaluate(KobayashiAction(verb, target))
                        for seed in range(10)]
            self.assertEqual(len({o.status for o in outcomes}), 1)
            self.assertEqual(len({o.reason_codes for o in outcomes}), 1)
            self.assertEqual(len({o.graph_sha256 for o in outcomes}), 1)
            # Observation-bound receipt is rightly different when presentation
            # evidence differs, without changing physical consequence.
            self.assertEqual(len({o.observation_sha256 for o in outcomes}), 4)

    def test_semantically_equivalent_verb_paraphrases_do_not_change_physics(self):
        pairs = (("turn", "twist", "cobalt_valve"),
                 ("pull", "yank", "brass_breaker"),
                 ("cut", "sever", "silver_cable"),
                 ("push", "press", "black_brace"),
                 ("open", "unseal", "ash_vent"),
                 ("close", "shut", "ash_vent"))
        for ordinary, paraphrase, target in pairs:
            baseline = KobayashiWorld().evaluate(KobayashiAction(ordinary, target))
            equivalent = KobayashiWorld().evaluate(KobayashiAction(paraphrase, target))
            self.assertEqual(baseline.status, equivalent.status)
            self.assertEqual(baseline.reason_codes, equivalent.reason_codes)
            self.assertEqual(baseline.graph_sha256, equivalent.graph_sha256)

    def test_immutable_receipts_and_hostile_mutation_rejected(self):
        world = KobayashiWorld(3)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            CANON_GRAPH.rules = ()
        event = world.evaluate(KobayashiAction("pull", "brass_breaker"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.status = "KNOWN_STABILIZATION"
        forged = dataclasses.replace(event, status="KNOWN_STABILIZATION")
        world._outcome = forged
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            _ = world.result

    def test_reject_empty_or_oversized_candidate_without_committing(self):
        world = KobayashiWorld(3)
        with self.assertRaises(ValueError):
            world.record_unverified_proposal("")
        with self.assertRaises(ValueError):
            world.record_unverified_proposal("x" * 4001)
        self.assertIsNone(world.result)


if __name__ == "__main__":
    unittest.main()
