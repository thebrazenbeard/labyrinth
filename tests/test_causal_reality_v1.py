"""Focused, real execution tests for the bounded toy causal micro-world."""

import dataclasses
import json
import unittest

from labyrinth.causal_reality_v1 import (
    ACTION_VERBS,
    CausalBlueprint,
    CausalLink,
    CausalWorld,
    MechanismAction,
    UnresolvedManipulation,
)


FORM_VERB = {"valve": "turn", "lever": "pull", "cable": "cut", "brace": "push"}


class CausalRealityTests(unittest.TestCase):
    def test_seeded_reproducibility_and_no_memorized_static_safe_target(self):
        same1 = CausalBlueprint.from_seed(197, "solvable")
        same2 = CausalBlueprint.from_seed(197, "solvable")
        self.assertEqual(same1, same2)
        self.assertEqual(same1.sha256, same2.sha256)
        solutions = set()
        commitments = set()
        for seed in range(30):
            blueprint = CausalBlueprint.from_seed(seed, "solvable")
            world = CausalWorld(blueprint)
            action = next(MechanismAction(FORM_VERB[a.form], a.label)
                          for a in blueprint.actuators
                          if any(link.actuator == a.label and
                                 link.subsystem == "cooling_engaged"
                                 for link in blueprint.links))
            world.step(action)
            verdict = world.private_verdict()
            self.assertEqual(verdict.outcome, "STATION_STABILIZED")
            self.assertEqual(verdict.possible_safe_manipulations, (action,))
            self.assertEqual(verdict.epistemic_quality, "NOT_EVALUATED")
            self.assertEqual(verdict.oracle_checked_action_count,
                             len(blueprint.actuators) * len(ACTION_VERBS))
            solutions.add(action.target)
            commitments.add(blueprint.sha256)
        self.assertGreater(len(solutions), 1)
        self.assertEqual(len(commitments), 30)

    def test_unknown_target_non_destructive_then_valid_action(self):
        blueprint = CausalBlueprint.from_seed(37, "solvable")
        world = CausalWorld(blueprint)
        starting_hash = world.initial_observation_sha256
        with self.assertRaises(UnresolvedManipulation):
            world.step(MechanismAction("cut", "unknown_pipe"))
        self.assertIsNone(world.result)
        self.assertFalse(world.state.manipulated)
        self.assertEqual(world.observe().sha256, starting_hash)
        possible = next(a for a in blueprint.actuators
                        if any(link.actuator == a.label and
                               link.subsystem == "cooling_engaged"
                               for link in blueprint.links))
        event = world.execute(MechanismAction(FORM_VERB[possible.form], possible.label))
        self.assertEqual(event.observed_outcome, "STATION_STABILIZED")
        with self.assertRaisesRegex(ValueError, "only one"):
            world.step(MechanismAction("push", "white_brace"))

    def test_wrong_verb_is_bounded_one_shot_failure_not_a_retry(self):
        world = CausalWorld(CausalBlueprint.from_seed(1, "solvable"))
        event = world.step(MechanismAction("open", "amber_valve"))
        self.assertEqual(event.observed_outcome, "CATASTROPHIC_FAILURE")
        self.assertEqual(world.private_verdict().failure_causes,
                         ("THERMAL_COLLAPSE",))
        with self.assertRaises(ValueError):
            world.step(MechanismAction("turn", "amber_valve"))

    def test_uncertain_pair_public_indistinguishable_despite_outcome_difference(self):
        for base in (0, 1, 14, 72):
            left_bp = CausalBlueprint.from_seed(base * 2, "uncertain")
            right_bp = CausalBlueprint.from_seed(base * 2 + 1, "uncertain")
            left = CausalWorld(left_bp)
            right = CausalWorld(right_bp)
            # The answer, class, model-facing blueprint fingerprint and all
            # hidden links are absent from public evidence.
            self.assertEqual(left.observe(), right.observe())
            self.assertEqual(left.observe().as_user_message(),
                             right.observe().as_user_message())
            self.assertNotEqual(left_bp.sha256, right_bp.sha256)
            trial = MechanismAction(FORM_VERB[left_bp.actuators[0].form],
                                    left_bp.actuators[0].label)
            left.step(trial)
            right.step(trial)
            self.assertNotEqual(left.private_verdict().outcome,
                                right.private_verdict().outcome)

    def test_impossible_bounded_exhaustive_not_auto_reasoning_failure(self):
        bp = CausalBlueprint.from_seed(91, "impossible")
        for actuator in bp.actuators:
            for verb in ACTION_VERBS:
                world = CausalWorld(bp)
                event = world.step(MechanismAction(verb, actuator.label))
                self.assertEqual(event.observed_outcome, "CATASTROPHIC_FAILURE")
                verdict = world.private_verdict()
                self.assertEqual(verdict.possible_safe_manipulations, ())
                self.assertEqual(verdict.epistemic_quality, "NOT_EVALUATED")
                self.assertEqual(verdict.oracle_checked_action_count, 24)
                self.assertEqual(len(verdict.oracle_certificate_sha256), 64)

    def test_blueprint_and_event_precommitted_and_tamper_detected(self):
        bp = CausalBlueprint.from_seed(42, "solvable")
        world = CausalWorld(bp)
        initial_sha = world.initial_observation_sha256
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bp.links = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            world.state.cooling_engaged = True
        self.assertEqual(world.blueprint_sha256, bp.sha256)
        safe_actuator = next(a for a in bp.actuators
                             if any(link.actuator == a.label and
                                    link.subsystem == "cooling_engaged"
                                    for link in bp.links))
        action = MechanismAction(FORM_VERB[safe_actuator.form], safe_actuator.label)
        event = world.step(action)
        self.assertEqual(event.observation_sha256, initial_sha)
        self.assertEqual(event.blueprint_sha256, bp.sha256)
        self.assertEqual(event.state_after_sha256,
                         __import__("hashlib").sha256(
                             json.dumps(dataclasses.asdict(world.state),
                                        sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False).encode("ascii")
                         ).hexdigest())
        self.assertEqual(event, world.result)
        tampered = dataclasses.replace(event, observed_outcome="CATASTROPHIC_FAILURE")
        world._event = tampered  # hostile mutation of private runtime memory
        with self.assertRaisesRegex(ValueError, "receipt integrity"):
            world.private_verdict()

    def test_no_private_graph_or_answer_menu_in_public_prompt(self):
        for challenge_class in ("solvable", "uncertain", "impossible"):
            bp = CausalBlueprint.from_seed(113, challenge_class)
            message = CausalWorld(bp).observe().as_user_message()
            public = json.loads(message)
            self.assertEqual(set(public.keys()),
                             {"scenario", "visible_objects", "appearances",
                              "public_clues", "terminal_temperature_c",
                              "remaining_manipulations"})
            self.assertNotIn(challenge_class, message)
            self.assertNotIn("blueprint_sha256", message)
            self.assertNotIn("reservoir_seal_live", message)
            self.assertNotIn("possible_safe_manipulations", message)
            self.assertNotIn("allowed_actions", message)

    def test_invalid_class_graph_rejected_from_forgery(self):
        bp = CausalBlueprint.from_seed(5, "solvable")
        poisoned = tuple(CausalLink(link.actuator, link.subsystem, False)
                         if link.subsystem == "cooling_engaged" else link
                         for link in bp.links)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            dataclasses.replace(bp, links=poisoned)


if __name__ == "__main__":
    unittest.main()
