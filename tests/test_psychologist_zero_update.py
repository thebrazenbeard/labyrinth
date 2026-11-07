"""A non-learning observational policy that defeats all public diverse worlds.

This test is a *negative control*, not a Minotaur competency test. The policy
never reads Exam internals, transition functions, search, or a gold trace.
It relies only on candidate-visible room labels and route descriptions.
"""

import re
import unittest

from labyrinth.diverse_exam import CandidateSession, make_exam, public_view


def _routes(observation):
    """Read candidate-visible room exits; no evaluator state is consulted."""
    return {room: direction for direction, room in
            re.findall(r"(north|east|south|west) to ([a-z_]+)", observation)}


def frozen_observation_policy(opening, act):
    """One immutable script chosen solely from visible environmental landmarks."""
    observation = opening["observation"]
    start_routes = _routes(observation)
    if len(start_routes) != 1:
        raise AssertionError("frozen policy assumes exactly one starting exit")
    result = act(next(iter(start_routes.values())))
    observation = result["observation"]

    nearby = _routes(observation)
    if "archive" in nearby:
        plan = ("archive", "take", "junction", "gate", "offer", "leave")
    elif "lever" in nearby:
        plan = ("lever", "activate", "junction", "bridge", "leave")
    elif "west_shrine" in nearby:
        plan = ("west_shrine", "take", "hub", "east_shrine", "take", "hub",
                "portal", "combine", "leave")
    else:
        raise AssertionError("unknown visible landmark ontology")

    for destination_or_action in plan:
        action = _routes(observation).get(destination_or_action,
                                           destination_or_action)
        result = act(action)
        observation = result["observation"]
    return result["verdict"], opening["remaining_actions"] - result["remaining_actions"]


class ZeroUpdateBaselineTests(unittest.TestCase):
    def test_frozen_policy_wins_all_open_rotations_paraphrases_and_splits(self):
        outcomes = []
        for seed in range(12):
            for split in ("train", "dev", "test"):
                for variant in range(20):
                    exam = make_exam(seed, split=split, variant=variant)
                    session = CandidateSession(exam)
                    outcomes.append(frozen_observation_policy(public_view(exam), session.act))
        self.assertEqual(len(outcomes), 720)
        self.assertTrue(all(status == "VERIFIED_WIN" and cost <= 12
                            for status, cost in outcomes))

    def test_same_frozen_policy_does_not_win_sealed_counterfactuals(self):
        outcomes = []
        for seed in range(12):
            for split in ("train", "dev", "test"):
                for variant in range(20):
                    exam = make_exam(seed, split=split, variant=variant, sealed=True)
                    session = CandidateSession(exam)
                    outcomes.append(frozen_observation_policy(public_view(exam), session.act))
        self.assertEqual(len(outcomes), 720)
        self.assertTrue(all(status != "VERIFIED_WIN" and cost <= 12
                            for status, cost in outcomes))

    def test_split_label_does_not_protect_identical_cases(self):
        for seed in range(12):
            for variant in range(20):
                train = make_exam(seed, split="train", variant=variant)
                test = make_exam(seed, split="test", variant=variant)
                self.assertNotEqual(train.case_id, test.case_id)
                self.assertEqual(public_view(train), public_view(test))


if __name__ == "__main__":
    unittest.main()
