"""Independent no-learning hostile controls for the 'novel discovery' reviewer.

They do not claim a verified discovery or modify the evaluator. They expose
how a downstream metric based on NOVEL_RULESET_CHALLENGE alone would reward
unsupported or independently falsified proposals.
"""

from collections import Counter
from itertools import product
import unittest

from labyrinth.novel_discovery_world import CandidateSession, make_world
from labyrinth.novel_discovery_reviewer import (
    FIELDS, review_proposal, verify_receipt_replay,
)


def all_conjunctions():
    """Independently enumerate the complete 3^5-1 declared DSL hypothesis set."""
    for values in product((None, 0, 1), repeat=len(FIELDS)):
        literals = [{"field": field, "equals": value}
                    for field, value in zip(FIELDS, values) if value is not None]
        if literals:
            yield {"form": "and_literals", "literals": literals}


class HostileNoveltyMetricTests(unittest.TestCase):
    def test_all_241_falsified_rules_are_now_verified_failures(self):
        world = make_world(3)
        counts = Counter()
        for proposal in all_conjunctions():
            # Zero interventions or model calls. A fake claimant may flood
            # the review channel solely by enumerating legal DSL proposals.
            receipt = review_proposal(world, (), (), proposal)
            counts[(receipt["verdict"], receipt["review"]["fit_status"],
                    receipt["review"]["novelty_status"])] += 1
            self.assertFalse(receipt["verified_success"])
            self.assertTrue(verify_receipt_replay(receipt))
        self.assertEqual(sum(counts.values()), 242)
        self.assertEqual(counts[("VERIFIED_FAILURE", "FALSIFIED",
                                 "REFUTED_BY_COUNTEREXAMPLE")], 241)
        self.assertEqual(counts[("NOVEL_RULESET_CHALLENGE", "COUNTERFACTUAL_MATCH",
                                 "OUTSIDE_CATALOG")], 1)
        self.assertEqual(sum(value for (verdict, _, _), value in counts.items()
                             if verdict == "NOVEL_RULESET_CHALLENGE"), 1)

    def test_known_control_world_has_no_manufactured_novel_matches(self):
        world = make_world(11, control=True)
        for proposal in all_conjunctions():
            receipt = review_proposal(world, (), (), proposal)
            self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
            self.assertEqual(receipt["review"]["fit_status"], "FALSIFIED")
            self.assertTrue(verify_receipt_replay(receipt))
            self.assertFalse(receipt["independent_custody"])
            self.assertFalse(receipt["verified_success"])

    def test_perfect_counterfactual_match_needs_no_actual_exploration(self):
        # Public code fixes this particular synthetic rule. A hardcoded
        # candidate wins the exhaustive synthetic fit test before observing
        # a single action consequence; that isn't adaptive learning.
        proposal = {"form": "and_literals", "literals": [
            {"field": "prev2_x", "equals": 1},
            {"field": "prev_x", "equals": 0},
            {"field": "x", "equals": 1},
            {"field": "y", "equals": 1},
        ]}
        for seed in range(100):
            session = CandidateSession(make_world(seed))
            receipt = session.submit_proposal(proposal)
            self.assertEqual(session.attempts(), ())
            self.assertEqual(receipt["review"]["fit_status"], "COUNTERFACTUAL_MATCH")
            self.assertEqual(receipt["verdict"], "NOVEL_RULESET_CHALLENGE")
            self.assertFalse(receipt["verified_success"])

    def test_untyped_unobserved_narrative_remains_unknown(self):
        session = CandidateSession(make_world(0))
        receipt = session.submit_proposal("Invisible turtles may govern the output.")
        self.assertEqual(session.attempts(), ())
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["review"]["novelty_status"], "UNTYPED_PENDING")
        self.assertEqual(receipt["review"]["contexts_checked"], 0)
        self.assertEqual(receipt["review"]["fit_status"], "UNKNOWN")
        self.assertFalse(receipt["verified_success"])
        self.assertFalse(receipt["independent_custody"])
        self.assertTrue(verify_receipt_replay(receipt))


if __name__ == "__main__":
    unittest.main()
