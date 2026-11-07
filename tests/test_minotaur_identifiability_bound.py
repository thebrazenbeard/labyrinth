"""Independent finite-state bound on ten-action rebound rule identification.

Exhaustively groups action prefixes by (observed state, compatible hypothesis
mask). This is exact dynamic programming, not random search or a fixed script.
The reference truth table and 242-conjunction evaluator are independently
encoded here, without importing the implementation's posterior helper.
"""

from itertools import product
import unittest

from labyrinth.minotaur_observation_controls import (
    FIXED_PUBLIC_RULE, assess_observational_identification,
)
from labyrinth.novel_discovery_reviewer import replay_world
from labyrinth.novel_discovery_world import CandidateSession, make_world


CONTEXT_FIELDS = ("x", "y", "prev_x", "prev2_x", "prev_y")
ACTION_PAIRS = tuple(product((0, 1), repeat=2))
CONJUNCTIONS = tuple(
    terms for terms in product((-1, 0, 1), repeat=len(CONTEXT_FIELDS))
    if any(term != -1 for term in terms)
)


def _conjunction_result(hypothesis, context):
    return int(all(required == -1 or required == observed
                   for required, observed in zip(hypothesis, context)))


def _rebound_result(context):
    x, y, prev_x, prev2_x, _prev_y = context
    return int((prev2_x, prev_x, x, y) == (1, 0, 1, 1))


def _compatible_masks():
    result = {}
    for context in product((0, 1), repeat=len(CONTEXT_FIELDS)):
        expected = _rebound_result(context)
        result[context] = sum(
            1 << offset for offset, hypothesis in enumerate(CONJUNCTIONS)
            if _conjunction_result(hypothesis, context) == expected
        )
    return result


def _state_space_exhaustive_bound(attempt_limit=10, change_after=3):
    """Each reachable (last-two x, last y, hypothesis mask) is exact state.

    No irreversible pruning, sampling, posterior approximation, or access to
    hidden environment variables is used. Four possible interventions are
    enumerated at every step. A prefix with an invalid action cannot gain
    additional observations within the same budget, so need not be searched.
    """
    matching = _compatible_masks()
    all_hypotheses = (1 << len(CONJUNCTIONS)) - 1
    states = {(0, 0, 0, all_hypotheses)}
    metrics = []
    for step in range(attempt_limit):
        next_states = set()
        for prev2_x, prev_x, prev_y, possible in states:
            for x, y in ACTION_PAIRS:
                context = (x, y, prev_x, prev2_x, prev_y)
                narrowed = possible
                if step >= change_after:
                    narrowed &= matching[context]
                next_states.add((prev_x, x, y, narrowed))
        states = next_states
        metrics.append((len(states),
                        min(mask.bit_count() for _, _, _, mask in states)))
    return metrics


class MinotaurFiniteStateIdentificationBoundTests(unittest.TestCase):
    def test_all_legal_ten_action_traces_cannot_identify_unique_rule(self):
        self.assertEqual(len(CONJUNCTIONS), 242)
        metrics = _state_space_exhaustive_bound()
        self.assertEqual(len(metrics), 10)
        self.assertEqual(metrics[-1][1], 4)
        self.assertTrue(all(minimum > 1 for _states, minimum in metrics))

    def test_concrete_four_survivor_trace_agrees_with_public_evaluator(self):
        # Independently supplied minimizing witness; exact unmodified 10-action
        # legal trace yields four surviving conjunctions, not one.
        actions = ((0, 0), (0, 0), (0, 0), (1, 1), (0, 0),
                   (1, 0), (0, 0), (1, 1), (0, 0), (0, 1))
        world = make_world(0)
        session = CandidateSession(world)
        for x, y in actions:
            session.intervene(x, y)
        effects = tuple(row["effect"] for row in session.view()["observations"])
        receipt = assess_observational_identification(world, actions, effects,
                                                      FIXED_PUBLIC_RULE)
        self.assertEqual(receipt.consistent_hypotheses, 4)
        self.assertEqual(receipt.status, "UNDERDETERMINED")
        self.assertFalse(receipt.neural_learning_established)

    def test_hypothetical_thirteen_action_counterexample_to_unbounded_claim(self):
        # This sequence is NOT permitted by the real ten-action examination.
        # It refutes any overbroad claim that the causal rule is intrinsically
        # unidentifiable across all action budgets.
        actions = ((0, 0), (1, 0), (0, 0), (0, 1), (1, 1),
                   (0, 0), (1, 0), (0, 0), (1, 1), (0, 1),
                   (1, 1), (1, 0), (1, 1))
        self.assertIsNone(replay_world(make_world(0), actions))
        prev2_x = prev_x = prev_y = 0
        surviving = list(CONJUNCTIONS)
        for offset, (x, y) in enumerate(actions):
            context = (x, y, prev_x, prev2_x, prev_y)
            if offset >= 3:
                actual = _rebound_result(context)
                surviving = [hypothesis for hypothesis in surviving
                             if _conjunction_result(hypothesis, context) == actual]
            prev2_x, prev_x, prev_y = prev_x, x, y
        self.assertEqual(len(surviving), 1)

    def test_single_correct_public_rule_is_not_identified_from_no_data(self):
        receipt = assess_observational_identification(make_world(0), (), (),
                                                      FIXED_PUBLIC_RULE)
        self.assertTrue(receipt.proposal_consistent)
        self.assertEqual(receipt.consistent_hypotheses, 242)
        self.assertEqual(receipt.status, "NO_POST_CHANGE_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
