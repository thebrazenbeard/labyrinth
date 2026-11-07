"""Independent finite-horizon observation-identifiability certificate.

Proves the minimum posterior for every legal rebound trace with <=10 charged
actions without enumerating each of the 4**10 raw sequences. The DP groups
prefixes by the Markov history and exact 242-bit remaining-hypotheses mask.
"""

from itertools import product
from collections import deque
import unittest

from labyrinth.minotaur_observation_controls import (
    TOTAL_HYPOTHESES, _posterior, _post_contexts,
)
from labyrinth.novel_discovery_world import (
    CHANGE_AFTER, INPUTS, MAX_ATTEMPTS, make_world,
)
from labyrinth.novel_discovery_reviewer import replay_world


FIELDS_INDEPENDENT = ("x", "y", "prev_x", "prev2_x", "prev_y")


def _hypothesis_masks_by_context():
    """Independent three-valued truth table, no production hypothesis iterator."""
    states = [combination for combination in product((-1, 0, 1), repeat=5)
              if any(v != -1 for v in combination)]
    assert len(states) == 242
    masks = {}
    for context in product((0, 1), repeat=5):
        mask = 0
        for index, hypothesis in enumerate(states):
            if all(value == -1 or context[field] == value
                   for field, value in enumerate(hypothesis)):
                mask |= 1 << index
        masks[context] = mask
    return states, masks


def _exact_dp():
    hypotheses, positive_masks = _hypothesis_masks_by_context()
    all_hypotheses = (1 << len(hypotheses)) - 1
    # First three legal interventions predate the hidden rule change and
    # therefore provide NO observations for post-change hypothesis fitting.
    pre = {(0, 0, 0): ()}  # (prev2_x, prev_x, prev_y): one shortest witness
    for _ in range(CHANGE_AFTER):
        following = {}
        for (prev2, prev, prev_y), trace in pre.items():
            for x, y in INPUTS:
                following.setdefault((prev, x, y), trace + ((x, y),))
        pre = following
    assert len(pre) == 8
    # DP state contains all and only the variables determining future
    # observations, plus the set of hypotheses consistent with the past.
    frontier = {(p2, p1, py, all_hypotheses): trace
                for (p2, p1, py), trace in pre.items()}
    cardinalities = [len(frontier)]
    min_remaining_by_depth = [len(hypotheses)]
    for _ in range(MAX_ATTEMPTS - CHANGE_AFTER):
        following = {}
        for (p2, p1, py, posterior), trace in frontier.items():
            for x, y in INPUTS:
                context = (x, y, p1, p2, py)
                rebound = int((p2, p1, x, y) == (1, 0, 1, 1))
                predicted_ones = positive_masks[context]
                compatible = (predicted_ones if rebound
                              else all_hypotheses ^ predicted_ones)
                next_posterior = posterior & compatible
                state = (p1, x, y, next_posterior)
                following.setdefault(state, trace + ((x, y),))
        frontier = following
        cardinalities.append(len(frontier))
        min_remaining_by_depth.append(min(s[3].bit_count() for s in frontier))
    final_state = min(frontier, key=lambda state: state[3].bit_count())
    return {
        "total_hypotheses": len(hypotheses),
        "cardinalities": tuple(cardinalities),
        "min_remaining_by_depth": tuple(min_remaining_by_depth),
        "minimum_posterior": final_state[3].bit_count(),
        "one_minimizing_trace": frontier[final_state],
        "unique_terminal_states": len(frontier),
    }


def _identification_trace_outside_budget():
    """BFS a six-context separating design; no posterior-search implementation.

    Two positive observations differing in prev_y exclude adding a fifth
    literal. Four single-predicate deletions each need a negative observation
    to exclude a weaker three-literal law. This six-context design distinguishes
    the rebound rule from all other conjunctions in the declared 242 DSL.
    """
    def covered(context):
        x, y, px, p2, py = context
        matches = (
            (p2, px, x, y, py) == (1, 0, 1, 1, 0),
            (p2, px, x, y, py) == (1, 0, 1, 1, 1),
            (p2, px, x, y) == (0, 0, 1, 1),
            (p2, px, x, y) == (1, 1, 1, 1),
            (p2, px, x, y) == (1, 0, 0, 1),
            (p2, px, x, y) == (1, 0, 1, 0),
        )
        return sum((1 << bit) for bit, result in enumerate(matches) if result)

    pre = {(0, 0, 0): ()}
    for _ in range(CHANGE_AFTER):
        next_pre = {}
        for (_, px, _), trace in pre.items():
            for x, y in INPUTS:
                next_pre.setdefault((px, x, y), trace + ((x, y),))
        pre = next_pre
    # Source-derived pre-change states all have 3 already charged actions.
    front = deque((p2, px, py, 0, trace)
                  for (p2, px, py), trace in pre.items())
    seen = {(p2, px, py, 0) for p2, px, py in pre}
    while front:
        p2, px, py, hit, trace = front.popleft()
        if hit == 63:
            return trace
        for x, y in INPUTS:
            newhit = hit | covered((x, y, px, p2, py))
            next_state = (px, x, y, newhit)
            if next_state not in seen:
                seen.add(next_state)
                front.append((*next_state, trace + ((x, y),)))
    raise AssertionError("finite six-context separator never reachable")


class ObservabilityFiniteProofTests(unittest.TestCase):
    def test_exact_markov_posterior_dp_proves_no_unique_rebound_rule(self):
        certificate = _exact_dp()
        self.assertEqual(certificate["total_hypotheses"], TOTAL_HYPOTHESES)
        self.assertEqual(certificate["minimum_posterior"], 4)
        self.assertEqual(len(certificate["one_minimizing_trace"]), MAX_ATTEMPTS)
        self.assertEqual(len(certificate["cardinalities"]), MAX_ATTEMPTS - CHANGE_AFTER + 1)
        # Cross-check one witness by the independent world transition/replay
        # and the production source's string-tuple posterior for parity.
        witness = certificate["one_minimizing_trace"]
        effects = replay_world(make_world(control=False), witness)[1]
        direct = _posterior(_post_contexts(witness, effects))
        self.assertEqual(len(direct), 4)
        print("OBSERVABILITY_DP_CERTIFICATE", certificate)

    def test_no_legal_prefix_could_be_more_identifying_than_full_horizon(self):
        certificate = _exact_dp()
        self.assertEqual(certificate["min_remaining_by_depth"][-1], 4)
        self.assertGreaterEqual(min(certificate["min_remaining_by_depth"]), 4)
        # Invalid attempted actions do not change physical state and charge
        # budget. Removing them leaves <=10 valid actions with the SAME
        # contexts/effects; any short valid sequence can be extended to ten,
        # and each additional valid observation can only shrink a posterior.
        self.assertEqual(MAX_ATTEMPTS, 10)
        self.assertEqual(CHANGE_AFTER, 3)

    def test_longer_trace_can_uniquely_identify_rule_in_same_declared_dsl(self):
        witness = _identification_trace_outside_budget()
        self.assertGreater(len(witness), MAX_ATTEMPTS)
        effects = replay_world(make_world(control=False), witness)
        # Production replay enforces the 10-attempt cap, so use our own
        # independent truth-function generator for this *hypothetical* >10
        # intervention extension. No actual live session can execute it.
        self.assertIsNone(effects)
        p2 = px = py = 0
        extended_rows = []
        for index, (x, y) in enumerate(witness):
            if index >= CHANGE_AFTER:
                row = {"x": x, "y": y, "prev_x": px,
                       "prev2_x": p2, "prev_y": py}
                extended_rows.append((row, int((p2, px, x, y) == (1, 0, 1, 1))))
            p2, px, py = px, x, y
        self.assertEqual(len(_posterior(extended_rows)), 1)
        print("UNBOUNDED_SEPARATOR_TRACE", len(witness), witness)


if __name__ == "__main__":
    unittest.main()
