"""Independent Wave17 novel-router finite, observational and leakage audit."""

from collections import Counter
from dataclasses import replace
from itertools import product
import unittest

from labyrinth import novel_router_analysis as mathref
from labyrinth.switchback_novel_transfer import (
    MAX_PROBES, ProbeSession, RoutingRule, make_case, search,
)


class NovelRouterWave17AnalysisTests(unittest.TestCase):
    def test_complete_rule_tables_and_alias_classes(self):
        groups = mathref.all_law_equivalence()
        multiplicities = Counter(map(len, groups.values()))
        self.assertEqual(sum(k * n for k, n in multiplicities.items()), 24)
        self.assertTrue(all(len(signature) == 32 * (1 + 3 * 2)
                            for signature in groups))
        self.assertEqual({law for rules in groups.values() for law in rules},
                         {RoutingRule(a, b, c) for a, b, c in
                          product(range(2), range(4), range(3))})
        print("W17_COMPLETE_ROUTING_LAW_EQUIVALENCE", {
            "parameters": 24, "distinct_complete_laws": len(groups),
            "multiplicity_histogram": dict(sorted(multiplicities.items())),
        })

    def test_all_rule_pairs_collapse_only_if_bounded_observations_equal(self):
        base = make_case(0)
        result = mathref.counterfactual_pair_observability(base)
        self.assertEqual(result["counterfactual_worlds"], 552)
        self.assertLessEqual(result["distinct_bounded_observation_classes"], 552)
        self.assertEqual(sum(size * count for size, count in
                             result["class_size_histogram"].items()), 552)
        example = result["example_equivalent_pair"]
        if example is not None:
            before_a, after_a = example[0]
            before_b, after_b = example[1]
            a = replace(base, before=before_a, after=after_a)
            b = replace(base, before=before_b, after=after_b)
            # Separate pair-product graph must concur with structural IDs.
            self.assertIsNone(mathref.shortest_public_discriminator(a, b))
        print("W17_COUNTERFACTUAL_PAIR_OBSERVABILITY", result)

    def test_bounded_independent_dp_matches_source_for_1000_generated_cases(self):
        blocks = Counter()
        maximum_dp_states = 0
        maximum_source_nodes = 0
        for seed in range(1000):
            case = make_case(seed)
            independent, states = mathref.reference_witness(case)
            maximum_dp_states = max(maximum_dp_states, states)
            claimed = search(case)
            maximum_source_nodes = max(maximum_source_nodes, claimed["nodes_explored"])
            self.assertTrue(claimed["complete"], seed)
            if independent is None:
                self.assertEqual(claimed["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES", seed)
                self.assertIsNone(claimed["witness"])
                blocks[(seed // 100, "UNSAT")] += 1
            else:
                self.assertEqual(claimed["verdict"], "VERIFIED_WIN", seed)
                self.assertTrue(mathref.reference_play(case, independent))
                self.assertTrue(mathref.reference_play(case, claimed["witness"]))
                blocks[(seed // 100, "WIN")] += 1
        self.assertEqual(sum(blocks.values()), 1000)
        self.assertGreater(sum(n for (_, verdict), n in blocks.items()
                               if verdict == "WIN"), 0)
        self.assertGreater(sum(n for (_, verdict), n in blocks.items()
                               if verdict == "UNSAT"), 0)
        print("W17_INDEPENDENT_DP1000", {
            "seed_universe": "[0,1000)",
            "train_0_99": {v: blocks[(0, v)] for v in ("WIN", "UNSAT")},
            "heldout_100_199": {v: blocks[(1, v)] for v in ("WIN", "UNSAT")},
            "all_wins": sum(n for (_, v), n in blocks.items() if v == "WIN"),
            "all_unsat": sum(n for (_, v), n in blocks.items() if v == "UNSAT"),
            "max_reference_states": maximum_dp_states,
            "max_source_nodes": maximum_source_nodes,
        })

    def test_public_equivalence_pair_product_for_thousand_seed_universe(self):
        cases = [make_case(seed) for seed in range(1000)]
        result = mathref.analyze_public_collision_classes(cases)
        self.assertEqual(result["cases"], 1000)
        self.assertLessEqual(result["max_shortest_discriminator"], MAX_PROBES)
        self.assertEqual(result["same_initial_public_pairs"],
                         result["bounded_probe_distinguishable_pairs"] +
                         len(result["bounded_probe_equivalent_pairs"]))
        if result["first_discriminator"]:
            seed_a, seed_b, actions = result["first_discriminator"]
            a, b = make_case(seed_a), make_case(seed_b)
            self.assertEqual(mathref.public_initial_signature(a),
                             mathref.public_initial_signature(b))
            session_a, session_b = ProbeSession(a), ProbeSession(b)
            for action in actions:
                session_a.probe(action)
                session_b.probe(action)
            # Transcript rows record lane levels, while the current public
            # view additionally reveals pending cargo and orientation.
            # A legal discriminator may therefore separate full views even
            # when the narrow history rows alone happen to match.
            fields = ("current_levels", "orientation", "pending_packet",
                      "ticks", "loads_used", "routes_used",
                      "post_delivery_observed", "observations")
            first_view, second_view = session_a.view(), session_b.view()
            self.assertNotEqual(tuple(first_view[field] for field in fields),
                                tuple(second_view[field] for field in fields))
        print("W17_PUBLIC_PROBE_PAIR_PRODUCT", result)

    def test_mechanism_family_overlap_and_retrospective_seed_recovery(self):
        train = [make_case(seed) for seed in range(100)]
        heldout = [make_case(seed) for seed in range(100, 200)]
        train_rules = {rule for case in train for rule in (case.before, case.after)}
        heldout_rules = {rule for case in heldout for rule in (case.before, case.after)}
        train_pairs = {(c.before, c.after) for c in train}
        heldout_pairs = {(c.before, c.after) for c in heldout}
        train_table_pairs = {(mathref.full_law_signature(c.before),
                              mathref.full_law_signature(c.after)) for c in train}
        heldout_table_pairs = {(mathref.full_law_signature(c.before),
                                mathref.full_law_signature(c.after)) for c in heldout}
        self.assertGreater(len(train_rules & heldout_rules), 0)
        self.assertGreater(len(train_pairs & heldout_pairs), 0)
        self.assertEqual(len(train_table_pairs & heldout_table_pairs),
                         len(train_pairs & heldout_pairs))
        heldout_unseen_pair_cases = sum((case.before, case.after) not in
                                        train_pairs for case in heldout)

        allowed = ("LOAD", "TICK", "ROUTE", "PURGE")
        zero_probe = mathref.retrospective_unique_count(heldout, ())
        one_probe = {action: mathref.retrospective_unique_count(
            heldout, (action,)) for action in allowed}
        best_two = max((mathref.retrospective_unique_count(heldout, actions), actions)
                       for actions in product(allowed, repeat=2))
        best_three = max((mathref.retrospective_unique_count(heldout, actions), actions)
                         for actions in product(allowed, repeat=3))
        self.assertLessEqual(zero_probe, 100)
        self.assertGreaterEqual(max(one_probe.values()), zero_probe)
        self.assertGreaterEqual(best_three[0], best_two[0])
        print("W17_SOURCE_PUBLIC_CATALOGUE_LEAKAGE", {
            "train": "[0,100)", "heldout": "[100,200)",
            "train_rule_values": len(train_rules),
            "heldout_rule_values": len(heldout_rules),
            "shared_rule_values": len(train_rules & heldout_rules),
            "train_raw_rule_pairs": len(train_pairs),
            "heldout_raw_rule_pairs": len(heldout_pairs),
            "shared_raw_rule_pairs": len(train_pairs & heldout_pairs),
            "heldout_new_pair_combinations": len(heldout_pairs - train_pairs),
            "heldout_cases_with_new_pair_combinations": heldout_unseen_pair_cases,
            "unique_with_zero_probes": zero_probe,
            "unique_after_fixed_one_probe": one_probe,
            "best_fixed_two_probe": best_two,
            "best_fixed_three_probe": best_three,
        })

    def test_independent_probe_trace_matches_real_public_observations(self):
        # A cross-check: the analysis is a source-independent prediction of
        # what the candidate can *actually* observe, not an abstract label.
        for seed in range(100):
            case = make_case(seed)
            real = ProbeSession(case)
            state = mathref.initial(case)
            for action in ("LOAD", "TICK", "ROUTE", "LOAD", "TICK", "RESET",
                           "ROUTE", "LOAD", "PURGE", "TICK"):
                expected = mathref.reference_advance(case, state, action)
                applied = expected is not None
                if applied:
                    state = expected
                public = real.probe(action)
                record = public["observations"][-1]
                self.assertIs(record["applied"], applied, (seed, action))
                self.assertEqual(public["current_levels"], list(state[:2]))
                self.assertEqual(public["orientation"], state[2])
                self.assertEqual(public["pending_packet"], state[3])
                self.assertEqual(public["ticks"], state[4])
                self.assertEqual(public["loads_used"], state[5])
                self.assertEqual(public["routes_used"], state[6])
                self.assertIs(public["post_delivery_observed"], state[7])


if __name__ == "__main__":
    unittest.main()
