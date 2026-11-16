"""Independent Switchback negative controls; no model updates or host runs.

These exact-source tests challenge *generalization claims*, not the usefulness
of a finite search oracle. Public-source rules, seeded parameter variation,
and privileged `Case` access must never be called neural learning evidence.
"""

import unittest

from labyrinth.switchback_circuit import (
    CONTROLS, MAX_MOVES, ProbeSession, State, kernel, make_case, public_view, step,
)
from labyrinth.switchback_controls import (
    MEMORIZED_SEED0_PATH, memorized_control, privileged_symbolic_control,
)
from labyrinth.switchback_oracle import (
    adjudicate, search, verify_plan, verify_receipt_replay,
)


def _independent_engine_reachability(case):
    """Enumerate all legal action sequences using simulator path, not oracle.

    Full finite dynamic programming across nine actions, with state merging
    only when the complete engine State is equal. A self-loop cannot improve a
    future plan at the same depth, but its inclusion is harmless.
    """
    frontier = {State(case.initial)}
    for _ in range(MAX_MOVES):
        next_states = set()
        for state in frontier:
            for action in CONTROLS:
                next_state = step(case, state, action)
                if next_state.won:
                    return True
                if not next_state.committed:
                    next_states.add(next_state)
        frontier = next_states
    return False


def _public_metadata_key(case):
    view = public_view(case)
    return (view["initial_register"], view["target_register"],
            view["switch_at"], view["maximum_pulses"],
            view["maximum_moves"],
            tuple(view.get("plan_actions", view.get("actions", ()))),
            tuple(view.get("probe_actions", ())))


def _precomputed_public_catalog(first_seed=0, stop_seed=100):
    """Source-reading attack: offline cache of publicly generated cases."""
    by_public_key = {}
    for seed in range(first_seed, stop_seed):
        case = make_case(seed)
        by_public_key.setdefault(_public_metadata_key(case), []).append(case)
    return by_public_key


def _zero_probe_catalog_policy(session, catalog):
    """Sees ONLY public metadata, not target private Case or source seed.

    If one offline cached case shares this exact public signature, submits a
    precomputed, untrained answer. Otherwise it abstains. This is a deliberate
    training/evaluation contamination demonstration, not a reliable solver.
    """
    visible = session.view()
    public_key = (visible["initial_register"], visible["target_register"],
                  visible["switch_at"], visible["maximum_pulses"],
                  visible["maximum_moves"],
                  tuple(visible.get("plan_actions", visible.get("actions", ()))),
                  tuple(visible.get("probe_actions", ())))
    candidates = catalog.get(public_key, [])
    if len(candidates) != 1:
        return session.submit({"kind": "ABSTAIN"})
    precomputed = search(candidates[0])
    if precomputed["verdict"] == "VERIFIED_WIN":
        return session.submit({"kind": "PLAN", "actions": precomputed["witness"]})
    if precomputed["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        return session.submit({"kind": "CLAIM_UNSAT"})
    return session.submit({"kind": "ABSTAIN"})


_PRECOMMITTED_DISAMBIGUATING_PROBES = (
    "0", "1", "0", "WAIT", "1", "WAIT", "RESET", "1", "0", "1", "WAIT", "0",
)


def _offline_probe_fingerprint(case):
    simulation = ProbeSession(case)
    return tuple(simulation.probe(action)["current_register"]
                 for action in _PRECOMMITTED_DISAMBIGUATING_PROBES)


def _probe_catalog_policy(session, catalog):
    """Source-indexed, zero-update policy with only *public live* observations.

    The 12-action sequence is precommitted, not selected by learning. For each
    offline source case, run its published observation protocol to obtain an
    expected fingerprint. The policy never accesses the live session's `_case`
    (unlike the intentionally privileged negative control).
    """
    visible = session.view()
    key = (visible["initial_register"], visible["target_register"],
           visible["switch_at"], visible["maximum_pulses"],
           visible["maximum_moves"],
           tuple(visible.get("plan_actions", visible.get("actions", ()))),
           tuple(visible.get("probe_actions", ())))
    possible = catalog.get(key, [])
    if len(possible) > 1:
        observed = tuple(session.probe(action)["current_register"]
                         for action in _PRECOMMITTED_DISAMBIGUATING_PROBES)
        possible = [case for case in possible
                    if _offline_probe_fingerprint(case) == observed]
    if len(possible) != 1:
        return session.submit({"kind": "ABSTAIN"})
    cached = search(possible[0])
    if cached["verdict"] == "VERIFIED_WIN":
        return session.submit({"kind": "PLAN", "actions": cached["witness"]})
    if cached["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
        return session.submit({"kind": "CLAIM_UNSAT"})
    return session.submit({"kind": "ABSTAIN"})


class SwitchbackWave9PsychologistTests(unittest.TestCase):
    def test_precommitted_source_catalog_probes_can_fit_all_100_in_sample(self):
        catalog = _precomputed_public_catalog()
        outcomes = {"VERIFIED_WIN": 0,
                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0}
        lengths = set()
        for seed in range(100):
            session = ProbeSession(make_case(seed))
            result = _probe_catalog_policy(session, catalog)
            outcomes[result["verdict"]] += 1
            lengths.add(len(session.view()["observations"]))
            self.assertTrue(verify_receipt_replay(result))
            self.assertFalse(result["neural_learning_verified"])
        self.assertEqual(outcomes, {"VERIFIED_WIN": 67,
                                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 33})
        self.assertEqual(lengths, {0, len(_PRECOMMITTED_DISAMBIGUATING_PROBES)})
        # Perfect catalogue fit is not neural parameter learning or transfer.

        unseen = {"UNKNOWN": 0, "VERIFIED_FAILURE": 0,
                  "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0,
                  "VERIFIED_WIN": 0}
        for seed in range(100, 200):
            result = _probe_catalog_policy(ProbeSession(make_case(seed)), catalog)
            unseen[result["verdict"]] += 1
        self.assertEqual(unseen, {"UNKNOWN": 86, "VERIFIED_FAILURE": 12,
                                  "PROVEN_UNSAT_UNDER_DECLARED_RULES": 1,
                                  "VERIFIED_WIN": 1})

    def test_public_source_catalog_identifies_83_of_100_without_probes(self):
        # Train/test overlap is deliberate to prove how the benchmark can be
        # gamed, even when the candidate cannot inspect live hidden state.
        catalog = _precomputed_public_catalog()
        self.assertEqual(sum(len(cases) == 1 for cases in catalog.values()), 83)
        self.assertEqual(sum(len(cases) for cases in catalog.values()
                             if len(cases) > 1), 17)
        outcomes = {"VERIFIED_WIN": 0,
                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0,
                    "UNKNOWN": 0}
        for seed in range(100):
            session = ProbeSession(make_case(seed))
            visible = session.view()
            self.assertNotIn("seed", visible)
            self.assertNotIn("before", visible)
            self.assertNotIn("after", visible)
            receipt = _zero_probe_catalog_policy(session, catalog)
            outcomes[receipt["verdict"]] += 1
            self.assertEqual(session.view()["observations"], [])
            if receipt["verdict"] != "UNKNOWN":
                self.assertTrue(verify_receipt_replay(receipt))
            self.assertFalse(receipt["neural_learning_verified"])
        self.assertEqual(sum(outcomes.values()), 100)
        self.assertEqual(outcomes["UNKNOWN"], 17)
        self.assertEqual(outcomes["VERIFIED_WIN"] +
                         outcomes["PROVEN_UNSAT_UNDER_DECLARED_RULES"], 83)

    def test_new_seeds_are_not_independent_holdout_of_public_catalog(self):
        catalog = _precomputed_public_catalog()
        classified = false_or_different = 0
        receipts = {"UNKNOWN": 0, "VERIFIED_FAILURE": 0,
                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0,
                    "VERIFIED_WIN": 0}
        for seed in range(100, 200):
            case = make_case(seed)
            result = _zero_probe_catalog_policy(ProbeSession(case), catalog)
            receipts[result["verdict"]] += 1
            candidate = catalog.get(_public_metadata_key(case), [])
            if len(candidate) != 1:
                continue
            classified += 1
            old = search(candidate[0])
            new = search(case)
            if (old["verdict"] != new["verdict"] or
                    (old["witness"] is not None and
                     verify_plan(case, old["witness"])["verdict"] != "VERIFIED_WIN")):
                false_or_different += 1
        self.assertGreater(classified, 0)
        self.assertGreater(false_or_different, 0)
        # Exact negative-control test on the distinct set of held-out seeds:
        # the contaminated lookup collapses to 2 correct, 12 false and 86
        # abstentions. No novel mechanics have been learned in that policy.
        self.assertEqual(receipts, {"UNKNOWN": 86, "VERIFIED_FAILURE": 12,
                                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 1,
                                    "VERIFIED_WIN": 1})

    def test_zero_probe_privileged_oracle_is_a_contamination_control(self):
        for seed in range(30):
            case = make_case(seed)
            session = ProbeSession(case)
            # Test-only deliberate violation: reading session._case would
            # bypass all candidate-public evidence and probe requirements.
            hidden = session._case
            result = search(hidden)
            if result["verdict"] == "VERIFIED_WIN":
                receipt = session.submit({"kind": "PLAN", "actions": result["witness"]})
            else:
                receipt = session.submit({"kind": "CLAIM_UNSAT"})
            self.assertEqual(receipt["verdict"], result["verdict"])
            self.assertEqual(session.view()["observations"], [])
            self.assertFalse(receipt.get("private_custody",
                                         receipt.get("adjudication", {}).get(
                                             "private_custody", True)))

    def test_identical_public_state_has_opposite_satisfiability(self):
        # A public-only zero-probe candidate sees literally identical data.
        # Returning WIN or UNSAT without probing must be wrong in one world.
        winning = make_case(13)
        unsat = make_case(14)
        self.assertEqual(public_view(winning), public_view(unsat))
        self.assertTrue(_independent_engine_reachability(winning))
        self.assertFalse(_independent_engine_reachability(unsat))
        self.assertEqual(search(winning)["verdict"], "VERIFIED_WIN")
        self.assertEqual(search(unsat)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        for case in (winning, unsat):
            receipt = adjudicate(case, {"kind": "ABSTAIN"})
            self.assertEqual(receipt["verdict"], "UNKNOWN")
            self.assertFalse(receipt["neural_learning_verified"])
            self.assertFalse(receipt["graduation_eligible"])
            self.assertTrue(verify_receipt_replay(receipt))

    def test_source_aware_seed0_script_generalizes_by_chance_not_updating(self):
        self.assertEqual(MEMORIZED_SEED0_PATH, ("0", "0", "WAIT", "0", "COMMIT"))
        wins = []
        for seed in range(100):
            receipt = memorized_control(make_case(seed))
            if receipt["verdict"] == "VERIFIED_WIN":
                wins.append(seed)
            self.assertEqual(receipt["proposal"]["actions"],
                             list(MEMORIZED_SEED0_PATH))
            self.assertFalse(receipt["neural_learning_verified"])
        self.assertEqual(wins, [0, 27, 31, 60, 62, 84, 95])

    def test_hidden_case_oracle_perfectly_classifies_without_public_observation(self):
        outcomes = {"VERIFIED_WIN": 0, "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0}
        for seed in range(100):
            case = make_case(seed)
            receipt = privileged_symbolic_control(case)
            self.assertTrue(verify_receipt_replay(receipt))
            self.assertEqual(receipt["verdict"], search(case)["verdict"])
            outcomes[receipt["verdict"]] += 1
            self.assertEqual(receipt["source_commit"], "UNBOUND")
            self.assertFalse(receipt["private_custody"])
            self.assertFalse(receipt["neural_learning_verified"])
            self.assertFalse(receipt["graduation_eligible"])
        self.assertEqual(outcomes, {"VERIFIED_WIN": 67,
                                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 33})

    def test_independent_engine_and_symbolic_reachability_agree_on_100_seeds(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                case = make_case(seed)
                independent_win = _independent_engine_reachability(case)
                result = search(case)
                self.assertEqual(result["verdict"] == "VERIFIED_WIN",
                                 independent_win)
                self.assertTrue(result["complete"])
                if result["witness"] is not None:
                    self.assertEqual(verify_plan(case, result["witness"])["verdict"],
                                     "VERIFIED_WIN")

    def test_distinct_case_signatures_are_not_independent_mechanic_families(self):
        cases = [make_case(seed) for seed in range(100)]
        # The generator truly varies observable truth tables, unlike the
        # Cipher Wheel's mere color permutations. But all are synthesized by
        # five PUBLIC kernel families, not 100 independently invented laws.
        signatures = {case.mechanic_signature for case in cases}
        pairs = {(case.before.family, case.after.family) for case in cases}
        underlying = {r.family for case in cases for r in (case.before, case.after)}
        table = lambda rule: tuple(kernel(r, bit, rule)
                                   for r in range(8) for bit in (0, 1))
        exact_rule_tables = {table(rule) for case in cases
                             for rule in (case.before, case.after)}
        exact_rule_pairs = {(table(case.before), table(case.after)) for case in cases}
        self.assertEqual(len(signatures), 100)
        self.assertEqual(len(pairs), 25)
        self.assertEqual(len(underlying), 5)
        self.assertEqual(len(exact_rule_tables), 132)
        self.assertEqual(len(exact_rule_pairs), 99)
        self.assertTrue(all(case.mechanic_signature != case.case_id for case in cases))
        # Neither 100 unique hashes nor 25 generator pairs are an established
        # independent sampling unit for *neural transfer*.
        self.assertTrue(all(adjudicate(case, {"kind": "ABSTAIN"})[
            "semantic_family_independence"] == "UNQUALIFIED" for case in cases))


if __name__ == "__main__":
    unittest.main()
