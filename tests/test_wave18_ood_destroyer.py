"""Independent hostile single-fault injections for causal-latch OOD family.

Frozen source: 8c512fb71b44a3572576406c37c5256d6f1c71e9.
These fault injections are source TCB tests, not claims of remote OS attacks.
"""

from copy import deepcopy
from hashlib import sha256
import unittest
from unittest import mock

from labyrinth import novel_router_ood_family as ood


class Wave18OodDestroyer(unittest.TestCase):
    @staticmethod
    def _independent_reachability(case: ood.Case) -> bool:
        """Separate graph search using only canonical case data, no source solvers."""
        beginning = (case.initial_bits, 0, 0, 0, False)
        frontier = [beginning]
        visited = {beginning}
        while frontier:
            bits, sealed, ticks, seals_used, after_seals = frontier.pop()
            if ticks >= ood.MAX_MOVES:
                continue  # A final COMMIT itself consumes a move.
            if (bits == case.target_bits and sealed == case.target_sealed and
                    seals_used == case.max_seals and after_seals):
                return True
            candidates = []
            for node in range(ood.NODES):
                mask = 1 << node
                if not sealed & mask:
                    candidates.append((bits ^ mask, sealed, ticks + 1,
                                       seals_used, after_seals))
                    if seals_used < case.max_seals:
                        candidates.append((bits, sealed | mask, ticks + 1,
                                           seals_used + 1, after_seals))
            pulsed = 0
            for node in range(ood.NODES):
                mask = 1 << node
                if sealed & mask:
                    bit_on = bool(bits & mask)
                else:
                    on_neighbors = (bits & case.parents[node]).bit_count()
                    bit_on = on_neighbors >= case.thresholds[node]
                if bit_on:
                    pulsed |= mask
            candidates.append((pulsed, sealed, ticks + 1,
                               seals_used,
                               after_seals or seals_used == case.max_seals))
            for next_state in candidates:
                if next_state not in visited:
                    visited.add(next_state)
                    frontier.append(next_state)
        return False

    def test_independent_finite_closure_disagrees_with_no_seed0_to_63_verdict(self):
        """Independent source-only reachability challenges win and UNSAT labels."""
        outcomes = {"VERIFIED_WIN": 0, "PROVEN_UNSAT_UNDER_DECLARED_RULES": 0}
        for seed in range(64):
            case = ood.make_case(seed)
            report = ood.search(case)
            self.assertTrue(report["complete"], f"seed={seed} incomplete closure")
            self.assertIn(report["verdict"], outcomes, f"seed={seed}")
            independent_win = self._independent_reachability(case)
            self.assertEqual(report["verdict"] == "VERIFIED_WIN", independent_win,
                             f"seed={seed} source verdict contradicted independent "
                             "finite-state reachability")
            outcomes[report["verdict"]] += 1
        self.assertEqual(sum(outcomes.values()), 64)

    def test_session_submit_must_reject_forged_adjudicator_win(self):
        """One faulty adjudicator must not promote an invalid plan to a win."""
        case = ood.make_case(0)
        invalid = {"kind": "PLAN", "actions": ["COMMIT"]}
        source = ood.adjudicate(case, invalid)
        self.assertEqual(source["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(ood._raw_plan_verdict(case, invalid["actions"]),
                         "VERIFIED_FAILURE")
        forged = deepcopy(source)
        forged["verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["reason"] = "TWO_PLAN_CHECKS_AGREE"
        forged["sha256"] = sha256(ood._canonical({
            key: value for key, value in forged.items()
            if key != "sha256"}).encode()).hexdigest()

        session = ood.ProbeSession(case)
        with mock.patch.object(ood, "adjudicate", return_value=forged):
            receipt = session.submit(invalid)
        self.assertIs(receipt["verified_success"], False,
                      "source session accepted a single falsified adjudicator "
                      "WIN without its own primitive goal replay")
        self.assertNotEqual(receipt["verdict"], "VERIFIED_WIN")

    def test_reset_revalidates_independent_genesis_after_mutable_constructor_fault(self):
        """Post-init RESET must not silently install a corrupted initial state."""
        case = ood.make_case(0)
        session = ood.ProbeSession(case)
        session.probe("PULSE")
        corrupted_bits = case.initial_bits ^ 1
        with mock.patch.object(ood, "initial_state", return_value=ood.State(corrupted_bits)):
            feedback = session.probe("RESET")
        self.assertTrue(feedback["finished"],
                        "reset installed a mutated genesis and returned normal "
                        "public feedback instead of fail-closed quarantine")
        self.assertEqual(feedback["current_bits"], f"{case.initial_bits:05b}")


if __name__ == "__main__":
    unittest.main()
