"""Source-only semantic identity attacks; does not execute a language model.

Prompt-only overlays below are NOT runtime paraphrase support. They are
negative controls proving that many strings on one Case cannot create new
mechanics, while similar prose can conceal materially different transition
laws or distinct objective reachability. Language-model generalization UNKNOWN.
"""

from dataclasses import replace
import unittest

from labyrinth.switchback_circuit import (
    MAX_MOVES, State, make_case, public_view, step,
)
from labyrinth.switchback_oracle import (
    adjudicate, search, verify_plan, verify_receipt_replay,
)


_PARAPHRASES = (
    "Operate a three-bit register whose update law changes after the declared tick. "
    "Use 0, 1, WAIT, and then COMMIT at the target after a post-change input.",
    "Two binary signals manipulate a 3-bit device; midway its transition rule "
    "is revised. WAIT consumes time and COMMIT accepts only a target with a "
    "later input. Find a legal path or express uncertainty.",
    "The machine displays three bits. Commands 0/1 are pulses, WAIT advances "
    "time, COMMIT judges the declared objective after the regime shift. "
    "Respect both move and pulse budgets.",
    "In a bounded causal circuit, the first and second input laws differ. "
    "Reach the given register value by an intervention after the switch, "
    "then commit; otherwise say what remains unknown.",
)


def _overlay(case, wording):
    """A test-only presentation overlay. No new case generation occurs."""
    result = public_view(case)
    result["prompt"] = wording
    return result


def _source_lookup_ignores_prompt(visible, cache):
    """An explicitly memorizing script, not a natural-language reasoner."""
    key = tuple((name, str(value)) for name, value in sorted(visible.items())
                if name != "prompt")
    return cache.get(key)


def _engine_has_win(case):
    frontier = {State(case.initial)}
    for _ in range(MAX_MOVES):
        next_frontier = set()
        for state in frontier:
            for action in ("0", "1", "WAIT", "COMMIT"):
                result = step(case, state, action)
                if result.won:
                    return True
                if not result.committed:
                    next_frontier.add(result)
        frontier = next_frontier
    return False


class MinotaurWave10SemanticDiscrimination(unittest.TestCase):
    def test_prompt_paraphrases_are_not_new_mechanics_or_independent_trials(self):
        case = make_case(13)
        source = public_view(case)
        key = tuple((name, str(value)) for name, value in sorted(source.items())
                    if name != "prompt")
        cached = {key: search(case)}
        outcomes = []
        for wording in _PARAPHRASES:
            shown = _overlay(case, wording)
            self.assertNotEqual(shown["prompt"], source["prompt"])
            self.assertEqual({k: v for k, v in shown.items() if k != "prompt"},
                             {k: v for k, v in source.items() if k != "prompt"})
            outcomes.append(_source_lookup_ignores_prompt(shown, cached))
            self.assertEqual(case.mechanic_signature, make_case(13).mechanic_signature)
            self.assertEqual(case.case_id, make_case(13).case_id)
        self.assertTrue(all(outcome == outcomes[0] for outcome in outcomes))
        self.assertEqual(outcomes[0]["verdict"], "VERIFIED_WIN")
        self.assertEqual(len({case.mechanic_signature for _ in _PARAPHRASES}), 1)
        # Merely showing four different texts exercises zero model semantics.

    def test_same_prose_metadata_and_same_family_pair_hide_opposite_outcomes(self):
        win, unsat = make_case(3), make_case(2513)
        self.assertEqual(public_view(win), public_view(unsat))
        self.assertEqual((win.before.family, win.after.family),
                         (unsat.before.family, unsat.after.family))
        self.assertNotEqual(win.mechanic_signature, unsat.mechanic_signature)
        self.assertNotEqual(win.case_id, unsat.case_id)
        self.assertTrue(_engine_has_win(win))
        self.assertFalse(_engine_has_win(unsat))
        self.assertEqual(search(win)["verdict"], "VERIFIED_WIN")
        self.assertEqual(search(unsat)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_same_exact_mechanism_can_have_opposite_case_outcomes(self):
        win, unsat = make_case(44), make_case(3566)
        self.assertEqual(win.mechanic_signature, unsat.mechanic_signature)
        self.assertNotEqual((win.initial, win.target),
                            (unsat.initial, unsat.target))
        self.assertNotEqual(win.case_id, unsat.case_id)
        self.assertTrue(_engine_has_win(win))
        self.assertFalse(_engine_has_win(unsat))
        known_winning_path = search(win)["witness"]
        self.assertEqual(verify_plan(win, known_winning_path)["verdict"],
                         "VERIFIED_WIN")
        self.assertEqual(verify_plan(unsat, known_winning_path)["verdict"],
                         "VERIFIED_FAILURE")
        self.assertEqual(search(unsat)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_invalid_identity_reuse_rejected_independent_of_wording(self):
        case = make_case(44)
        forged = replace(case, target=(case.target + 1) % 8)
        self.assertEqual(forged.case_id, case.case_id)
        self.assertNotEqual(forged.target, case.target)
        with self.assertRaises(ValueError):
            search(forged)
        with self.assertRaises(ValueError):
            adjudicate(forged, {"kind": "ABSTAIN"})

    def test_frozen_source_identity_not_modified_by_public_view_mutation(self):
        case = make_case(13)
        printed = public_view(case)
        original = dict(printed)
        printed["prompt"] = "Contradictory mutation: WAIT now resets the register."
        self.assertNotEqual(printed, public_view(case))
        self.assertEqual(public_view(case), original)
        receipt = adjudicate(case, {"kind": "PLAN", "actions": search(case)["witness"]})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(verify_receipt_replay(receipt))
        self.assertFalse(receipt["neural_learning_verified"])
        self.assertFalse(receipt["graduation_eligible"])
        # Such a contradictory *text* is not an authorized mechanical change.


if __name__ == "__main__":
    unittest.main()
