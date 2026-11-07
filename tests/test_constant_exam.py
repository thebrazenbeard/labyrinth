"""Stdlib tests for deterministic constant-exam puzzles and hostile controls."""

import json
import unittest
from unittest.mock import patch

import labyrinth.constant_exam as exam_module
from labyrinth.constant_exam import (ACTIONS, MAX_ACTIONS, CandidateSession, State,
                           make_exam, observe, public_view, ruleset_source_fingerprint,
                           search, step, world_fingerprint)
from labyrinth.constant_exam_verifier import verify_trace, verify_unsat_certificate


class ConstantExamTests(unittest.TestCase):
    def setUp(self):
        self.open_case = make_exam(42, split="test", variant=1)
        self.sealed_case = make_exam(42, split="test", variant=1, kind="sealed_control")

    def test_paraphrase_changes_wording_but_not_semantics(self):
        prompts = [public_view(make_exam(42, split="test", variant=i))["prompt"]
                   for i in range(16)]
        self.assertGreater(len(set(prompts)), 1)
        for variant in range(16):
            other = make_exam(42, split="test", variant=variant)
            self.assertEqual(other.case_id, self.open_case.case_id)
            self.assertEqual(world_fingerprint(other), world_fingerprint(self.open_case))
            self.assertEqual(search(other).witness, search(self.open_case).witness)

    def test_seed_and_split_are_reproducible_and_separated(self):
        self.assertEqual(make_exam(42, split="dev", variant=3),
                         make_exam(42, split="dev", variant=3))
        self.assertNotEqual(make_exam(42, split="test").case_id,
                            make_exam(42, split="train").case_id)
        self.assertNotEqual(public_view(make_exam(41))["prompt"],
                            public_view(make_exam(42))["prompt"])

    def test_only_whitelisted_candidate_keys_and_no_oracle_label(self):
        candidate = public_view(self.open_case)
        self.assertEqual(set(candidate), {
            "case_id", "trial_id", "prompt", "legal_action_vocabulary", "maximum_actions",
            "initial_observation"})
        payload = json.dumps(candidate).lower()
        for secret in ("apparently_sealed", "sealed_control", "witness", "answer",
                       "verified_win", "unsat_in_declared", "kind", "token"):
            self.assertNotIn(secret, payload)

    def test_capped_search_returns_unknown_not_false_impossibility(self):
        result = search(self.open_case, max_expansions=2)
        self.assertEqual(result.status, "UNSOLVED")
        self.assertEqual(result.witness, ())
        self.assertNotEqual(search(self.open_case).status, "UNSAT_IN_DECLARED_FINITE_MODEL")

    def test_unexpected_valid_witness_is_accepted(self):
        # Independent verifier must not require equality to a memorized key.
        legitimate = ("north", "east", "take", "west", "south", "north", "north",
                      "offer", "leave")
        self.assertEqual(verify_trace(kind=self.open_case.kind, actions=legitimate).outcome,
                         "VERIFIED_WIN")

    def test_search_witness_replayed_independently(self):
        oracle = search(self.open_case)
        self.assertEqual(oracle.status, "VERIFIED_WIN")
        self.assertLessEqual(len(oracle.witness), MAX_ACTIONS)
        self.assertEqual(verify_trace(kind=self.open_case.kind, actions=oracle.witness).outcome,
                         "VERIFIED_WIN")

    def test_formal_unreachable_only_for_fully_exhausted_control(self):
        bounded = search(self.sealed_case, max_expansions=1)
        self.assertEqual(bounded.status, "UNSOLVED")
        complete = search(self.sealed_case)
        self.assertEqual(complete.status, "UNSAT_IN_DECLARED_FINITE_MODEL")
        self.assertFalse(any(s.place == "freedom" for s in complete.reachable))
        for state in complete.reachable:
            for action in ACTIONS:
                self.assertIn(step(self.sealed_case, state, action), complete.reachable)
        self.assertTrue(verify_unsat_certificate(kind=self.sealed_case.kind,
                                                states=complete.reachable))

    def test_independent_unsat_certificate_rejects_forgery(self):
        closed = search(self.sealed_case).reachable
        self.assertFalse(verify_unsat_certificate(kind=self.open_case.kind,
                                                states=closed))
        self.assertFalse(verify_unsat_certificate(kind=self.sealed_case.kind,
                                                states=closed - {State()}))
        self.assertFalse(verify_unsat_certificate(kind=self.sealed_case.kind,
                                                states={State()}))
        self.assertFalse(verify_unsat_certificate(kind=self.sealed_case.kind,
                                                states=search(self.open_case).reachable))

    def test_single_rule_counterfactual_falsifies_unsat_transfer(self):
        self.assertNotEqual(world_fingerprint(self.open_case),
                            world_fingerprint(self.sealed_case))
        self.assertEqual(search(self.sealed_case).status, "UNSAT_IN_DECLARED_FINITE_MODEL")
        self.assertEqual(search(self.open_case).status, "VERIFIED_WIN")
        self.assertEqual(
            verify_trace(kind=self.sealed_case.kind, actions=search(self.open_case).witness).outcome,
            "NOT_WON")

    def test_forged_win_and_illegal_actions_rejected(self):
        for actions in (("WIN",), ("teleport",), ("north", "offer", "leave"),
                        ("north", "east", "take", "west", "north", "leave")):
            self.assertNotEqual(verify_trace(kind=self.open_case.kind, actions=actions).outcome,
                                "VERIFIED_WIN")
        self.assertEqual(verify_trace(kind=self.open_case.kind,
                         actions=("north",) * (MAX_ACTIONS + 1)).outcome, "INVALID_TRACE")

    def test_transition_model_and_independent_trace_replay_agree(self):
        for exam in (self.open_case, self.sealed_case):
            frontier = [(State(), ())]
            seen = {State()}
            while frontier:
                state, path = frontier.pop(0)
                if state.place == "freedom":
                    continue
                for action in ACTIONS:
                    nxt = step(exam, state, action)
                    actions = path + (action,)
                    if len(actions) <= MAX_ACTIONS:
                        independent = verify_trace(kind=exam.kind, actions=actions)
                        self.assertEqual(independent.final_room, nxt.place)
                        self.assertEqual(independent.outcome == "VERIFIED_WIN",
                                         nxt.place == "freedom")
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append((nxt, actions))

    def test_validate_inputs(self):
        for seed in (True, 1.3, "1"):
            with self.assertRaises(ValueError):
                make_exam(seed)
        with self.assertRaises(ValueError):
            make_exam(1, kind="anything")
        with self.assertRaises(ValueError):
            search(self.open_case, max_expansions=-1)
        with self.assertRaises(ValueError):
            step(self.open_case, State(), "jump")

    def test_trial_id_changes_without_inflating_semantic_family(self):
        zero = public_view(make_exam(42, variant=0))
        nine = public_view(make_exam(42, variant=9))
        self.assertEqual(zero["case_id"], nine["case_id"])
        self.assertNotEqual(zero["trial_id"], nine["trial_id"])
        self.assertEqual(len(ruleset_source_fingerprint()), 64)

    def test_candidate_session_observes_and_escapes_without_kind_control(self):
        session = CandidateSession(self.open_case)
        self.assertIn("initial_observation", session.start())
        transcript = []
        for action in search(self.open_case).witness:
            result = session.act(action)
            transcript.append(result)
            self.assertNotIn("kind", result)
            self.assertNotIn("coin", result)
            self.assertNotIn("open_gate", result)
        self.assertTrue(any("token" in r["observation"] for r in transcript))
        self.assertEqual(transcript[-1]["status"], "VERIFIED_WIN")
        with self.assertRaises(ValueError):
            session.act("north")
        with self.assertRaises(ValueError):
            session.start()

    def test_budget_change_changes_fingerprint_without_false_global_unsat(self):
        normal = ruleset_source_fingerprint()
        with patch.object(exam_module, "MAX_ACTIONS", 3):
            self.assertNotEqual(ruleset_source_fingerprint(), normal)
            self.assertEqual(search(self.open_case).status, "NO_WIN_WITHIN_ACTION_BUDGET")

    def test_session_binds_private_kind_and_has_strict_budget(self):
        sealed = CandidateSession(self.sealed_case)
        open_witness = search(self.open_case).witness
        last = None
        observations = []
        for action in open_witness:
            last = sealed.act(action)
            observations.append(last["observation"])
        self.assertNotEqual(last["status"], "VERIFIED_WIN")
        self.assertTrue(any("Bare shelves" in observation for observation in observations))
        invalid = CandidateSession(self.open_case)
        for _ in range(MAX_ACTIONS):
            exhausted = invalid.act("BREAK REALM")
            self.assertEqual(exhausted["status"], "INVALID_ACTION")
        self.assertEqual(exhausted["actions_remaining"], 0)
        with self.assertRaises(ValueError):
            invalid.act("north")


if __name__ == "__main__":
    unittest.main()
