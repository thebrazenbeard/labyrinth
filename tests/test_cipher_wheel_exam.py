"""Hostile finite causal exam tests: no model or private evaluator execution."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from itertools import product
import io
import json
import unittest
from unittest import mock

from labyrinth import cipher_wheel_world as dut
from labyrinth.cipher_wheel_cli import main
from labyrinth.cipher_wheel_controls import (MEMORIZED_TRAIN_SEED0,
                                             memorized_control, symbolic_explorer)
from labyrinth.cipher_wheel_oracle import (_canonical, oracle_labels, oracle_table,
                                           replay_trace, verify_receipt_replay,
                                           verify_table)


def _resign(receipt):
    cloned = deepcopy(receipt)
    cloned.pop("receipt_sha256", None)
    cloned["receipt_sha256"] = sha256(_canonical(cloned).encode()).hexdigest()
    return cloned


def _group_order(table):
    labels = tuple(dut.LABELS)
    generators = [tuple(labels.index(table[action][label]) for label in labels)
                  for action in ("A", "B")]
    identity = tuple(range(4))
    group = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for gen in generators:
            composite = tuple(gen[current[index]] for index in range(4))
            if composite not in group:
                group.add(composite)
                frontier.append(composite)
    return len(group)


class CipherWheelHostileTests(unittest.TestCase):
    def test_exact_frozen_world_and_seed_type_gates(self):
        for seed in (None, True, 0.0, -1, 2**31, "0"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                dut.make_world(seed)
        for kwargs in ({"heldout": 1}, {"heldout": "true"},
                       {"variant": True}, {"variant": -1}, {"variant": 10000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dut.make_world(0, **kwargs)
        with self.assertRaises(FrozenInstanceError):
            dut.make_world(0).seed = 1

    def test_candidate_only_public_view_has_no_latent_regime_seed_or_table(self):
        for heldout in (False, True):
            public = dut.candidate_view(dut.make_world(17, heldout=heldout))
            self.assertEqual(set(public), {"prompt", "labels", "actions", "current",
                                           "remaining_attempts", "history", "verdict", "finished"})
            self.assertNotIn("heldout", str(public))
            self.assertNotIn("seed", str(public))
            self.assertNotIn("world_id", str(public))

    def test_semantic_world_id_is_variant_invariant_but_split_sensitive(self):
        base = dut.make_world(17)
        self.assertEqual(base.world_id, dut.make_world(17, variant=100).world_id)
        self.assertNotEqual(base.world_id, dut.make_world(17, heldout=True).world_id)
        self.assertNotEqual(base.world_id, dut.make_world(18).world_id)

    def test_seeded_label_binding_matches_separately_implemented_prf(self):
        for seed in (0, 1, 2, 3, 15, 100, 123456, 2**31 - 1):
            self.assertEqual(dut.seeded_labels(seed), oracle_labels(seed))
            self.assertEqual(set(oracle_labels(seed)), set(dut.LABELS))

    def test_train_and_heldout_have_different_group_orders_eight_vs_24(self):
        for seed in range(12):
            self.assertEqual(_group_order(oracle_table(dut.make_world(seed))), 8)
            self.assertEqual(_group_order(oracle_table(dut.make_world(seed, heldout=True))), 24)

    def test_train_and_heldout_are_mechanically_not_lexically_distinct(self):
        for seed in range(12):
            train = oracle_table(dut.make_world(seed))
            heldout = oracle_table(dut.make_world(seed, heldout=True))
            self.assertNotEqual(train, heldout)
            self.assertEqual(sum(train["A"][x] == x for x in dut.LABELS), 0)
            self.assertEqual(sum(heldout["A"][x] == x for x in dut.LABELS), 1)

    def test_candidate_seed_and_split_cannot_be_read_from_public_history(self):
        for heldout in (False, True):
            session = dut.CandidateSession(dut.make_world(13, heldout=heldout))
            for action in ("B", "A", "RESET", "A"):
                session.act(action)
            view = session.view()
            self.assertEqual(set(view), set(dut.candidate_view(dut.make_world(0))))
            self.assertNotIn("heldout", str(view))
            self.assertNotIn("world_id", str(view))

    def test_all_public_four_action_traces_parity_with_separate_oracle(self):
        for heldout in (False, True):
            for suffix in product(dut.ACTIONS, repeat=4):
                session = dut.CandidateSession(dut.make_world(7, heldout=heldout))
                for action in suffix:
                    view = session.act(action)
                state, trace = replay_trace(dut.make_world(7, heldout=heldout),
                                            view["history"])
                self.assertEqual(view["current"], oracle_labels(7)[state])
                self.assertEqual(trace, tuple(view["history"]))

    def test_invalid_actions_are_charged_with_no_state_change(self):
        session = dut.CandidateSession(dut.make_world(0))
        start = session.view()["current"]
        for action in ("BACKDOOR", "a", True, 1, None, "X" * 100):
            view = session.act(action)
            self.assertEqual(view["current"], start)
            self.assertFalse(view["history"][-1]["valid"])
        self.assertEqual(session.view()["remaining_attempts"], dut.MAX_ATTEMPTS - 6)
        self.assertIsNotNone(replay_trace(dut.make_world(0), session.view()["history"]))

    def test_python_equality_spoof_does_not_activate_controls(self):
        class FakeAction:
            def __eq__(self, other):
                return other == "A"
        session = dut.CandidateSession(dut.make_world(0))
        self.assertEqual(session.act(FakeAction())["current"], session.view()["current"])
        self.assertFalse(session.view()["history"][0]["valid"])
        self.assertEqual(session.view()["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(session.receipt()["integrity_status"], "BLOCKED_UNSERIALIZABLE_TRACE")
        self.assertFalse(verify_receipt_replay(session.receipt()))

    def test_invalid_action_values_have_distinct_exact_typed_receipts(self):
        for left, right in (("X" * 65, "Y" * 65), (True, None), (False, 0),
                            ("X" * 100, "X" * 101)):
            with self.subTest(left=str(left)[:10], right=str(right)[:10]):
                l = dut.CandidateSession(dut.make_world(0))
                r = dut.CandidateSession(dut.make_world(0))
                self.assertFalse(l.act(left)["history"][-1]["valid"])
                self.assertFalse(r.act(right)["history"][-1]["valid"])
                a, b = l.receipt(), r.receipt()
                self.assertNotEqual(a["attempts"], b["attempts"])
                self.assertNotEqual(a["receipt_sha256"], b["receipt_sha256"])
                self.assertTrue(verify_receipt_replay(a))
                self.assertTrue(verify_receipt_replay(b))
                self.assertEqual(len(a["attempts"]), 1)
                self.assertEqual(len(b["attempts"]), 1)

    def test_unrepresentable_invalid_action_consumes_attempt_and_blocks_receipt(self):
        session = dut.CandidateSession(dut.make_world(0))
        result = session.act("x" * 10000)
        self.assertTrue(result["finished"])
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(len(result["history"]), 1)
        receipt = session.receipt()
        self.assertEqual(receipt["integrity_status"], "BLOCKED_UNSERIALIZABLE_TRACE")
        self.assertFalse(receipt["verified_success"])
        self.assertFalse(verify_receipt_replay(receipt))

    def test_nested_json_coercion_attempts_are_quarantined_not_laundered(self):
        # Native json.dumps maps Python int/None/bool mapping keys to strings
        # and tuples to arrays; losing these types previously forged custody.
        attacks = (({1: "x"}, {"1": "x"}),
                   ({None: 7}, {"null": 7}),
                   ({True: "x"}, {"true": "x"}),
                   ([(1, 2)], [[1, 2]]))
        for hostile, json_safe in attacks:
            with self.subTest(hostile=repr(hostile)):
                blocked = dut.CandidateSession(dut.make_world(0))
                observation = blocked.act(hostile)
                self.assertEqual(observation["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertTrue(observation["finished"])
                self.assertEqual(len(observation["history"]), 1)
                r = blocked.receipt()
                self.assertEqual(r["integrity_status"], "BLOCKED_UNSERIALIZABLE_TRACE")
                self.assertFalse(r["verified_success"])
                self.assertFalse(verify_receipt_replay(r))
                later = blocked.submit_table(oracle_table(dut.make_world(0)))
                self.assertEqual(later["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertFalse(later["verified_success"])
                legal_json = dut.CandidateSession(dut.make_world(0))
                legal_view = legal_json.act(json_safe)
                self.assertEqual(legal_view["verdict"], "UNKNOWN")
                self.assertEqual(len(legal_view["history"]), 1)
                self.assertTrue(verify_receipt_replay(legal_json.receipt()))

    def test_attempt_budget_exhausts_and_prevents_late_table_submission(self):
        session = dut.CandidateSession(dut.make_world(0))
        for _ in range(dut.MAX_ATTEMPTS):
            session.act("bogus")
        receipt = session.submit_table(oracle_table(dut.make_world(0)))
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertFalse(receipt["verified_success"])
        self.assertEqual(len(receipt["attempts"]), dut.MAX_ATTEMPTS)
        self.assertTrue(verify_receipt_replay(receipt))

    def test_correct_finite_table_proves_only_exact_world_win(self):
        world = dut.make_world(0)
        receipt = dut.CandidateSession(world).submit_table(oracle_table(world))
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertEqual(receipt["table_review"]["full_rows_checked"], 8)
        self.assertTrue(receipt["verified_success"])
        self.assertFalse(receipt["neural_learning_verified"])
        self.assertFalse(receipt["independent_custody"])
        self.assertFalse(receipt["graduation_eligible"])
        self.assertTrue(receipt["public_source_oracle"])
        self.assertEqual(receipt["semantic_family_id"], "cipher-wheel.D4")
        self.assertEqual(receipt["effective_semantic_families_in_receipt"], 1)
        self.assertEqual(len(receipt["attempts"]), 0)  # public source, not discovery
        self.assertTrue(verify_receipt_replay(receipt))

    def test_wrong_table_gives_replayable_specific_falsifier(self):
        world = dut.make_world(0, heldout=True)
        wrong = deepcopy(oracle_table(world))
        wrong["A"]["amber"] = "amber" if wrong["A"]["amber"] != "amber" else "jade"
        receipt = dut.CandidateSession(world).submit_table(wrong)
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["table_review"]["reason"], "FALSIFIED_TRANSITION")
        self.assertEqual(receipt["table_review"]["counterexample"]["from"], "amber")
        self.assertTrue(verify_receipt_replay(receipt))

    def test_malformed_serializable_proposal_is_falsified_and_preserved(self):
        for candidate in ({}, {"A": []}, {"A": {}, "B": {}}, None, "guess"):
            with self.subTest(candidate=candidate):
                receipt = dut.CandidateSession(dut.make_world(0)).submit_table(candidate)
                self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
                self.assertEqual(receipt["reason"], "INVALID_TABLE")
                self.assertEqual(receipt["submitted_raw"], candidate)
                self.assertTrue(verify_receipt_replay(receipt))

    def test_unserializable_or_oversized_submission_is_terminal_no_free_retry(self):
        for candidate in (object(), {"bad": {1, 2}}, "x" * 9000):
            with self.subTest(candidate=type(candidate).__name__):
                session = dut.CandidateSession(dut.make_world(0))
                rejected = session.submit_table(candidate)
                self.assertEqual(rejected["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertEqual(rejected["reason"], "BLOCKED_UNSERIALIZABLE_SUBMISSION")
                self.assertTrue(rejected["submission_made"])
                self.assertFalse(rejected["verified_success"])
                self.assertTrue(session.view()["finished"])
                self.assertFalse(verify_receipt_replay(rejected))
                later = session.submit_table(oracle_table(dut.make_world(0)))
                self.assertEqual(later, rejected)
                self.assertEqual(len(later["attempts"]), 0)

    def test_repeated_submission_and_postterminal_actions_do_not_overwrite_win(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        first = session.submit_table(oracle_table(world))
        self.assertEqual(session.submit_table({})["receipt_sha256"], first["receipt_sha256"])
        self.assertEqual(session.act("bogus")["verdict"], "VERIFIED_WIN")
        self.assertEqual(len(session.view()["history"]), 0)

    def test_independent_replay_rejects_mutated_engine_transition(self):
        session = dut.CandidateSession(dut.make_world(0))
        with mock.patch.object(dut, "transition", side_effect=lambda world, state, action: state):
            result = session.act("A")
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        receipt = session.receipt()
        self.assertFalse(receipt["verified_success"])

    def test_world_hidden_label_mutation_fails_oracle_identity(self):
        world = dut.make_world(0)
        forged = replace(world, labels=tuple(reversed(world.labels)))
        self.assertIsNone(replay_trace(forged, ()))
        with self.assertRaises(ValueError):
            oracle_table(forged)

    def test_str_subclass_equality_spoof_cannot_forge_world_or_win(self):
        class LyingStr(str):
            def __eq__(self, other):
                return True
            __hash__ = str.__hash__

        real = dut.make_world(0)
        fake = replace(real, labels=tuple(LyingStr("not-a-color") for _ in range(4)),
                       world_id=LyingStr("not-a-world-id"))
        self.assertIsNone(replay_trace(fake, ()))
        self.assertEqual(verify_table(fake, oracle_table(real))["verdict"],
                         "EVALUATOR_DISAGREEMENT")
        with self.assertRaises(ValueError):
            dut.CandidateSession(fake)
        with self.assertRaises(ValueError):
            dut.transition(fake, 0, "A")
        with self.assertRaises(ValueError):
            dut.candidate_view(fake)

    def test_mutated_public_observation_cannot_be_replayed(self):
        session = dut.CandidateSession(dut.make_world(0))
        session.act("A")
        corrupted = deepcopy(session.view()["history"])
        corrupted[0]["after"] = "violet" if corrupted[0]["after"] != "violet" else "amber"
        self.assertIsNone(replay_trace(dut.make_world(0), corrupted))

    def test_receipt_digest_change_blocks_replay(self):
        world = dut.make_world(0)
        receipt = dut.CandidateSession(world).submit_table(oracle_table(world))
        tampered = deepcopy(receipt)
        tampered["verdict"] = "VERIFIED_FAILURE"
        self.assertFalse(verify_receipt_replay(tampered))

    def test_resigned_fake_graduation_or_success_cannot_be_replayed(self):
        receipt = dut.CandidateSession(dut.make_world(0)).submit_table({})
        tampered = deepcopy(receipt)
        tampered["verdict"] = "VERIFIED_WIN"
        tampered["verified_success"] = True
        self.assertFalse(verify_receipt_replay(_resign(tampered)))

    def test_resigned_forged_candidate_table_cannot_be_replayed(self):
        world = dut.make_world(0)
        receipt = dut.CandidateSession(world).submit_table(oracle_table(world))
        tampered = deepcopy(receipt)
        tampered["submitted_table"]["A"]["amber"] = "amber"
        self.assertFalse(verify_receipt_replay(_resign(tampered)))

    def test_resigned_forged_observation_cannot_be_replayed(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        session.act("A")
        receipt = session.submit_table(oracle_table(world))
        tampered = deepcopy(receipt)
        tampered["attempts"][0]["after"] = "amber" if tampered["attempts"][0]["after"] != "amber" else "jade"
        self.assertFalse(verify_receipt_replay(_resign(tampered)))

    def test_public_view_nested_action_cannot_rewrite_private_charged_ledger(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        viewed = session.act("B")  # B and RESET both leave initial state unchanged
        self.assertEqual(viewed["history"][0]["action"]["json"], '"B"')
        viewed["history"][0]["action"]["json"] = '"RESET"'
        viewed["history"][0]["action"]["type"] = "forged-type"
        self.assertEqual(session.view()["history"][0]["action"]["json"], '"B"')
        receipt = session.submit_table(oracle_table(world))
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertEqual(receipt["attempts"][0]["action"]["json"], '"B"')
        self.assertTrue(verify_receipt_replay(receipt))

    def test_returned_receipt_nested_mutation_cannot_poison_session_or_later_receipt(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        session.act("B")
        first = session.submit_table(oracle_table(world))
        first["attempts"][0]["action"]["json"] = '"RESET"'
        first["submitted_raw"]["A"]["amber"] = "violet"
        first["submitted_table"]["A"]["amber"] = "violet"
        later = session.receipt()
        self.assertEqual(later["attempts"][0]["action"]["json"], '"B"')
        self.assertEqual(later["submitted_raw"], oracle_table(world))
        self.assertEqual(later["submitted_table"], oracle_table(world))
        self.assertTrue(verify_receipt_replay(later))

    def test_exact_table_verification_is_seed_bound_not_template_answer(self):
        known = oracle_table(dut.make_world(0))
        self.assertEqual(known, MEMORIZED_TRAIN_SEED0)
        self.assertEqual(verify_table(dut.make_world(0), known)["verdict"], "VERIFIED_WIN")
        self.assertEqual(verify_table(dut.make_world(0, heldout=True), known)["verdict"],
                         "VERIFIED_FAILURE")
        self.assertEqual(verify_table(dut.make_world(1), known)["verdict"],
                         "VERIFIED_FAILURE")

    def test_zero_update_memorization_control_fails_structural_transfer(self):
        self.assertEqual(memorized_control(dut.CandidateSession(dut.make_world(0)))["verdict"],
                         "VERIFIED_WIN")
        for heldout, seed in ((True, 0), (True, 1), (False, 1), (False, 2)):
            receipt = memorized_control(dut.CandidateSession(dut.make_world(seed, heldout=heldout)))
            self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
            self.assertTrue(verify_receipt_replay(receipt))

    def test_100_seed_no_update_control_shows_relabeling_collision_not_learning(self):
        counts = {}
        for heldout in (False, True):
            wins = 0
            for seed in range(100):
                receipt = memorized_control(dut.CandidateSession(
                    dut.make_world(seed, heldout=heldout)))
                wins += int(receipt["verdict"] == "VERIFIED_WIN")
            counts[heldout] = wins
        # Six equivalent surface mappings accidentally match the frozen
        # answer; no new structural mechanism was learned in those cases.
        self.assertEqual(counts, {False: 6, True: 0})

    def test_public_only_symbolic_explorer_solves_many_relabels_and_both_mechanics(self):
        for heldout in (False, True):
            for seed in range(25):
                with self.subTest(heldout=heldout, seed=seed):
                    receipt = symbolic_explorer(
                        dut.CandidateSession(dut.make_world(seed, heldout=heldout)))
                    self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
                    self.assertLess(len(receipt["attempts"]), dut.MAX_ATTEMPTS)
                    self.assertTrue(verify_receipt_replay(receipt))

    def test_cli_symbolic_vs_memorized_reports_reproducible_controls(self):
        for arguments, expected in ((["--seed", "0"], "VERIFIED_WIN"),
                                    (["--seed", "0", "--heldout"], "VERIFIED_WIN"),
                                    (["--seed", "0", "--heldout", "--control", "memorized"],
                                     "VERIFIED_FAILURE")):
            first = io.StringIO()
            with redirect_stdout(first):
                self.assertEqual(main(arguments), 0)
            receipt = json.loads(first.getvalue())
            self.assertEqual(receipt["verdict"], expected)
            self.assertTrue(verify_receipt_replay(receipt))
            second = io.StringIO()
            with redirect_stdout(second):
                main(arguments)
            self.assertEqual(first.getvalue(), second.getvalue())

    def test_verifier_rejects_python_equality_spoofed_submission_values(self):
        class FakeColor:
            def __eq__(self, other):
                return other == "amber"
        incorrect = deepcopy(MEMORIZED_TRAIN_SEED0)
        incorrect["A"]["amber"] = FakeColor()
        self.assertEqual(verify_table(dut.make_world(0), incorrect)["verdict"],
                         "VERIFIED_FAILURE")


if __name__ == "__main__":
    unittest.main()
