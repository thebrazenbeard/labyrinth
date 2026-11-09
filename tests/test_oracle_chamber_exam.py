"""Hostile, independent-oracle checks for Minotaur's noisy evidence chamber."""

from itertools import combinations, product
import unittest
from unittest import mock

from labyrinth import oracle_chamber_exam as dut
from labyrinth.oracle_chamber_baseline import (SCHEDULE, decode_single_lie,
                                               run_public_parser)
from labyrinth.oracle_chamber_verifier import (
    adjudicate_claim, hypothetical_attested_pair_query, replay_answers,
    verify_policy, verify_unsat_bound)


def all_patterns(max_lies):
    return [frozenset(p) for count in range(max_lies + 1)
            for p in combinations(range(dut.MAX_QUERIES), count)]


def exact_case(secret, lie_slots, *, lie_budget=1):
    return dut.Case(seed=0, variant=0, lie_budget=lie_budget,
                    secret=secret, lie_slots=frozenset(lie_slots),
                    case_id="manually-enumerated-adversary")


class OracleChamberHostileTests(unittest.TestCase):
    def test_seed_factory_is_deterministic(self):
        self.assertEqual(dut.make_case(37), dut.make_case(37))

    def test_seed_type_rejects_bool_string_negative_and_overflow(self):
        for seed in (True, "0", -1, 2**31, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                dut.make_case(seed)

    def test_variant_and_sealed_require_exact_types(self):
        for kwargs in ({"variant": True}, {"variant": -1},
                       {"variant": 10000}, {"sealed": 1}, {"sealed": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dut.make_case(0, **kwargs)

    def test_public_view_never_exposes_case_identity_or_hidden_sample(self):
        for seed in range(50):
            case = dut.make_case(seed, sealed=bool(seed % 2))
            view = dut.candidate_view(case)
            self.assertEqual(set(view), {"prompt", "questions", "remaining_queries",
                                         "submissions", "answers"})
            self.assertNotIn("case_id", str(view))
            self.assertNotIn("lie_slots", str(view))
            self.assertNotIn(case.case_id, str(view))

    def test_paraphrase_varies_wording_without_semantic_rule_change(self):
        for seed in range(10):
            variants = {dut.candidate_view(dut.make_case(seed, variant=v))["prompt"]
                        for v in range(60)}
            self.assertGreater(len(variants), 3)
            self.assertEqual(dut.world_fingerprint(dut.make_case(seed)),
                             dut.world_fingerprint(dut.make_case(seed, variant=59)))

    def test_open_and_sealed_are_true_one_rule_near_twins(self):
        for seed in range(50):
            open_case, sealed = dut.make_case(seed), dut.make_case(seed, sealed=True)
            self.assertEqual(open_case.secret, sealed.secret)
            self.assertEqual(open_case.lie_budget, 1)
            self.assertEqual(sealed.lie_budget, 2)
            self.assertNotEqual(dut.world_fingerprint(open_case),
                                dut.world_fingerprint(sealed))

    def test_engine_and_independent_answers_agree_for_every_open_adversary(self):
        for secret in dut.SECRETS:
            for lies in all_patterns(1):
                with self.subTest(secret=secret, lies=lies):
                    case = exact_case(secret, lies)
                    session = dut.CandidateSession(case)
                    for question in SCHEDULE:
                        observation = session.ask(question)
                        self.assertEqual(observation["integrity_status"], "OK")
                    self.assertEqual(replay_answers(case, session.attempts()),
                                     tuple(x["answer"] for x in observation["answers"]))

    def test_engine_and_independent_answers_agree_for_every_sealed_adversary(self):
        for secret in dut.SECRETS:
            for lies in all_patterns(2):
                case = exact_case(secret, lies, lie_budget=2)
                session = dut.CandidateSession(case)
                for question in SCHEDULE:
                    observation = session.ask(question)
                self.assertEqual(observation["integrity_status"], "OK")
                self.assertEqual(replay_answers(case, session.attempts()),
                                 tuple(x["answer"] for x in observation["answers"]))

    def test_third_oracle_hardcoded_bit_truth_table_catches_shared_assumptions(self):
        expected = {0: (0, 0, 0), 1: (1, 0, 1),
                    2: (0, 1, 1), 3: (1, 1, 0)}
        for secret, triplet in expected.items():
            case = exact_case(secret, ())
            session = dut.CandidateSession(case)
            for question in ("alpha", "beta", "parity"):
                result = session.ask(question)
            self.assertEqual(tuple(x["answer"] for x in result["answers"]), triplet)

    def test_five_query_error_correcting_policy_guarantees_all_24_open_cases(self):
        review = verify_policy(1, {"questions": list(SCHEDULE),
                                   "decoder": "nearest_codeword"})
        self.assertEqual(review.verdict, "VERIFIED_WIN")
        self.assertEqual(review.cases_checked, 24)

    def test_same_policy_fails_a_sealed_counterexample(self):
        review = verify_policy(2, {"questions": list(SCHEDULE),
                                   "decoder": "nearest_codeword"})
        self.assertEqual(review.verdict, "VERIFIED_FAILURE")
        self.assertIsNotNone(review.counterexample)
        self.assertLessEqual(review.cases_checked, 64)

    def test_all_243_fixed_query_schedules_fail_the_two_lie_twin(self):
        for schedule in product(dut.PREDICATES, repeat=dut.MAX_QUERIES):
            review = verify_policy(2, {"questions": list(schedule),
                                       "decoder": "nearest_codeword"})
            self.assertNotEqual(review.verdict, "VERIFIED_WIN", schedule)

    def test_closed_world_counting_bound_is_proof_not_no_known_win(self):
        sealed = dut.make_case(0, sealed=True)
        claim = {"declared_rules": {"secrets": 4, "binary_answers": 2,
                                    "max_queries": 5, "lie_budget": 2}}
        reviewed = verify_unsat_bound(sealed, claim)
        self.assertEqual(reviewed["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(reviewed["certificate_verified"])
        self.assertEqual((reviewed["required_paths"], reviewed["available_paths"]),
                         (64, 32))

    def test_open_counting_bound_does_not_prove_impossibility(self):
        opened = dut.make_case(0)
        claim = {"declared_rules": {"secrets": 4, "binary_answers": 2,
                                    "max_queries": 5, "lie_budget": 1}}
        reviewed = verify_unsat_bound(opened, claim)
        self.assertEqual(reviewed["verdict"], "UNKNOWN")
        self.assertFalse(reviewed["certificate_verified"])
        self.assertEqual((reviewed["required_paths"], reviewed["available_paths"]),
                         (24, 32))

    def test_forged_unsat_assumptions_never_certify_an_open_world(self):
        opened = dut.make_case(0)
        forged = {"kind": "UNSAT_CERTIFICATE", "declared_rules": {
            "secrets": 4, "binary_answers": 2, "max_queries": 5, "lie_budget": 2}}
        reviewed = dut.CandidateSession(opened).submit(forged)
        self.assertEqual(reviewed["verdict"], "UNKNOWN")
        self.assertFalse(reviewed["certificate_verified"])

    def test_public_only_parser_wins_120_open_policy_instances_without_learning(self):
        for seed in range(120):
            case = dut.make_case(seed)
            session = dut.CandidateSession(case)
            reviewed = run_public_parser(dut.candidate_view(case), session.ask, session.submit)
            self.assertEqual(reviewed["verdict"], "VERIFIED_WIN", seed)
            self.assertTrue(reviewed["policy_verified"])
            self.assertFalse(reviewed["verified_success"])

    def test_parser_distinguishes_instance_guess_from_universal_success(self):
        for seed in range(120):
            case = dut.make_case(seed)
            session = dut.CandidateSession(case)
            result = run_public_parser(dut.candidate_view(case), session.ask,
                                       session.submit, instance_only=True)
            self.assertEqual(result["verdict"], "VERIFIED_WIN", seed)
            self.assertTrue(result["instance_win_verified"])
            self.assertFalse(result["policy_verified"])
            self.assertFalse(result["verified_success"])

    def test_parser_certifies_120_sealed_worlds_without_victory_credit(self):
        for seed in range(120):
            case = dut.make_case(seed, sealed=True)
            session = dut.CandidateSession(case)
            reviewed = run_public_parser(dut.candidate_view(case), session.ask, session.submit)
            self.assertEqual(reviewed["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES", seed)
            self.assertTrue(reviewed["certificate_verified"])
            self.assertFalse(reviewed["verified_success"])
            self.assertEqual(session.attempts(), ())

    def test_all_single_corruptions_decode_without_secret_access(self):
        for secret in dut.SECRETS:
            truth = tuple((secret & 1, secret & 1, (secret >> 1) & 1,
                           (secret >> 1) & 1, (secret & 1) ^ ((secret >> 1) & 1)))
            for lies in all_patterns(1):
                corrupted = tuple(bit ^ int(index in lies)
                                  for index, bit in enumerate(truth))
                self.assertEqual(decode_single_lie(corrupted), secret)

    def test_invalid_query_costs_budget_and_leaks_no_secret_answer(self):
        case = dut.make_case(2)
        session = dut.CandidateSession(case)
        for index in range(dut.MAX_QUERIES):
            after = session.ask({"bad": index} if index % 2 else "teleport")
            self.assertIsNone(after["answers"][-1]["answer"])
            self.assertEqual(after["remaining_queries"], 4 - index)
        with self.assertRaises(ValueError):
            session.ask("alpha")
        self.assertEqual(len(session.attempts()), 5)
        self.assertEqual(after["integrity_status"], "OK")

    def test_nonstring_equality_spoof_is_invalid_query(self):
        class FakeQuery:
            def __eq__(self, other):
                return other == "alpha"

        session = dut.CandidateSession(dut.make_case(1))
        after = session.ask(FakeQuery())
        self.assertIsNone(after["answers"][-1]["answer"])
        self.assertEqual(after["integrity_status"], "OK")

    def test_submission_terminal_and_cannot_be_overwritten(self):
        case = dut.make_case(0)
        session = dut.CandidateSession(case)
        result = session.submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertTrue(session.view()["finished"])
        with self.assertRaises(ValueError):
            session.submit({"kind": "POLICY"})
        with self.assertRaises(ValueError):
            session.ask("alpha")

    def test_corrupted_environment_answer_is_detected_immediately(self):
        case = exact_case(1, ())
        session = dut.CandidateSession(case)
        with mock.patch.object(dut, "_true_answer", return_value=0):
            observation = session.ask("alpha")
        self.assertEqual(observation["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(observation["finished"])
        self.assertEqual(session.attempts(), ("alpha",))
        with self.assertRaises(ValueError):
            session.ask("beta")

    def test_independent_verifier_exception_is_terminal(self):
        session = dut.CandidateSession(dut.make_case(0))
        with mock.patch("labyrinth.oracle_chamber_verifier.replay_answers",
                        side_effect=RuntimeError("oracle down")):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])

    def test_mutated_independent_truth_table_is_detected_by_environment(self):
        case = exact_case(1, ())
        session = dut.CandidateSession(case)
        with mock.patch.dict("labyrinth.oracle_chamber_verifier._TABLE",
                             {"alpha": (0, 0, 0, 0)}):
            result = session.ask("alpha")
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["remaining_queries"], dut.MAX_QUERIES - 1)
        self.assertEqual(session.attempts(), ("alpha",))

    def test_single_correct_guess_does_not_pass_graduation_gate(self):
        case = dut.make_case(0)
        result = dut.CandidateSession(case).submit({"kind": "COMMIT", "secret": case.secret})
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertTrue(result["instance_win_verified"])
        self.assertFalse(result["verified_success"])

    def test_bool_or_int_subclass_cannot_spoof_exact_secret(self):
        case = exact_case(1, ())
        self.assertEqual(dut.CandidateSession(case).submit(
            {"kind": "COMMIT", "secret": True})["verdict"], "VERIFIED_FAILURE")
        class FakeInt(int):
            pass
        self.assertEqual(dut.CandidateSession(case).submit(
            {"kind": "COMMIT", "secret": FakeInt(1)})["verdict"], "VERIFIED_FAILURE")

    def test_incomplete_adaptive_query_tree_returns_unknown(self):
        review = verify_policy(1, {"query_tree": {"": "alpha"}, "decoder": {}})
        self.assertEqual(review.verdict, "UNKNOWN")

    def test_explicit_adaptive_tree_and_leaf_decoder_are_verifiable(self):
        tree = {"".join(prefix): SCHEDULE[len(prefix)]
                for depth in range(5) for prefix in product("01", repeat=depth)}
        decoder = {"".join(bits): decode_single_lie(tuple(map(int, bits)))
                   for bits in product("01", repeat=5)}
        decoder = {bits: value for bits, value in decoder.items() if value is not None}
        review = verify_policy(1, {"query_tree": tree, "decoder": decoder})
        self.assertEqual(review.verdict, "VERIFIED_WIN")
        self.assertEqual(review.cases_checked, 24)

    def test_modified_decoder_leaf_yields_replayable_counterexample(self):
        tree = {"".join(prefix): SCHEDULE[len(prefix)]
                for depth in range(5) for prefix in product("01", repeat=depth)}
        decoder = {"".join(bits): decode_single_lie(tuple(map(int, bits)))
                   for bits in product("01", repeat=5)}
        decoder["00000"] = 3
        review = verify_policy(1, {"query_tree": tree, "decoder": decoder})
        self.assertEqual(review.verdict, "VERIFIED_FAILURE")
        self.assertEqual(review.counterexample["answers"], "00000")

    def test_novel_rule_proposal_is_preserved_not_accepted(self):
        case = dut.make_case(0, sealed=True)
        proposal = "Permit one authenticated four-valued response that cannot be corrupted."
        result = dut.CandidateSession(case).submit({"kind": "NOVEL_RULE",
                                                    "rule_change": proposal})
        self.assertEqual(result["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertFalse(result["verified_success"])
        self.assertEqual(result["proposal"], proposal)

    def test_hypothetical_new_sensor_does_not_retroactively_refute_old_proof(self):
        case = dut.make_case(0, sealed=True)
        proposed = hypothetical_attested_pair_query(case)
        self.assertEqual(proposed["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertTrue(proposed["conditional_full_identification"])
        self.assertEqual(proposed["hypothetical_answer"], case.secret)
        self.assertEqual(verify_unsat_bound(case, {"declared_rules": {
            "secrets": 4, "binary_answers": 2, "max_queries": 5,
            "lie_budget": 2}})["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_abstention_labels_do_not_grant_score(self):
        case = dut.make_case(0, sealed=True)
        for assessment in ("UNKNOWN", "NO_KNOWN_WIN"):
            result = dut.CandidateSession(case).submit({"kind": "ABSTAIN",
                                                        "assessment": assessment})
            self.assertEqual(result["verdict"], assessment)
            self.assertFalse(result["verified_success"])

    def test_malformed_policy_is_not_presumed_failure_or_success(self):
        for bad in (None, "nearest", {}, {"questions": ["alpha"]},
                    {"questions": ["alpha"] * 5, "decoder": "wrong"}):
            review = verify_policy(1, bad)
            self.assertIn(review.verdict, ("UNKNOWN", "VERIFIED_FAILURE"))
            self.assertNotEqual(review.verdict, "VERIFIED_WIN")


if __name__ == "__main__":
    unittest.main()
