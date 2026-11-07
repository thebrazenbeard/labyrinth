"""Independent hostile falsification of out-of-catalog Minotaur discoveries."""

from contextlib import redirect_stdout
from copy import deepcopy
from itertools import product
import io
import json
import unittest
from unittest import mock

from labyrinth import novel_discovery_world as dut
from labyrinth.epistemic_shift_exam import ALL_KINDS
from labyrinth.novel_discovery_cli import DEMO_PROPOSAL, main
from labyrinth.novel_discovery_reviewer import (
    FIELDS, KNOWN_CATALOG, decode_attempts, encode_attempts,
    exhaustive_rule_review, normalize_proposal, replay_world, review_proposal,
    verify_receipt_digest, verify_receipt_replay)


def played(world):
    session = dut.CandidateSession(world)
    for pair in dut.PRE_PROBES + dut.POST_PROBES:
        session.intervene(*pair)
    return session


class NovelDiscoveryHostileTests(unittest.TestCase):
    def test_unified_catalog_is_exactly_the_seven_old_epistemic_kernels(self):
        self.assertEqual(set(KNOWN_CATALOG), set(ALL_KINDS))
        self.assertNotIn("rebound", KNOWN_CATALOG)

    def test_factory_strict_types_and_seed_bounds(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"seed": 2**31},
                       {"seed": "0"}, {"seed": 0, "variant": 10000},
                       {"seed": 0, "variant": True}, {"seed": 0, "control": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dut.make_world(**kwargs)

    def test_public_view_omits_hidden_identity_and_mode(self):
        for seed in range(20):
            world = dut.make_world(seed, control=bool(seed % 2))
            candidate = dut.candidate_view(world)
            self.assertEqual(set(candidate), {"prompt", "inputs", "remaining_attempts",
                                              "observations"})
            self.assertNotIn(world.world_id, str(candidate))
            self.assertNotIn("control", str(candidate))

    def test_paraphrases_share_identical_semantic_world_identity(self):
        case = dut.make_world(0)
        variants = {dut.candidate_view(dut.make_world(0, variant=v))["prompt"]
                    for v in range(90)}
        self.assertGreater(len(variants), 1)
        self.assertEqual(case.world_id, dut.make_world(0, variant=89).world_id)

    def test_hand_calculated_two_step_rebound_trace_is_not_edge_control(self):
        actual = played(dut.make_world(0))
        control = played(dut.make_world(0, control=True))
        outputs = lambda s: tuple(item["effect"] for item in s.view()["observations"])
        self.assertEqual(outputs(actual), (1, 0, 1, 0, 0, 0, 0, 1, 0, 0))
        self.assertEqual(outputs(control), (1, 0, 1, 0, 0, 1, 0, 1, 0, 1))

    def test_original_engine_and_second_codepath_agree_on_all_64_short_traces(self):
        for control in (False, True):
            world = dut.make_world(0, control=control)
            for suffix in product(dut.INPUTS, repeat=3):
                session = dut.CandidateSession(world)
                for pair in dut.PRE_PROBES + suffix:
                    view = session.intervene(*pair)
                self.assertEqual(view["verdict"], "UNKNOWN")
                self.assertEqual(replay_world(world, session.attempts())[1],
                                 tuple(item["effect"] for item in view["observations"]))

    def test_manual_32_context_truth_table_has_only_two_rebound_ones(self):
        normalized, status = normalize_proposal(DEMO_PROPOSAL)
        self.assertEqual(status, "EXECUTABLE")
        cases = [dict(zip(FIELDS, bits)) for bits in product((0, 1), repeat=5)]
        true_contexts = [c for c in cases if all(c[lit["field"]] == lit["equals"]
                    for lit in normalized["literals"])]
        self.assertEqual(len(true_contexts), 2)  # prev_y is irrelevant
        self.assertTrue(all((c["prev2_x"], c["prev_x"], c["x"], c["y"]) ==
                            (1, 0, 1, 1) for c in true_contexts))

    def test_novel_proposal_survives_all_32_counterfactuals_without_auto_certification(self):
        receipt = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL)
        self.assertEqual(receipt["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(receipt["review"]["fit_status"], "COUNTERFACTUAL_MATCH")
        self.assertTrue(receipt["review"]["counterfactual_match"])
        self.assertEqual(receipt["review"]["contexts_checked"], 32)
        self.assertEqual(receipt["review"]["catalog_equivalent"], [])
        self.assertFalse(receipt["verified_success"])
        self.assertFalse(receipt["independent_custody"])
        self.assertEqual(receipt["review_status"], "PENDING_INDEPENDENT_REVIEW")
        self.assertEqual(receipt["subject"]["source_commit"], "UNBOUND")
        self.assertTrue(verify_receipt_replay(receipt))

    def test_same_novel_claim_is_falsified_by_one_rule_control_twin(self):
        receipt = played(dut.make_world(0, control=True)).submit_proposal(DEMO_PROPOSAL)
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["review"]["fit_status"], "FALSIFIED")
        self.assertEqual(receipt["review"]["novelty_status"], "REFUTED_BY_COUNTEREXAMPLE")
        self.assertFalse(receipt["review"]["counterfactual_match"])
        self.assertIsNotNone(receipt["review"]["counterexample"])
        self.assertTrue(verify_receipt_replay(receipt))

    def test_exhaustive_242_zero_probe_hypotheses_refuted_before_novelty(self):
        """A guessed outside-catalog formula cannot earn discovery by failing."""
        world = dut.make_world(0)
        counts = {"NOVEL_RULESET_CHALLENGE": 0, "VERIFIED_FAILURE": 0}
        for pattern in product((None, 0, 1), repeat=len(FIELDS)):
            if all(value is None for value in pattern):
                continue
            proposal = {"form": "and_literals", "literals": [
                {"field": field, "equals": value}
                for field, value in zip(FIELDS, pattern) if value is not None]}
            receipt = review_proposal(world, (), (), proposal)
            counts[receipt["verdict"]] += 1
            self.assertTrue(verify_receipt_replay(receipt))
            self.assertEqual(receipt["observed_attempt_count"], 0)
            self.assertFalse(receipt["verified_success"])
            self.assertEqual(receipt["review_status"], "PENDING_INDEPENDENT_REVIEW")
            if receipt["verdict"] == "VERIFIED_FAILURE":
                self.assertEqual(receipt["review"]["fit_status"], "FALSIFIED")
                self.assertIsNotNone(receipt["review"]["counterexample"])
            else:
                self.assertEqual(receipt["review"]["fit_status"], "COUNTERFACTUAL_MATCH")
                self.assertEqual(receipt["review"]["contexts_checked"], 32)
        self.assertEqual(counts, {"NOVEL_RULESET_CHALLENGE": 1, "VERIFIED_FAILURE": 241})

    def test_canonical_receipt_is_deterministic_and_unchanged_by_literal_order(self):
        first = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL)
        reversed_literals = {"form": "and_literals",
                             "literals": list(reversed(DEMO_PROPOSAL["literals"]))}
        second = played(dut.make_world(0)).submit_proposal(reversed_literals)
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        self.assertTrue(verify_receipt_digest(first))

    def test_receipt_digest_detects_post_hoc_mutation_but_is_not_signature(self):
        receipt = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL)
        changed = deepcopy(receipt)
        changed["review"]["counterfactual_match"] = False
        self.assertFalse(verify_receipt_digest(changed))
        self.assertFalse(verify_receipt_digest({"receipt_sha256": True}))

    def test_serialized_receipt_roundtrip_preserves_candidate_and_evidence(self):
        receipt = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL,
                                                             author="Model-X")
        replayed = json.loads(json.dumps(receipt))
        self.assertEqual(replayed["candidate"]["author"], "Model-X")
        self.assertTrue(verify_receipt_digest(replayed))
        self.assertTrue(verify_receipt_replay(replayed))

    def test_charged_invalid_input_values_remain_distinguishable_in_receipts(self):
        world = dut.make_world(0)
        first = dut.CandidateSession(world)
        first.intervene("north", None)
        a = first.submit_proposal(DEMO_PROPOSAL)
        second = dut.CandidateSession(world)
        second.intervene("teleport", None)
        b = second.submit_proposal(DEMO_PROPOSAL)
        self.assertNotEqual(a["observed_trace_sha256"], b["observed_trace_sha256"])
        self.assertNotEqual(a["receipt_sha256"], b["receipt_sha256"])
        self.assertTrue(verify_receipt_replay(a))
        self.assertTrue(verify_receipt_replay(b))

    def test_typed_canonical_trace_distinguishes_true_from_integer_one(self):
        self.assertNotEqual(encode_attempts(((True, 1),)),
                            encode_attempts(((1, 1),)))
        self.assertEqual(decode_attempts(encode_attempts(((True, 1),))),
                         ((True, 1),))

    def test_unserializable_attempt_blocks_receipt_instead_of_losing_provenance(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        session.intervene({"arbitrary": "object"}, None)
        result = session.submit_proposal(DEMO_PROPOSAL)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["review_status"], "BLOCKED_UNSERIALIZABLE_TRACE")
        self.assertFalse(result["verified_success"])

    def test_unbounded_integer_attempt_is_quarantined_from_receipt(self):
        session = dut.CandidateSession(dut.make_world(0))
        session.intervene(2**2048, 0)
        receipt = session.submit_proposal(DEMO_PROPOSAL)
        self.assertEqual(receipt["review_status"], "BLOCKED_UNSERIALIZABLE_TRACE")
        self.assertFalse(receipt["verified_success"])

    def test_recomputed_digest_cannot_hide_semantically_tampered_trace(self):
        receipt = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL)
        tampered = deepcopy(receipt)
        tampered["trace"]["effects"][-1] = 1
        from labyrinth.novel_discovery_reviewer import _canonical
        from hashlib import sha256
        tampered["observed_trace_sha256"] = sha256(
            _canonical(tampered["trace"]).encode()).hexdigest()
        unsigned = {key: val for key, val in tampered.items() if key != "receipt_sha256"}
        tampered["receipt_sha256"] = sha256(_canonical(unsigned).encode()).hexdigest()
        self.assertTrue(verify_receipt_digest(tampered))
        self.assertFalse(verify_receipt_replay(tampered))

    def test_recomputed_digest_cannot_forge_graduation_verdict(self):
        receipt = played(dut.make_world(0)).submit_proposal(DEMO_PROPOSAL)
        tampered = deepcopy(receipt)
        tampered["verdict"] = "VERIFIED_WIN"
        tampered["verified_success"] = True
        from labyrinth.novel_discovery_reviewer import _canonical
        from hashlib import sha256
        unsigned = {key: val for key, val in tampered.items() if key != "receipt_sha256"}
        tampered["receipt_sha256"] = sha256(_canonical(unsigned).encode()).hexdigest()
        self.assertTrue(verify_receipt_digest(tampered))
        self.assertFalse(verify_receipt_replay(tampered))

    def test_narrative_receipt_cannot_be_relabelled_executable(self):
        receipt = played(dut.make_world(0)).submit_proposal(
            "A second-order hidden causal rule may exist.")
        self.assertTrue(verify_receipt_replay(receipt))
        tampered = deepcopy(receipt)
        tampered["candidate"]["proposal_status"] = "EXECUTABLE"
        tampered["review"]["novelty_status"] = "OUTSIDE_CATALOG"
        from labyrinth.novel_discovery_reviewer import _canonical
        from hashlib import sha256
        unsigned = {key: val for key, val in tampered.items() if key != "receipt_sha256"}
        tampered["receipt_sha256"] = sha256(_canonical(unsigned).encode()).hexdigest()
        self.assertTrue(verify_receipt_digest(tampered))
        self.assertFalse(verify_receipt_replay(tampered))

    def test_typed_dsl_rejects_duplicate_or_contradictory_literals(self):
        broken = {"form": "and_literals", "literals": [
            {"field": "x", "equals": 0}, {"field": "x", "equals": 1}]}
        normalized, status = normalize_proposal(broken)
        self.assertIsNone(normalized)
        self.assertEqual(status, "INVALID_LITERAL")

    def test_dsl_rejects_undeclared_fields_and_nonbinary_values(self):
        for literal in ({"field": "third_previous_x", "equals": 1},
                        {"field": "x", "equals": True},
                        {"field": "x", "equals": 2},
                        {"field": "x", "equals": "1"}):
            proposal = {"form": "and_literals", "literals": [literal]}
            normalized, status = normalize_proposal(proposal)
            self.assertIsNone(normalized)
            self.assertEqual(status, "INVALID_LITERAL")

    def test_nonstring_equality_spoof_cannot_be_dsl_field(self):
        class FakeField:
            def __eq__(self, other):
                return other == "x"

        proposal = {"form": "and_literals", "literals": [
            {"field": FakeField(), "equals": 1}]}
        normalized, status = normalize_proposal(proposal)
        self.assertIsNone(normalized)
        self.assertEqual(status, "INVALID_LITERAL")

    def test_narrative_proposal_is_preserved_pending_without_fake_falsification(self):
        text = "Perhaps causal state needs two previous interventions, not one."
        receipt = played(dut.make_world(0)).submit_proposal(text)
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["candidate"]["proposal"]["description"], text)
        self.assertEqual(receipt["review"]["fit_status"], "UNKNOWN")
        self.assertEqual(receipt["review"]["novelty_status"], "UNTYPED_PENDING")
        self.assertEqual(receipt["review"]["contexts_checked"], 0)
        self.assertFalse(receipt["verified_success"])
        self.assertTrue(verify_receipt_replay(receipt))

    def test_invalid_schema_is_unknown_not_automatically_novel(self):
        receipt = played(dut.make_world(0)).submit_proposal({"execute": "python"})
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["candidate"]["proposal_status"], "INVALID_SCHEMA")
        self.assertEqual(receipt["review_status"], "PENDING_INDEPENDENT_REVIEW")

    def test_known_catalog_copy_rule_retains_falsifier_but_not_novel_label(self):
        trivial = {"form": "and_literals", "literals": [{"field": "x", "equals": 1}]}
        receipt = played(dut.make_world(0)).submit_proposal(trivial)
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["review"]["novelty_status"], "REFUTED_BY_COUNTEREXAMPLE")
        self.assertIn("copy", receipt["review"]["catalog_equivalent"])
        self.assertEqual(receipt["review"]["fit_status"], "FALSIFIED")
        self.assertIsNotNone(receipt["review"]["counterexample"])

    def test_attack_all_seven_catalog_functions_cannot_equal_rebound_truth_table(self):
        normalized, _ = normalize_proposal(DEMO_PROPOSAL)
        reviewed = exhaustive_rule_review(dut.make_world(0), normalized)
        self.assertEqual(reviewed["catalog_equivalent"], [])
        self.assertTrue(reviewed["counterfactual_match"])

    def test_invalid_interventions_consume_budget_without_affecting_hidden_history(self):
        session = dut.CandidateSession(dut.make_world(0))
        for index in range(dut.MAX_ATTEMPTS):
            result = session.intervene("teleport" if index % 2 else True, None)
            self.assertIsNone(result["observations"][-1]["effect"])
            self.assertEqual(result["remaining_attempts"], 9 - index)
        self.assertEqual(replay_world(dut.make_world(0), session.attempts())[0], dut.State())
        with self.assertRaises(ValueError):
            session.intervene(1, 1)

    def test_nonstring_action_equality_spoof_is_charged_invalid(self):
        class FakeBit:
            def __eq__(self, other):
                return other == 1

        session = dut.CandidateSession(dut.make_world(0))
        result = session.intervene(FakeBit(), 1)
        self.assertEqual(result["observations"][0]["input"], "INVALID")
        self.assertIsNone(result["observations"][0]["effect"])

    def test_submitted_outcome_is_terminal_and_cannot_be_overwritten(self):
        session = played(dut.make_world(0))
        session.submit_proposal(DEMO_PROPOSAL)
        self.assertTrue(session.view()["finished"])
        with self.assertRaises(ValueError):
            session.submit_proposal("a different answer")
        with self.assertRaises(ValueError):
            session.intervene(1, 1)

    def test_corrupted_engine_transition_is_detected_before_receipt(self):
        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        original = dut.transition

        def mutant(config, state, pair):
            next_state, output = original(config, state, pair)
            return next_state, 1 - output if type(output) is int else output

        with mock.patch.object(dut, "transition", side_effect=mutant):
            result = session.intervene(1, 0)
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        self.assertEqual(len(session.attempts()), 1)

    def test_state_equality_spoof_cannot_hide_incorrect_transition(self):
        class FakeCounter:
            def __eq__(self, other):
                return True

        world = dut.make_world(0)
        session = dut.CandidateSession(world)
        original = dut.transition

        def forged(config, state, pair):
            next_state, output = original(config, state, pair)
            return dut.State(FakeCounter(), next_state.prev2_x,
                             next_state.prev_x, next_state.prev_y), output

        with mock.patch.object(dut, "transition", side_effect=forged):
            result = session.intervene(1, 0)
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])

    def test_independent_replay_exception_is_terminal(self):
        session = dut.CandidateSession(dut.make_world(0))
        with mock.patch("labyrinth.novel_discovery_reviewer.replay_world",
                        side_effect=RuntimeError("broken verifier")):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])

    def test_review_cannot_accept_tampered_observation_transcript(self):
        world = dut.make_world(0)
        receipt = review_proposal(world, dut.PRE_PROBES, (9, 9, 9), DEMO_PROPOSAL)
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(receipt["review_status"], "BLOCKED_INTEGRITY")

    def test_cli_prints_json_receipt_with_deterministic_digest(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--seed", "0", "--author", "fixture"])
        self.assertEqual(code, 0)
        receipt = json.loads(out.getvalue())
        self.assertEqual(receipt["review"]["fit_status"], "COUNTERFACTUAL_MATCH")
        self.assertEqual(receipt["candidate"]["author"], "fixture")
        self.assertTrue(verify_receipt_digest(receipt))

    def test_cli_control_twin_emits_falsifier_not_an_acceptance(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--seed", "0", "--control"])
        self.assertEqual(code, 0)
        receipt = json.loads(out.getvalue())
        self.assertEqual(receipt["review"]["fit_status"], "FALSIFIED")
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["review_status"], "PENDING_INDEPENDENT_REVIEW")


if __name__ == "__main__":
    unittest.main()
