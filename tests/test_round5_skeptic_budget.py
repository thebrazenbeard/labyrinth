"""Adversarial attempt-budget witness tests, separate from implementation tests."""

import unittest

from labyrinth.epistemic_shift_exam import CandidateSession, MAX_ATTEMPTS, make_case
from labyrinth.epistemic_shift_verifier import adjudicate_claim, independent_replay


class AttemptBudgetTrust(unittest.TestCase):
    def test_subclass_length_lies_cannot_exceed_examination_budget(self):
        class LyingList(list):
            def __len__(self):
                return 4

        case = make_case(0, split="test")
        attempts = LyingList([(0, 0)] * (MAX_ATTEMPTS + 12))
        self.assertEqual(len(attempts), 4)
        self.assertEqual(list.__len__(attempts), 22)
        self.assertIsNone(independent_replay(case, attempts))
        result = adjudicate_claim(case,
                                  {"kind": "HYPOTHESIS", "kernel": "edge"},
                                  attempts, [0] * 22, [])
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(result["verified_success"])

    def test_tuple_subclass_iteration_lies_cannot_alter_replay(self):
        class LyingTuple(tuple):
            def __len__(self):
                return 1

            def __iter__(self):
                return iter(((0, 0),) * 30)

        case = make_case(1, split="test")
        attempts = LyingTuple(((0, 0),))
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(independent_replay(case, attempts))

    def test_exact_builtins_preserve_ten_attempt_boundary(self):
        case = make_case(0, split="test")
        for constructor in (list, tuple):
            for amount in (0, 4, MAX_ATTEMPTS):
                replay = independent_replay(case, constructor([(0, 0)] * amount))
                self.assertIsNotNone(replay)
                self.assertEqual(replay[0].interventions, amount)
            self.assertIsNone(independent_replay(
                case, constructor([(0, 0)] * (MAX_ATTEMPTS + 1))))


class ContradictionIsNotIdentification(unittest.TestCase):
    def test_indistinguishable_edge_lag_worlds_cannot_certify_lucky_revision(self):
        intervention_prefix = ((0, 0), (0, 0), (0, 0), (1, 0))
        receipts = []
        observation_traces = []
        for seed in (0, 1):
            session = CandidateSession(make_case(seed, split="test"))
            session.hypothesize("copy")
            for pair in intervention_prefix:
                view = session.intervene(*pair)
            observation_traces.append(view["observations"])
            session.hypothesize("edge")
            receipts.append(session.submit({"kind": "REVISION",
                                            "revised_kernel": "edge"}))
        self.assertEqual(observation_traces[0], observation_traces[1])
        self.assertEqual(receipts[0]["verdict"], "UNKNOWN")
        self.assertFalse(receipts[0]["revision_verified"])
        self.assertFalse(receipts[0]["observationally_identified"])
        self.assertEqual(receipts[1]["verdict"], "VERIFIED_FAILURE")

    def test_oracle_hypothesis_accuracy_is_kept_separate_from_identification(self):
        session = CandidateSession(make_case(0, split="test"))
        for pair in ((0, 0), (0, 0), (0, 0), (1, 0)):
            session.intervene(*pair)
        receipt = session.submit({"kind": "HYPOTHESIS", "kernel": "edge"})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(receipt["hypothesis_verified"])
        self.assertFalse(receipt["verified_success"])

    def test_exhaustively_discriminative_probes_retain_revision_credit(self):
        from labyrinth.epistemic_shift_baseline import run_public_symbolic_revision

        for split in ("train", "dev", "test"):
            for seed in (0, 1):
                session = CandidateSession(make_case(seed, split=split))
                receipt = run_public_symbolic_revision(
                    session.intervene, session.hypothesize, session.submit)
                self.assertEqual(receipt["verdict"], "VERIFIED_WIN", (split, seed))
                self.assertTrue(receipt["revision_verified"])
                self.assertTrue(receipt["observationally_identified"])


if __name__ == "__main__":
    unittest.main()
