"""Independent hostile RED controls for Epistemic Shift at 644244f67aeccc3."""

import inspect
import unittest
from unittest.mock import patch

from labyrinth import epistemic_shift_exam as exam


class EpistemicShiftDestroyerTests(unittest.TestCase):
    def test_sealed_copy_law_has_one_semantic_fingerprint(self):
        # Sealing disables every post-change kind; the executable world is
        # identically copy for even/odd seeds and all declared split labels.
        cases = [exam.make_case(seed, split=split, sealed=True)
                 for split in exam.SPLIT_KINDS for seed in (0, 1)]
        self.assertEqual(len({exam.semantic_fingerprint(case) for case in cases}), 1)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_nonbinary_equality_object_cannot_pass_independent_replay(self):
        class LyingOutput:
            def __eq__(self, other):
                return True

            def __repr__(self):
                return "FORGED-OUTPUT"

        case = exam.make_case(0, split="test")
        session = exam.CandidateSession(case)
        original = exam.transition

        def corrupt(config, state, pair):
            next_state, _ = original(config, state, pair)
            return next_state, LyingOutput()

        with patch.object(exam, "transition", side_effect=corrupt):
            result = session.intervene(1, 0)
        self.assertEqual(result["integrity_status"], "EVALUATOR_DISAGREEMENT", result)
        self.assertTrue(result["finished"])

    def test_sealed_no_change_world_cannot_claim_forged_revision(self):
        class Chameleon:
            def __eq__(self, other):
                frame = inspect.currentframe()
                while frame is not None and frame.f_code.co_name != "adjudicate_claim":
                    frame = frame.f_back
                # Equal for replay verification, unequal when the same bogus
                # value is interpreted as a post-change contradiction.
                return not (frame is not None and "valid_count" in frame.f_locals)

        case = exam.make_case(0, split="test", sealed=True)
        session = exam.CandidateSession(case)
        session.hypothesize("copy")
        original = exam.transition
        count = 0

        def corrupt_fourth(config, state, pair):
            nonlocal count
            count += 1
            next_state, output = original(config, state, pair)
            return next_state, Chameleon() if count == 4 else output

        with patch.object(exam, "transition", side_effect=corrupt_fourth):
            for pair in ((0, 0), (1, 0), (0, 1), (1, 1)):
                last = session.intervene(*pair)

        if last["integrity_status"] == "OK":
            session.hypothesize("copy")
            receipt = session.submit({"kind": "REVISION", "revised_kernel": "copy"})
            self.assertFalse(receipt.get("revision_verified", False), receipt)
        else:
            self.assertEqual(last["verdict"], "EVALUATOR_DISAGREEMENT")
            self.assertTrue(last["finished"])

    def test_missing_or_invalid_law_is_not_a_novel_ruleset_challenge(self):
        for kind, key in (("HYPOTHESIS", "kernel"),
                          ("REVISION", "revised_kernel")):
            for invalid in (None, 7, "", "   "):
                with self.subTest(kind=kind, invalid=invalid):
                    session = exam.CandidateSession(exam.make_case(0, split="test"))
                    for pair in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        session.intervene(*pair)
                    receipt = session.submit({"kind": kind, key: invalid})
                    self.assertEqual(receipt["verdict"], "UNKNOWN", receipt)
                    self.assertNotIn("proposal", receipt)


if __name__ == "__main__":
    unittest.main()
