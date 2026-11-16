"""Independent hostile regression against epistemic-shift semantic identity.

An unobservable suppressed post-law cannot distinguish two sealed worlds.
The case ID may differ for provenance, but their semantic fingerprints must
agree when every possible intervention history produces the same outputs.
"""

import json
import unittest

from labyrinth.epistemic_shift_exam import (
    CandidateSession,
    INPUTS,
    make_case,
    semantic_fingerprint,
)
from labyrinth.epistemic_shift_verifier import independent_replay, adjudicate_claim


class SealedSemanticsFingerprint(unittest.TestCase):
    def test_all_suppressed_mechanisms_have_one_observable_semantics(self):
        cases = [make_case(seed, split=split, sealed=True)
                 for split in ("train", "dev", "test")
                 for seed in (0, 1)]
        self.assertEqual(len({case.case_id for case in cases}), 6)

        # The evaluator witness confirms physical equivalence independently
        # of semantic_fingerprint: no sealed machine ever stops copying x.
        for case in cases:
            for pair in INPUTS:
                session = CandidateSession(case)
                for _ in range(10):
                    view = session.intervene(*pair)
                    self.assertEqual(view["observations"][-1]["output"], pair[0])
                    self.assertEqual(view["integrity_status"], "OK")

        fingerprints = {semantic_fingerprint(case) for case in cases}
        self.assertEqual(len(fingerprints), 1,
                         "sealed suppressed mechanisms are not distinct skills")


class NovelMechanismReceiptIntegrity(unittest.TestCase):
    def test_malformed_hypotheses_are_not_laundered_as_novel_rules(self):
        for kind in ("HYPOTHESIS", "REVISION"):
            for malformed in (object(), b"not-text", [], None, " \t "):
                with self.subTest(kind=kind, malformed=repr(malformed)):
                    session = CandidateSession(make_case(0, split="test"))
                    for _ in range(4):
                        session.intervene(0, 0)
                    field = "kernel" if kind == "HYPOTHESIS" else "revised_kernel"
                    receipt = session.submit({"kind": kind, field: malformed})
                    self.assertEqual(receipt["verdict"], "UNKNOWN")
                    self.assertFalse(receipt["verified_success"])
                    json.dumps(receipt)  # Auditable and exportable exact receipt.


class ReplayAttemptBudgetIntegrity(unittest.TestCase):
    def test_evil_length_list_cannot_exceed_ten_attempts(self):
        class FalseLength(list):
            def __len__(self):
                return 4

        case = make_case(0, split="test")
        attempted = FalseLength([(0, 0)] * 22)
        self.assertIsNone(independent_replay(case, attempted))
        receipt = adjudicate_claim(
            case, {"kind": "HYPOTHESIS", "kernel": "edge"},
            attempted, [0] * 22, [])
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")


if __name__ == "__main__":
    unittest.main()
