"""Independent Wave8 meta-check of trusted launcher, not custody evidence.

Exact source baseline eb70aa531ceac25a8fe4a05aee014d8e263d2f05.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from labyrinth.minotaur_cross_exam import ExamSpec, run_cross_exam
from labyrinth.minotaur_trusted_launcher import (
    ProcessCandidateConfig, candidate_public_summary, run_trusted_cross_exam,
)


def reseal_self_hash(run):
    unsigned = asdict(run)
    unsigned.pop("receipt_sha256")
    digest = sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return replace(run, receipt_sha256=digest)


class UnqualifiedHostCheck(unittest.TestCase):
    def test_exact_typed_self_resealed_fake_win_stays_labeled_unattested(self):
        genuine = run_trusted_cross_exam(
            (ExamSpec("oracle", 739, sealed=True),),
            ProcessCandidateConfig(mode="probe", generation_seed=17))[0]
        fake_run = reseal_self_hash(replace(genuine.cross_case.candidate,
                                            final_verdict="VERIFIED_WIN",
                                            verified_escape=True))
        fake_host = replace(genuine, cross_case=replace(genuine.cross_case,
                                                       candidate=fake_run))
        projected = candidate_public_summary((fake_host,))[0]
        # PASS-as-EXPLOIT: SHA is not a signature; forged host says win.
        self.assertEqual(projected["host_reported_candidate_verdict"], "VERIFIED_WIN")
        self.assertTrue(projected["host_reported_candidate_escape"])
        # Essential scope boundary: no authenticated or promoted status.
        self.assertEqual(projected["evidence_status"], "UNATTESTED_HOST_SELF_HASH")
        self.assertNotIn("candidate_escape_verified", projected)
        self.assertNotIn("graduated", projected)

    def test_untrusted_factory_rejected_before_private_exam_construction(self):
        def malicious(public):
            raise AssertionError("untrusted factory ran")
        with patch("labyrinth.minotaur_cross_exam.make_public_case",
                   side_effect=AssertionError("private exam materialized")) as private:
            with self.assertRaises(PermissionError):
                run_cross_exam((ExamSpec("oracle", 739, sealed=True),),
                               candidate_factory=malicious,
                               authorize_local_adapter=True)
            private.assert_not_called()


if __name__ == "__main__":
    unittest.main()
