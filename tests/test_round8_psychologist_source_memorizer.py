"""Independent PSYCHOLOGIST negative controls for Cipher Wheel claims.

All code is test-only. It runs *no* neural model, optimizer, or private host.
It demonstrates that published mechanisms plus a few public observations can
produce excellent per-seed scores even without learned structural transfer.
"""

import unittest

from labyrinth.cipher_wheel_controls import (
    memorized_control, symbolic_explorer,
)
from labyrinth.cipher_wheel_oracle import oracle_table, verify_receipt_replay
from labyrinth.cipher_wheel_world import (
    CandidateSession, make_world, seeded_labels,
)


# Public source, not learned weights; the same constants as the open exam.
_KNOWN_ACTIONS = {
    "D4": {"A": (1, 2, 3, 0), "B": (0, 3, 2, 1)},
    "S4": {"A": (1, 2, 0, 3), "B": (0, 1, 3, 2)},
}


def _source_aware_public_memorizer(session):
    """Prewritten intervention policy for the two public source mechanisms.

    Uses only session.view/act/submit_table, not seed or private attributes;
    hardcodes the entire fixed source mechanism family. This is a replayable
    zero-update memorization-control, not induction from unknown mechanics.
    """
    label0 = session.view()["current"]
    label1 = session.act("A")["current"]
    label2 = session.act("A")["current"]
    label3 = session.act("A")["current"]
    if label3 == label0:
        # A has period 3; source-known S4. The missing state is A^2 then B.
        session.act("RESET")
        session.act("A")
        session.act("A")
        label3 = session.act("B")["current"]
        source_family = "S4"
    else:
        source_family = "D4"
    internal_to_color = (label0, label1, label2, label3)
    table = {
        action: {internal_to_color[source]: internal_to_color[target]
                 for source, target in enumerate(targets)}
        for action, targets in _KNOWN_ACTIONS[source_family].items()
    }
    return session.submit_table(table)


def _privileged_zero_probe_source_oracle(session):
    """Intentional forbidden-side-channel control (NOT candidate compliance)."""
    return session.submit_table(oracle_table(session._world))


class _ThirdMechanismSession:
    """Independent toy counterfactual: S4-like A but changed B generator.

    Source-aware two-family script does not know B=(3,1,2,0). Candidate
    observations retain identical four labels, actions and RESET semantics.
    No production exam, real custody or graduation claims.
    """

    def __init__(self, seed):
        self.labels = seeded_labels(seed)
        self.state = 0
        self.history = []
        self.submitted = False

    def view(self):
        return {"current": self.labels[self.state],
                "history": list(self.history), "finished": self.submitted}

    def act(self, action):
        if self.submitted:
            raise ValueError("already terminal")
        if action == "RESET":
            self.state = 0
        elif action == "A":
            self.state = (1, 2, 0, 3)[self.state]
        elif action == "B":
            self.state = (3, 1, 2, 0)[self.state]
        else:
            raise ValueError("not a valid control")
        self.history.append((action, self.labels[self.state]))
        return self.view()

    def submit_table(self, table):
        self.submitted = True
        expected = {
            action: {self.labels[position]: self.labels[target]
                     for position, target in enumerate(targets)}
            for action, targets in {
                "A": (1, 2, 0, 3), "B": (3, 1, 2, 0)
            }.items()
        }
        return {"verdict": "VERIFIED_WIN" if table == expected else "VERIFIED_FAILURE",
                "verified_success": table == expected,
                "evidence_class": "UNSIGNED_TEST_ONLY",
                "observations": len(self.history)}


def _learning_gate_test_only(receipts, *, independent_mechanisms_required=8):
    """Fail-closed semantics, intentionally has no live promotion authority.

    A numerical win-rate cannot become neural-transfer proof without certified
    source blindness, a model/update/custody record, and independent mechanism
    units. A thousand label permutations within a family count as ONE.
    """
    if not receipts or not all(verify_receipt_replay(r) for r in receipts):
        return "BLOCKED_REPLAY"
    if any(r["verdict"] != "VERIFIED_WIN" for r in receipts):
        return "BLOCKED_INCOMPLETE_SUCCESS"
    families = {r["semantic_family_id"] for r in receipts}
    if len(families) < independent_mechanisms_required:
        return "BLOCKED_EFFECTIVE_SAMPLE_SIZE"
    if any(r["public_source_oracle"] or not r["independent_custody"] or
           not r["neural_learning_verified"] or r["source_commit"] == "UNBOUND"
           for r in receipts):
        return "BLOCKED_UNQUALIFIED_EVIDENCE"
    return "SOURCE_ONLY_GATE_UNAUTHORIZED"  # test-only, never GRADUATED


class CipherWheelWave8PsychologistControls(unittest.TestCase):
    def test_public_mechanism_memo_scores_100_each_split_with_tiny_fixed_script(self):
        success = {"D4": 0, "S4": 0}
        attempts = {"D4": set(), "S4": set()}
        for heldout, family in ((False, "D4"), (True, "S4")):
            for seed in range(100):
                receipt = _source_aware_public_memorizer(
                    CandidateSession(make_world(seed, heldout=heldout)))
                success[family] += receipt["verdict"] == "VERIFIED_WIN"
                attempts[family].add(len(receipt["attempts"]))
                self.assertTrue(verify_receipt_replay(receipt))
                self.assertFalse(receipt["neural_learning_verified"])
                self.assertFalse(receipt["graduation_eligible"])
        self.assertEqual(success, {"D4": 100, "S4": 100})
        self.assertEqual(attempts, {"D4": {3}, "S4": {7}})

    def test_privileged_zero_probe_oracle_wins_but_is_custody_violation(self):
        for heldout in (False, True):
            for seed in range(100):
                session = CandidateSession(make_world(seed, heldout=heldout))
                receipt = _privileged_zero_probe_source_oracle(session)
                self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
                self.assertEqual(len(receipt["attempts"]), 0)
                self.assertTrue(receipt["public_source_oracle"])
                self.assertFalse(receipt["independent_custody"])
                self.assertFalse(receipt["neural_learning_verified"])

    def test_source_aware_memorizer_collapses_on_unseen_generator(self):
        memo_wins = symbolic_wins = 0
        for seed in range(25):
            memo = _source_aware_public_memorizer(_ThirdMechanismSession(seed))
            symbolic = symbolic_explorer(_ThirdMechanismSession(seed))
            memo_wins += memo["verdict"] == "VERIFIED_WIN"
            symbolic_wins += symbolic["verdict"] == "VERIFIED_WIN"
        self.assertEqual(memo_wins, 0)
        self.assertEqual(symbolic_wins, 25)

    def test_200_successful_relabels_have_only_two_semantic_families(self):
        receipts = [
            _source_aware_public_memorizer(CandidateSession(
                make_world(seed, heldout=heldout)))
            for heldout in (False, True) for seed in range(100)
        ]
        self.assertEqual(len(receipts), 200)
        self.assertTrue(all(r["verdict"] == "VERIFIED_WIN" for r in receipts))
        self.assertEqual({r["semantic_family_id"] for r in receipts},
                         {"cipher-wheel.D4", "cipher-wheel.S4"})
        self.assertEqual(_learning_gate_test_only(receipts),
                         "BLOCKED_EFFECTIVE_SAMPLE_SIZE")
        # Even waiving the family-count requirement cannot elevate public
        # source and same-process, unverifiable neural provenance.
        self.assertEqual(_learning_gate_test_only(
            receipts, independent_mechanisms_required=2),
            "BLOCKED_UNQUALIFIED_EVIDENCE")

    def test_legacy_frozen_script_is_strawman_not_learning_control(self):
        wins = {"D4": 0, "S4": 0}
        for heldout, family in ((False, "D4"), (True, "S4")):
            for seed in range(100):
                result = memorized_control(CandidateSession(
                    make_world(seed, heldout=heldout)))
                wins[family] += result["verdict"] == "VERIFIED_WIN"
        self.assertEqual(wins, {"D4": 6, "S4": 0})


if __name__ == "__main__":
    unittest.main()
