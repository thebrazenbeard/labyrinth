"""Hostile negative controls: benchmark score != demonstrated learning.

This file deliberately runs no model, optimizer, or private evaluator.  Its
public-view-only lookup exposes seed leakage, and its paraphrase perturbation
tests whether the reference policy understands the rule rather than its regex.
"""

import re
import unittest

from labyrinth import timegate_exam as exam
from labyrinth.timegate_baseline import run_public_rule_parser


def public_configuration_key(view):
    """Candidate-visible rule parameters; no access to Case or private state."""
    prompt = view["prompt"]
    observations = view["observation"]
    match = re.search(
        r"Charging requires clock phase (\d+) in a cycle of (\d+); "
        r"the gate accepts escape only at phase (\d+)\. "
        r"Relay power persists for (\d+) subsequent",
        prompt,
    )
    if match is None:
        raise ValueError("unknown rule description")
    charge, period, goal, duration = map(int, match.groups())
    first_clock = int(re.search(r"Clock: (\d+)/", observations).group(1))
    return period, charge, goal, first_clock, duration


def train_observation_lookup(seeds):
    """Memorize observed rule signatures and labelled TRAIN witnesses only."""
    table = {}
    for seed in seeds:
        case = exam.make_case(seed)
        key = public_configuration_key(exam.candidate_view(case))
        labelled_witness = exam.search(case).witness  # TRAIN-side oracle only
        if key in table:
            if table[key] != labelled_witness:
                raise AssertionError("identical public configuration had conflicting labels")
        table[key] = labelled_witness
    return table


def lookup_candidate(view, submit, memory):
    """No Case, test witness oracle, evaluator transition, or parameter update."""
    actions = memory.get(public_configuration_key(view))
    if actions is None:
        return submit({"kind": "ABSTAIN", "assessment": "NO_KNOWN_WIN"})
    return submit({"kind": "WITNESS", "actions": actions})


class RoundTwoTransferControls(unittest.TestCase):
    def test_nominal_wording_variants_have_only_twelve_distinct_full_prompts(self):
        # Four preambles times three admonitions: the rules themselves retain
        # identical wording across all ten thousand labelled variants.
        variants = {exam.candidate_view(exam.make_case(7, variant=i))["prompt"]
                    for i in range(10000)}
        self.assertEqual(len(variants), 12)
        self.assertEqual(len({public_configuration_key(exam.candidate_view(
            exam.make_case(7, variant=i))) for i in range(10000)}), 1)

    def test_disjoint_seed_holdout_reuses_exact_train_world_configurations(self):
        # Different integer seeds are NOT different underlying mechanics.
        train = train_observation_lookup(range(120))
        self.assertEqual(len(train), 86)
        statuses = []
        for seed in range(120, 240):
            case = exam.make_case(seed, variant=31)
            session = exam.CandidateSession(case)
            receipt = lookup_candidate(exam.candidate_view(case),
                                       session.submit, train)
            statuses.append(receipt["verdict"])
        self.assertEqual(statuses.count("VERIFIED_WIN"), 76)
        # The remaining cases are all OPEN and solvable by an oracle, yet a
        # memory-only candidate may legitimately say *it* knows no solution.
        # That self-report is not an independently verified UNSAT certificate.
        self.assertEqual(statuses.count("NO_KNOWN_WIN"), 44)

    def test_same_seed_new_surface_variant_is_perfectly_memorized(self):
        train = train_observation_lookup(range(120))
        successes = 0
        for seed in range(120):
            case = exam.make_case(seed, variant=9999)
            session = exam.CandidateSession(case)
            receipt = lookup_candidate(exam.candidate_view(case),
                                       session.submit, train)
            successes += receipt["verdict"] == "VERIFIED_WIN"
        self.assertEqual(successes, 120)

    def test_semantically_equal_rule_paraphrase_breaks_public_parser(self):
        # We change only the presentation, not any game semantics or answer.
        # Existing 'paraphrases' vary only introductory and exhortatory text.
        case = exam.make_case(17)
        view = exam.candidate_view(case)
        original = "Charging requires clock phase "
        self.assertIn(original, view["prompt"])
        modified = dict(view)
        modified["prompt"] = view["prompt"].replace(
            f"Charging requires clock phase {case.charge_phase} in a cycle of {case.period};",
            f"The relay may be armed only on phase {case.charge_phase} of the "
            f"{case.period}-step clock;",
        )
        self.assertNotEqual(modified["prompt"], view["prompt"])
        session = exam.CandidateSession(case)
        with self.assertRaises(AttributeError):
            run_public_rule_parser(modified, session.act, session.submit)
        self.assertEqual(session.attempts(), ())
        # The identical model-free parser still succeeds on the usual phrasing.
        normal = exam.CandidateSession(case)
        self.assertEqual(run_public_rule_parser(view, normal.act,
                                                normal.submit).verdict, "VERIFIED_WIN")

    def test_abstention_label_is_not_objective_unsat_or_calibration_evidence(self):
        # Both cases receive the very same accepted self-report even though
        # one has an independently verified winning path under declared rules.
        for sealed in (False, True):
            case = exam.make_case(8, sealed=sealed)
            review = exam.CandidateSession(case).submit(
                {"kind": "ABSTAIN", "assessment": "NO_KNOWN_WIN"})
            self.assertEqual(review["verdict"], "NO_KNOWN_WIN")
            self.assertFalse(review["verified_success"])
        self.assertEqual(exam.search(exam.make_case(8)).verdict, "VERIFIED_WIN")
        self.assertEqual(exam.search(exam.make_case(8, sealed=True)).verdict,
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")


if __name__ == "__main__":
    unittest.main()
