"""Remediation must remain inside the operator-approved TRAIN inventory."""

from dataclasses import replace
import unittest

from labyrinth.minotaur_policy_learning import MinotaurTrainingPolicy
from tests.test_minotaur_policy_effectiveness_l8 import (
    APPROVALS as DUAL_APPROVALS,
    OUTCOMES as DUAL_OUTCOMES,
    SCOPE as DUAL_SCOPE,
    context as dual_context,
)
from tests.test_minotaur_policy_learning import (
    SCOPE as SINGLE_SCOPE,
    approval,
    context as single_context,
    outcome,
)


class MinotaurRemediationInventoryTests(unittest.TestCase):
    def test_stalled_transfer_without_unused_contrast_stays_on_weak_tactic(self):
        learner = MinotaurTrainingPolicy(
            scope=DUAL_SCOPE, approved_training_cases=DUAL_APPROVALS
        )
        for visit in DUAL_OUTCOMES[:3] + DUAL_OUTCOMES[6:9]:
            learner.observe(visit)
        for visit in DUAL_OUTCOMES[9:13]:
            learner.observe(replace(visit, was_correct=False))
        # The three approved contrast cases were already consumed. A proposed
        # remedial contrast room would have no unused approved case behind it.
        self.assertEqual(
            learner.propose_room(dual_context("arithmetic").public_metadata()),
            {"tactic": "transfer", "difficulty": 1},
        )

    def test_unused_approved_contrast_permits_bounded_remediation(self):
        approved = tuple(
            approval(i, tactic=("practice" if i < 3 else
                                "contrast" if i < 6 or i == 10 else "transfer"))
            for i in range(14)
        )
        learner = MinotaurTrainingPolicy(scope=SINGLE_SCOPE,
                                         approved_training_cases=approved)
        for i in range(3):
            learner.observe(outcome(i, correct=True))
        for i in range(3, 6):
            learner.observe(outcome(i, tactic="contrast", correct=True))
        for i in range(6, 10):
            learner.observe(outcome(i, tactic="transfer", correct=False))
        proposal = learner.propose_room(single_context().public_metadata())
        self.assertEqual(proposal["tactic"], "contrast")
        self.assertLessEqual(proposal["difficulty"], SINGLE_SCOPE.specificity)


if __name__ == "__main__":
    unittest.main()
