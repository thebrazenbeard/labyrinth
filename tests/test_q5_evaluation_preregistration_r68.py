"""R68: a source screen must not turn its own post-hoc roster into a precommit."""

from dataclasses import replace
from hashlib import sha256
import inspect
import unittest

from labyrinth.q5_evaluation_preregistration_r68 import (
    EvaluationPrecommitError, PredeclaredSeedRoster,
    build_frozen_evaluation_precommit, screen_predeclared_evaluation,
)
from tests.test_five_arm_statistical_screen import sample_outcomes


def _fixture():
    outcomes = sample_outcomes()
    rosters = tuple(PredeclaredSeedRoster(
        run_id=item.plan.run_id,
        scope_seed=item.plan.scope_seed,
        plan_sha256=item.plan.source_contract_sha256,
        development_cases=tuple(x.case_sha256 for x in item.development),
        transfer_cases=tuple(x.case_sha256 for x in item.transfer),
    ) for item in outcomes)
    return outcomes, build_frozen_evaluation_precommit(rosters)


def _screen(outcomes, precommit, *, expected=None, verifier=None, witness=b"externally pinned witness bytes"):
    return screen_predeclared_evaluation(
        precommit=precommit,
        external_expected_precommit_sha256=expected or precommit.precommit_sha256,
        preoutcome_witness=witness,
        verify_external_preoutcome_witness=verifier or (lambda data, digest: True),
        outcomes=outcomes,
    )


class Q5EvaluationPreregistrationR68Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcomes, cls.precommit = _fixture()

    def test_exact_external_source_precommit_and_screen_never_attests_or_graduates(self):
        received = []
        result = _screen(
            self.outcomes, self.precommit,
            verifier=lambda witness, digest: received.append((witness, digest)) or True,
        )
        self.assertEqual(received[0][1], self.precommit.precommit_sha256)
        self.assertTrue(result["source_screen_pass"])
        self.assertEqual(result["screen"]["seed_count"], 8)
        for flag in ("independent_preoutcome_witness_custody_verified",
                     "independent_evaluator_custody_verified",
                     "physical_pretrained_model_verified",
                     "semantic_holdout_verified", "retention_authorized",
                     "learning_authorized", "graduation_authorized"):
            self.assertIs(result[flag], False, flag)

    def test_source_self_digest_recomputed_after_case_swap_still_screen_passes_but_new_gate_rejects(self):
        source = self.outcomes[0]
        new_case = replace(source.transfer[0], case_sha256=sha256(b"post-hoc-swap").hexdigest())
        changed_cases = (new_case,) + source.transfer[1:]
        from labyrinth.five_arm_statistical_screen import _case_digest, screen_five_arm_seed_outcomes
        altered = (replace(source, transfer=changed_cases,
                           transfer_case_digest=_case_digest(changed_cases)),) + self.outcomes[1:]
        self.assertTrue(screen_five_arm_seed_outcomes(altered)["source_screen_pass"])
        with self.assertRaisesRegex(EvaluationPrecommitError, "roster"):
            _screen(altered, self.precommit)

    def test_precommit_digest_cannot_be_replaced_by_operator_after_observing_results(self):
        mutated = replace(self.precommit.seed_rosters[0],
                          transfer_cases=(sha256(b"new-posthoc-case").hexdigest(),)
                          + self.precommit.seed_rosters[0].transfer_cases[1:])
        posthoc = build_frozen_evaluation_precommit(
            (mutated,) + self.precommit.seed_rosters[1:])
        with self.assertRaisesRegex(EvaluationPrecommitError, "external.*digest"):
            _screen(self.outcomes, posthoc, expected=self.precommit.precommit_sha256)

    def test_witness_callback_cannot_swap_frozen_roster_after_initial_digest_check(self):
        """object.__setattr__ bypasses frozen dataclass's assignment guard."""
        source = self.outcomes[0]
        changed_case = replace(source.transfer[0],
                               case_sha256=sha256(b"r69-callback-replaced-case").hexdigest())
        changed_cases = (changed_case,) + source.transfer[1:]
        from labyrinth.five_arm_statistical_screen import _case_digest
        altered_outcomes = (
            replace(source, transfer=changed_cases,
                    transfer_case_digest=_case_digest(changed_cases)),
        ) + self.outcomes[1:]
        original_roster = self.precommit.seed_rosters[0]
        replaced_roster = replace(original_roster,
                                  transfer_cases=tuple(c.case_sha256 for c in changed_cases))

        for mutation in ("outer", "nested"):
            with self.subTest(mutation=mutation):
                precommit = replace(self.precommit, seed_rosters=tuple(
                    replace(seed) for seed in self.precommit.seed_rosters))

                def mutable_witness(_witness, _digest):
                    if mutation == "outer":
                        object.__setattr__(precommit, "seed_rosters",
                                           (replaced_roster,) + precommit.seed_rosters[1:])
                    else:
                        object.__setattr__(precommit.seed_rosters[0], "transfer_cases",
                                           replaced_roster.transfer_cases)
                    return True

                with self.assertRaisesRegex(EvaluationPrecommitError, "mutated.*witness"):
                    _screen(altered_outcomes, precommit, verifier=mutable_witness)

    def test_frame_inspecting_callback_cannot_mutate_both_original_and_snapshot(self):
        """Same-interpreter callback can inspect caller locals, including `sealed`."""
        source = self.outcomes[0]
        replacement = replace(source.development[0],
                              case_sha256=sha256(b"r69-frame-posthoc-case").hexdigest())
        changed_cases = (replacement,) + source.development[1:]
        from labyrinth.five_arm_statistical_screen import _case_digest
        altered = (replace(source, development=changed_cases,
                           development_case_digest=_case_digest(changed_cases)),) + self.outcomes[1:]
        precommit = replace(self.precommit, seed_rosters=tuple(
            replace(seed) for seed in self.precommit.seed_rosters))
        altered_roster = replace(precommit.seed_rosters[0],
                                 development_cases=tuple(c.case_sha256 for c in changed_cases))

        def malicious_callback(_witness, _digest):
            frame = inspect.currentframe()
            assert frame is not None
            caller = frame.f_back
            assert caller is not None
            sealed = caller.f_locals["sealed"]
            # Mutate BOTH objects identically; equality alone cannot detect it.
            changed_rosters = (altered_roster,) + precommit.seed_rosters[1:]
            object.__setattr__(precommit, "seed_rosters", changed_rosters)
            object.__setattr__(sealed, "seed_rosters", changed_rosters)
            return True

        with self.assertRaisesRegex(EvaluationPrecommitError, "post-witness.*digest"):
            _screen(altered, precommit, verifier=malicious_callback)

    def test_outcome_seed_reorder_or_omission_is_rejected(self):
        with self.assertRaisesRegex(EvaluationPrecommitError, "seed order"):
            _screen((self.outcomes[1], self.outcomes[0]) + self.outcomes[2:], self.precommit)
        with self.assertRaisesRegex(EvaluationPrecommitError, "seed order"):
            _screen(self.outcomes[:-1], self.precommit)

    def test_untrusted_witness_and_missing_witness_are_rejected(self):
        with self.assertRaisesRegex(EvaluationPrecommitError, "external witness"):
            _screen(self.outcomes, self.precommit, verifier=lambda *_: False)
        with self.assertRaisesRegex(EvaluationPrecommitError, "external witness"):
            _screen(self.outcomes, self.precommit, witness=b"")
        with self.assertRaisesRegex(EvaluationPrecommitError, "external witness"):
            _screen(self.outcomes, self.precommit, verifier=lambda *_: 1)

    def test_policy_or_commitment_mutation_fails_closed(self):
        for change in (
            dict(familywise_alpha=0.1),
            dict(minimum_mean_gain=0.0),
            dict(method="unadjusted_p_value"),
            dict(precommit_sha256="0" * 64),
        ):
            with self.subTest(change=change), self.assertRaises(EvaluationPrecommitError):
                _screen(self.outcomes, replace(self.precommit, **change))

    def test_duplicate_predeclared_case_and_omitted_control_roster_fail(self):
        first = self.precommit.seed_rosters[0]
        duplicated = replace(first, development_cases=(first.development_cases[0],) * 20)
        with self.assertRaisesRegex(EvaluationPrecommitError, "duplicate"):
            build_frozen_evaluation_precommit((duplicated,) + self.precommit.seed_rosters[1:])
        truncated = replace(first, transfer_cases=first.transfer_cases[:19])
        with self.assertRaisesRegex(EvaluationPrecommitError, "case count"):
            build_frozen_evaluation_precommit((truncated,) + self.precommit.seed_rosters[1:])

    def test_reused_plan_identity_and_malformed_primitives_rejected(self):
        first = self.precommit.seed_rosters[0]
        repeated = replace(self.precommit.seed_rosters[1], plan_sha256=first.plan_sha256)
        with self.assertRaisesRegex(EvaluationPrecommitError, "duplicate"):
            build_frozen_evaluation_precommit(
                (first, repeated) + self.precommit.seed_rosters[2:])
        bad = replace(first, scope_seed=True)
        with self.assertRaisesRegex(EvaluationPrecommitError, "seed"):
            build_frozen_evaluation_precommit((bad,) + self.precommit.seed_rosters[1:])


if __name__ == "__main__":
    unittest.main()
