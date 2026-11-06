"""R68 source-only preregistration manifest tests (no model execution)."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import sys
import unittest

import labyrinth.retention_transfer_cohort_r68 as cohort_module
from labyrinth.retention_qualification import (
    PROMPT_COMMITMENT_VERSION,
    RetentionPair,
    RetentionPolicy,
    normalized_prompt_sha256,
    retention_roster_sha256,
)
from labyrinth.retention_sample_planning_r57 import plan_prior_skill_retention_samples
from labyrinth.retention_transfer_cohort_r68 import (
    CohortManifestError,
    StructuralTransferUnit,
    build_fixed_cohort_manifest,
    verify_fixed_cohort_manifest,
    verify_fixed_cohort_outcome_selection,
)


def _digest(char: str) -> str:
    return char * 64


def _fixture():
    rows = []
    for skill in ("logic", "arithmetic"):
        for i in range(72):
            before = f"{skill} before semantic situation {i}"
            after = f"{skill} after semantic situation {i}"
            rows.append(RetentionPair(
                skill_id=skill,
                situation_id=f"s-{skill}-{i}",
                structure_family_id=f"family-{skill}-{i}",
                before_case_id=f"b-{skill}-{i}",
                after_case_id=f"a-{skill}-{i}",
                before_prompt_text=before,
                after_prompt_text=after,
                before_prompt_sha256=normalized_prompt_sha256(before),
                after_prompt_sha256=normalized_prompt_sha256(after),
            ))
    roster = tuple(rows)
    training_prompt = "canonical training prompt"
    policy = RetentionPolicy(
        objective_id="retention-transfer-demo",
        scope_sha256=_digest("1"),
        before_checkpoint_sha256=_digest("2"),
        after_checkpoint_sha256=_digest("3"),
        evaluator_sha256=_digest("4"),
        conditions_sha256=_digest("5"),
        session_nonce="a" * 32,
        prior_skill_ids=("logic", "arithmetic"),
        expected_roster_sha256=retention_roster_sha256(roster),
        prompt_commitment_version=PROMPT_COMMITMENT_VERSION,
        excluded_training_prompts=(training_prompt,),
        excluded_training_prompt_sha256s=(normalized_prompt_sha256(training_prompt),),
        excluded_training_structure_family_ids=("training-family",),
    )
    plan = plan_prior_skill_retention_samples(
        ("logic", "arithmetic"), (0, 0)
    )
    transfer = tuple(StructuralTransferUnit(
        source_skill_id=source,
        target_skill_id="novel-symbolic",
        case_id=f"xfer-{i}",
        situation_id=f"transfer-sit-{i}",
        structure_family_id=f"transfer-family-{i}",
        prompt_text=f"Novel symbolic structure transfer prompt {i}",
        prompt_sha256=normalized_prompt_sha256(
            f"Novel symbolic structure transfer prompt {i}"
        ),
    ) for i, source in enumerate(("logic", "arithmetic")))
    return policy, roster, plan, ("novel-symbolic",), (2,), transfer


def _manifest(args):
    return build_fixed_cohort_manifest(*args)


class RetentionTransferCohortR68Tests(unittest.TestCase):
    def test_exact_precommit_and_full_outcome_coverage_is_source_only(self):
        args = _fixture()
        manifest = _manifest(args)
        pinned_digest = sha256(manifest).hexdigest()
        review = verify_fixed_cohort_manifest(pinned_digest, manifest, *args)
        self.assertEqual(review.manifest_sha256, pinned_digest)
        self.assertEqual(review.retention_counts, (72, 72))
        self.assertEqual(review.transfer_counts, (2,))
        self.assertEqual(review.status, "SOURCE_ONLY_UNATTESTED_FIXED_COHORT")
        self.assertFalse(review.precommitted_before_outcomes)
        self.assertFalse(review.independent_case_custody_verified)
        self.assertFalse(review.model_execution_authenticated)
        self.assertFalse(review.graduation_authorized)
        with self.assertRaises((AttributeError, TypeError)):
            object.__setattr__(review, "graduation_authorized", True)
        observed = verify_fixed_cohort_outcome_selection(
            pinned_digest, manifest, *args,
            retention_situation_ids=tuple(p.situation_id for p in args[1]),
            transfer_case_ids=tuple(p.case_id for p in args[-1]),
        )
        self.assertEqual(observed.manifest_sha256, pinned_digest)

    def test_late_subset_extra_duplicate_and_unpinned_manifest_fail_closed(self):
        args = _fixture()
        manifest = _manifest(args)
        pin = sha256(manifest).hexdigest()
        situations = tuple(p.situation_id for p in args[1])
        transfer_ids = tuple(p.case_id for p in args[-1])
        for retention_ids, transfer_selection in (
            (situations[:-1], transfer_ids),
            (situations + (situations[0],), transfer_ids),
            (situations[:-1] + ("late-new-situation",), transfer_ids),
            (situations, transfer_ids[:-1]),
            (situations, transfer_ids + transfer_ids[:1]),
        ):
            with self.subTest(retention_ids=len(retention_ids), transfer_ids=transfer_selection):
                with self.assertRaises(CohortManifestError):
                    verify_fixed_cohort_outcome_selection(
                        pin, manifest, *args,
                        retention_situation_ids=retention_ids,
                        transfer_case_ids=transfer_selection,
                    )
        with self.assertRaises(CohortManifestError):
            verify_fixed_cohort_manifest(_digest("e"), manifest, *args)
        with self.assertRaises(CohortManifestError):
            verify_fixed_cohort_manifest(pin, manifest + b" ", *args)
        with self.assertRaises(CohortManifestError):
            verify_fixed_cohort_manifest(pin, manifest, *args[:-1], args[-1][::-1])

    def test_sampling_counts_must_exactly_match_r57_plan(self):
        policy, roster, plan, targets, target_counts, transfer = _fixture()
        extra_before = "arithmetic before independent situation 72"
        extra_after = "arithmetic after independent situation 72"
        extra = replace(
            roster[-1], situation_id="s-arithmetic-72",
            structure_family_id="family-arithmetic-72",
            before_case_id="b-arithmetic-72",
            after_case_id="a-arithmetic-72",
            before_prompt_text=extra_before, after_prompt_text=extra_after,
            before_prompt_sha256=normalized_prompt_sha256(extra_before),
            after_prompt_sha256=normalized_prompt_sha256(extra_after),
        )
        for edited in (roster[:-1], roster + (extra,)):
            with self.assertRaises(CohortManifestError):
                build_fixed_cohort_manifest(
                    replace(policy, expected_roster_sha256=retention_roster_sha256(edited)),
                    edited, plan, targets, target_counts, transfer,
                )
        with self.assertRaises(CohortManifestError):
            build_fixed_cohort_manifest(policy, roster, plan, targets, (1,), transfer)
        weaker_plan = plan_prior_skill_retention_samples(
            ("arithmetic", "logic"), (0, 0)
        )
        with self.assertRaises(CohortManifestError):
            build_fixed_cohort_manifest(policy, roster, weaker_plan, targets, target_counts, transfer)

    def test_manifest_is_deterministic_and_never_exports_raw_prompts(self):
        args = _fixture()
        first = _manifest(args)
        second = _manifest(args)
        self.assertEqual(first, second)
        self.assertNotIn(b"canonical training prompt", first)
        self.assertNotIn(b"Novel symbolic structure transfer prompt", first)
        self.assertNotIn(b"logic before semantic situation", first)
        # Fixed retention unit count is based on situations, not both forms.
        self.assertEqual(len(args[1]), 144)
        self.assertEqual(len({p.situation_id for p in args[1]}), 144)
        self.assertEqual(len({p.before_case_id for p in args[1]}), 144)
        self.assertEqual(len({p.after_case_id for p in args[1]}), 144)

    def test_post_validation_trace_mutation_cannot_change_reviewed_roster_digest(self):
        """A late caller mutation cannot change a digest from validated bytes.

        Simulates a hostile in-process caller at the exact point after the
        canonical manifest equality check and before return field creation.
        This does not assert that mutable caller input itself stays fixed.
        """
        args = _fixture()
        manifest = _manifest(args)
        pin = sha256(manifest).hexdigest()
        reviewed_roster_digest = retention_roster_sha256(args[1])
        reviewed_function = cohort_module.verify_fixed_cohort_manifest
        source, start_line = inspect.getsourcelines(reviewed_function)
        review_line = next(
            start_line + i for i, line in enumerate(source)
            if line.lstrip().startswith("return FixedCohortSourceReview(")
        )
        mutations = []

        def trace(frame, event, arg):
            if (not mutations and frame.f_code is reviewed_function.__code__ and event == "line"
                    and frame.f_lineno == review_line):
                object.__setattr__(
                    args[1][0], "before_prompt_sha256", "f" * 64
                )
                mutations.append(True)
            return trace

        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            review = reviewed_function(pin, manifest, *args)
        finally:
            sys.settrace(previous_trace)
        self.assertEqual(mutations, [True])
        self.assertNotEqual(retention_roster_sha256(args[1]), reviewed_roster_digest)
        self.assertEqual(review.retention_roster_sha256, reviewed_roster_digest)
        self.assertEqual(review.manifest_sha256, pin)
        self.assertFalse(review.independent_case_custody_verified)

    def test_post_validation_trace_mutation_cannot_admit_late_outcome_id(self):
        """Observation roster must come from sealed manifest, not live pairs."""
        args = _fixture()
        manifest = _manifest(args)
        pin = sha256(manifest).hexdigest()
        late_id = "attacker-substituted-after-manifest-check"
        selected_retention = (late_id,) + tuple(
            case.situation_id for case in args[1][1:]
        )
        selected_transfer = tuple(case.case_id for case in args[-1])
        reviewed_function = cohort_module.verify_fixed_cohort_outcome_selection
        source, start_line = inspect.getsourcelines(reviewed_function)
        selection_line = next(
            start_line + i for i, line in enumerate(source)
            if line.lstrip().startswith("(retention_situation_ids,")
        )
        mutations = []

        def trace(frame, event, arg):
            if (not mutations and frame.f_code is reviewed_function.__code__
                    and event == "line" and frame.f_lineno == selection_line):
                object.__setattr__(args[1][0], "situation_id", late_id)
                mutations.append(True)
            return trace

        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(CohortManifestError):
                reviewed_function(
                    pin, manifest, *args,
                    retention_situation_ids=selected_retention,
                    transfer_case_ids=selected_transfer,
                )
        finally:
            sys.settrace(previous_trace)
        self.assertEqual(mutations, [True])

    def test_cross_domain_situation_family_case_and_prompt_collisions(self):
        args = _fixture()
        policy, roster, plan, targets, counts, transfer = args
        first = transfer[0]
        attacks = (
            first._replace(situation_id=roster[0].situation_id),
            first._replace(structure_family_id=roster[0].structure_family_id),
            first._replace(case_id=roster[0].before_case_id),
            first._replace(prompt_text=roster[0].before_prompt_text,
                           prompt_sha256=roster[0].before_prompt_sha256),
            first._replace(structure_family_id="training-family"),
            first._replace(prompt_text=policy.excluded_training_prompts[0],
                           prompt_sha256=policy.excluded_training_prompt_sha256s[0]),
            first._replace(situation_id=transfer[1].situation_id),
            first._replace(structure_family_id=transfer[1].structure_family_id),
        )
        for attacker in attacks:
            with self.subTest(attacker=attacker):
                with self.assertRaises(CohortManifestError):
                    build_fixed_cohort_manifest(
                        policy, roster, plan, targets, counts,
                        (attacker, transfer[1]),
                    )

    def test_bad_source_target_and_mutated_policy_fail_closed(self):
        policy, roster, plan, targets, counts, transfer = _fixture()
        for altered in (
            transfer[0]._replace(source_skill_id="undeclared"),
            transfer[0]._replace(target_skill_id="logic"),
            transfer[0]._replace(target_skill_id="undeclared"),
            transfer[0]._replace(prompt_sha256=_digest("f")),
        ):
            with self.assertRaises(CohortManifestError):
                build_fixed_cohort_manifest(
                    policy, roster, plan, targets, counts, (altered, transfer[1])
                )
        object.__setattr__(policy, "significance_alpha", 1.0)
        with self.assertRaises(CohortManifestError):
            build_fixed_cohort_manifest(policy, roster, plan, targets, counts, transfer)


if __name__ == "__main__":
    unittest.main()
