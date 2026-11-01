"""Q6 source-only toy extrapolation controls. Not a private qualification."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import unittest
from unittest.mock import patch

from labyrinth.q6_stratified_toy_study_r93 import (
    Q6StudyError, Q6ExposureExhaustion, Q6ExposureReuse,
    construct_q6_cohorts, run_q6_toy_study,
    validate_q6_cohorts, semantic_case_key, q6_exposure_history_sha256, _profile,
)
from labyrinth.toy_backend import ToyLinearAdapter


class Q6StratifiedToyStudyTests(unittest.TestCase):
    def test_fixed_balanced_semantic_profiles_and_range_shift(self):
        for seed in (1, 7, 9103):
            with self.subTest(seed=seed):
                cohort = construct_q6_cohorts(seed=seed, per_profile=5)
                validation = validate_q6_cohorts(cohort, per_profile=5)
                self.assertEqual(validation["total"], 80)
                for name in ("train_a", "train_b", "heldout", "transfer"):
                    cases = getattr(cohort, name)
                    self.assertEqual(len(cases), 20)
                    self.assertEqual(set(Counter(_profile(c.prompt) for c in cases).values()), {5})
                self.assertTrue(all(max(int(c.prompt.split()[1]), int(c.prompt.split()[3][:-1])) <= 499
                                    for group in (cohort.train_a, cohort.train_b, cohort.heldout)
                                    for c in group))
                self.assertTrue(all(min(int(c.prompt.split()[1]), int(c.prompt.split()[3][:-1])) >= 800
                                    for c in cohort.transfer))

    def test_semantically_equivalent_commutative_alias_fails_closed(self):
        cohorts = construct_q6_cohorts(seed=8, per_profile=3)
        original = next(c for c in cohorts.train_a if " + " in c.prompt)
        a, b = original.prompt.splitlines()[0].removeprefix("Calculate ").removesuffix(".").split(" + ")
        forged = replace(cohorts.heldout[0],
                         prompt=f"Calculate {b} + {a}.\nRespond with exactly one line: ANSWER: <integer>.",
                         answer=original.answer)
        self.assertNotEqual(forged.prompt, original.prompt)
        self.assertEqual(semantic_case_key(forged), semantic_case_key(original))
        with self.assertRaisesRegex(Q6StudyError, "semantic overlap"):
            validate_q6_cohorts(replace(cohorts, heldout=(forged,) + cohorts.heldout[1:]),
                                per_profile=3)

    def test_inverse_equation_alias_across_operators_fails_closed(self):
        cohorts = construct_q6_cohorts(seed=0, per_profile=6)
        source = next(c for c in cohorts.train_a if " + " in c.prompt)
        a, b = (int(x) for x in source.prompt.splitlines()[0]
                .removeprefix("Calculate ").removesuffix(".").split(" + "))
        inverse_prompt = (f"Calculate {a+b} - {a}.\n"
                          "Respond with exactly one line: ANSWER: <integer>.")
        inverse_profile = _profile(inverse_prompt)
        replacement = next(i for i, c in enumerate(cohorts.heldout)
                           if _profile(c.prompt) == inverse_profile)
        forged = replace(cohorts.heldout[replacement], prompt=inverse_prompt, answer=str(b))
        self.assertNotEqual(source.prompt, forged.prompt)
        with self.assertRaisesRegex(Q6StudyError, "semantic overlap"):
            validate_q6_cohorts(replace(
                cohorts, heldout=cohorts.heldout[:replacement] + (forged,)
                + cohorts.heldout[replacement+1:],
            ), per_profile=6)

    def test_eval_gold_and_score_cannot_be_rewritten(self):
        cohorts = construct_q6_cohorts(seed=13, per_profile=3)
        case = cohorts.transfer[0]
        forged = replace(case, answer="9999")
        with self.assertRaisesRegex(Q6StudyError, "gold"):
            validate_q6_cohorts(replace(cohorts, transfer=(forged,) + cohorts.transfer[1:]),
                                per_profile=3)

    def test_preregistered_replay_is_deterministic_not_secret(self):
        a = construct_q6_cohorts(seed=15, per_profile=3)
        b = construct_q6_cohorts(seed=15, per_profile=3)
        self.assertEqual(a, b)
        exposed = {case.prompt for case in a.heldout}
        with self.assertRaisesRegex(Q6StudyError, "previously exposed"):
            validate_q6_cohorts(b, per_profile=3, previously_exposed_eval_prompts=exposed)
        # The history is caller-supplied and nonauthorizing, but its content
        # still needs semantic alias matching, not only exact prompt bytes.
        add_case = next(c for c in b.heldout if " + " in c.prompt)
        a, b_operand = (add_case.prompt.splitlines()[0]
                        .removeprefix("Calculate ").removesuffix(".").split(" + "))
        alias = (f"Calculate {b_operand} + {a}.\n"
                 "Respond with exactly one line: ANSWER: <integer>.")
        self.assertNotEqual(alias, add_case.prompt)
        with self.assertRaisesRegex(Q6StudyError, "previously exposed"):
            validate_q6_cohorts(b, per_profile=3, previously_exposed_eval_prompts={alias})

    def test_real_updates_sham_matched_exposure_and_restart(self):
        report = run_q6_toy_study(seed=7, per_profile=3, epochs_stage_a=24,
                                  epochs_stage_b=36, learning_rate=0.5)
        self.assertEqual(report["status"], "SOURCE_ONLY_TOY_Q6_UNQUALIFIED")
        self.assertEqual(report["cohort_counts"],
                         {"train_a": 12, "train_b": 12, "heldout": 12, "transfer": 12})
        self.assertNotEqual(report["trained_hash_initial"], report["trained_hash_after_stage_b"])
        self.assertEqual(report["trained_hash_after_stage_b"], report["fresh_reload_sha256"])
        self.assertEqual(report["zero_hash_initial"], report["zero_hash_after_stage_b"])
        self.assertEqual(report["matched_visits"], 12 * (24 + 36))
        self.assertEqual(report["zero_parameter_updates"], 0)
        self.assertGreater(report["trained_parameter_updates"], 0)
        self.assertGreater(report["inverted_parameter_updates"], 0)
        self.assertFalse(report["q6_independently_qualified"])
        self.assertFalse(report["graduation_authorized"])
        self.assertFalse(report["pretrained_learning_established"])
        for arm in ("trained", "zero_update", "inverted_labels"):
            self.assertEqual(set(report["scores"][arm]), {"initial", "after_a", "after_b"})
            for stage in report["scores"][arm].values():
                self.assertEqual(set(stage), {"heldout", "transfer"})
                self.assertTrue(all(0 <= value <= 12 for value in stage.values()))

    def test_default_fixed_schedule_converges_without_fabricating_stage_b_updates(self):
        report = run_q6_toy_study()
        self.assertGreater(report["trained_parameter_updates"], 0)
        self.assertEqual(report["matched_visits"], 24 * 160)
        self.assertFalse(report["q6_independently_qualified"])

    def test_runner_refuses_cross_seed_semantic_eval_replay(self):
        # A later fixed public seed can reuse an *earlier* algebraic problem.
        # The runner must admit a supplied exposure register before training,
        # and must not silently drop it at its internal validator call.
        previous = set()
        for seed in range(12):
            cohorts = construct_q6_cohorts(seed=seed, per_profile=6)
            previous.update(case.prompt for case in cohorts.heldout + cohorts.transfer)
        seed12 = construct_q6_cohorts(seed=12, per_profile=6)
        prior_keys = {
            semantic_case_key(case)
            for seed in range(12)
            for cohorts in (construct_q6_cohorts(seed=seed, per_profile=6),)
            for case in cohorts.heldout + cohorts.transfer
        }
        self.assertTrue(prior_keys & {
            semantic_case_key(case) for case in seed12.heldout + seed12.transfer
        })
        with patch.object(ToyLinearAdapter, "train",
                          side_effect=AssertionError("optimizer ran before exposure refusal")):
            with self.assertRaisesRegex(Q6StudyError, "previously exposed"):
                run_q6_toy_study(seed=12, previously_exposed_eval_prompts=previous)

    def test_runner_refuses_prior_eval_reused_in_later_train_before_optimizer(self):
        # A held-out arithmetic problem from one public seed can become TRAIN
        # on another seed, even when neither new evaluation partition collides.
        previous_cohorts = [
            construct_q6_cohorts(seed=seed, per_profile=6) for seed in range(10)
        ]
        previous = {
            case.prompt for cohorts in previous_cohorts
            for case in cohorts.heldout + cohorts.transfer
        }
        previous_semantics = {
            semantic_case_key(case) for cohorts in previous_cohorts
            for case in cohorts.heldout + cohorts.transfer
        }
        later = construct_q6_cohorts(seed=10, per_profile=6)
        self.assertTrue(previous_semantics & {
            semantic_case_key(case) for case in later.train_a + later.train_b
        })
        self.assertFalse(previous_semantics & {
            semantic_case_key(case) for case in later.heldout + later.transfer
        })
        with patch.object(ToyLinearAdapter, "train",
                          side_effect=AssertionError("optimizer ran before exposure refusal")):
            with self.assertRaisesRegex(Q6ExposureReuse, "previously exposed"):
                run_q6_toy_study(seed=10, per_profile=6,
                                 previously_exposed_eval_prompts=previous)

    def test_prior_train_exposure_blocks_later_public_eval_before_optimizer(self):
        previous = construct_q6_cohorts(seed=2, per_profile=6)
        prior_train = {c.prompt for c in previous.train_a + previous.train_b}
        later = construct_q6_cohorts(seed=7, per_profile=6)
        prior_keys = {semantic_case_key(c) for c in previous.train_a + previous.train_b}
        self.assertTrue(prior_keys & {
            semantic_case_key(c) for c in later.heldout + later.transfer
        })
        with patch.object(ToyLinearAdapter, "train",
                          side_effect=AssertionError("optimizer ran on exposed evaluation")):
            with self.assertRaisesRegex(Q6ExposureReuse, "previously trained"):
                run_q6_toy_study(seed=7, per_profile=6,
                                 previously_exposed_train_prompts=prior_train)

    def test_prior_training_replay_is_allowed_only_inside_current_train(self):
        current = construct_q6_cohorts(seed=9, per_profile=3)
        result = validate_q6_cohorts(
            current, per_profile=3,
            previously_exposed_train_prompts={current.train_a[0].prompt},
        )
        self.assertEqual(result["total"], 48)

    def test_runner_snapshots_train_history_before_mutating_generator(self):
        later = construct_q6_cohorts(seed=7, per_profile=6)
        former = construct_q6_cohorts(seed=2, per_profile=6)
        mutable = {c.prompt for c in former.train_a + former.train_b}

        def adversarial_construct(*, seed, per_profile):
            mutable.clear()
            return later

        with patch("labyrinth.q6_stratified_toy_study_r93.construct_q6_cohorts",
                   side_effect=adversarial_construct):
            with patch.object(ToyLinearAdapter, "train",
                              side_effect=AssertionError("optimizer ran after history mutation")):
                with self.assertRaisesRegex(Q6ExposureReuse, "previously trained"):
                    run_q6_toy_study(seed=7, per_profile=6,
                                     previously_exposed_train_prompts=mutable)

    def test_runner_rejects_wrong_train_history_pin_before_generation(self):
        former = construct_q6_cohorts(seed=2, per_profile=6)
        history = {former.train_a[0].prompt}
        # The evaluation-history domain must never authenticate TRAIN history.
        wrong_domain_digest = q6_exposure_history_sha256(history)
        with patch("labyrinth.q6_stratified_toy_study_r93.construct_q6_cohorts",
                   side_effect=AssertionError("generated before TRAIN history pin check")):
            with self.assertRaisesRegex(Q6StudyError, "train exposure history pin mismatch"):
                run_q6_toy_study(previously_exposed_train_prompts=history,
                                 expected_train_exposure_history_sha256=wrong_domain_digest)
            with self.assertRaisesRegex(Q6StudyError, "malformed train exposure history pin"):
                run_q6_toy_study(previously_exposed_train_prompts=history,
                                 expected_train_exposure_history_sha256="not-a-digest")
            with self.assertRaisesRegex(Q6StudyError, "canonical arithmetic prompt"):
                run_q6_toy_study(previously_exposed_train_prompts={"not arithmetic"})
            with self.assertRaisesRegex(Q6StudyError, "explicit exposed prompt set"):
                run_q6_toy_study(previously_exposed_train_prompts=[next(iter(history))])

    def test_runner_accepts_matching_train_history_pin_without_granting_custody(self):
        from labyrinth.q6_stratified_toy_study_r93 import q6_train_exposure_history_sha256

        previous = construct_q6_cohorts(seed=3, per_profile=3)
        history = {previous.train_a[0].prompt}
        digest = q6_train_exposure_history_sha256(history)
        report = run_q6_toy_study(
            seed=4, per_profile=3, epochs_stage_a=2, epochs_stage_b=2,
            previously_exposed_train_prompts=history,
            expected_train_exposure_history_sha256=digest,
        )
        self.assertEqual(report["supplied_train_exposure_history_sha256"], digest)
        self.assertEqual(report["supplied_train_exposure_prompt_count"], 1)
        self.assertTrue(report["caller_supplied_train_history_pin_checked"])
        self.assertFalse(report["history_custody_verified"])

    def test_runner_exports_complete_public_training_exposure_non_authorizing(self):
        from labyrinth.q6_stratified_toy_study_r93 import q6_train_exposure_history_sha256

        report = run_q6_toy_study(seed=4, per_profile=3,
                                  epochs_stage_a=2, epochs_stage_b=2)
        cohort = construct_q6_cohorts(seed=4, per_profile=3)
        expected = [c.prompt for c in cohort.train_a + cohort.train_b]
        self.assertEqual(report["trained_prompts"], expected)
        self.assertEqual(report["training_prompt_sha256"], [
            sha256(prompt.encode("ascii")).hexdigest() for prompt in expected
        ])
        self.assertNotEqual(q6_train_exposure_history_sha256(set(expected)),
                            q6_exposure_history_sha256(set(expected)))
        self.assertEqual(report["supplied_train_exposure_prompt_count"], 0)
        self.assertFalse(report["history_custody_verified"])
        self.assertFalse(report["q6_independently_qualified"])

    def test_runner_exports_public_eval_prompt_exposure_for_next_seed(self):
        first = run_q6_toy_study(seed=4, per_profile=3)
        exposures = first["evaluated_prompts"]
        self.assertEqual(len(exposures), 24)
        self.assertEqual(len(first["evaluation_prompt_sha256"]), 24)
        self.assertEqual(len(set(exposures)), 24)
        with self.assertRaisesRegex(Q6StudyError, "previously exposed"):
            run_q6_toy_study(seed=4, per_profile=3,
                             previously_exposed_eval_prompts=set(exposures))

    def test_runner_snapshots_history_before_caller_mutates_during_generation(self):
        cohorts = construct_q6_cohorts(seed=7, per_profile=3)
        mutable_history = {cohorts.heldout[0].prompt}

        def adversarial_construct(*, seed, per_profile):
            mutable_history.clear()
            return cohorts

        with patch("labyrinth.q6_stratified_toy_study_r93.construct_q6_cohorts",
                   side_effect=adversarial_construct):
            with patch.object(ToyLinearAdapter, "train",
                              side_effect=AssertionError("optimizer ran after replay")):
                with self.assertRaisesRegex(Q6StudyError, "previously exposed"):
                    run_q6_toy_study(seed=7, per_profile=3,
                                     previously_exposed_eval_prompts=mutable_history)

    def test_runner_rejects_malformed_history_before_cohort_generation(self):
        with patch("labyrinth.q6_stratified_toy_study_r93.construct_q6_cohorts",
                   side_effect=AssertionError("cohort generated before history validation")):
            with self.assertRaisesRegex(Q6StudyError, "canonical arithmetic prompt"):
                run_q6_toy_study(previously_exposed_eval_prompts={"invalid"})

    def test_runner_rejects_external_history_pin_mismatch_before_generation(self):
        valid = construct_q6_cohorts(seed=3, per_profile=3).heldout[0].prompt
        expected = q6_exposure_history_sha256({valid})
        changed = construct_q6_cohorts(seed=4, per_profile=3).heldout[0].prompt
        self.assertNotEqual(expected, q6_exposure_history_sha256({changed}))
        with patch("labyrinth.q6_stratified_toy_study_r93.construct_q6_cohorts",
                   side_effect=AssertionError("cohort generated before pin check")):
            with self.assertRaisesRegex(Q6StudyError, "exposure history pin mismatch"):
                run_q6_toy_study(previously_exposed_eval_prompts={changed},
                                 expected_exposure_history_sha256=expected)

    def test_typed_reuse_and_bounded_history_exhaustion(self):
        prior = construct_q6_cohorts(seed=7, per_profile=3).heldout[0].prompt
        with self.assertRaises(Q6ExposureReuse):
            run_q6_toy_study(seed=7, per_profile=3,
                             previously_exposed_eval_prompts={prior})
        # The public arithmetic grammar has enough valid distinct examples to
        # hit the bounded history limit without mocks or fabricated authority.
        prompts = {
            f"Calculate {a} + {b}.\nRespond with exactly one line: ANSWER: <integer>."
            for a in range(1, 502) for b in range(1, 201)
        }
        self.assertGreater(len(prompts), 100_000)
        with self.assertRaisesRegex(Q6ExposureExhaustion, "history exhausted"):
            q6_exposure_history_sha256(prompts)

    def test_runner_reports_caller_pinned_source_digest_without_authority(self):
        history = {construct_q6_cohorts(seed=3, per_profile=3).heldout[0].prompt}
        digest = q6_exposure_history_sha256(history)
        report = run_q6_toy_study(seed=4, per_profile=3, epochs_stage_a=2,
                                  epochs_stage_b=2,
                                  previously_exposed_eval_prompts=history,
                                  expected_exposure_history_sha256=digest)
        self.assertEqual(report["supplied_exposure_history_sha256"], digest)
        self.assertEqual(report["supplied_exposure_prompt_count"], 1)
        self.assertTrue(report["caller_supplied_history_pin_checked"])
        self.assertFalse(report["q6_independently_qualified"])


if __name__ == "__main__":
    unittest.main()
