"""R28 public toy reward-hacking falsifiers; no private Minotaur answers."""

from dataclasses import replace
import unittest

from labyrinth.r28_reward_hacking import (
    ChallengeObservation, FrozenChallengePlan, ChallengeSpec,
    audit_reward_shortcuts, build_public_toy_challenge,
)
from labyrinth.trainingday_objective_metamorphic import (
    FormalObjectiveContract, FormalSkillCriterion,
)


CONTRACT = FormalObjectiveContract(
    "toy-mastery-v1", "skill_mastery",
    (FormalSkillCriterion("arithmetic", "exact_result", 2, 2),
     FormalSkillCriterion("source_priority", "correct_rank", 2, 1)),
    ("private_ood",), "a" * 64, "b" * 64, "c" * 64,
    "Toy public practice only",
)


def fixture():
    plan = build_public_toy_challenge(CONTRACT, seed="r28-reward-attack", per_skill=2)
    observations = tuple(
        ChallengeObservation(
            case_id=case.case_id,
            objective_id=plan.objective_id,
            skill_id=case.skill_id,
            prompt_sha256=case.prompt_sha256,
            correct=True,
            format_valid=True,
            evidence_source="PUBLIC_TOY_ORACLE",
        ) for case in plan.cases
    )
    selected_ids = tuple(case.case_id for case in plan.cases)
    return plan, observations, selected_ids


def audit(plan=None, observations=None, selected_ids=None, contract=CONTRACT):
    good_plan, good_obs, good_selection = fixture()
    return audit_reward_shortcuts(
        contract, good_plan if plan is None else plan,
        good_obs if observations is None else observations,
        good_selection if selected_ids is None else selected_ids,
    )


class RewardHackingToyTests(unittest.TestCase):
    def test_fixed_public_generation_replays_exactly_and_changes_by_seed(self):
        a, _, _ = fixture()
        self.assertEqual(a, build_public_toy_challenge(CONTRACT, "r28-reward-attack", 2))
        other_seed = build_public_toy_challenge(CONTRACT, "another-seed", 2)
        self.assertNotEqual(a, other_seed)
        self.assertTrue({c.case_id for c in a.cases}.isdisjoint(
            {c.case_id for c in other_seed.cases}))
        self.assertEqual(len({c.case_id for c in a.cases}), 4)
        self.assertEqual(len({c.prompt_sha256 for c in a.cases}), 4)
        reordered = replace(CONTRACT, skills=tuple(reversed(CONTRACT.skills)))
        self.assertEqual(a, build_public_toy_challenge(reordered, "r28-reward-attack", 2))

    def test_clean_toy_mastery_stays_non_authorizing(self):
        decision = audit()
        self.assertEqual(decision.status, "PUBLIC_TOY_ONLY")
        self.assertTrue(decision.public_toy_mastery)
        self.assertFalse(decision.graduation_authorized)
        self.assertFalse(decision.promotion_allowed)
        self.assertFalse(decision.independent_exam_verified)

    def test_format_only_correctness_is_not_mastery(self):
        _, observations, _ = fixture()
        spoof = (replace(observations[0], correct=False, format_valid=True),) + observations[1:]
        self.assertEqual(audit(observations=spoof).status, "PUBLIC_TOY_ONLY")
        self.assertFalse(audit(observations=spoof).public_toy_mastery)
        heur = (replace(observations[0], evidence_source="FORMAT_ONLY"),) + observations[1:]
        self.assertEqual(audit(observations=heur).status, "REWARD_HACK_BLOCKED")
        self.assertIn("FORMAT_HEURISTIC_OR_UNVERIFIED_ORACLE", audit(observations=heur).findings)

    def test_hint_template_and_reused_training_example_each_block(self):
        _, observations, _ = fixture()
        for flag, expected in (
            ("hint_embedded", "PROMPT_HINT_LEAKAGE"),
            ("grading_template_imitation", "GRADING_TEMPLATE_IMITATION"),
            ("training_example_reused", "TRAINING_EXEMPLAR_REUSE"),
        ):
            with self.subTest(flag=flag):
                modified = (replace(observations[0], **{flag: True}),) + observations[1:]
                v = audit(observations=modified)
                self.assertEqual(v.status, "REWARD_HACK_BLOCKED")
                self.assertIn(expected, v.findings)
                self.assertIsNone(v.public_toy_mastery)
                self.assertFalse(v.graduation_authorized)

    def test_exact_training_prompt_overlap_blocks_even_without_claimed_reuse(self):
        plan, _, _ = fixture()
        overlapping_plan = build_public_toy_challenge(
            CONTRACT, "r28-reward-attack", 2,
            training_prompt_sha256=(plan.cases[0].prompt_sha256,),
        )
        v = audit(plan=overlapping_plan)
        self.assertEqual(v.status, "REWARD_HACK_BLOCKED")
        self.assertIn("TRAINING_PROMPT_OVERLAP", v.findings)

    def test_curriculum_reselection_and_reordering_block(self):
        _, _, selected = fixture()
        for changed in (selected[:-1], selected[1:] + selected[:1], selected + ("unexpected",),
                        (selected[0],) * len(selected)):
            with self.subTest(selected=changed):
                v = audit(selected_ids=changed)
                self.assertEqual(v.status, "CURRICULUM_SELECTION_CHANGED")
                self.assertFalse(v.graduation_authorized)

    def test_mixed_objective_and_case_mutation_block(self):
        plan, observations, _ = fixture()
        variants = (
            replace(plan, objective_id="other-objective"),
            replace(plan, evaluator_sha256="d" * 64),
            replace(plan, objective_fingerprint="e" * 64),
            replace(plan, cases=(replace(plan.cases[0], skill_id="source_priority"),) + plan.cases[1:]),
        )
        for p in variants:
            with self.subTest(plan=p):
                self.assertEqual(audit(plan=p).status, "PLAN_SCOPE_MISMATCH")
        for o in (
            replace(observations[0], objective_id="other-objective"),
            replace(observations[0], prompt_sha256="f" * 64),
            replace(observations[0], skill_id="source_priority"),
        ):
            with self.subTest(observation=o):
                self.assertEqual(audit(observations=(o,) + observations[1:]).status,
                                 "REWARD_HACK_BLOCKED")

    def test_duplicate_observations_and_split_reassignment_block(self):
        _, obs, _ = fixture()
        self.assertEqual(audit(observations=(obs[0], obs[0]) + obs[2:]).status,
                         "REWARD_HACK_BLOCKED")
        private = (replace(obs[0], split="private_exam"),) + obs[1:]
        self.assertEqual(audit(observations=private).status, "REWARD_HACK_BLOCKED")

    def test_malformed_caller_controlled_containers_reject_boundedly(self):
        plan, obs, selected = fixture()
        self.assertEqual(audit(observations=list(obs)).status, "MALFORMED_CHALLENGE")
        self.assertEqual(audit(selected_ids=list(selected)).status, "MALFORMED_CHALLENGE")
        self.assertEqual(audit(plan=replace(plan, cases=plan.cases * 200)).status,
                         "MALFORMED_CHALLENGE")
        self.assertEqual(audit(observations=(replace(obs[0], correct=1),) + obs[1:]).status,
                         "MALFORMED_CHALLENGE")

    def test_generator_rejects_noncanonical_and_oversize_inputs(self):
        for seed, per_skill in (("", 2), ("\U0001f400", 2), ("good\x00bad", 2),
                                ("good", True), ("good", 0), ("good", 1), ("good", 400),
                                ("good", 2.5)):
            with self.subTest(seed=seed, per_skill=per_skill):
                with self.assertRaises(ValueError):
                    build_public_toy_challenge(CONTRACT, seed, per_skill)

        impossible = replace(CONTRACT, skills=(
            FormalSkillCriterion("large-skill", "exact", 300, 300),
        ))
        with self.assertRaises(ValueError):
            build_public_toy_challenge(impossible, "good", 1)

    def test_nonprimitive_plan_fields_fail_closed_without_overloaded_equality(self):
        plan, _, _ = fixture()

        class BadEqual:
            def __eq__(self, other):
                raise AssertionError("untrusted equality executed")

        for field in ("objective_id", "objective_fingerprint", "evaluator_sha256",
                      "conditions_sha256", "commitment_sha256"):
            with self.subTest(field=field):
                verdict = audit(plan=replace(plan, **{field: BadEqual()}))
                self.assertEqual(verdict.status, "MALFORMED_CHALLENGE")


if __name__ == "__main__":
    unittest.main()
