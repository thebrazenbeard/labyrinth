"""CPU/source contracts for scoped HF program TRAIN versus public heldout.

These tests never load a pretrained model or execute neural optimizer steps.
They do not make public development cases an independent final examination.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import unittest

from labyrinth.hf_pilot import (
    PilotConfig, _arithmetic_semantic_commitments, _arithmetic_history_witnesses,
    _arithmetic_case_binding_records, _arithmetic_case_binding_root,
    _resume_training_cases, _scoped_training_cases,
)
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_reserved_structural_transfer import _independent_expected_answer


class HFProgramScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = TrainingScopeSpec(
            objective_id="program-scope-fixture", domain="reasoning",
            skills=("program_reasoning",), specificity=3, scale=3,
            max_cycles=3, seed=71,
        )
        self.config = PilotConfig(
            model_path="C:/fixture/base", output_dir="C:/fixture/run",
            seed=89, train_count=4, eval_count=8, tiers=(4,), steps=4,
        )

    def _cases(self, cycle: int):
        return _scoped_training_cases(self.config, self.scope, cycle, ())

    def test_train_semantic_programs_match_distinct_public_structural_holdouts(self):
        train, heldout, plan = self._cases(0)
        self.assertEqual(len(train), self.scope.total_chambers)
        self.assertEqual(plan.cycle_index, 0)
        self.assertEqual({case.split for case in train}, {"train"})
        self.assertEqual({case.split for case in heldout}, {"eval"})
        self.assertEqual({case.tier for case in train + heldout}, {4})
        self.assertEqual({case.kind for case in train + heldout}, {"register_program"})
        self.assertEqual({dict(c.metadata)["profile"] for c in train}, {"train"})
        self.assertEqual({dict(c.metadata)["semantic_selector"] for c in train},
                         {"labyrinth.program-focus.v1"})
        self.assertEqual({dict(c.metadata)["profile"] for c in heldout},
                         {"structural_holdout"})
        train_structures = {dict(case.metadata)["structure"] for case in train}
        heldout_structures = {dict(case.metadata)["structure"] for case in heldout}
        self.assertFalse(train_structures & heldout_structures)
        self.assertEqual({len(s.split(";")) for s in train_structures}, {4})
        self.assertEqual({len(s.split(";")) for s in heldout_structures}, {5, 6})
        self.assertEqual({case.id for case in train} & {case.id for case in heldout}, set())
        self.assertEqual({case.prompt for case in train} & {case.prompt for case in heldout}, set())
        for case in heldout:
            with self.subTest(case=case.id):
                self.assertEqual(str(_independent_expected_answer(case)), case.answer)

    def test_cycle_regeneration_replays_predecessor_and_rejects_old_prompts(self):
        prior_train, prior_eval, _ = self._cases(0)
        witnesses = _arithmetic_history_witnesses(prior_train + prior_eval)
        bindings = _arithmetic_case_binding_records(witnesses)
        prior_report = {
            "config": {**asdict(self.config), "tiers": list(self.config.tiers)},
            "training_case_ids": [case.id for case in prior_train],
            "heldout_case_ids": [case.id for case in prior_eval],
            "resumed_journey": {
                "scope": self.scope.to_dict(), "scope_fingerprint": self.scope.fingerprint,
                "cycle_index": 0, "weak_skills": [],
                "historical_case_ids": sorted(case.id for case in prior_train + prior_eval),
                "historical_prompt_sha256": sorted(hashlib.sha256(case.prompt.encode()).hexdigest()
                                                   for case in prior_train + prior_eval),
                "historical_arithmetic_sha256": sorted(
                    _arithmetic_semantic_commitments(prior_train + prior_eval)
                ),
                "historical_arithmetic_witnesses_v1": witnesses,
                "historical_arithmetic_case_bindings_v1": bindings,
                "historical_arithmetic_binding_root_v1": _arithmetic_case_binding_root(bindings),
            },
        }
        train, heldout, _ = _resume_training_cases(
            self.config, self.scope, 1, (), prior_report,
        )
        self.assertFalse({case.id for case in prior_train + prior_eval} &
                         {case.id for case in train + heldout})
        self.assertFalse({case.prompt for case in prior_train + prior_eval} &
                         {case.prompt for case in train + heldout})
        prior_report["resumed_journey"]["historical_prompt_sha256"].append(
            hashlib.sha256(train[0].prompt.encode()).hexdigest()
        )
        # A cycle-zero predecessor cannot legitimately know any successor
        # prompt. Reject its forged history before the later overlap check.
        with self.assertRaisesRegex(ValueError, "initial predecessor has unexplained"):
            _resume_training_cases(self.config, self.scope, 1, (), prior_report)

    def test_wrong_tier_or_unsupported_goal_fails_closed(self):
        wrong_tier = PilotConfig(**{**asdict(self.config), "tiers": (3,)})
        with self.assertRaisesRegex(ValueError, "tiers must exactly match"):
            _scoped_training_cases(wrong_tier, self.scope, 0, ())
        unsupported = TrainingScopeSpec(
            objective_id="empathy-not-implemented", domain="social",
            skills=("empathy",), specificity=1, scale=3, max_cycles=2, seed=8,
        )
        with self.assertRaisesRegex(ValueError, "matching heldout diagnostic"):
            _scoped_training_cases(self.config, unsupported, 0, ())


if __name__ == "__main__":
    unittest.main()
