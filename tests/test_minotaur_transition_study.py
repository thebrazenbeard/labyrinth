"""Contract tests for V4 training study; not neural qualification."""
from __future__ import annotations
import unittest
from labyrinth.minotaur_transition_study import TransitionStudyConfig, _score
from labyrinth.minotaur_transition_curriculum import CurriculumConfig, build_transition_splits
from labyrinth.minotaur_transition_v4 import (
    TransitionConfig, build_transition_model, encode_transition_cases,
)


class TransitionStudyTests(unittest.TestCase):
    def test_bounded_config_and_causal_structure_holdout(self):
        with self.assertRaises(ValueError):
            TransitionStudyConfig(train_cases=31).validate()
        with self.assertRaises(ValueError):
            TransitionStudyConfig(epochs=51).validate()
        c = TransitionStudyConfig(train_cases=32, dev_cases=32, transfer_cases=32, epochs=1)
        c.validate()
        splits = build_transition_splits(CurriculumConfig(
            seed=c.data_seed, train_cases=c.train_cases, dev_cases=c.dev_cases,
            transfer_cases=c.transfer_cases))
        structures = [{case.canonical_sequence for case in x.cases} for x in splits.values()]
        for i, a in enumerate(structures):
            for b in structures[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(sum(case.first_error_step is None for case in splits["train"].cases), 16)

    def test_real_random_model_scores_without_oracle_predictions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Neural CI provides optional Torch")
        torch.set_num_threads(min(2, torch.get_num_threads()))
        splits = build_transition_splits(CurriculumConfig(
            seed=333, train_cases=16, dev_cases=16, transfer_cases=16))
        cfg=TransitionConfig(seed=333, width=16)
        m=build_transition_model(cfg)
        train={c.canonical_sequence for c in splits["train"].cases}
        report=_score(m, splits["dev"].cases,cfg,torch,tuple(train))
        self.assertEqual(report["disposition"],"SCORED")
        self.assertEqual(report["qualification"],"NOT_ESTABLISHED")
        self.assertEqual(report["metrics"]["cases"],16)
        self.assertIn("per_step_state_accuracy",report["metrics"])
        self.assertFalse(report["overlaps"]["structural"])

if __name__=="__main__":
    unittest.main()
