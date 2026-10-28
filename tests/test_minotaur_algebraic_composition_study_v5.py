"""Explicitly bounded neural-addition composition study configuration."""
import unittest
from labyrinth.minotaur_algebraic_composition_study_v5 import (
    CompositionStudyConfig, _model_visible_rows,
)

class CompositionStudyContracts(unittest.TestCase):
    def test_only_addition_training_rows(self):
        from labyrinth.minotaur_algebraic_data_v5 import atomic_split
        split=atomic_split(718,train_fraction=.8)
        add=split.op_ids[0]
        rows=_model_visible_rows(split.train,add)
        self.assertGreater(len(rows),5000)
        self.assertTrue(all(r[0]==add for r in rows))
        self.assertFalse({r[:3] for r in rows} & {r[:3] for r in split.holdout})

    def test_config_bounded(self):
        CompositionStudyConfig(seed=4).validate()
        with self.assertRaises(ValueError):
            CompositionStudyConfig(steps=0).validate()
        with self.assertRaises(ValueError):
            CompositionStudyConfig(seed=True).validate()
