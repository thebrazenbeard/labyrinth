import json
import unittest
from labyrinth.curriculum import CurriculumSpec, CurriculumState, TrainingObservation, next_training_case, record_training_result

class StringChild(str):
    pass

class FloatChild(float):
    pass

class ListChild(list):
    pass

class ObservationChild(TrainingObservation):
    pass

class BoundaryReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.spec = CurriculumSpec(levels=['easy', 'hard'], window_size=2, distinct_challenges_required=2)
        self.catalog = {'easy': ['a', 'b'], 'hard': ['c', 'd']}

    def test_plain_lists_and_json_replay_remain_supported(self):
        levels = ['easy', 'hard']
        spec = CurriculumSpec(levels=levels, window_size=2, distinct_challenges_required=2)
        obs = TrainingObservation('a', 1, 'easy', 1.0)
        history = [obs]
        state = CurriculumState(observations=history)
        levels.reverse()
        history.clear()
        self.assertEqual(spec.levels, ('easy', 'hard'))
        restored = CurriculumState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(state, restored)
        self.assertEqual(next_training_case(spec, state, self.catalog, run_seed=7), next_training_case(spec, restored, self.catalog, run_seed=7))
        advanced, promoted = record_training_result(spec, restored, TrainingObservation('b', 2, 'easy', 1))
        self.assertTrue(promoted)
        self.assertEqual(advanced.level_index, 1)

    def test_reject_nonprimitive_plan_inputs(self):
        for kwargs in ({'levels': ListChild(['easy','hard'])}, {'levels': [StringChild('easy'),'hard']}, {'promotion_score': FloatChild(.75)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CurriculumSpec(**kwargs)

    def test_reject_nonprimitive_observation_inputs(self):
        for args in ((StringChild('a'),1,'easy',1.), ('a',1,StringChild('easy'),1.), ('a',1,'easy',FloatChild(1.)), ('a',1,'easy',1.,StringChild('train'))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TrainingObservation(*args)

    def test_reject_nonprimitive_history(self):
        valid = TrainingObservation('a',1,'easy',1.)
        for value in (ListChild([valid]), [object()], [ObservationChild('a',1,'easy',1.)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CurriculumState(observations=value)

    def test_revalidate_forged_stored_observation(self):
        obs = TrainingObservation('a',1,'easy',1.)
        state = CurriculumState(observations=[obs])
        object.__setattr__(obs, 'score', 2.)
        with self.assertRaises(ValueError):
            next_training_case(self.spec, state, self.catalog, run_seed=7)

    def test_revalidate_forged_new_observation(self):
        obs = TrainingObservation('a',1,'easy',1.)
        object.__setattr__(obs, 'score', 2.)
        with self.assertRaises(ValueError):
            record_training_result(self.spec, CurriculumState(), obs)

    def test_revalidate_retained_plan_without_renormalizing(self):
        object.__setattr__(self.spec, 'levels', ['easy','hard'])
        with self.assertRaises(ValueError):
            next_training_case(self.spec, CurriculumState(), self.catalog, run_seed=7)

    def test_reject_nonprimitive_catalog_identifier(self):
        with self.assertRaises(ValueError):
            next_training_case(self.spec, CurriculumState(), {'easy':[StringChild('a'),'b']}, run_seed=7)

    def test_reject_forged_state_shape(self):
        state = CurriculumState()
        object.__setattr__(state, 'level_index', False)
        with self.assertRaises(ValueError):
            next_training_case(self.spec, state, self.catalog, run_seed=7)

if __name__ == '__main__':
    unittest.main()
