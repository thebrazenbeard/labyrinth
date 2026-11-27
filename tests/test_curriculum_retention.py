import unittest
from labyrinth.curriculum import CurriculumSpec, CurriculumState, TrainingObservation, next_training_case, record_training_result

class CurriculumRetentionTests(unittest.TestCase):
    def test_caller_mutation_does_not_reorder_validated_levels(self):
        levels = ["easy", "hard"]
        spec = CurriculumSpec(levels=levels, window_size=2, distinct_challenges_required=2)
        catalog = {"easy": ("a", "b"), "hard": ("c", "d")}
        first = next_training_case(spec, CurriculumState(), catalog, run_seed=17)
        levels.reverse()
        self.assertEqual(next_training_case(spec, CurriculumState(), catalog, run_seed=17), first)

    def test_history_is_detached_and_can_continue_after_caller_mutation(self):
        original = TrainingObservation("a", 1, "easy", 1.0)
        history = [original]
        state = CurriculumState(observations=history)
        history.clear()
        self.assertEqual(state.observations, (original,))
        spec = CurriculumSpec(levels=("easy", "hard"), window_size=2, distinct_challenges_required=2)
        advanced, promoted = record_training_result(spec, state, TrainingObservation("b", 2, "easy", 1.0))
        self.assertTrue(promoted)
        self.assertEqual(advanced.level_index, 1)
        self.assertEqual(state.observations, (original,))

    def test_reject_string_as_level_sequence(self):
        with self.assertRaises(ValueError):
            CurriculumSpec(levels="ab", window_size=2, distinct_challenges_required=2)

    def test_reject_boolean_state_index(self):
        with self.assertRaises(ValueError):
            CurriculumState(level_index=False)
    def test_reject_boolean_schema_version(self):
        with self.assertRaises(ValueError):
            CurriculumState.from_dict({"schema_version": True, "level_index": 0, "observations": []})
