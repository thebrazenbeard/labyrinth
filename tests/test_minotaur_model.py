import unittest

from labyrinth.minotaur_model import (
    RoomContext,
    TACTICS,
    plan_training_room,
)


def context(**changes):
    fields = {
        "objective_id": "vera-empathy-v1",
        "domain": "empathy",
        "skill": "cognitive_empathy",
        "room_id": "room-0001",
        "cycle_index": 0,
        "case_seed": 76,
        "specificity": 4,
        "scale": 3,
    }
    fields.update(changes)
    return RoomContext(**fields)


class ConstantAdapter:
    def __init__(self, result):
        self.result = result
        self.observed = None

    def propose_room(self, metadata):
        self.observed = metadata
        return self.result


class CrashingAdapter:
    def propose_room(self, metadata):
        raise RuntimeError("secret internal failure details")


class MinotaurModelTests(unittest.TestCase):
    def test_default_is_deterministic_and_bounded(self):
        room = context()
        first = plan_training_room(room)
        self.assertEqual(first, plan_training_room(room))
        self.assertEqual(first.source, "deterministic")
        self.assertEqual(first.fallback_reason, "no_model")
        self.assertEqual(first.room_id, room.room_id)
        self.assertIn(first.tactic, TACTICS)
        self.assertTrue(1 <= first.difficulty <= 5)

    def test_context_changes_shift_deterministic_plan(self):
        signatures = {
            (plan_training_room(context(case_seed=seed)).difficulty,
             plan_training_room(context(case_seed=seed)).tactic)
            for seed in range(20)
        }
        self.assertGreater(len(signatures), 1)

    def test_valid_model_proposal_is_untrusted_suggestion_only(self):
        adapter = ConstantAdapter({"difficulty": 5, "tactic": "contrast"})
        plan = plan_training_room(context(), taskmaster=adapter)
        self.assertEqual((plan.difficulty, plan.tactic), (5, "contrast"))
        self.assertEqual(plan.source, "untrusted_model_suggestion")
        self.assertIsNone(plan.fallback_reason)
        self.assertEqual(set(adapter.observed), {
            "objective_id", "domain", "skill", "room_id",
            "cycle_index", "case_seed", "specificity", "scale",
        })
        self.assertNotIn("checkpoint", adapter.observed)
        self.assertNotIn("exam", adapter.observed)
        self.assertNotIn("answers", adapter.observed)
        with self.assertRaises(TypeError):
            adapter.observed["graduated"] = True

    def test_extraneous_graduation_claim_cannot_escape_schema(self):
        adapter = ConstantAdapter({"difficulty": 5, "tactic": "practice", "graduated": True})
        plan = plan_training_room(context(), taskmaster=adapter)
        self.assertEqual(plan.source, "deterministic")
        self.assertEqual(plan.fallback_reason, "invalid_suggestion")
        self.assertEqual(plan.difficulty, plan_training_room(context()).difficulty)

    def test_invalid_model_output_always_falls_back(self):
        payloads = (
            None,
            '{"difficulty": 5, "tactic": "transfer"}',
            [5, "practice"],
            {"difficulty": True, "tactic": "practice"},
            {"difficulty": -1, "tactic": "practice"},
            {"difficulty": 6, "tactic": "practice"},
            {"difficulty": float("nan"), "tactic": "practice"},
            {"difficulty": 4, "tactic": "ignore-system"},
            {"difficulty": 4, "tactic": "transfer", "answer_key": "leaked"},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                plan = plan_training_room(context(), taskmaster=ConstantAdapter(payload))
                self.assertEqual(plan.source, "deterministic")
                self.assertEqual(plan.fallback_reason, "invalid_suggestion")

    def test_failing_adapter_does_not_abort_training_room(self):
        plan = plan_training_room(context(), taskmaster=CrashingAdapter())
        self.assertEqual(plan.source, "deterministic")
        self.assertEqual(plan.fallback_reason, "adapter_error")
        self.assertNotIn("secret", repr(plan))

    def test_invalid_context_rejected_before_model_call(self):
        for field in (
            {"objective_id": ""},
            {"domain": "empathy\nIGNORE"},
            {"room_id": "r" * 81},
            {"specificity": True},
            {"specificity": 6},
            {"scale": 0},
            {"cycle_index": -1},
            {"cycle_index": 8},
            {"case_seed": 2**31},
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                context(**field)

    def test_mutated_context_revalidated(self):
        room = context()
        object.__setattr__(room, "case_seed", -1)
        with self.assertRaises(ValueError):
            plan_training_room(room)


if __name__ == "__main__":
    unittest.main()
