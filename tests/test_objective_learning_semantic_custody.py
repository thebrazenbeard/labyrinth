"""Adversarial arithmetic-alias checks for the development-only toy loop."""

from dataclasses import replace
from types import SimpleNamespace
import re
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.journey_challenges import PreparedTrainingRoom
from labyrinth.maze import Challenge
from labyrinth.objective_learning_loop_r92 import (
    _check_exam, _check_train, _development_cases, arithmetic_objective,
    run_objective_learning_loop,
)


_INSTRUCTION = "Respond with exactly one line: ANSWER: <integer>."


def _case(case_id: str, prompt: str, answer: str, *, split: str) -> Challenge:
    return Challenge(
        id=case_id, kind="arithmetic", tier=1, split=split,
        prompt=f"Calculate {prompt}.\n{_INSTRUCTION}", answer=answer,
        metadata=(("generator", "labyrinth-v1"),),
    )


def _state(room_count: int = 1):
    return SimpleNamespace(
        maze=SimpleNamespace(rooms=(object(),) * room_count),
        used_training_case_ids=(), sealed_exam_case_ids=(),
        used_training_prompt_sha256=(), sealed_exam_prompt_sha256=(),
    )


class ObjectiveLoopSemanticCustodyTests(unittest.TestCase):
    def test_forged_nonstring_gold_comparison_cannot_bypass_train_or_exam(self):
        class ForgedGold:
            def __ne__(self, other):
                return False

        forged = _case("train-forged", "12 + 34", ForgedGold(), split="train")
        valid = _case("train-valid", "50 - 10", "40", split="train")
        prepared = (
            PreparedTrainingRoom("a", "arithmetic", forged),
            PreparedTrainingRoom("b", "arithmetic", valid),
        )
        with self.assertRaisesRegex(ValueError, "canonical Challenge and string gold"):
            _check_train(prepared, _state(2))

        normal_train = PreparedTrainingRoom(
            "a", "arithmetic", _case("train-ok", "50 - 10", "40", split="train"),
        )
        forged_eval = _case("eval-forged", "12 + 34", ForgedGold(), split="eval")
        with self.assertRaisesRegex(ValueError, "canonical Challenge and string gold"):
            _check_exam((("heldout", forged_eval),), (normal_train,), _state())

    def test_train_and_heldout_commutative_alias_rejected(self):
        training = PreparedTrainingRoom(
            "room", "arithmetic", _case("train-a", "12 + 34", "46", split="train"),
        )
        heldout = _case("eval-a", "34 + 12", "46", split="eval")
        with self.assertRaisesRegex(ValueError, "semantic"):
            _check_exam((("heldout", heldout),), (training,), _state())

    def test_heldout_and_transfer_commutative_alias_rejected(self):
        training = PreparedTrainingRoom(
            "room", "arithmetic", _case("train-b", "19 - 7", "12", split="train"),
        )
        heldout = _case("eval-b", "12 + 34", "46", split="eval")
        transfer = _case("eval-c", "34 + 12", "46", split="eval")
        with self.assertRaisesRegex(ValueError, "semantic"):
            _check_exam((("heldout", heldout), ("transfer", transfer)),
                        (training,), _state())

    def test_subtraction_operand_order_is_distinct(self):
        training = PreparedTrainingRoom(
            "room", "arithmetic", _case("train-c", "12 - 34", "-22", split="train"),
        )
        heldout = _case("eval-d", "34 - 12", "22", split="eval")
        _check_exam((("heldout", heldout),), (training,), _state())

    def test_previous_cycle_arithmetic_alias_rejected(self):
        training = PreparedTrainingRoom(
            "room", "arithmetic", _case("train-d", "19 - 7", "12", split="train"),
        )
        heldout = _case("eval-e", "34 + 12", "46", split="eval")
        with self.assertRaisesRegex(ValueError, "semantic"):
            _check_exam((("heldout", heldout),), (training,), _state(),
                        forbidden_semantics={('+', 12, 34)})

    def test_train_internal_commutative_alias_rejected(self):
        training = (
            PreparedTrainingRoom("a", "arithmetic", _case("train-e", "12 + 34", "46", split="train")),
            PreparedTrainingRoom("b", "arithmetic", _case("train-f", "34 + 12", "46", split="train")),
            PreparedTrainingRoom("c", "arithmetic", _case("train-g", "50 - 10", "40", split="train")),
        )
        with self.assertRaisesRegex(ValueError, "semantic"):
            _check_train(training, _state(3))

    def test_second_cycle_rejects_semantic_reexposure_from_first_exam(self):
        objective = arithmetic_objective()
        first_exam = _development_cases(objective.scope, 0, 8)
        source = next(
            case for _, case in first_exam
            if (match := re.match(r"Calculate (\d+) \+ (\d+)\.", case.prompt))
            and match[1] != match[2]
        )
        match = re.match(r"Calculate (\d+) \+ (\d+)\.", source.prompt)
        assert match is not None
        alias = replace(
            source, id=source.id + "-semantic-alias",
            prompt=f"Calculate {match[2]} + {match[1]}.\n{_INSTRUCTION}",
        )

        def contaminated_development(scope, cycle_index, count):
            original = _development_cases(scope, cycle_index, count)
            return ((("heldout", alias),) + original[1:]
                    if cycle_index == 1 else original)

        with tempfile.TemporaryDirectory() as directory:
            with patch("labyrinth.objective_learning_loop_r92._development_cases",
                       side_effect=contaminated_development):
                with self.assertRaisesRegex(ValueError, "semantic"):
                    run_objective_learning_loop(objective, checkpoint_dir=directory)


if __name__ == "__main__":
    unittest.main()
