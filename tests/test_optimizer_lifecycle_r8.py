"""Dependency-free contracts for explicitly owned HF optimizer continuation.

The fakes exercise orchestration, not physical Torch checkpoint recovery.
"""

from types import SimpleNamespace
from hashlib import sha256
import random
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import (
    PilotConfig, _train_steps, _training_visit_exposures, _training_visit_indices,
)
from labyrinth.maze import Challenge


class Parameter:
    requires_grad = True
    device = SimpleNamespace(type="cpu")


class Loss:
    def backward(self):
        pass

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return 1.0


class Model:
    def __init__(self):
        self.param = Parameter()

    def parameters(self):
        return iter((self.param,))

    def train(self):
        pass

    def named_buffers(self):
        return iter(())

    def __call__(self, *, input_ids, labels):
        return SimpleNamespace(loss=Loss())


class AdamW:
    created = 0

    def __init__(self, params, *, lr):
        self.param_groups = [{"params": list(params)}]
        self.lr = lr
        self.steps = 0
        AdamW.created += 1

    def zero_grad(self, *, set_to_none):
        pass

    def step(self):
        self.steps += 1


class Torch:
    long = "long"

    def __init__(self):
        self.seeds = []
        self.optim = SimpleNamespace(AdamW=AdamW)
        self.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: None))

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def tensor(self, data, *, dtype, device):
        return data

    def isfinite(self, value):
        return SimpleNamespace(all=lambda: True)


def inputs():
    cases = [
        Challenge("train-t1-0", "arithmetic", 1, "train", "Question 0", "5"),
        Challenge("train-t1-1", "arithmetic", 1, "train", "Question 1", "6"),
    ]
    config = PilotConfig("unused", "unused", seed=17, steps=2, device="cpu")
    return cases, config


def issued_exposures(cases, *, seed, cursor, steps):
    all_indices = _training_visit_indices(cases, seed=seed, steps=cursor + steps)
    return tuple((cases[index].id, sha256(cases[index].prompt.encode("utf-8")).hexdigest())
                 for index in all_indices[cursor:])


class HFExternallyOwnedOptimizer(unittest.TestCase):
    def setUp(self):
        AdamW.created = 0

    def test_external_adamw_preserves_moments_and_existing_rng_across_calls(self):
        model, torch = Model(), Torch()
        optimizer = AdamW(model.parameters(), lr=0.0001)
        cases, config = inputs()
        random.seed(12345)
        expected_random = random.random()
        random.seed(12345)
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            first = _train_steps(model, object(), torch, cases, ["5", "6"], config,
                                 optimizer=optimizer, rng_policy="preserve",
                                 visit_cursor=0,
                                 expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))
            second = _train_steps(model, object(), torch, cases, ["5", "6"], config,
                                  optimizer=optimizer, rng_policy="preserve",
                                  visit_cursor=2,
                                  expected_visit_exposures=issued_exposures(cases, seed=17, cursor=2, steps=2))
        self.assertEqual((first["steps"], second["steps"]), (2, 2))
        self.assertEqual(AdamW.created, 1)
        self.assertEqual(optimizer.steps, 4)
        self.assertEqual(torch.seeds, [])
        self.assertEqual(random.random(), expected_random)

    def test_split_continuation_visits_same_cases_in_same_order_as_uninterrupted(self):
        cases = [
            Challenge(f"train-t1-{i}", "arithmetic", 1, "train", f"Question {i}", str(i))
            for i in range(5)
        ]
        torch = Torch()
        full = PilotConfig("unused", "unused", seed=17, steps=8, device="cpu")
        half = PilotConfig("unused", "unused", seed=17, steps=4, device="cpu")
        model, resumed_model = Model(), Model()
        optimizer = AdamW(resumed_model.parameters(), lr=0.0001)
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            uninterrupted = _train_steps(model, object(), torch, cases,
                                         [case.answer for case in cases], full)
            first = _train_steps(resumed_model, object(), torch, cases,
                                 [case.answer for case in cases], half,
                                 optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                                 expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=4))
            second = _train_steps(resumed_model, object(), torch, cases,
                                  [case.answer for case in cases], half,
                                  optimizer=optimizer, rng_policy="preserve", visit_cursor=4,
                                  expected_visit_exposures=issued_exposures(cases, seed=17, cursor=4, steps=4))
        self.assertEqual(first["visited_case_ids"] + second["visited_case_ids"],
                         uninterrupted["visited_case_ids"])
        self.assertNotEqual(second["visited_case_ids"], first["visited_case_ids"])
        self.assertEqual(optimizer.steps, 8)
        self.assertEqual(
            first["visited_exposures"] + second["visited_exposures"],
            uninterrupted["visited_exposures"],
        )

    def test_fresh_optimizer_records_completed_exposures_in_step_loop(self):
        cases, config = inputs()
        model, torch = Model(), Torch()
        precommitted = _training_visit_exposures(cases, seed=config.seed,
                                                 steps=config.steps)
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            completed = _train_steps(
                model, object(), torch, cases, [c.answer for c in cases], config,
                expected_visit_exposures=precommitted,
            )
        self.assertEqual(completed["visited_exposures"], list(precommitted))
        self.assertEqual(completed["visited_case_ids"], [pair[0] for pair in precommitted])
        self.assertEqual(AdamW.created, 1)

    def test_mutating_caller_train_list_during_encoding_cannot_change_encoded_catalog(self):
        cases, config = inputs()
        original = cases[1]
        pinned = _training_visit_exposures(cases, seed=config.seed, steps=config.steps)
        seen_prompts = []

        def encode_and_mutate(_tokenizer, prompt, _answer, _max_tokens, *, answer_only):
            seen_prompts.append(prompt)
            if len(seen_prompts) == 1:
                cases[1] = cases[0]
            else:
                cases[1] = original
            return ([1], [1])

        with patch("labyrinth.hf_pilot._encode_pair", side_effect=encode_and_mutate):
            result = _train_steps(
                Model(), object(), Torch(), cases, ["5", "6"], config,
                expected_visit_exposures=pinned,
            )
        self.assertEqual(seen_prompts, ["Question 0", "Question 1"])
        self.assertEqual(result["visited_exposures"], list(pinned))

    def test_fresh_optimizer_rejects_altered_case_or_prompt_before_update(self):
        cases, config = inputs()
        precommitted = _training_visit_exposures(cases, seed=config.seed,
                                                 steps=config.steps)
        wrong_case = (("forged-case", precommitted[0][1]), *precommitted[1:])
        wrong_prompt = ((precommitted[0][0], "0" * 64), *precommitted[1:])
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            for wrong in (wrong_case, wrong_prompt):
                with self.subTest(wrong=wrong):
                    AdamW.created = 0
                    with self.assertRaisesRegex(ValueError, "pinned slice"):
                        _train_steps(Model(), object(), Torch(), cases,
                                     [c.answer for c in cases], config,
                                     expected_visit_exposures=wrong)
                    self.assertEqual(AdamW.created, 0)

    def test_external_optimizer_requires_preserve_rng_and_correct_labels(self):
        model, torch = Model(), Torch()
        optimizer = AdamW(model.parameters(), lr=0.0001)
        cases, config = inputs()
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config, optimizer=optimizer)
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config, rng_policy="preserve")
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["6", "5"], config,
                             optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", learning_rate=0.0,
                             visit_cursor=0,
                             expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))
            with self.assertRaisesRegex(ValueError, "scheduler"):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", scheduler=object(),
                             visit_cursor=0,
                             expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))

    def test_external_optimizer_must_reference_exact_model_parameters(self):
        model, torch = Model(), Torch()
        wrong_optimizer = AdamW(Model().parameters(), lr=0.0001)
        cases, config = inputs()
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=wrong_optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))

    def test_external_optimizer_forbids_cuda(self):
        model, torch = Model(), Torch()
        optimizer = AdamW(model.parameters(), lr=0.0001)
        cases, _ = inputs()
        config = PilotConfig("unused", "unused", seed=17, steps=2, device="cuda")
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=issued_exposures(cases, seed=17, cursor=0, steps=2))

    def test_uncheckpointed_buffer_and_forged_schedule_are_rejected(self):
        model, torch = Model(), Torch()
        optimizer = AdamW(model.parameters(), lr=0.0001)
        cases, config = inputs()
        expected = issued_exposures(cases, seed=17, cursor=0, steps=2)
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=(("forged-case", expected[0][1]), *expected[1:]))
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=((expected[0][0], "0" * 64), *expected[1:]))
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve",
                             expected_visit_exposures=expected)
            model.named_buffers = lambda: iter((("running_mean", object()),))
            with self.assertRaises(ValueError):
                _train_steps(model, object(), torch, cases, ["5", "6"], config,
                             optimizer=optimizer, rng_policy="preserve", visit_cursor=0,
                             expected_visit_exposures=expected)
        self.assertEqual(optimizer.steps, 0)


if __name__ == "__main__":
    unittest.main()
