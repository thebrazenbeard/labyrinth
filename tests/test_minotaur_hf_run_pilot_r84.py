"""Source-only, no-model qualification for staged adaptive TRAIN to HF entrypoint."""

from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
import re
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest import mock

from labyrinth.hf_pilot import (
    PilotConfig, _arithmetic_equivalence_key, _scoped_public_heldout,
    reload_evaluate, run_pilot,
)
from labyrinth.hf_resume_bridge import ResumeContext, preflight_resume
from labyrinth.hf_report_audit import audit_hf_report
from labyrinth.evaluation import score_response
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.maze import Challenge
from labyrinth.minotaur_hf_roster_bridge import (
    _source_arithmetic, collect_minotaur_hf_roster, replay_staged_roster,
    snapshot_staged_roster,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


def staged_fixture(root: Path, *, count: int = 15):
    # A real feedback-conditioned corridor: 15 rooms allow practice, contrast,
    # and transfer to occur. The learner only sees prompt text, never gold.
    scope = TrainingScopeSpec(
        "r84-hf-arithmetic", "reasoning", ("arithmetic",), 5, 5, 2, 321,
    )
    runner = TrainingCorridorRunner.begin(
        scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
    )
    config = PilotConfig(str(root / "model"), str(root / "new-run"),
                         seed=31, train_count=count, eval_count=5,
                         steps=19, tiers=(1,))

    def learner(view):
        kind = "arithmetic_contrast" if view.prompt.startswith("Expression A") else "arithmetic"
        case = Challenge(view.challenge_id, kind, 1, "train", view.prompt, "")
        answer, _ = _source_arithmetic(case)
        return "ANSWER: " + answer

    return scope, config, collect_minotaur_hf_roster(runner, config, learner)


class StagedHfRunPilotTests(unittest.TestCase):
    def test_real_entrypoint_consumes_staged_cases_in_all_three_arms_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope, config, roster = staged_fixture(Path(tmp))
            model_root = Path(config.model_path)
            model_root.mkdir()
            (model_root / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
            (model_root / "model.safetensors").write_bytes(b"not-real-weights")

            class Param:
                requires_grad = True
                def __init__(self, model, snapshot=None):
                    self.model, self.snapshot = model, snapshot
                def detach(self):
                    return self
                def clone(self):
                    return Param(self.model, self.model.value)
                def copy_(self, other):
                    self.model.value = other.snapshot

            class Model:
                def __init__(self):
                    self.value = "a" * 64
                    self.parameter = Param(self)
                def named_parameters(self):
                    return [("layer.lora_A.default.weight", self.parameter)]
                def save_pretrained(self, directory, *, safe_serialization):
                    path = Path(directory)
                    path.mkdir()
                    (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")

            source_cases = tuple(roster.train_cases)
            original_cases = tuple((c.id, c.prompt, c.answer) for c in source_cases)
            stages = []
            seen_eval_rosters = []
            model = Model()
            def fake_train(current, _tokenizer, _torch, train, labels, cfg, *,
                           learning_rate=None, expected_visit_exposures=None):
                stages.append((tuple((c.id, c.prompt, c.answer) for c in train),
                               tuple(labels), tuple(expected_visit_exposures)))
                if len(stages) == 1:
                    # A hostile sham callback mutates even a frozen Challenge;
                    # neither subsequent optimizer arm nor original external
                    # roster mutation may redirect later supervision.
                    object.__setattr__(train[0], "answer", "999999")
                    object.__setattr__(roster.train_cases[0], "answer", "888888")
                elif len(stages) == 2:
                    current.value = "b" * 64
                else:
                    current.value = "c" * 64
                return {
                    "steps": cfg.steps, "example_visits": cfg.steps,
                    "input_tokens_seen": cfg.steps * 5,
                    "losses": [1.5] * cfg.steps,
                    "visited_case_ids": [cid for cid, _ in expected_visit_exposures],
                    "visited_exposures": list(expected_visit_exposures),
                }

            def fake_score(_model, _tokenizer, _torch, cases, **_kwargs):
                seen_eval_rosters.append(tuple((c.id, c.prompt, c.answer) for c in cases))
                trials = []
                for case in cases:
                    response = "ANSWER: 0"
                    scored = score_response(case, response)
                    trials.append({"id": case.id, "correct": scored.correct,
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                        "response_preview": response, "score_reason": scored.reason})
                correct = sum(t["correct"] for t in trials)
                if len(seen_eval_rosters) == 1:
                    # The public diagnostics are not protected final gold,
                    # but one scorer must not poison later control comparisons.
                    object.__setattr__(cases[0], "answer", "999999")
                return {"correct": correct, "total": len(cases),
                        "accuracy": correct / len(cases),
                        "format_valid": len(cases), "format_rate": 1.0,
                        "trials": trials}

            torch = SimpleNamespace(
                get_num_threads=lambda: 4, set_num_threads=lambda n: None,
                manual_seed=lambda seed: None,
                use_deterministic_algorithms=lambda yes: None,
                no_grad=lambda: nullcontext(),
            )
            deps = (torch, object(), object(),
                    (lambda **kw: object(), None, lambda base, _cfg: base), "cpu")
            with (
                patch("labyrinth.hf_pilot._deps", return_value=deps),
                patch("labyrinth.hf_pilot._device_headroom", return_value=None),
                patch("labyrinth.hf_pilot._load_base", return_value=(model, object())),
                patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])),
                patch("labyrinth.hf_pilot._score_model", side_effect=fake_score),
                patch("labyrinth.hf_pilot._train_steps", side_effect=fake_train),
                patch("labyrinth.hf_pilot._parameter_digest", side_effect=lambda obj, _: obj.value),
                patch("labyrinth.hf_pilot._lora_value_digest", side_effect=lambda obj, _: obj.value),
                patch("labyrinth.hf_pilot._frozen_parameter_digest", return_value="d" * 64),
                patch("labyrinth.hf_pilot._buffer_digest", return_value="e" * 64),
                patch("labyrinth.hf_pilot.package_version", return_value="test"),
            ):
                report = run_pilot(config, scope=scope, cycle_index=0,
                                   minotaur_roster=roster)
            self.assertEqual(len(stages), 3)
            self.assertEqual(len(seen_eval_rosters), 4)
            self.assertTrue(all(r == seen_eval_rosters[0] for r in seen_eval_rosters))
            self.assertTrue(all(stage[0] == original_cases for stage in stages))
            self.assertTrue(all(stage[2] == roster.ordered_optimizer_exposures for stage in stages))
            self.assertEqual(stages[0][1], stages[1][1])
            self.assertNotEqual(stages[1][1], stages[2][1])
            self.assertFalse("resumed_journey" in report)
            self.assertFalse(any("answer" in case for case in
                                 report["minotaur_staged_train"]["train"]))
            audited = audit_hf_report(json.loads(json.dumps(report)))
            self.assertEqual(audited["status"], "INCOMPLETE", audited)
            self.assertFalse(audited["authenticated_model_execution"])
            report_file = Path(config.output_dir) / "pilot_report.json"
            self.assertTrue(report_file.is_file())
            saved = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(audit_hf_report(saved)["status"], "INCOMPLETE")

            # A separate reload must use the actual staged TRAIN catalog to
            # reproduce public heldout; it may not silently regenerate the
            # ordinary fixed-maze TRAIN set and evaluate a different split.
            with patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError("deps reached")):
                with self.assertRaisesRegex(RuntimeError, "deps reached"):
                    reload_evaluate(config.output_dir, Path(tmp) / "reload.json")

            # Later-cycle checkpoint continuation has no authenticated
            # Minotaur source/case lineage, even if the ordinary LoRA report
            # and saved adapter are internally self-consistent.
            import hashlib as _hashlib
            successor = replace(config, output_dir=str(Path(tmp) / "successor"))
            context = ResumeContext(
                scope.fingerprint, 1,
                _hashlib.sha256(report_file.read_bytes()).hexdigest(),
                "b" * 64, "operator-source-only",
            )
            with self.assertRaisesRegex(ValueError, "staged Minotaur TRAIN"):
                preflight_resume(successor, config.output_dir, context)

            tampered = json.loads(json.dumps(saved))
            tampered["minotaur_staged_train"]["train"][0]["prompt"] += "\nUNTRUSTED"
            self.assertEqual(audit_hf_report(tampered)["status"], "FAIL")
            tampered = json.loads(json.dumps(saved))
            del tampered["minotaur_staged_train"]
            self.assertEqual(audit_hf_report(tampered)["status"], "FAIL")

    def test_real_fifteen_room_practice_contrast_transfer_are_answerless_replayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope, config, roster = staged_fixture(Path(tmp))
            tactics = {dict(case.metadata)["tactic"] for case in roster.train_cases}
            self.assertEqual(tactics, {"practice", "contrast", "transfer"})
            cases, public = snapshot_staged_roster(roster, config, scope)
            self.assertEqual(len(cases), 15)
            self.assertTrue(all("answer" not in case for case in public["train"]))
            self.assertTrue(all("answer_sha256" not in case for case in public["train"]))
            copy_scope, replay, exposures = replay_staged_roster(
                json.loads(json.dumps(public)), config,
            )
            self.assertEqual(copy_scope, scope)
            self.assertEqual([case.answer for case in replay],
                             [case.answer for case in roster.train_cases])
            self.assertEqual(exposures, roster.ordered_optimizer_exposures)
            for invalid_cycle in (False, True, 0.0, "0"):
                forged_cycle = json.loads(json.dumps(public))
                forged_cycle["cycle_index"] = invalid_cycle
                with self.subTest(invalid_cycle=invalid_cycle):
                    with self.assertRaisesRegex(ValueError, "scope or budget"):
                        replay_staged_roster(forged_cycle, config)
            false_roster = replace(roster, cycle_index=False)
            with self.assertRaisesRegex(ValueError, "cycle-zero"):
                snapshot_staged_roster(false_roster, config, scope)
            heldout = _scoped_public_heldout(config, scope, 0, replay)
            self.assertTrue(heldout)
            self.assertFalse({_arithmetic_equivalence_key(case) for case in replay}
                             & {_arithmetic_equivalence_key(case) for case in heldout})
            contrast = next(c for c in replay if c.kind == "arithmetic_contrast")
            match = re.search(r"Expression A is \d+ \+ (\d+)\.", contrast.prompt)
            self.assertIsNotNone(match)
            b = int(match[1])
            equivalent = Challenge(
                "eval-public-equivalent", "arithmetic", 1, "eval",
                f"Calculate {b} + {b}.\nRespond with exactly one line: ANSWER: <integer>.",
                str(2 * b),
            )
            with patch("labyrinth.hf_pilot.generate_maze", return_value=[equivalent]):
                with self.assertRaisesRegex(ValueError, "semantic arithmetic"):
                    _scoped_public_heldout(config, scope, 0, replay)
            noncanonical = Challenge(
                "train-taskmaster-noncanonical", "arithmetic", 1, "train",
                "Calculate 00004 + 00005.\nRespond with exactly one line: ANSWER: <integer>.",
                "9",
            )
            with self.assertRaisesRegex(ValueError, "generator bounds"):
                _source_arithmetic(noncanonical)

    def test_staged_entry_rejects_non_initial_scope_and_modified_roster_before_ml_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope, config, roster = staged_fixture(Path(tmp))
            model = Path(config.model_path)
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"local-fake")
            with patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError("deps reached")):
                for kwargs in ({"cycle_index": 1}, {"cycle_index": 0, "weak_skills": ("arithmetic",)},
                               {"cycle_index": 0, "unauthenticated_research_resume": True}):
                    with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                        run_pilot(config, scope=scope, minotaur_roster=roster, **kwargs)
                with self.assertRaises(ValueError):
                    tampered = replace(roster, train_cases=(replace(roster.train_cases[0],
                        split="eval"),) + roster.train_cases[1:])
                    run_pilot(config, scope=scope, cycle_index=0, minotaur_roster=tampered)
                with self.assertRaisesRegex(ValueError, "answer"):
                    tampered = replace(roster, train_cases=(replace(roster.train_cases[0],
                        answer="999999"),) + roster.train_cases[1:])
                    run_pilot(config, scope=scope, cycle_index=0, minotaur_roster=tampered)
                with self.assertRaisesRegex(RuntimeError, "deps reached"):
                    run_pilot(config, scope=scope, cycle_index=0, minotaur_roster=roster)


if __name__ == "__main__":
    unittest.main()
