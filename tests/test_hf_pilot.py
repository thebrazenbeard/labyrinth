"""Dependency-free contract tests; these do not claim that a HF model trained."""

import json
import hashlib
import re
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import (
    PilotConfig,
    SCHEMA,
    LEGACY_SCHEMA,
    PRE_EXPOSURE_SCHEMA,
    _arithmetic_equivalence_key,
    EVAL_SEED_XOR,
    EVAL_SPLIT_MAX_ATTEMPTS,
    _metadata_digest,
    _training_visit_indices,
    _weights_digest,
    _device_headroom,
    _encode_pair,
    _lora_value_digest,
    _summarize_trials,
    derange_answers,
    reload_evaluate,
    run_pilot,
    training_cases,
    validate_config,
)
from labyrinth.maze import Challenge, generate_maze
from tests.test_optimizer_lifecycle_r8 import AdamW, Model, Torch, inputs


class DummyTokenizer:
    eos_token = "~"

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [1] + [ord(c) for c in text] if add_special_tokens
                else [ord(c) for c in text]}


class HfPilotContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "local_model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"small-fixture-not-real-weights")
        self.output = root / "new_run"

    def spec(self, **kwargs):
        return PilotConfig(model_path=str(self.model), output_dir=str(self.output), **kwargs)

    def _adaptive_staged_fixture(self):
        from tests.test_minotaur_hf_roster_bridge import _runner
        from tests.test_journey_policy_runner import toy_arithmetic_learner
        from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster

        config = self.spec(seed=31, train_count=3, eval_count=3,
                           steps=5, tiers=(1,))
        runner = _runner()
        roster = collect_minotaur_hf_roster(
            runner, config, toy_arithmetic_learner,
        )
        self.assertEqual([dict(case.metadata)["tactic"] for case in roster.train_cases],
                         ["practice", "practice", "contrast"])
        return config, runner.state.scope, roster

    def test_adaptive_pilot_requires_explicit_cycle_zero_source_consent_before_deps(self):
        config, scope, roster = self._adaptive_staged_fixture()
        invalid = (
            {"scope": scope, "cycle_index": 0,
             "adaptive_train_roster": roster},
            {"scope": scope, "cycle_index": 0,
             "allow_unattested_adaptive_train": True},
            {"adaptive_train_roster": roster,
             "allow_unattested_adaptive_train": True},
            {"scope": scope, "cycle_index": 1,
             "adaptive_train_roster": roster,
             "allow_unattested_adaptive_train": True},
            {"scope": replace(scope, seed=scope.seed + 1), "cycle_index": 0,
             "adaptive_train_roster": roster,
             "allow_unattested_adaptive_train": True},
            {"scope": scope, "cycle_index": 0,
             "adaptive_train_roster": replace(
                 roster, ordered_optimizer_exposures=tuple(reversed(
                     roster.ordered_optimizer_exposures))),
             "allow_unattested_adaptive_train": True},
        )
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
            "heavy model initialization must not begin")):
            for options in invalid:
                with self.subTest(options=tuple(options)), self.assertRaises(ValueError):
                    run_pilot(config, **options)
        self.assertFalse(self.output.exists())

    def test_staged_and_adaptive_roster_modes_reject_co_supply_before_config(self):
        config = self.spec()
        with patch("labyrinth.hf_pilot.validate_config", side_effect=AssertionError(
            "model/config validation must not precede source-mode exclusion")):
            for extra in (False, True):
                with self.subTest(adaptive_opt_in=extra), self.assertRaisesRegex(
                    ValueError, "mutually exclusive"
                ):
                    run_pilot(
                        config, minotaur_roster=object(),
                        adaptive_train_roster=object(),
                        allow_unattested_adaptive_train=extra,
                    )
        self.assertFalse(self.output.exists())

    def test_staged_contrast_heldout_rejects_internal_plus_expression_alias(self):
        from labyrinth.hf_pilot import _scoped_public_heldout

        config, scope, roster = self._adaptive_staged_fixture()
        contrast = next(case for case in roster.train_cases
                        if case.kind == "arithmetic_contrast")
        matched = re.search(r"Expression A is (\d+) \+ (\d+)\.", contrast.prompt)
        self.assertIsNotNone(matched)
        left, right = int(matched[1]), int(matched[2])
        self.assertNotEqual(left, right)

        def public_case(case_id, a, b):
            return Challenge(
                case_id, "arithmetic", 1, "eval",
                f"Calculate {a} + {b}.\nRespond with exactly one line: ANSWER: <integer>.",
                str(a + b),
            )

        internal_plus_alias = public_case("eval-alias", left, right)
        disjoint = [public_case(f"eval-disjoint-{i}", 99900 + i, 99800 + i)
                    for i in range(config.eval_count)]
        with patch("labyrinth.hf_pilot.generate_maze", side_effect=(
            [internal_plus_alias, *disjoint[1:]], disjoint,
        )) as generate:
            heldout = _scoped_public_heldout(config, scope, 0, roster.train_cases)
        self.assertEqual(heldout, disjoint)
        self.assertEqual(generate.call_count, 2,
                         "contrast's a+b expression must invalidate first heldout draw")

    def test_strict_reload_rejects_incomplete_source_audit_before_neural_load(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(
            json.dumps({"schema": SCHEMA}), encoding="utf-8",
        )
        with (
            patch("labyrinth.hf_report_audit.audit_hf_report", return_value={
                "status": "INCOMPLETE", "failures": [],
                "unchecked": ["unverified strict exposure fields"],
            }),
            patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
                "neural dependency loading must not be reached")),
        ):
            with self.assertRaisesRegex(ValueError, "strict internal source audit"):
                reload_evaluate(self.output, self.output / "reloaded.json")

    def test_schema_downgrade_cannot_bypass_reload_exposure_gate(self):
        self.output.mkdir()
        report_path = self.output / "pilot_report.json"
        for historical_schema in (LEGACY_SCHEMA, PRE_EXPOSURE_SCHEMA):
            # A caller can erase the strict marker and three-arm in-loop
            # exposure receipts and substitute an old schema string.
            report_path.write_text(json.dumps({
                "schema": historical_schema,
                "training": {arm: {"steps": 2} for arm in (
                    "sham", "candidate", "corrupted",
                )},
            }), encoding="utf-8")
            with (
                self.subTest(schema=historical_schema),
                patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
                    "downgraded report reached model loading")),
                patch("labyrinth.hf_report_audit.audit_hf_report", side_effect=AssertionError(
                    "legacy report reached audit without explicit compatibility opt-in")),
            ):
                with self.assertRaisesRegex(ValueError, "explicit compatibility opt-in"):
                    reload_evaluate(self.output, self.output / "reloaded.json")
        self.assertFalse((self.output / "reloaded.json").exists())

    def test_legacy_reload_opt_in_still_requires_pass_compatibility_audit(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(
            json.dumps({"schema": PRE_EXPOSURE_SCHEMA}), encoding="utf-8",
        )
        with (
            patch("labyrinth.hf_report_audit.audit_hf_report", return_value={
                "status": "INCOMPLETE", "failures": [], "unchecked": ["missing receipts"],
            }) as audit,
            patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
                "neural loading must not proceed on incomplete legacy evidence")),
        ):
            with self.assertRaisesRegex(ValueError, "strict internal source audit"):
                reload_evaluate(self.output, self.output / "reloaded.json",
                                allow_legacy_reload=True)
        audit.assert_called_once()
        self.assertEqual(audit.call_args.kwargs, {"allow_legacy": True})

    def test_staged_reload_rejects_unrelated_incompleteness_before_neural_load(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(
            json.dumps({"schema": SCHEMA, "minotaur_staged_train": {}}),
            encoding="utf-8",
        )
        with (
            patch("labyrinth.hf_report_audit.audit_hf_report", return_value={
                "status": "INCOMPLETE", "failures": [],
                "unchecked": [
                    "Minotaur staged TRAIN originates in the operator process, not independent custody",
                    "truncated candidate trial evidence",
                ],
            }),
            patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
                "neural dependency loading must not be reached")),
        ):
            with self.assertRaisesRegex(ValueError, "strict internal source audit"):
                reload_evaluate(self.output, self.output / "reloaded.json")

    def test_adaptive_pilot_uses_staged_source_not_legacy_generator(self):
        config, scope, roster = self._adaptive_staged_fixture()
        with (
            patch("labyrinth.hf_pilot.training_cases", side_effect=AssertionError(
                "legacy static TRAIN generator was called")),
            patch("labyrinth.hf_pilot._scoped_training_cases", side_effect=AssertionError(
                "legacy scoped TRAIN generator was called")),
            patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError(
                "reached heavy dependency boundary")),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached heavy dependency boundary"):
                run_pilot(
                    config, scope=scope, cycle_index=0,
                    adaptive_train_roster=roster,
                    allow_unattested_adaptive_train=True,
                )
        self.assertFalse(self.output.exists())

    def test_adaptive_pilot_rejects_heldout_mutation_before_any_train_arm(self):
        config, scope, roster = self._adaptive_staged_fixture()

        class Parameter:
            requires_grad = True

            def detach(self):
                return self

            def clone(self):
                return self

        class FakeModel:
            def named_parameters(self):
                return [("layer.lora_A", Parameter())]

        fake_torch = SimpleNamespace(
            get_num_threads=lambda: 4, set_num_threads=lambda _n: None,
            manual_seed=lambda _n: None,
            use_deterministic_algorithms=lambda _value: None,
        )

        def mutate_diagnostic(_model, _tokenizer, _torch, cases, **_kwargs):
            object.__setattr__(cases[0], "answer", "tampered after evaluation")
            return {"total": len(cases)}

        with (
            patch("labyrinth.hf_pilot._deps", return_value=(
                fake_torch, object(), object(),
                (lambda **_kw: object(), object(), lambda base, _cfg: base), "cpu",
            )),
            patch("labyrinth.hf_pilot._load_base", return_value=(FakeModel(), object())),
            patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])),
            patch("labyrinth.hf_pilot._score_model", side_effect=mutate_diagnostic),
            patch("labyrinth.hf_pilot._device_headroom", return_value=None),
            patch("labyrinth.hf_pilot._parameter_digest", return_value="a" * 64),
            patch("labyrinth.hf_pilot._frozen_parameter_digest", return_value="b" * 64),
            patch("labyrinth.hf_pilot._buffer_digest", return_value="c" * 64),
            patch("labyrinth.hf_pilot._train_steps", side_effect=AssertionError(
                "optimizer must not start after diagnostic source mutation")),
        ):
            with self.assertRaisesRegex(RuntimeError, "case content changed"):
                run_pilot(
                    config, scope=scope, cycle_index=0,
                    adaptive_train_roster=roster,
                    allow_unattested_adaptive_train=True,
                )
        self.assertFalse(self.output.exists())

    def test_adaptive_pilot_three_arms_train_on_staged_source_with_sealed_report(self):
        from contextlib import nullcontext
        from labyrinth.evaluation import score_response
        from labyrinth.hf_adaptive_source_v1 import ADAPTIVE_SCHEMA
        from labyrinth.hf_report_audit import audit_hf_report

        config, scope, roster = self._adaptive_staged_fixture()
        initial, trained, corrupted = "a" * 64, "b" * 64, "c" * 64

        class Parameter:
            requires_grad = True

            def __init__(self, owner, snapshot=None):
                self.owner = owner
                self.snapshot = snapshot

            def detach(self):
                return self

            def clone(self):
                return Parameter(self.owner, self.owner.value)

            def copy_(self, other):
                self.owner.value = other.snapshot

        class Model:
            def __init__(self):
                self.value = initial
                self.parameter = Parameter(self)

            def named_parameters(self):
                return [("layer.lora_A.default.weight", self.parameter)]

            def save_pretrained(self, directory, *, safe_serialization):
                self.saved = self.value
                path = Path(directory)
                path.mkdir()
                (path / "adapter_model.safetensors").write_bytes(b"fake-adaptive-weights")

        model = Model()
        torch = SimpleNamespace(
            get_num_threads=lambda: 4, set_num_threads=lambda _n: None,
            manual_seed=lambda _seed: None,
            use_deterministic_algorithms=lambda _value: None,
            no_grad=lambda: nullcontext(),
        )
        visits = []

        def fake_train(current, _tokenizer, _torch, train, labels, pilot,
                       *, learning_rate=None, expected_visit_exposures=None):
            self.assertEqual(tuple(train), roster.train_cases)
            self.assertEqual(tuple(expected_visit_exposures),
                             roster.ordered_optimizer_exposures)
            self.assertEqual(current.value, initial)
            if learning_rate == 0.0:
                arm = "sham"
            elif tuple(labels) == tuple(case.answer for case in train):
                arm = "candidate"
                current.value = trained
            else:
                arm = "corrupted"
                current.value = corrupted
            visits.append(arm)
            return {
                "steps": pilot.steps, "example_visits": pilot.steps,
                "visited_case_ids": [case_id for case_id, _ in expected_visit_exposures],
                "visited_exposures": list(expected_visit_exposures),
                "input_tokens_seen": 200,
                "losses": [2.0] * pilot.steps,
            }

        def fake_score(_model, _tokenizer, _torch, cases, **_kwargs):
            records = []
            for case in cases:
                answer = "ANSWER: 0"
                scored = score_response(case, answer)
                records.append({
                    "id": case.id, "correct": scored.correct,
                    "score_reason": scored.reason,
                    "response_preview": answer,
                    "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                })
            correct = sum(record["correct"] for record in records)
            return {
                "correct": correct, "total": len(records),
                "accuracy": correct / len(records),
                "format_valid": len(records), "format_rate": 1.0,
                "trials": records,
            }

        with (
            patch("labyrinth.hf_pilot.training_cases", side_effect=AssertionError(
                "legacy static TRAIN called")),
            patch("labyrinth.hf_pilot._scoped_training_cases", side_effect=AssertionError(
                "legacy scoped TRAIN called")),
            patch("labyrinth.hf_pilot._deps", return_value=(
                torch, object(), object(),
                (lambda **_kw: object(), object(), lambda base, _cfg: base), "cpu",
            )),
            patch("labyrinth.hf_pilot._load_base", return_value=(model, object())),
            patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])),
            patch("labyrinth.hf_pilot._score_model", side_effect=fake_score),
            patch("labyrinth.hf_pilot._device_headroom", return_value=None),
            patch("labyrinth.hf_pilot._train_steps", side_effect=fake_train),
            patch("labyrinth.hf_pilot._parameter_digest", side_effect=lambda obj, _t: obj.value),
            patch("labyrinth.hf_pilot._lora_value_digest", side_effect=lambda obj, _t: obj.value),
            patch("labyrinth.hf_pilot._frozen_parameter_digest", return_value="d" * 64),
            patch("labyrinth.hf_pilot._buffer_digest", return_value="e" * 64),
            patch("labyrinth.hf_pilot.package_version", return_value="synthetic-test"),
        ):
            result = run_pilot(
                config, scope=scope, cycle_index=0,
                adaptive_train_roster=roster,
                allow_unattested_adaptive_train=True,
            )

        self.assertEqual(visits, ["sham", "candidate", "corrupted"])
        self.assertEqual(result["schema"], ADAPTIVE_SCHEMA)
        self.assertEqual(result["training_case_ids"], [c.id for c in roster.train_cases])
        self.assertEqual(result["training"]["sham"]["visited_exposures"],
                         list(roster.ordered_optimizer_exposures))
        self.assertIn("adaptive_train_source_v1", result)
        self.assertFalse(result.get("resumed_journey"))
        self.assertEqual(model.saved, trained)
        saved = json.loads((self.output / "pilot_report.json").read_text(encoding="utf-8"))
        self.assertEqual(audit_hf_report(saved)["status"], "PASS")
        # The same adaptive manifest is reconstructed for a separate-process
        # reload, including the identical heldout source and adapter bytes.
        # Changed control evidence must fail before a model is imported.
        reload_path = self.output / "reload-evaluation.json"
        reloaded_model = Model()
        reloaded_model.value = trained
        peft_loader = SimpleNamespace(
            from_pretrained=lambda _base, _path, **_kwargs: reloaded_model,
        )
        with (
            patch("labyrinth.hf_pilot._deps", return_value=(
                torch, object(), object(), (object(), peft_loader, object()), "cpu",
            )),
            patch("labyrinth.hf_pilot._load_base", return_value=(object(), object())),
            patch("labyrinth.hf_pilot._lora_value_digest", side_effect=lambda obj, _t: obj.value),
            patch("labyrinth.hf_pilot._score_model", side_effect=fake_score),
        ):
            reloaded = reload_evaluate(self.output, reload_path)
        self.assertTrue(reloaded["matches_original_candidate_trials"])
        self.assertEqual(reloaded["reloaded_lora_tensor_value_sha256"], trained)
        self.assertTrue(reload_path.is_file())
        saved["control_data"]["sham_updates"] = 1
        (self.output / "pilot_report.json").write_text(
            json.dumps(saved), encoding="utf-8",
        )
        tampered_reload = self.output / "tampered-reload-evaluation.json"
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError(
            "tampered adaptive report reached model load")):
            with self.assertRaisesRegex(ValueError, "strict internal source audit"):
                reload_evaluate(self.output, tampered_reload)
        self.assertFalse(tampered_reload.exists())

    def _source_visit_fixture(self):
        from labyrinth import hf_pilot

        cases, config = inputs()
        # This independent oracle deliberately does not use the helper under test.
        planned = tuple(
            (cases[index].id, hashlib.sha256(cases[index].prompt.encode("utf-8")).hexdigest())
            for index in (1, 0)
        )
        return hf_pilot, cases, config, planned

    def test_source_visit_helper_and_fresh_control_arms_record_exact_optimizer_visits(self):
        hf_pilot, cases, config, planned = self._source_visit_fixture()
        self.assertEqual(hf_pilot._training_visit_exposures(cases, seed=17, steps=2), planned)
        for learning_rate in (None, 0.0):
            with self.subTest(learning_rate=learning_rate):
                AdamW.created = 0
                model, torch = Model(), Torch()
                with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
                    result = hf_pilot._train_steps(
                        model, object(), torch, cases, [case.answer for case in cases], config,
                        learning_rate=learning_rate, expected_visit_exposures=planned,
                    )
                self.assertEqual(result["visited_exposures"], list(planned))
                self.assertEqual(result["visited_case_ids"], [case_id for case_id, _ in planned])
                self.assertEqual(result["example_visits"], 2)
                self.assertEqual(AdamW.created, 1)

    def test_pinned_fresh_training_rejects_changed_case_id_prompt_and_order(self):
        hf_pilot, cases, config, planned = self._source_visit_fixture()
        invalid_slices = (
            (("substituted-id", planned[0][1]), planned[1]),
            ((planned[0][0], "0" * 64), planned[1]),
            tuple(reversed(planned)),
        )
        for invalid in invalid_slices:
            with self.subTest(invalid=invalid):
                AdamW.created = 0
                with patch("labyrinth.hf_pilot._encode_pair", side_effect=AssertionError("encoding began")):
                    with self.assertRaisesRegex(ValueError, "exposure schedule"):
                        hf_pilot._train_steps(
                            Model(), object(), Torch(), cases, [case.answer for case in cases], config,
                            expected_visit_exposures=invalid,
                        )
                self.assertEqual(AdamW.created, 0)

    def test_pinned_fresh_catalog_rejects_heldout_and_duplicate_case_ids(self):
        hf_pilot, cases, config, _ = self._source_visit_fixture()
        bad_catalogs = (
            [Challenge(cases[0].id, cases[0].kind, cases[0].tier, "eval", cases[0].prompt, cases[0].answer), cases[1]],
            [cases[0], Challenge(cases[0].id, cases[1].kind, cases[1].tier,
                                 "train", cases[1].prompt, cases[1].answer)],
        )
        for bad in bad_catalogs:
            with self.subTest(cases=bad):
                expected = hf_pilot._training_visit_exposures(bad, seed=17, steps=2)
                AdamW.created = 0
                with patch("labyrinth.hf_pilot._encode_pair", side_effect=AssertionError("encoding began")):
                    with self.assertRaisesRegex(ValueError, "training catalog"):
                        hf_pilot._train_steps(Model(), object(), Torch(), bad,
                                              [c.answer for c in bad], config,
                                              expected_visit_exposures=expected)
                self.assertEqual(AdamW.created, 0)

    def test_tokenizer_cannot_rewrite_pinned_source_or_supervision_lists(self):
        hf_pilot, cases, config, planned = self._source_visit_fixture()
        labels = [case.answer for case in cases]
        seen = []

        def encode(_tokenizer, prompt, answer, _max_tokens, **_kwargs):
            seen.append((prompt, answer))
            if len(seen) == 1:
                cases[1] = cases[0]
                labels[1] = "tampered supervision"
            return [1], [1]

        with patch("labyrinth.hf_pilot._encode_pair", side_effect=encode):
            result = hf_pilot._train_steps(Model(), object(), Torch(), cases, labels, config,
                                           expected_visit_exposures=planned)
        self.assertEqual(seen, [("Question 0", "5"), ("Question 1", "6")])
        self.assertEqual(result["visited_exposures"], list(planned))

    def test_partial_optimizer_failure_never_reports_full_success(self):
        hf_pilot, cases, _, _ = self._source_visit_fixture()
        config = PilotConfig("unused", "unused", seed=17, steps=3, device="cpu")
        planned = hf_pilot._training_visit_exposures(cases, seed=17, steps=3)

        class FailOnSecondStep(AdamW):
            latest = None

            def __init__(self, params, *, lr):
                super().__init__(params, lr=lr)
                FailOnSecondStep.latest = self

            def step(self):
                if self.steps == 1:
                    raise RuntimeError("injected optimizer failure")
                return super().step()

        torch = Torch()
        torch.optim.AdamW = FailOnSecondStep
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaisesRegex(RuntimeError, "injected optimizer failure") as raised:
                hf_pilot._train_steps(Model(), object(), torch, cases,
                                       [c.answer for c in cases], config,
                                       expected_visit_exposures=planned)
        self.assertEqual(FailOnSecondStep.latest.steps, 1)
        self.assertEqual(raised.exception.completed_visit_exposures, planned[:1])

    def test_step_boundary_rejects_case_mutation_after_first_optimizer_update(self):
        hf_pilot, cases, config, planned = self._source_visit_fixture()

        class MutatingAdamW(AdamW):
            latest = None

            def __init__(self, params, *, lr):
                super().__init__(params, lr=lr)
                MutatingAdamW.latest = self

            def step(self):
                super().step()
                if self.steps == 1:
                    # Deliberately bypass the frozen dataclass, simulating a
                    # hostile callback that tampers with a source object.
                    object.__setattr__(cases[0], "prompt", "changed after first update")

        torch = Torch()
        torch.optim.AdamW = MutatingAdamW
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaisesRegex(RuntimeError, "TRAIN case content changed") as raised:
                hf_pilot._train_steps(Model(), object(), torch, cases,
                                       [case.answer for case in cases], config,
                                       expected_visit_exposures=planned)
        self.assertEqual(MutatingAdamW.latest.steps, 1)
        self.assertEqual(raised.exception.completed_visit_exposures, planned[:1])

    def test_final_optimizer_update_cannot_hide_changed_case_content(self):
        hf_pilot, _, _, _ = self._source_visit_fixture()
        config = PilotConfig("unused", "unused", seed=17, steps=1, device="cpu")

        for field, replacement in (("id", "tampered-after-final-step"),
                                   ("prompt", "tampered after final step"),
                                   ("answer", "tampered answer after final step")):
            with self.subTest(field=field):
                current, _ = inputs()
                planned = hf_pilot._training_visit_exposures(current, seed=17, steps=1)

                class FinalMutatingAdamW(AdamW):
                    latest = None

                    def __init__(self, params, *, lr):
                        super().__init__(params, lr=lr)
                        FinalMutatingAdamW.latest = self

                    def step(self):
                        super().step()
                        object.__setattr__(current[1], field, replacement)

                torch = Torch()
                torch.optim.AdamW = FinalMutatingAdamW
                with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
                    with self.assertRaisesRegex(RuntimeError, "TRAIN case content changed") as raised:
                        hf_pilot._train_steps(Model(), object(), torch, current,
                                               [case.answer for case in current], config,
                                               expected_visit_exposures=planned)
                self.assertEqual(FinalMutatingAdamW.latest.steps, 1)
                self.assertEqual(raised.exception.completed_visit_exposures, planned)

    def test_tokenizer_callback_cannot_change_non_exposure_case_fields(self):
        hf_pilot, _, config, _ = self._source_visit_fixture()
        for field, replacement in (("split", "eval"), ("tier", 2),
                                   ("answer", "wrong answer"),
                                   ("metadata", (("altered", "yes"),))):
            with self.subTest(field=field):
                current, _ = inputs()
                planned = hf_pilot._training_visit_exposures(current, seed=17, steps=2)

                def encode(_tokenizer, _prompt, _answer, _max_tokens, **_kwargs):
                    object.__setattr__(current[0], field, replacement)
                    return [1], [1]

                AdamW.created = 0
                with patch("labyrinth.hf_pilot._encode_pair", side_effect=encode):
                    with self.assertRaisesRegex(RuntimeError, "TRAIN case content changed"):
                        hf_pilot._train_steps(Model(), object(), Torch(), current,
                                               [case.answer for case in current], config,
                                               expected_visit_exposures=planned)
                self.assertEqual(AdamW.created, 0)

    def test_mutated_unvisited_case_fails_at_final_boundary(self):
        hf_pilot, cases, _, _ = self._source_visit_fixture()
        config = PilotConfig("unused", "unused", seed=17, steps=1, device="cpu")
        planned = hf_pilot._training_visit_exposures(cases, seed=17, steps=1)

        class MutatingAdamW(AdamW):
            def step(self):
                super().step()
                object.__setattr__(cases[0], "split", "eval")

        torch = Torch()
        torch.optim.AdamW = MutatingAdamW
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaisesRegex(RuntimeError, "TRAIN case content changed") as raised:
                hf_pilot._train_steps(Model(), object(), torch, cases,
                                       [case.answer for case in cases], config,
                                       expected_visit_exposures=planned)
        self.assertEqual(raised.exception.completed_visit_exposures, planned)

    def test_nonfinite_after_optimizer_update_retains_completed_prefix(self):
        hf_pilot, cases, _, _ = self._source_visit_fixture()
        config = PilotConfig("unused", "unused", seed=17, steps=1, device="cpu")
        planned = hf_pilot._training_visit_exposures(cases, seed=17, steps=1)
        model = Model()

        class NonfiniteParamTorch(Torch):
            def isfinite(self, value):
                if value is model.param:
                    return type("FiniteCheck", (), {"all": lambda self: False})()
                return super().isfinite(value)

        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            with self.assertRaisesRegex(RuntimeError, "non-finite trained LoRA parameter") as raised:
                hf_pilot._train_steps(model, object(), NonfiniteParamTorch(), cases,
                                       [case.answer for case in cases], config,
                                       expected_visit_exposures=planned)
        self.assertEqual(raised.exception.completed_visit_exposures, planned)

    def test_run_pilot_rejects_sham_answer_mutation_before_candidate_arm(self):
        config = self.spec()
        train, heldout = training_cases(config)
        scheduled = tuple(
            (train[index].id, hashlib.sha256(train[index].prompt.encode("utf-8")).hexdigest())
            for index in _training_visit_indices(train, seed=config.seed, steps=config.steps)
        )

        class FakeParameter:
            requires_grad = True

            def detach(self):
                return self

            def clone(self):
                return self

        class FakeModel:
            def named_parameters(self):
                return [("layer.lora_A", FakeParameter())]

        fake_model = FakeModel()
        torch = SimpleNamespace(
            get_num_threads=lambda: 4, set_num_threads=lambda _n: None,
            manual_seed=lambda _seed: None,
            use_deterministic_algorithms=lambda _enabled: None,
        )

        def sham_tampers(_model, _tokenizer, _torch, cases, _labels, _config,
                         *, learning_rate=None, expected_visit_exposures=None):
            self.assertEqual(learning_rate, 0.0)
            self.assertEqual(expected_visit_exposures, scheduled)
            object.__setattr__(cases[0], "answer", "tampered after sham")
            return {"visited_exposures": list(scheduled)}

        with (
            patch("labyrinth.hf_pilot.training_cases", return_value=(train, heldout)),
            patch("labyrinth.hf_pilot._deps", return_value=(
                torch, object(), object(),
                (lambda **_kwargs: object(), object(), lambda base, _cfg: base), "cpu",
            )),
            patch("labyrinth.hf_pilot._load_base", return_value=(fake_model, object())),
            patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])),
            patch("labyrinth.hf_pilot._score_model", return_value={"total": 0}),
            patch("labyrinth.hf_pilot._device_headroom", return_value=None),
            patch("labyrinth.hf_pilot._parameter_digest", return_value="a" * 64),
            patch("labyrinth.hf_pilot._frozen_parameter_digest", return_value="b" * 64),
            patch("labyrinth.hf_pilot._buffer_digest", return_value="c" * 64),
            patch("labyrinth.hf_pilot._train_steps", side_effect=sham_tampers) as trainer,
        ):
            with self.assertRaisesRegex(RuntimeError, "TRAIN case content changed"):
                run_pilot(config)
        self.assertEqual(trainer.call_count, 1)
        self.assertFalse(self.output.exists())

    def test_preflight_does_not_create_output_or_download(self):
        model, output, weights = validate_config(self.spec())
        self.assertEqual(model, self.model)
        self.assertEqual(output, self.output)
        self.assertEqual([p.name for p in weights], ["model.safetensors"])
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output_and_model_directory(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            validate_config(self.spec())
        with self.assertRaises(ValueError):
            validate_config(PilotConfig(model_path=str(self.model), output_dir=str(self.model)))

    def test_refuses_oversized_weights_and_bin_only(self):
        (self.model / "model.safetensors").unlink()
        (self.model / "pytorch_model.bin").write_bytes(b"pickle")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            validate_config(self.spec())

    def test_refuses_invalid_resource_bounds_before_heavy_import(self):
        for field, value in [("steps", 0), ("steps", 129), ("max_tokens", 10),
                             ("train_count", 100), ("seed", True), ("lora_rank", 99),
                             ("learning_rate", float("nan")), ("device", "auto")]:
            with self.subTest(field=field):
                payload = {**self.spec().__dict__, field: value}
                with self.assertRaises(ValueError):
                    validate_config(PilotConfig(**payload))

    def test_train_split_is_disjoint_and_pinned(self):
        config = self.spec(seed=13, train_count=5, eval_count=5, tiers=(1, 2))
        train, heldout = training_cases(config)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(heldout), 10)
        self.assertEqual({c.split for c in train}, {"train"})
        self.assertEqual({c.split for c in heldout}, {"eval"})
        self.assertFalse({c.id for c in train} & {c.id for c in heldout})
        self.assertEqual(train, training_cases(config)[0])
        self.assertEqual(heldout, generate_maze(
            seed=config.seed ^ EVAL_SEED_XOR, count=config.eval_count,
            tiers=config.tiers, split="eval",
        ))

    def test_train_eval_split_excludes_commutative_arithmetic_leakage(self):
        # Reproduced public train/eval collision before the repair: the two
        # texts differ but the addition expression is the same up to operand
        # order, a stronger equivalence than an exact prompt hash.
        config = self.spec(seed=1073, train_count=32, eval_count=32, tiers=(1,))
        train, heldout = training_cases(config)
        self.assertEqual((len(train), len(heldout)), (32, 32))
        train_keys = {_arithmetic_equivalence_key(case) for case in train}
        heldout_keys = {_arithmetic_equivalence_key(case) for case in heldout}
        self.assertFalse(train_keys & heldout_keys)
        self.assertEqual((train, heldout), training_cases(config))
        raw_heldout = generate_maze(
            seed=config.seed ^ EVAL_SEED_XOR, count=config.eval_count,
            tiers=config.tiers, split="eval",
        )
        self.assertTrue(train_keys & {
            _arithmetic_equivalence_key(case) for case in raw_heldout
        })  # Confirms the original seed-1073 false independence.
        self.assertNotEqual(raw_heldout, heldout)

    def test_arithmetic_split_equivalence_is_narrow_not_answer_equality(self):
        make = lambda prompt: Challenge("test", "arithmetic", 1, "train", prompt, "0")
        instruction = "\nRespond with exactly one line: ANSWER: <integer>."
        self.assertEqual(
            _arithmetic_equivalence_key(make("Calculate 991 + 807." + instruction)),
            _arithmetic_equivalence_key(make("Calculate 807 + 991." + instruction)),
        )
        self.assertNotEqual(
            _arithmetic_equivalence_key(make("Calculate 991 - 807." + instruction)),
            _arithmetic_equivalence_key(make("Calculate 807 - 991." + instruction)),
        )
        self.assertNotEqual(
            _arithmetic_equivalence_key(make("Calculate 1 + 3." + instruction)),
            _arithmetic_equivalence_key(make("Calculate 2 + 2." + instruction)),
        )
        self.assertEqual(
            _arithmetic_equivalence_key(make("Calculate 9 * 3." + instruction)),
            _arithmetic_equivalence_key(make("Calculate 3 * 9." + instruction)),
        )
        with self.assertRaisesRegex(ValueError, "unrecognized arithmetic prompt"):
            _arithmetic_equivalence_key(make("Calculate 9 / 3." + instruction))

    def test_arithmetic_split_reroll_budget_fails_closed_without_dropping_cases(self):
        instruction = "\nRespond with exactly one line: ANSWER: <integer>."
        train = [Challenge("t1", "arithmetic", 1, "train",
                           "Calculate 991 + 807." + instruction, "1798")]
        eval_cases = [Challenge("e1", "arithmetic", 1, "eval",
                                "Calculate 807 + 991." + instruction, "1798")]

        def never_separates(*, split, **kwargs):
            return train if split == "train" else eval_cases

        with patch("labyrinth.hf_pilot.generate_maze", side_effect=never_separates) as generate:
            with self.assertRaisesRegex(ValueError, "could not avoid semantic arithmetic"):
                training_cases(self.spec(seed=1073, train_count=2, eval_count=2))
            self.assertEqual(generate.call_count, 1 + EVAL_SPLIT_MAX_ATTEMPTS)

    def test_derangement_preserves_multiset_without_collisions(self):
        cases = [Challenge(str(i), "test", 1, "train", str(i), answer)
                 for i, answer in enumerate(["7", "7", "8", "8", "9", "10"])]
        changed = derange_answers(cases)
        self.assertEqual(sorted(changed), sorted(c.answer for c in cases))
        self.assertTrue(all(a.answer != b for a, b in zip(cases, changed)))
        with self.assertRaisesRegex(ValueError, "derange"):
            derange_answers(cases[:3])

    def test_supervision_masks_entire_prompt(self):
        tokenizer = DummyTokenizer()
        ids, labels = _encode_pair(tokenizer, "Compute 1+1", "2", 256)
        visible = len(ids) - len([c for c in "ANSWER: 2~"])
        self.assertEqual(labels[:visible], [-100] * visible)
        self.assertEqual(labels[visible:], ids[visible:])
        with self.assertRaisesRegex(ValueError, "token budget"):
            _encode_pair(tokenizer, "X" * 200, "2", 64)

    def test_no_torch_import_neither_creates_nor_changes_run_dir(self):
        with patch("labyrinth.hf_pilot._deps", side_effect=RuntimeError("optional deps unavailable")):
            with self.assertRaisesRegex(RuntimeError, "optional deps"):
                run_pilot(self.spec())
        self.assertFalse(self.output.exists())

    def test_pilot_rejects_base_weight_replacement_during_initial_model_load(self):
        class FakeTorch:
            @staticmethod
            def set_num_threads(value):
                pass

            @staticmethod
            def get_num_threads():
                return 1

            @staticmethod
            def manual_seed(value):
                pass

            @staticmethod
            def use_deterministic_algorithms(value):
                pass

        for change in ("replace-existing", "add-new-shard"):
            with self.subTest(change=change):
                (self.model / "model.safetensors").write_bytes(b"small-fixture-not-real-weights")
                extra = self.model / "extra.safetensors"
                if extra.exists():
                    extra.unlink()

                def swapped_load(*args, **kwargs):
                    if change == "replace-existing":
                        (self.model / "model.safetensors").write_bytes(b"tampered-after-initial-snapshot")
                    else:
                        extra.write_bytes(b"new-unpinned-weight")
                    return object(), object()

                with patch("labyrinth.hf_pilot._deps", return_value=(FakeTorch, None, None, (None, None, None), "cpu")):
                    with patch("labyrinth.hf_pilot._load_base", side_effect=swapped_load):
                        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
                            with patch("labyrinth.hf_pilot._score_model", side_effect=AssertionError("model inference should not run")):
                                with self.assertRaisesRegex(RuntimeError, "base model.*changed"):
                                    run_pilot(self.spec())
                self.assertFalse(self.output.exists())

    def test_tokenizer_metadata_symlink_rejected(self):
        external = Path(self.temp.name) / "other"
        external.write_text("read me?", encoding="utf-8")
        try:
            (self.model / "tokenizer.json").symlink_to(external)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            validate_config(self.spec())

    def test_sharded_weight_index_rejects_parent_traversal_before_heavy_import(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 123},
            "weight_map": {"model.layer": "../outside.safetensors"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index"):
            validate_config(self.spec())

    def test_valid_local_sharded_index_is_included_in_model_metadata_hash(self):
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": 123},
            "weight_map": {"model.layer": "model.safetensors"},
        }), encoding="utf-8")
        validate_config(self.spec())
        before = _metadata_digest(self.model)
        self.assertIn(index.name, before)
        index.write_text(json.dumps({
            "metadata": {"total_size": 124},
            "weight_map": {"model.layer": "model.safetensors"},
        }), encoding="utf-8")
        self.assertNotEqual(before[index.name], _metadata_digest(self.model)[index.name])

    def _reload_legacy_storage_fixture(self):
        # These old sparse v2 fixtures exercise on-disk adapter integrity,
        # not report-audit completeness. Explicitly isolate that layer: live
        # compatibility reloads require a real PASS legacy audit.
        with patch("labyrinth.hf_report_audit.audit_hf_report", return_value={
            "status": "PASS", "failures": [], "unchecked": [],
        }):
            return reload_evaluate(
                self.output, self.output / "verify.json", allow_legacy_reload=True,
            )

    def test_reload_refuses_symlink_adapter_root_before_loading_model(self):
        external = Path(self.temp.name) / "outside_adapter"
        external.mkdir()
        (external / "adapter_model.safetensors").write_bytes(b"fake-adapter")
        self.output.mkdir()
        try:
            (self.output / "candidate_adapter").symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        from labyrinth.hf_pilot import _prompt_commitment, training_cases
        from labyrinth.runner import _source_state

        config = self.spec()
        _, heldout = training_cases(config)
        report = {
            "schema": "labyrinth.hf-lora-pilot.v2",
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": {
                "adapter_model.safetensors": hashlib.sha256(b"fake-adapter").hexdigest()
            },
            "heldout_case_ids": [case.id for case in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        }
        (self.output / "pilot_report.json").write_text(json.dumps(report), encoding="utf-8")
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("loaded heavy dependencies")):
            with self.assertRaisesRegex(ValueError, "symlink"):
                self._reload_legacy_storage_fixture()

    def test_reload_detects_base_weight_changes_while_loading(self):
        from labyrinth.hf_pilot import _prompt_commitment, training_cases
        from labyrinth.runner import _source_state

        self.output.mkdir()
        candidate = self.output / "candidate_adapter"
        candidate.mkdir()
        (candidate / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
        config = self.spec()
        _, heldout = training_cases(config)
        (self.output / "pilot_report.json").write_text(json.dumps({
            "schema": "labyrinth.hf-lora-pilot.v2",
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": {
                "adapter_model.safetensors": hashlib.sha256(b"fixture-adapter").hexdigest()},
            "heldout_case_ids": [c.id for c in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        }), encoding="utf-8")

        def swapped_load(*args, **kwargs):
            (self.model / "model.safetensors").write_bytes(b"swapped-during-reloader")
            return object(), object()

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, None, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", side_effect=swapped_load):
                with self.assertRaisesRegex(RuntimeError, "base model.*changed"):
                    self._reload_legacy_storage_fixture()
        self.assertFalse((self.output / "verify.json").exists())

    def _write_reload_adapter_fixture(self):
        from labyrinth.hf_pilot import _prompt_commitment
        from labyrinth.runner import _source_state

        self.output.mkdir()
        candidate = self.output / "candidate_adapter"
        candidate.mkdir()
        (candidate / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
        config_file = candidate / "adapter_config.json"
        config_file.write_text('{"lora_alpha":8}', encoding="utf-8")
        config = self.spec()
        _, heldout = training_cases(config)
        adapter_sha256 = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in candidate.iterdir()}
        (self.output / "pilot_report.json").write_text(json.dumps({
            "schema": "labyrinth.hf-lora-pilot.v2",
            "source": _source_state(),
            "config": {**asdict(config), "tiers": list(config.tiers)},
            "base_weights_sha256": _weights_digest([self.model / "model.safetensors"]),
            "model_metadata_sha256": _metadata_digest(self.model),
            "candidate_adapter_sha256": adapter_sha256,
            "candidate_lora_tensor_value_sha256": "pretend-lora-digest",
            "heldout_case_ids": [case.id for case in heldout],
            "heldout_public_prompt_sha256": _prompt_commitment(heldout),
            "evaluations": {"candidate": {"trials": []}},
        }), encoding="utf-8")
        return candidate, config_file

    def test_reload_refuses_adapter_config_swap_during_peft_load(self):
        candidate, config_file = self._write_reload_adapter_fixture()

        class ReplacingPeft:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                config_file.write_text('{"lora_alpha":64}', encoding="utf-8")
                return object()

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, ReplacingPeft, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", return_value=(object(), object())):
                with patch("labyrinth.hf_pilot._lora_value_digest", return_value="pretend-lora-digest"):
                    with patch("labyrinth.hf_pilot._score_model", side_effect=AssertionError("tampered adapter must not be evaluated")):
                        with self.assertRaisesRegex(RuntimeError, "adapter.*changed"):
                            self._reload_legacy_storage_fixture()
        self.assertFalse((self.output / "verify.json").exists())

    def test_reload_refuses_new_adapter_file_during_evaluation(self):
        candidate, config_file = self._write_reload_adapter_fixture()

        class UnchangedPeft:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        def changed_during_score(*args, **kwargs):
            (candidate / "unexpected.safetensors").write_bytes(b"injected")
            return {"trials": []}

        with patch("labyrinth.hf_pilot._deps", return_value=(None, None, None, (None, UnchangedPeft, None), "cpu")):
            with patch("labyrinth.hf_pilot._load_base", return_value=(object(), object())):
                with patch("labyrinth.hf_pilot._lora_value_digest", return_value="pretend-lora-digest"):
                    with patch("labyrinth.hf_pilot._score_model", side_effect=changed_during_score):
                        with self.assertRaisesRegex(RuntimeError, "adapter.*changed"):
                            self._reload_legacy_storage_fixture()
        self.assertFalse((self.output / "verify.json").exists())

    def test_reload_refuses_nested_adapter_directory_symlink_before_dependencies(self):
        candidate, _ = self._write_reload_adapter_fixture()
        outside = Path(self.temp.name) / "outside_directory"
        outside.mkdir()
        try:
            (candidate / "nested_link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("heavy deps must not load")):
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                self._reload_legacy_storage_fixture()
        self.assertFalse((self.output / "verify.json").exists())

    def test_unknown_report_fails_before_dependency_loading(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text(json.dumps({"schema": "invalid"}))
        with self.assertRaisesRegex(ValueError, "schema"):
            reload_evaluate(self.output, self.output / "verify.json")

    def test_oversized_report_fails_before_dependency_loading(self):
        self.output.mkdir()
        (self.output / "pilot_report.json").write_text('{"padding":"' + "x" * 128 + '"}', encoding="utf-8")
        with patch("labyrinth.runner._MAX_JSON_BYTES", 64):
            with patch("labyrinth.hf_pilot._deps", side_effect=AssertionError("loaded heavy dependencies")):
                with self.assertRaisesRegex(ValueError, "JSON input exceeds"):
                    reload_evaluate(self.output, self.output / "verify.json")

    def test_module_help_does_not_import_optional_packages(self):
        result = subprocess.run([sys.executable, "-m", "labyrinth.hf_pilot", "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("train", result.stdout)
        self.assertIn("eval", result.stdout)

    def test_top_level_hf_cli_delegates_without_optional_packages(self):
        result = subprocess.run([sys.executable, "-m", "labyrinth", "train-hf", "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("--model-path", result.stdout)

    def test_tiny_fixture_help_does_not_import_optional_packages(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "make_tiny_hf_fixture.py"
        result = subprocess.run([sys.executable, str(script), "--help"],
                                text=True, capture_output=True, check=True)
        self.assertIn("--output-dir", result.stdout)

    def test_cuda_preflight_fails_closed_for_small_free_vram(self):
        class TorchWithMemory:
            class cuda:
                @staticmethod
                def mem_get_info():
                    return 128 * 1024 * 1024, 4 * 1024 * 1024 * 1024

        weights = [self.model / "model.safetensors"]
        self.assertIsNone(_device_headroom(TorchWithMemory, "cpu", weights))
        with self.assertRaisesRegex(RuntimeError, "CUDA memory"):
            _device_headroom(TorchWithMemory, "cuda", weights)

    def test_frozen_reload_lora_values_are_still_hashed(self):
        class Tensor:
            requires_grad = False
            shape = (1,)
            dtype = "float32"

            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def flatten(self):
                return self

            def view(self, dtype):
                return self

            def numpy(self):
                return self

            def tobytes(self):
                return self.value

        class Model:
            def __init__(self, value):
                self.value = value

            def named_parameters(self):
                return iter([("base_model.layer.lora_A.default.weight", Tensor(self.value))])

        class FakeTorch:
            uint8 = object()

        self.assertNotEqual(_lora_value_digest(Model(b"before"), FakeTorch),
                            _lora_value_digest(Model(b"after"), FakeTorch))

    def test_format_rate_is_not_conflated_with_accuracy(self):
        trials = [
            {"correct": False, "score_reason": "invalid_format"},
            {"correct": False, "score_reason": "incorrect"},
        ]
        summary = _summarize_trials(trials)
        self.assertEqual((summary["correct"], summary["format_valid"]), (0, 1))
        self.assertEqual((summary["accuracy"], summary["format_rate"]), (0.0, 0.5))
        with self.assertRaises(ValueError):
            _summarize_trials([])


if __name__ == "__main__":
    unittest.main()
