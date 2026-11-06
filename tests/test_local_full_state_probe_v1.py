"""Source-only matched public full-state inference and fake local APIs."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from labyrinth.ariadne_recovery_v1 import UnsupportedJournalPlatform
from labyrinth.evolutionary_population_v1 import EvaluationPlan
from labyrinth.local_full_state_probe_v1 import (
    DeterministicProbeConfig, LocalFullStateProbeError, LocalProbeTarget,
    PublicPrompt, probe_local_full_state_population, public_probe_suite_sha256,
)
from labyrinth.weight_space_campaign_bridge_v1 import run_weight_space_generation
from labyrinth.weight_space_spark_v1 import WeightSpaceEditPolicy, WeightSpaceEditProposal
from tests.test_weight_space_spark_v1 import _capture, _tensor


class FullStateLocalProbeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        founder = self.root / "w0"
        for role in ("base", "optimizer", "scheduler", "rng"):
            (founder / role).mkdir(parents=True)
        (founder / "base" / "config.json").write_text('{"model_type":"tiny"}')
        (founder / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
        (founder / "base" / "model.safetensors").write_bytes(
            _tensor((1., 2., 3., 4., 5., 6., 7., 8.)))
        for role in ("optimizer", "scheduler", "rng"):
            (founder / role / "state.bin").write_bytes(role.encode())
        (founder / "cursor.json").write_text('{"step":0}')
        manifest, path = _capture(founder)
        self.w0_manifest = path
        self.w0_state = manifest["state_sha256"]
        self.probes = (PublicPrompt("identity", "Introduce yourself in one sentence."),
                       PublicPrompt("revision", "Correct: the moon is a planet."))
        self.suite_sha = public_probe_suite_sha256(self.probes)
        self.calls: list[tuple[str, dict, float]] = []
        self.targets = (
            LocalProbeTarget("parent", "http://127.0.0.1:5011/api/v1/generate", "koboldcpp"),
            LocalProbeTarget("control", "http://127.0.0.1:5012/api/v1/generate", "koboldcpp"),
            LocalProbeTarget("left", "http://127.0.0.1:5013/completion", "llamacpp"),
            LocalProbeTarget("right", "http://127.0.0.1:5014/v1/completions", "openai_local"),
        )
        policy = WeightSpaceEditPolicy(("model.safetensors",), ("proj.weight",),
                                      maximum_modified_fraction=.25,
                                      maximum_relative_norm=.15)

        def proposal(ctx):
            return WeightSpaceEditProposal(
                ctx.candidate_id, ctx.parent_state_sha256, ctx.public_train_source_sha256,
                "model.safetensors", "proj.weight", (0,), (1., 0.),
                .5 if ctx.candidate_id == "left" else 1.,
                "Public TRAIN-only local hypothesis; no private gold provided.",
            )

        try:
            self.generation = run_weight_space_generation(
                w0_manifest_path=path, campaign_root=self.root / "generation",
                candidate_ids=("left", "right"), public_train_source_sha256="f"*64,
                propose=proposal, policy=policy,
                evaluation_plan=EvaluationPlan("minotaur_dev", "a"*64, "b"*64,
                                               "c"*64, "d"*64),
            )
        except UnsupportedJournalPlatform:
            self.skipTest("Ariadne requires Linux/WSL")

    def fake(self, endpoint, payload, timeout):
        self.calls.append((endpoint, payload, timeout))
        prompt = payload["prompt"]
        if endpoint.endswith("/api/v1/generate"):
            return {"results": [{"text": "KO " + prompt}]}
        if endpoint.endswith("/completion"):
            return {"content": "LL " + prompt}
        return {"choices": [{"text": "OA " + prompt}]}

    def run_probes(self, **override):
        params = dict(
            population=self.generation.population, targets=self.targets,
            public_probes=self.probes,
            expected_public_suite_sha256=self.suite_sha,
            config=DeterministicProbeConfig(max_tokens=20, seed=31),
            transport=self.fake,
        )
        params.update(override)
        return probe_local_full_state_population(**params)

    def test_matched_public_probes_rehash_all_subjects_without_gold_leak(self):
        receipt = self.run_probes()
        self.assertEqual(len(receipt.subjects), 4)
        self.assertEqual(receipt.public_suite_sha256, self.suite_sha)
        self.assertEqual(receipt.classification,
                         "PUBLIC_LOCALHOST_OUTPUT_UNVERIFIED_LOADED_MODEL")
        self.assertFalse(receipt.neural_improvement_proven)
        by_id = {subject.subject_id: subject for subject in receipt.subjects}
        self.assertEqual(by_id["parent"].full_state_sha256,
                         by_id["control"].full_state_sha256)
        self.assertEqual(by_id["parent"].full_state_sha256, self.w0_state)
        self.assertNotEqual(by_id["left"].full_state_sha256,
                            by_id["right"].full_state_sha256)
        self.assertTrue(all(item.source_state_unchanged and not item.model_bytes_loaded_proven
                            for item in receipt.subjects))
        self.assertEqual(len(self.calls), 8)
        self.assertEqual([payload["prompt"] for _, payload, _ in self.calls],
                         [probe.prompt for probe in self.probes] * 4)
        for _, payload, timeout in self.calls:
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["top_p"], 1)
            self.assertEqual(payload["seed"], 31)
            self.assertEqual(timeout, 20.0)
            self.assertFalse(set(payload) & {
                "expected", "answer", "gold", "score", "examiner_id",
                "suite_sha256", "sealed_case_commitment", "manifest_path",
            })
        self.assertTrue(by_id["left"].output_texts[0].startswith("LL "))
        self.assertTrue(by_id["right"].output_texts[0].startswith("OA "))

    def test_uncommitted_cohort_and_nonloopback_service_fail_without_transport(self):
        with self.assertRaisesRegex(LocalFullStateProbeError, "precommitted suite"):
            self.run_probes(expected_public_suite_sha256="0"*64)
        self.assertEqual(self.calls, [])
        endpoints = (
            "https://127.0.0.1:5011/api/v1/generate",
            "http://localhost:5011/api/v1/generate",
            "http://192.168.1.10:5011/api/v1/generate",
            "http://127.0.0.1:5011/api/v1/generate?token=abc",
            "http://name:pass@127.0.0.1:5011/api/v1/generate",
            "http://127.0.0.1:5011/redirect",
        )
        for url in endpoints:
            with self.subTest(url=url), self.assertRaises(LocalFullStateProbeError):
                self.run_probes(targets=(replace(self.targets[0], endpoint=url),
                                         *self.targets[1:]))
            self.assertEqual(self.calls, [])

    def test_one_endpoint_requires_owner_activation_and_is_not_attestation(self):
        shared = tuple(replace(target, endpoint="http://127.0.0.1:5011/api/v1/generate",
                               backend="koboldcpp") for target in self.targets)
        with self.assertRaisesRegex(LocalFullStateProbeError, "owner reload"):
            self.run_probes(targets=shared)
        self.assertEqual(self.calls, [])
        activation_log = []

        def activate(context):
            activation_log.append(context)
            return "OPERATOR_REPORTS_ACTIVATED"

        receipt = self.run_probes(targets=shared, activate_target=activate)
        self.assertEqual(len(activation_log), 4)
        self.assertEqual(activation_log[0].full_state_sha256, self.w0_state)
        self.assertEqual(activation_log[0].subject_id, "parent")
        self.assertFalse(any(subject.model_bytes_loaded_proven for subject in receipt.subjects))

    def test_malformed_response_rejected_and_original_w0_preserved(self):
        def malformed(endpoint, payload, timeout):
            self.calls.append((endpoint, payload, timeout))
            return {"results": [{"text": "x"*5000}]}

        with self.assertRaisesRegex(LocalFullStateProbeError, "no bounded"):
            self.run_probes(transport=malformed)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.w0_state,
                         self.generation.population.parent_state_sha256)

    def test_modified_parent_or_nondeterministic_config_does_not_reach_inference(self):
        with self.assertRaisesRegex(LocalFullStateProbeError, "deterministic"):
            self.run_probes(config=DeterministicProbeConfig(temperature=.7))
        self.assertEqual(self.calls, [])
        # Changing even an optimizer byte defeats the complete physical
        # population before any prompt is sent to a model endpoint.
        optimizer = self.root / "w0" / "optimizer" / "state.bin"
        optimizer.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.run_probes()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
