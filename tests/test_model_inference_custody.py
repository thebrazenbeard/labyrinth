"""Source-level evaluator execution tests; predictor is deliberately a mock."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import hmac
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from labyrinth import blind_evaluation_custody as blind
from labyrinth import model_inference_custody as inference


class ToyPredictor:
    def predict(self, prompt: str) -> str:
        return "yes" if "sunny" in prompt else "no"


class InvalidPredictor:
    def predict(self, prompt: str) -> str:
        return None


class NoPredictor:
    pass


_PREDICTION_LOCK = threading.Lock()
_PREDICTION_COUNT = 0


class CountingPredictor:
    def predict(self, prompt: str) -> str:
        global _PREDICTION_COUNT
        with _PREDICTION_LOCK:
            _PREDICTION_COUNT += 1
        time.sleep(0.015)
        return "yes" if "sunny" in prompt else "no"


class MutatingPredictor:
    def __init__(self, artifact: Path):
        self.artifact = artifact

    def predict(self, prompt: str) -> str:
        self.artifact.write_bytes(b"mutation during private exam")
        return "yes"


def trusted_test_loader(root: Path, name: str) -> ToyPredictor:
    # A source-pinned test double, explicitly NOT an authenticated neural model.
    return ToyPredictor()


def invalid_result_loader(root: Path, name: str) -> InvalidPredictor:
    return InvalidPredictor()


def missing_predictor_loader(root: Path, name: str) -> NoPredictor:
    return NoPredictor()


def counting_loader(root: Path, name: str) -> CountingPredictor:
    return CountingPredictor()


def mutating_loader(root: Path, name: str) -> MutatingPredictor:
    return MutatingPredictor(root / name)


class EvaluatorInferenceCustodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint_name = "model.safetensors"
        self.artifact = self.root / self.checkpoint_name
        self.artifact.write_bytes(b"synthetic test adapter bytes only")
        self.checkpoint_sha = sha256(self.artifact.read_bytes()).hexdigest()
        self.source_file = Path(__file__).absolute()
        self.source_sha = sha256(self.source_file.read_bytes()).hexdigest()
        self.identity = inference.PinnedLoader(
            self.source_file, self.source_sha, self.checkpoint_sha)
        self.evaluator_key = b"v" * 32
        self.state_key = b"s" * 32
        self.run_id = "inference-test"
        self.db_path = self.root / "development.sqlite"
        self.cases = (
            blind.PrivateExamCase("h1", "sentiment", "heldout", "sunny park today",
                                  "hfamily1", "yes"),
            blind.PrivateExamCase("h2", "sentiment", "heldout", "cloudy garden now",
                                  "hfamily2", "no"),
            blind.PrivateExamCase("t1", "sentiment", "transfer", "sunny route home",
                                  "tfamily1", "yes"),
            blind.PrivateExamCase("t2", "sentiment", "transfer", "rainy drive away",
                                  "tfamily2", "no"),
        )
        blind.initialize_assessment(
            self.db_path, self.cases, run_id=self.run_id,
            objective_id="sentiment", checkpoint_sha256=self.checkpoint_sha,
            evaluator_key=self.evaluator_key, state_key=self.state_key,
        )

    def execute(self, loader=trusted_test_loader, **overrides):
        args = dict(db_path=self.db_path, cases=self.cases,
                    artifact_root=self.root, artifact_name=self.checkpoint_name,
                    pinned_loader=self.identity, loader=loader,
                    evaluator_key=self.evaluator_key, state_key=self.state_key,
                    run_id=self.run_id)
        args.update(overrides)
        return inference.execute_private_development_exam(**args)

    def verify(self, result, **overrides):
        args = dict(evaluator_key=self.evaluator_key, run_id=self.run_id,
                    checkpoint_sha256=self.checkpoint_sha,
                    loader_source_sha256=self.source_sha)
        args.update(overrides)
        return inference.verify_invocation(result, **args)

    def test_evaluator_invokes_loader_and_scores_without_supplied_responses(self):
        result = self.execute()
        trace = self.verify(result)
        self.assertEqual(4, trace["question_count"])
        self.assertEqual(4, sum(group["correct"] for group in trace["groups"]))
        self.assertTrue(trace["evaluator_invoked_predictor"])
        self.assertFalse(trace["caller_responses_accepted"])
        self.assertFalse(trace["model_execution_authenticated"])
        self.assertFalse(trace["graduation_authorized"])
        self.assertFalse(result.assessment.graduation_authorized)
        self.assertNotIn(b"sunny", result.trace_bytes)
        self.assertNotIn(b"cloudy", result.trace_bytes)

    def test_explicit_caller_responses_are_not_accepted(self):
        with self.assertRaises(TypeError):
            self.execute(responses={"h1": "yes"})

    def test_model_source_must_be_independently_pinned(self):
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(pinned_loader=replace(self.identity, source_sha256="0" * 64))
        with self.assertRaises(ValueError):
            self.execute(pinned_loader=None)

    def test_artifact_bytes_must_equal_precommit_before_loader(self):
        self.artifact.write_bytes(b"mutated adapter")
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute()

    def test_bad_predictor_output_irrevocably_spends_inference_reservation(self):
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(loader=invalid_result_loader)
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute()

    def test_missing_predictor_irrevocably_spends_inference_reservation(self):
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(loader=missing_predictor_loader)
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute()

    def test_checkpoint_mutation_during_predict_spends_inference_reservation(self):
        original = self.artifact.read_bytes()
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(loader=mutating_loader)
        self.artifact.write_bytes(original)
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute()

    def test_replay_rejected_after_atomic_scoring(self):
        self.verify(self.execute())
        with self.assertRaises(blind.BlindCustodyError):
            self.execute()
        # The preflight must reject replay before even querying the model.
        with self.assertRaises(blind.BlindCustodyError):
            self.execute(loader=invalid_result_loader)

    def test_concurrent_evaluators_cannot_expose_private_prompts_twice(self):
        global _PREDICTION_COUNT
        with _PREDICTION_LOCK:
            _PREDICTION_COUNT = 0
        barrier = threading.Barrier(2)
        original_preflight = inference._require_precommitted_pending_assessment

        def synchronized_preflight(*args, **kwargs):
            original_preflight(*args, **kwargs)
            barrier.wait(timeout=5)

        def attempted_exam():
            try:
                return self.execute(loader=counting_loader)
            except Exception as exc:
                return exc

        with patch.object(inference, "_require_precommitted_pending_assessment",
                          side_effect=synchronized_preflight):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: attempted_exam(), (0, 1)))
        self.assertEqual(1, sum(isinstance(r, inference.SignedEvaluatorInvocation)
                                for r in results))
        self.assertEqual(1, sum(isinstance(r, inference.InferenceCustodyError)
                                for r in results))
        with _PREDICTION_LOCK:
            self.assertEqual(4, _PREDICTION_COUNT)

    def test_wrong_run_does_not_consume_fresh_precommitted_exam(self):
        with self.assertRaises(blind.BlindCustodyError):
            self.execute(run_id="another-run")
        self.verify(self.execute())

    def test_trace_signature_and_assessment_binding_fail_closed(self):
        result = self.execute()
        with self.assertRaises(inference.InferenceCustodyError):
            self.verify(replace(result, signature_hex="0" * 64))
        with self.assertRaises(inference.InferenceCustodyError):
            self.verify(result, checkpoint_sha256="0" * 64)
        with self.assertRaises(inference.InferenceCustodyError):
            self.verify(result, loader_source_sha256="0" * 64)
        with self.assertRaises(inference.InferenceCustodyError):
            self.verify(replace(result, graduation_authorized=True))
        altered = replace(result.assessment, signature_hex="f" * 64)
        with self.assertRaises((inference.InferenceCustodyError,
                                blind.BlindCustodyError)):
            self.verify(replace(result, assessment=altered))

    def test_even_valid_evaluator_signature_cannot_grant_graduation(self):
        result = self.execute()
        forged = json.loads(result.trace_bytes)
        forged["graduation_authorized"] = True
        raw = inference._canonical(forged)
        tag = hmac.new(self.evaluator_key, inference._DOMAIN + raw, sha256).hexdigest()
        with self.assertRaises(inference.InferenceCustodyError):
            self.verify(replace(result, trace_bytes=raw, signature_hex=tag))

    def test_source_path_and_loader_function_must_agree(self):
        identity = replace(self.identity, source_file=self.root / "other.py")
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(pinned_loader=identity)
        with self.assertRaises(inference.InferenceCustodyError):
            self.execute(loader=lambda root, name: ToyPredictor())

    def test_symlink_artifact_is_rejected(self):
        replacement = self.root / "elsewhere.safetensors"
        replacement.write_bytes(self.artifact.read_bytes())
        self.artifact.unlink()
        try:
            self.artifact.symlink_to(replacement)
        except (OSError, NotImplementedError):
            self.skipTest("host cannot create a symlink in isolated fixture")
        with self.assertRaises(blind.BlindCustodyError):
            self.execute()


if __name__ == "__main__":
    unittest.main()
