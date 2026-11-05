"""Offline, synthetic-only conversation ordeal contract tests."""
import unittest

from labyrinth.conversation_ordeal import (
    Artifacts,
    LocalChatCompletionsAdapter,
    run_ordeal,
    synthetic_scenarios,
)


BASE_HASH = "a" * 64
ADAPTER_HASH = "b" * 64


class FakeChatModel:
    model_id = "synthetic:test"

    def __init__(self):
        self.calls = []

    def generate_chat(self, messages, *, seed):
        snapshot = tuple(dict(m) for m in messages)
        self.calls.append((snapshot, seed))
        return f"synthetic-answer-{len(self.calls)}"


class ConversationOrdealTests(unittest.TestCase):
    def artifacts(self):
        return Artifacts(
            model_id="synthetic:test",
            base_weights_sha256=BASE_HASH,
            adapter_sha256=ADAPTER_HASH,
        )

    def test_synthetic_scenarios_are_multiturn_and_review_foci_stay_hidden(self):
        model = FakeChatModel()
        report = run_ordeal(model, artifacts=self.artifacts(), include_responses=True)
        self.assertEqual(report["qualification"], "NONE")
        self.assertEqual(report["review"]["status"],
                         "UNSCORED_HUMAN_OR_INDEPENDENT_REVIEW_REQUIRED")
        self.assertEqual(len(model.calls),
                         sum(len(s.steps) for s in synthetic_scenarios()))
        self.assertGreaterEqual(len(synthetic_scenarios()), 3)
        # Exactly one system message starts each scenario; all later assistant
        # turns must enter the next model call, in their proper roles.
        first, _ = model.calls[0]
        second, _ = model.calls[1]
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual([m["role"] for m in second],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(second[2]["content"], "synthetic-answer-1")
        for scenario in synthetic_scenarios():
            for step in scenario.steps:
                self.assertFalse(any(
                    step.reviewer_focus in message["content"]
                    for messages, _ in model.calls for message in messages
                ))
        self.assertEqual(report["trials"][0]["turns"][0]["response"],
                         "synthetic-answer-1")

    def test_receipt_is_stable_and_hides_responses_by_default(self):
        a = run_ordeal(FakeChatModel(), artifacts=self.artifacts())
        b = run_ordeal(FakeChatModel(), artifacts=self.artifacts())
        self.assertEqual(a["report_sha256"], b["report_sha256"])
        self.assertEqual(a["trials"], b["trials"])
        self.assertFalse(a["responses_included"])
        for trial in a["trials"]:
            for turn in trial["turns"]:
                self.assertNotIn("response", turn)
                self.assertEqual(len(turn["sent_history_sha256"]), 64)
        self.assertEqual(a["declared_artifact_hashes"]["verification"],
                         "CALLER_DECLARED_NOT_INDEPENDENTLY_VERIFIED")
        self.assertTrue(a["review"]["rubric_is_not_a_training_target"])

    def test_runtime_identity_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            run_ordeal(FakeChatModel(), artifacts=Artifacts(
                model_id="unknown:different", base_weights_sha256=BASE_HASH
            ))

    def test_placeholders_and_malformed_hashes_rejected(self):
        for invalid in ("0" * 64, "z" * 64, "ab"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    Artifacts(model_id="m", base_weights_sha256=invalid)
        with self.assertRaises(ValueError):
            Artifacts(model_id="m", base_weights_sha256=BASE_HASH,
                      adapter_sha256="A" * 64)

    def test_empty_model_response_fails_closed(self):
        class EmptyModel:
            model_id = "synthetic:test"

            def generate_chat(self, messages, *, seed):
                return " "
        with self.assertRaisesRegex(ValueError, "empty"):
            run_ordeal(EmptyModel(), artifacts=self.artifacts())

    def test_transport_restricts_to_loopback_and_no_credentials(self):
        for url in ("https://127.0.0.1:8080", "http://example.com:8080",
                    "http://127.0.0.1:8080/alternate",
                    "http://user:password@127.0.0.1:8080"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    LocalChatCompletionsAdapter("model", base_url=url)
        client = LocalChatCompletionsAdapter("model", base_url="http://127.0.0.1:8080")
        self.assertEqual(client.model_id, "model")
        with self.assertRaises(ValueError):
            client.generate_chat([{"role": "tool", "content": "forbidden"}], seed=0)

    def test_seed_validation(self):
        for invalid in (-1, 2**31, True, "12"):
            with self.subTest(seed=invalid):
                with self.assertRaises(ValueError):
                    run_ordeal(FakeChatModel(), artifacts=self.artifacts(), seed=invalid)


if __name__ == "__main__":
    unittest.main()
