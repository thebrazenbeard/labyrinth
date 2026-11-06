"""Compatibility and opt-in Kobayashi tests using offline fake transports."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import unittest

from labyrinth.kobayashi_completion_gate_v1 import (
    CompletionGateError, KobayashiCompletionLedger,
)
from labyrinth.local_model_action_adapter_v1 import (
    LocalModelActionConfig, ModelActionBoundaryError,
)
from labyrinth.local_model_world_runner_v1 import (
    EVIDENCE_CLASS, SCHEMA, _trace_digest, main, run_local_model_world,
)
from tests.test_local_model_world_session_v1 import (
    _actor, _host_bound_appeal,
)
from tests.test_local_model_kobayashi_runner_v1 import FakeGuidedModelTransport


class FixedFakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, config, payload):
        self.requests.append((config, payload))
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


class LegacyWorldCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = LocalModelActionConfig(port=11434, model="fixture-local", retries=0)

    def test_existing_public_api_one_action_preserves_source_receipt_and_no_minotaur_claim(self):
        transport = FixedFakeTransport(['{"kind":"take","target":"ariadne_thread"}'])
        result = run_local_model_world(self.config, transport, max_actions=1)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["evidence_class"], EVIDENCE_CLASS)
        self.assertEqual(result["outcome"], "ACTION_BUDGET_EXHAUSTED")
        self.assertEqual(result["world_final_state"]["inventory"], ("ariadne_thread",))
        self.assertEqual(result["model_completion_requests"], 1)
        self.assertEqual(result["turns"][0]["event"]["outcome"], "ACQUIRED")
        self.assertFalse(result["kobayashi_enforced"])
        self.assertFalse(result["host_completion_accepted"])
        self.assertFalse(result["claims"]["minotaur_exam_passed"])
        self.assertEqual(result["trace_sha256"], _trace_digest({
            key: value for key, value in result.items() if key != "trace_sha256"
        }))

    def test_rejected_legacy_action_stops_source_world_unchanged(self):
        transport = FixedFakeTransport(['{"action":"graduate","score":100}'])
        result = run_local_model_world(self.config, transport, max_actions=2)
        self.assertEqual(result["outcome"], "MODEL_ACTION_REJECTED")
        self.assertEqual(result["world_final_state"]["steps"], 0)
        self.assertEqual(result["model_completion_requests"], 1)
        self.assertEqual(result["turns"], [])

    def test_historical_public_completion_is_real_bypass_but_not_host_acceptance(self):
        guided = FakeGuidedModelTransport()
        result = run_local_model_world(self.config, guided,
                                       public_seed=1234, max_actions=100)
        self.assertEqual(result["outcome"], "PUBLIC_QUEST_COMPLETED")
        self.assertTrue(result["world_final_state"]["completed"])
        self.assertIn("MINOTAUR_ENCOUNTER",
                      [turn["event"]["outcome"] for turn in result["turns"]])
        self.assertFalse(result["kobayashi_enforced"])
        self.assertFalse(result["host_completion_accepted"])
        self.assertFalse(result["claims"]["minotaur_exam_passed"])

    def test_strict_mode_refuses_missing_host_inputs_before_model_calls(self):
        guided, actor, proposal, proposal_config = _actor()
        with self.assertRaisesRegex(ModelActionBoundaryError, "strict Kobayashi.*requires"):
            run_local_model_world(actor.config, guided, require_kobayashi=True,
                                  max_actions=128)
        self.assertEqual(guided.requests, [])
        self.assertEqual(proposal.requests, [])

    def test_strict_evidence_refs_rejected_before_model_or_proposal_request(self):
        for refs in (("not-a-hash",), ("a" * 64, "a" * 64), ["a" * 64]):
            with self.subTest(refs=refs):
                guided, actor, proposal, proposal_config = _actor()
                with self.assertRaises(CompletionGateError):
                    run_local_model_world(
                        actor.config, guided, require_kobayashi=True,
                        proposal_config=proposal_config,
                        proposal_transport=proposal,
                        completion_ledger=KobayashiCompletionLedger(),
                        maze_id="preflight-evidence-test",
                        scenario_instance_sha256="e" * 64,
                        host_appeal=_host_bound_appeal,
                        evidence_refs=refs,
                    )
                self.assertEqual(guided.requests, [])
                self.assertEqual(proposal.requests, [])

    def test_strict_mode_preserves_legacy_arguments_and_host_accepted_completion(self):
        guided, actor, proposal, proposal_config = _actor()
        ledger = KobayashiCompletionLedger()
        result = run_local_model_world(
            actor.config, guided, public_seed=1234, difficulty=1,
            max_actions=128, require_kobayashi=True,
            proposal_config=proposal_config, proposal_transport=proposal,
            completion_ledger=ledger, maze_id="compat-instance-one",
            scenario_instance_sha256="a" * 64,
            host_appeal=_host_bound_appeal, variant_seed=2)
        self.assertTrue(result["kobayashi_enforced"])
        self.assertTrue(result["host_completion_accepted"])
        self.assertEqual(result["host_completion_acceptance"], "SOURCE_ONLY_UNATTESTED")
        self.assertEqual(result["session_schema"],
                         "labyrinth.host-owned-kobayashi-world-session.v1")
        self.assertNotIn("outcome", result)  # different strict receipt schema
        self.assertNotIn("trace_sha256", result)
        self.assertEqual(result["local_encounter_status"], "LOCAL_LOSS")
        self.assertTrue(result["maze_completed"])
        self.assertFalse(result["lifecycle_snapshot"]["maze_failed_due_to_local_loss"])
        accepted = ledger.lookup("compat-instance-one", "a" * 64)
        self.assertEqual(accepted.receipt_sha256, result["host_completion_receipt_sha256"])
        self.assertGreater(len(guided.requests), 3)
        self.assertEqual(len(proposal.requests), 1)

    def test_strict_mode_rejects_committed_instance_without_repeating_model_calls(self):
        ledger = KobayashiCompletionLedger()
        guided, actor, proposal, proposal_config = _actor()
        options = dict(require_kobayashi=True, public_seed=1234,
                       max_actions=128, proposal_config=proposal_config,
                       proposal_transport=proposal,
                       completion_ledger=ledger, maze_id="already-committed",
                       scenario_instance_sha256="c" * 64,
                       host_appeal=_host_bound_appeal, variant_seed=2)
        run_local_model_world(actor.config, guided, **options)
        guided_again, actor_again, proposal_again, proposal_config_again = _actor()
        with self.assertRaisesRegex(CompletionGateError, "already committed"):
            run_local_model_world(
                actor_again.config, guided_again,
                **{**options, "proposal_config": proposal_config_again,
                   "proposal_transport": proposal_again})
        self.assertEqual(guided_again.requests, [])
        self.assertEqual(proposal_again.requests, [])

    def test_strict_mode_rejects_closed_ledger_without_model_calls(self):
        guided, actor, proposal, proposal_config = _actor()
        ledger = KobayashiCompletionLedger()
        ledger.close()
        with self.assertRaisesRegex(CompletionGateError, "closed"):
            run_local_model_world(
                actor.config, guided, require_kobayashi=True, max_actions=128,
                proposal_config=proposal_config, proposal_transport=proposal,
                completion_ledger=ledger, maze_id="closed-ledger",
                scenario_instance_sha256="d" * 64,
                host_appeal=_host_bound_appeal)
        self.assertEqual(guided.requests, [])
        self.assertEqual(proposal.requests, [])

    def test_legacy_branch_rejects_host_gate_parameters_instead_of_silent_bypass(self):
        transport = FixedFakeTransport([])
        with self.assertRaisesRegex(ModelActionBoundaryError, "require_kobayashi=True"):
            run_local_model_world(self.config, transport,
                                  completion_ledger=KobayashiCompletionLedger())
        self.assertEqual(transport.requests, [])

    def test_cli_requires_explicit_live_opt_in_and_preserves_legacy_partial_exit(self):
        called = []
        def factory():
            called.append(True)
            raise AssertionError("should not construct transport")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as ctx:
            main(["--model", "fixture-local", "--port", "11434"],
                 transport_factory=factory)
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(called, [])
        out = StringIO()
        fake = FixedFakeTransport(['{"kind":"inspect","target":""}'])
        with redirect_stdout(out):
            exit_code = main([
                "--model", "fixture-local", "--port", "11434",
                "--allow-local-inference", "--max-actions", "1",
            ], transport_factory=lambda: fake)
        self.assertEqual(exit_code, 1)
        receipt = json.loads(out.getvalue())
        self.assertFalse(receipt["kobayashi_enforced"])
        self.assertFalse(receipt["host_completion_accepted"])

    def test_validation_blocks_malformed_budgets_before_model_calls(self):
        transport = FixedFakeTransport([])
        for changes in ({"max_actions": 0}, {"difficulty": 6},
                        {"public_seed": -1}, {"max_actions": True},
                        {"require_kobayashi": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_local_model_world(self.config, transport, **changes)
        self.assertEqual(transport.requests, [])


if __name__ == "__main__":
    unittest.main()
