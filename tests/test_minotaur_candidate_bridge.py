"""Public-only candidate/model bridge tests against a real finite examiner."""

from dataclasses import fields, replace
from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.adapters import ReplayAdapter
from labyrinth.timegate_exam import CandidateSession, candidate_view, make_case
from labyrinth.timegate_oracle import reachable_closure
from labyrinth.minotaur_candidate_bridge import (
    FixedActionAdapter,
    RunReceipt,
    TurnReceipt,
    audit_receipt,
    run_candidate,
    run_matched_pair,
    save_receipt_new,
    _public,
    _seal,
)


WINNING_SEED_ZERO = ("forward", "take", "forward", "take", "arm", "forward", "escape")


def context(seed=0, **flags):
    case = make_case(seed, **flags)
    return candidate_view(case), lambda: CandidateSession(case), case


class SpyLocalAdapter:
    model_id = "local:unattested-test-double"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, seed=None):
        self.prompts.append(prompt)
        return '{"kind":"ACTION","action":"wait"}'


class StartOnlyExam:
    """Legacy constant-exam-shaped public seam without a view() method."""

    def __init__(self):
        self.attempts = 0

    def start(self):
        return {"prompt": "Go forward once", "maximum_actions": 3,
                "initial_observation": "At entry"}

    def act(self, action):
        self.attempts += 1
        return {"observation": "Outside" if action == "forward" else "At entry",
                "actions_remaining": 3 - self.attempts,
                "status": "VERIFIED_WIN" if action == "forward" else "IN_PROGRESS"}


class CandidateBridgeTests(unittest.TestCase):
    def test_red_first_missing_bridge_then_fixed_script_replays_real_win(self):
        public, factory, _ = context()
        receipt = run_candidate(public, factory, FixedActionAdapter(WINNING_SEED_ZERO), seed=12)
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertEqual(len(receipt.turns), len(WINNING_SEED_ZERO))
        self.assertTrue(audit_receipt(receipt, public, factory))
        self.assertEqual(receipt.adapter_provenance, "DETERMINISTIC_CONTROL")

    def test_preexisting_terminal_escape_cannot_credit_zero_model_inference(self):
        public, _, _ = context()

        class PrewonSession:
            def public_start(self):
                return public

            def view(self):
                return {"status": "VERIFIED_WIN", "finished": True,
                        "escape_verified": True}

            def act(self, action):
                self.fail_on_action = True
                raise AssertionError("terminal examiner cannot be acted on")

        class ProbeAdapter:
            model_id = "local:zero-inference-probe"

            def __init__(self):
                self.calls = 0

            def generate(self, prompt, *, seed=None):
                self.calls += 1
                return '{"kind":"ACTION","action":"wait"}'

        probe = ProbeAdapter()
        receipt = run_candidate(public, PrewonSession, probe,
                                authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(probe.calls, 0)
        self.assertEqual(receipt.turns, ())
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, public, PrewonSession))
        forged_fields = {field.name: getattr(receipt, field.name)
                         for field in fields(RunReceipt)
                         if field.name != "receipt_sha256"}
        forged = _seal(**{**forged_fields, "verified_escape": True})
        self.assertFalse(audit_receipt(forged, public, PrewonSession))

        pair = run_matched_pair(public, PrewonSession,
                                candidate=FixedActionAdapter(("wait",)),
                                control=FixedActionAdapter(("wait",)))
        self.assertFalse(pair.candidate_win)
        self.assertFalse(pair.control_win)

    def test_claim_only_win_cannot_be_credited_as_executed_escape(self):
        public, _, _ = context()

        class ClaimGrantsFalseEscape:
            def public_start(self):
                return public

            def view(self):
                return {"verdict": "UNKNOWN", "finished": False,
                        "escape_verified": False}

            def act(self, action):
                return {"verdict": "UNKNOWN", "finished": False,
                        "escape_verified": False}

            def submit(self, claim):
                return {"verdict": "VERIFIED_WIN", "finished": True,
                        "escape_verified": True}

        for actions in ((), ("wait",)):
            with self.subTest(actions=actions):
                receipt = run_candidate(
                    public, ClaimGrantsFalseEscape, FixedActionAdapter(actions))
                self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
                self.assertEqual(receipt.turns[-1].disposition, "CLAIM")
                self.assertFalse(receipt.verified_escape)
                self.assertTrue(audit_receipt(receipt, public, ClaimGrantsFalseEscape))
                fields_without_digest = {
                    field.name: getattr(receipt, field.name)
                    for field in fields(RunReceipt)
                    if field.name != "receipt_sha256"
                }
                forged = _seal(**{**fields_without_digest, "verified_escape": True})
                self.assertFalse(audit_receipt(forged, public, ClaimGrantsFalseEscape))

        paired = run_matched_pair(public, ClaimGrantsFalseEscape,
                                  candidate=FixedActionAdapter(()),
                                  control=FixedActionAdapter(()))
        self.assertFalse(paired.candidate_win)
        self.assertFalse(paired.control_win)

        real_public, factory, _ = context()
        earned = run_candidate(real_public, factory,
                               FixedActionAdapter(WINNING_SEED_ZERO))
        self.assertTrue(earned.verified_escape)
        self.assertTrue(audit_receipt(earned, real_public, factory))

    def test_matched_pair_uses_fresh_sessions_and_records_no_learning(self):
        public, factory, _ = context()
        paired = run_matched_pair(
            public, factory,
            candidate=FixedActionAdapter(WINNING_SEED_ZERO, model_id="fixed:memorized-seed0"),
            control=FixedActionAdapter(("wait",) * 16, model_id="fixed:constant-wait"),
            seed=12,
        )
        self.assertTrue(paired.candidate_win)
        self.assertFalse(paired.control_win)
        self.assertFalse(paired.neural_training_claim)
        self.assertEqual(paired.session_isolation, "IN_PROCESS_UNATTESTED")
        self.assertFalse(paired.comparison_qualified)
        self.assertTrue(audit_receipt(paired.candidate, public, factory))
        self.assertTrue(audit_receipt(paired.control, public, factory))

    def test_matched_pair_rejects_single_reused_stateful_adapter(self):
        public, factory, _ = context()
        same = FixedActionAdapter(WINNING_SEED_ZERO)
        with self.assertRaisesRegex(ValueError, "independent adapter instances"):
            run_matched_pair(public, factory, candidate=same, control=same)

    def test_matched_pair_rejects_reused_examiner_session(self):
        public, factory, _ = context()
        singleton = factory()
        with self.assertRaisesRegex(ValueError, "independent examiner sessions"):
            run_matched_pair(public, lambda: singleton,
                             candidate=FixedActionAdapter(WINNING_SEED_ZERO),
                             control=FixedActionAdapter(("wait",)))

    def test_matched_pair_rejects_distinct_proxies_sharing_underlying_session(self):
        public, factory, _ = context()
        shared = factory()

        class Proxy:
            def view(self):
                return shared.view()

            def act(self, action):
                return shared.act(action)

            def submit(self, claim):
                return shared.submit(claim)

        with self.assertRaisesRegex(ValueError, "control examiner changed"):
            run_matched_pair(public, Proxy,
                             candidate=FixedActionAdapter(()),
                             control=FixedActionAdapter(WINNING_SEED_ZERO))

    def test_private_case_identity_never_in_adapter_prompt(self):
        public, factory, case = context(7, sealed=True)
        spy = SpyLocalAdapter()
        with self.assertRaises(PermissionError):
            run_candidate(public, factory, spy)
        self.assertEqual(spy.prompts, [])
        receipt = run_candidate(public, factory, spy, authorize_local_adapter=True)
        self.assertEqual(receipt.adapter_provenance, "MODEL_ADAPTER_UNATTESTED")
        self.assertEqual(len(spy.prompts), 16)
        self.assertTrue(all(case.case_id not in p for p in spy.prompts))
        self.assertTrue(all('"sealed"' not in p and '"case_id"' not in p for p in spy.prompts))
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_control_subclass_cannot_bypass_adapter_authorization(self):
        class Impostor(FixedActionAdapter):
            def generate(self, prompt, *, seed=None):
                return '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}'

        public, factory, _ = context()
        with self.assertRaises(PermissionError):
            run_candidate(public, factory, Impostor(()))
        permitted = run_candidate(public, factory, Impostor(()),
                                  authorize_local_adapter=True)
        self.assertEqual(permitted.adapter_provenance, "MODEL_ADAPTER_UNATTESTED")

    def test_oracle_public_protocol_gives_model_exact_actions_and_claim_shapes(self):
        from labyrinth.oracle_chamber_exam import (
            CandidateSession as OracleSession,
            candidate_view as oracle_view,
            make_case as oracle_case,
        )

        case = oracle_case(7, sealed=True)
        public = oracle_view(case)
        self.assertEqual(set(public), {"prompt", "questions", "remaining_queries",
                                       "submissions", "answers"})
        protocol = public["prompt"]
        self.assertIn("Malformed queries consume budget and return no binary answer.", public["prompt"])
        self.assertIs(type(protocol), str)
        self.assertNotIn("{...}", protocol)  # Never offer invalid JSON pseudo-examples.
        for required in ('"action":"alpha"', '"kind":"COMMIT"',
                         "POLICY", "UNSAT_CERTIFICATE",
                         '"kind":"ABSTAIN"', '"kind":"NOVEL_RULE"',
                         "INVALID"):
            with self.subTest(required=required):
                self.assertIn(required, protocol)

        seen_requests = []

        class ProtocolProbe:
            model_id = "synthetic:oracle-public-format-probe"

            def generate(self, prompt, *, seed=None):
                seen_requests.append(json.loads(prompt))
                return '{"kind":"ACTION","action":"alpha"}'

        factory = lambda: OracleSession(case)
        receipt = run_candidate(public, factory, ProtocolProbe(),
                                authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(receipt.turns[0].disposition, "ACTION")
        self.assertEqual(len(seen_requests), 1)
        self.assertEqual(seen_requests[0]["public_start"]["prompt"], protocol)
        self.assertIn("public_start.prompt", seen_requests[0]["response_instructions"])
        self.assertNotIn('"action":"one action or query from the public list"',
                         seen_requests[0]["response_instructions"])
        self.assertNotIn(case.case_id, json.dumps(seen_requests[0]))
        feedback = json.loads(receipt.turns[0].public_feedback_json)
        self.assertEqual(feedback["answers"][0]["question"], "alpha")
        self.assertIn(feedback["answers"][0]["answer"], (0, 1))
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_oracle_invalid_query_null_is_public_feedback_not_evaluator_failure(self):
        # Oracle Chamber explicitly charges malformed queries and publishes
        # an INVALID answer of null, rather than exposing a private answer.
        from labyrinth.oracle_chamber_exam import (
            CandidateSession as OracleSession,
            candidate_view as oracle_view,
            make_case as oracle_case,
        )

        case = oracle_case(7, sealed=True)
        public = oracle_view(case)
        factory = lambda: OracleSession(case)
        receipt = run_candidate(
            public, factory, FixedActionAdapter(("query",)),
            max_model_calls=1,
        )
        self.assertEqual(receipt.final_verdict, "UNKNOWN")
        self.assertEqual(receipt.turns[0].disposition, "ACTION")
        feedback = json.loads(receipt.turns[0].public_feedback_json)
        self.assertEqual(feedback["answers"], [{"question": "INVALID", "answer": None}])
        self.assertEqual(feedback["remaining_queries"], 4)
        self.assertTrue(audit_receipt(receipt, public, factory))

        # The exception must only admit a null under answers[index].answer.
        for value in (
            {"answer": None},
            {"nested": {"answer": None}},
            {"answers": [{"answer": 2}]},
            {"answers": [{"answer": "0"}]},
            {"answers": [{"answer": {"private": "hidden"}}]},
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "evaluator-only"):
                _public(value)

    def test_recursive_private_public_fields_fail_closed_before_model_call(self):
        public, factory, _ = context()
        public["nested"] = {"case_id": "private"}
        spy = SpyLocalAdapter()
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            run_candidate(public, factory, spy, authorize_local_adapter=True)
        self.assertEqual(spy.prompts, [])
        public, factory, _ = context()
        public["answer"] = 7
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            run_candidate(public, factory, spy, authorize_local_adapter=True)

    def test_process_only_does_not_invoke_python_mapping_callbacks_in_public_input(self):
        # A process boundary does not help when serializing a programmable
        # Mapping executes attacker-supplied Python on the examiner side.
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        effects = []

        class ProgrammablePublicDict(dict):
            def items(self):
                effects.append("python-callback-executed-in-examiner")
                return super().items()

        public, factory, _ = context()
        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c", 'import sys;sys.stdin.buffer.read();print(\'{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}\')'),
            model_id="synthetic:public-container-guard")
        with self.assertRaisesRegex(ValueError, "plain JSON containers"):
            run_candidate(ProgrammablePublicDict(public), factory, adapter,
                          authorize_local_adapter=True,
                          require_process_separation=True,
                          max_model_calls=1)
        self.assertEqual(effects, [])

    def test_subprocess_default_rejects_programmable_public_before_host_callbacks(self):
        # The exact subprocess adapter is process-labeled even without the
        # optional strict flag: it must not execute public-container Python.
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        callbacks = []

        class HostCallbackDict(dict):
            def items(self):
                callbacks.append("host-public-items")
                return super().items()

        public, factory, _ = context()
        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c",
             "import json,sys;json.load(sys.stdin);"
             "print(json.dumps({'kind':'ACTION','action':'wait'}))"),
            model_id="synthetic:default-process-public-guard")
        with self.assertRaisesRegex(ValueError, "plain JSON containers"):
            run_candidate(HostCallbackDict(public), factory, adapter,
                          authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(callbacks, [])

    def test_process_only_rejects_nested_list_subclass_without_iteration(self):
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        effects = []

        class ExecutableList(list):
            def __iter__(self):
                effects.append("iter-called-in-examiner")
                return super().__iter__()

        public, factory, _ = context()
        public["nested_extension"] = {"user_data": ExecutableList([1, 2])}
        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c", "print('{}')"),
            model_id="synthetic:nested-inert-guard")
        with self.assertRaisesRegex(ValueError, "plain JSON containers"):
            run_candidate(public, factory, adapter,
                          authorize_local_adapter=True,
                          require_process_separation=True,
                          max_model_calls=1)
        self.assertEqual(effects, [])

    def test_subprocess_default_rejects_programmable_examiner_feedback(self):
        # Default exact subprocess invocation must enforce the same inert
        # examiner-feedback boundary as explicit process-only mode.
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        callbacks = []

        class HostCallbackFeedback(dict):
            def items(self):
                callbacks.append("host-feedback-items")
                return super().items()

        public, factory, _ = context()

        class Host:
            def __init__(self):
                self.inner = factory()

            def view(self):
                return self.inner.view()

            def act(self, action):
                return HostCallbackFeedback(self.inner.act(action))

            def submit(self, claim):
                return self.inner.submit(claim)

        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c",
             "import json,sys;json.load(sys.stdin);"
             "print(json.dumps({'kind':'ACTION','action':'wait'}))"),
            model_id="synthetic:default-process-feedback-guard")
        receipt = run_candidate(public, Host, adapter,
                                authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(receipt.adapter_provenance, "MODEL_SUBPROCESS_UNATTESTED")
        self.assertEqual(receipt.final_verdict, "EVALUATOR_ERROR")
        self.assertEqual(callbacks, [],
                         "subprocess-labeled path ran host feedback callback")
        self.assertFalse(audit_receipt(receipt, public, Host))

    def test_process_only_rejects_programmable_examiner_feedback(self):
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        effects = []

        class ProgrammableFeedbackDict(dict):
            def items(self):
                effects.append("feedback-items-called-in-examiner")
                return super().items()

        public, factory, _ = context()

        class Host:
            def __init__(self):
                self.inner = factory()

            def view(self):
                return self.inner.view()

            def act(self, action):
                return ProgrammableFeedbackDict(self.inner.act(action))

            def submit(self, claim):
                return self.inner.submit(claim)

        response = '{"kind":"ACTION","action":"wait"}'
        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c",
             "import sys;sys.stdin.buffer.read();print(" + repr(response) + ")"),
            model_id="synthetic:feedback-inert-guard")
        receipt = run_candidate(public, Host, adapter,
                                authorize_local_adapter=True,
                                require_process_separation=True,
                                max_model_calls=1)
        self.assertEqual(receipt.final_verdict, "EVALUATOR_ERROR")
        self.assertEqual(effects, [])
        self.assertFalse(audit_receipt(receipt, public, Host))

    def test_process_only_refuses_unauthorized_adapter_before_public_traversal(self):
        effects = []

        class ProgrammableDict(dict):
            def items(self):
                effects.append("items-called-before-auth")
                return super().items()

        public, factory, _ = context()
        with self.assertRaisesRegex(PermissionError, "process-only"):
            run_candidate(ProgrammableDict(public), factory, SpyLocalAdapter(),
                          require_process_separation=True)
        self.assertEqual(effects, [])

    def test_process_receipt_audit_rejects_programmable_public_input(self):
        import sys
        from labyrinth.minotaur_candidate_bridge import SubprocessCandidateAdapter

        public, factory, _ = context()
        response = '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}'
        adapter = SubprocessCandidateAdapter(
            (sys.executable, "-c",
             "import sys;sys.stdin.buffer.read();print(" + repr(response) + ")"),
            model_id="synthetic:receipt-inert-guard")
        receipt = run_candidate(public, factory, adapter,
                                authorize_local_adapter=True,
                                require_process_separation=True,
                                max_model_calls=1)
        self.assertTrue(audit_receipt(receipt, public, factory))
        effects = []

        class ProgrammableDict(dict):
            def items(self):
                effects.append("replay-callback-executed-in-examiner")
                return super().items()

        self.assertFalse(audit_receipt(receipt, ProgrammableDict(public), factory))
        self.assertEqual(effects, [])

    def test_public_snapshot_cannot_serialize_different_mapping_than_it_inspected(self):
        """Public validation and JSON bytes must use the identical primitive snapshot."""
        class DeceptiveMapping(Mapping):
            def __iter__(self):
                return iter(("observation",))

            def __len__(self):
                return 1

            def __getitem__(self, key):
                return "SECRET" if key == "private" else "LOOK"

            def items(self):
                return [("observation", "LOOK")]

            def keys(self):
                return ["private"]

        class DeceptiveDict(dict):
            def items(self):
                return [("observation", "LOOK")]

        self.assertEqual(json.loads(_public(DeceptiveMapping())),
                         {"observation": "LOOK"})
        self.assertEqual(json.loads(_public({"nested": DeceptiveMapping()})),
                         {"nested": {"observation": "LOOK"}})
        self.assertEqual(json.loads(_public(DeceptiveDict(private="SECRET"))),
                         {"observation": "LOOK"})

    def test_examiner_feedback_logic_uses_same_snapshot_as_receipt(self):
        """A host Mapping cannot claim a win via get() while recording failure."""
        class SplitFeedback(Mapping):
            data = {"status": "UNSOLVED", "finished": True,
                    "escape_verified": False, "observation": "Still in maze"}

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, key):
                return self.data[key]

            def get(self, key, default=None):
                if key == "status":
                    return "VERIFIED_WIN"
                if key == "escape_verified":
                    return True
                return self.data.get(key, default)

        class Host:
            def start(self):
                return {"maximum_actions": 1, "initial_observation": "Still in maze"}

            def act(self, action):
                return SplitFeedback()

        public = Host().start()
        receipt = run_candidate(public, Host, FixedActionAdapter(("WAIT",)))
        self.assertEqual(receipt.final_verdict, "UNSOLVED")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, public, Host))
        self.assertEqual(json.loads(receipt.turns[0].public_feedback_json)["status"],
                         receipt.final_verdict)

    def test_post_action_view_cannot_inject_unrecorded_win(self):
        """Verdict and escape witness must come from one recorded observation."""
        class SplitViewHost:
            def __init__(self):
                self.acted = False

            def view(self):
                if self.acted:
                    return {"status": "VERIFIED_WIN", "finished": True,
                            "escape_verified": False, "observation": "different"}
                return {"status": "UNKNOWN", "finished": False,
                        "maximum_actions": 1, "observation": "start"}

            def act(self, action):
                self.acted = True
                return {"status": "UNSOLVED", "finished": True,
                        "escape_verified": True, "observation": "same"}

        public = SplitViewHost().view()
        receipt = run_candidate(public, SplitViewHost, FixedActionAdapter(("WAIT",)))
        self.assertEqual(receipt.final_verdict, "UNSOLVED")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, public, SplitViewHost))

    def test_legacy_start_only_exam_is_adaptable_without_private_case(self):
        public = StartOnlyExam().start()
        factory = StartOnlyExam
        receipt = run_candidate(public, factory, FixedActionAdapter(("forward",)))
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertFalse(receipt.verified_escape)  # Status-only is not escape evidence.
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_budget_and_invalid_actions_never_grant_extra_turns(self):
        public, factory, _ = context()
        receipt = run_candidate(public, factory, FixedActionAdapter(("teleport",) * 18))
        self.assertEqual(len(receipt.turns), 16)
        self.assertEqual(receipt.final_verdict, "VERIFIED_FAILURE")
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_malformed_adapter_response_fails_without_retries_or_effects(self):
        public, factory, _ = context()
        receipt = run_candidate(public, factory,
                                ReplayAdapter({}, default='{"kind":"ACTION","kind":"CLAIM"}'))
        self.assertEqual(receipt.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(len(receipt.turns), 1)
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_oversized_output_preserves_one_character_of_overflow_for_audit(self):
        public, factory, _ = context()
        # The first 4096 chars parse as a valid action; the real response
        # exceeds the cap and must still audit as malformed output.
        valid_prefix = '{"kind":"ACTION","action":"wait"}'.ljust(4096)
        oversized_response = valid_prefix + "!"

        class OversizedAdapter:
            model_id = "synthetic:overlong-adapter"

            def generate(self, prompt, *, seed=None):
                return oversized_response

        receipt = run_candidate(public, factory, OversizedAdapter(),
                                authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(receipt.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(receipt.turns[0].response_text, oversized_response)
        self.assertEqual(len(receipt.turns[0].response_text), 4097)
        self.assertTrue(audit_receipt(receipt, public, factory))

        # An arbitrarily longer output remains bounded at the same prefix.
        oversized_response = valid_prefix + "!" * 6000
        larger = run_candidate(public, factory, OversizedAdapter(),
                               authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(larger.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(larger.turns[0].response_text, oversized_response[:4097])
        self.assertTrue(audit_receipt(larger, public, factory))

    def test_rehashed_invalid_output_cannot_claim_unbounded_response_witness(self):
        public, factory, _ = context()
        real = run_candidate(
            public, factory, ReplayAdapter({}, default="!" * 9000),
            max_model_calls=1)
        self.assertEqual(real.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(len(real.turns[0].response_text), 4097)
        self.assertTrue(audit_receipt(real, public, factory))

        original = {
            field.name: getattr(real, field.name)
            for field in fields(RunReceipt)
            if field.name != "receipt_sha256"
        }
        for impossible_length in (4098, 9000):
            with self.subTest(length=impossible_length):
                forged_turn = replace(
                    real.turns[0], response_text="!" * impossible_length)
                rehashed = _seal(**{**original, "turns": (forged_turn,)})
                self.assertNotEqual(real.receipt_sha256, rehashed.receipt_sha256)
                self.assertFalse(
                    audit_receipt(rehashed, public, factory),
                    "replay accepted impossible unbounded model-output witness")

    def test_nonstring_adapter_response_cannot_run_programmable_string_conversion(self):
        """Malformed adapter output must not invoke attacker-owned __str__."""
        public, factory, _ = context()
        calls = []

        class PoisonedText:
            def __str__(self):
                calls.append("str-invoked")
                raise RuntimeError("attacker controlled string conversion")

        class PoisonedAdapter:
            model_id = "hostile:unattested-nonstring"

            def generate(self, prompt, *, seed=None):
                calls.append("generated")
                return PoisonedText()

        receipt = run_candidate(public, factory, PoisonedAdapter(),
                                authorize_local_adapter=True)
        self.assertEqual(receipt.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(receipt.turns[0].response_text, "")
        self.assertEqual(calls, ["generated"])
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_surrogate_json_action_is_invalid_and_receipt_still_replays(self):
        public, factory, _ = context()
        # Literal JSON escape is ASCII, but json.loads materializes a lone
        # surrogate that cannot be encoded as UTF-8 for a signed receipt.
        response = r'{"kind":"ACTION","action":"\ud800"}'
        receipt = run_candidate(public, factory, ReplayAdapter({}, default=response))
        self.assertEqual(receipt.final_verdict, "MODEL_OUTPUT_INVALID")
        self.assertEqual(len(receipt.turns), 1)
        self.assertIsNone(receipt.turns[0].public_feedback_json)
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_invalid_unicode_model_identity_rejected_before_examiner_use(self):
        public, factory, _ = context()
        called = []

        def counted_factory():
            called.append("session")
            return factory()

        with self.assertRaisesRegex(ValueError, "Unicode"):
            run_candidate(public, counted_factory,
                          FixedActionAdapter((), model_id="invalid:\ud800"))
        self.assertEqual(called, [])

    def test_invalid_unicode_public_string_rejected_before_examiner_use(self):
        public, factory, _ = context()
        public["note"] = "illformed:\ud800"
        called = []

        def counted_factory():
            called.append("session")
            return factory()

        with self.assertRaisesRegex(ValueError, "Unicode"):
            run_candidate(public, counted_factory, FixedActionAdapter(()))
        self.assertEqual(called, [])

    def test_cyclic_or_overdeep_host_public_view_fails_boundedly(self):
        public, factory, _ = context()
        public["cycle"] = public
        with self.assertRaisesRegex(ValueError, "cycle"):
            run_candidate(public, factory, FixedActionAdapter(()))

        public, factory, _ = context()
        cursor = public
        for index in range(40):
            child = {}
            cursor[f"layer-{index}"] = child
            cursor = child
        with self.assertRaisesRegex(ValueError, "nesting"):
            run_candidate(public, factory, FixedActionAdapter(()))

    def test_explicit_abstention_is_not_escape_success(self):
        public, factory, _ = context(17, sealed=True)
        adapter = ReplayAdapter({}, default=json.dumps(
            {"kind": "CLAIM", "claim": {"kind": "ABSTAIN", "assessment": "UNKNOWN"}}))
        receipt = run_candidate(public, factory, adapter)
        self.assertEqual(receipt.final_verdict, "UNKNOWN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_json_certificate_is_verified_but_not_mislabeled_escape(self):
        public, factory, case = context(17, sealed=True)
        claim = {"kind": "UNSAT_CERTIFICATE",
                 "states": [list(state) for state in reachable_closure(case)],
                 "declared_rules": {"period": case.period,
                                    "charge_phase": case.charge_phase,
                                    "pass_phase": case.pass_phase,
                                    "initial_phase": case.initial_phase,
                                    "lifetime": case.lifetime,
                                    "sealed": case.sealed}}
        response = json.dumps({"kind": "CLAIM", "claim": claim})
        receipt = run_candidate(public, factory, ReplayAdapter({}, default=response))
        self.assertEqual(receipt.final_verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(json.loads(receipt.turns[0].public_feedback_json)["certificate_verified"])
        self.assertTrue(audit_receipt(receipt, public, factory))

    def test_receipt_tampering_or_wrong_world_fails_independent_replay(self):
        public, factory, _ = context()
        receipt = run_candidate(public, factory, FixedActionAdapter(WINNING_SEED_ZERO))
        self.assertFalse(audit_receipt(replace(receipt, final_verdict="VERIFIED_FAILURE"), public, factory))
        other_public, other_factory, _ = context(1)
        self.assertFalse(audit_receipt(receipt, other_public, other_factory))
        self.assertFalse(audit_receipt(replace(receipt, source_tree_sha256="0" * 64), public, factory))
        self.assertFalse(audit_receipt(replace(receipt, max_model_calls=2), public, factory))

    def test_receipt_subclass_never_runs_host_callback_during_audit_or_save(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        callbacks = []

        class ProgrammableReceipt(RunReceipt):
            def __getattribute__(self, key):
                if key == "schema":
                    callbacks.append("receipt host callback")
                return super().__getattribute__(key)

        supplied = ProgrammableReceipt(**{
            field.name: getattr(valid, field.name) for field in fields(RunReceipt)
        })
        self.assertFalse(audit_receipt(supplied, public, factory))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            with self.assertRaises(TypeError):
                save_receipt_new(path, supplied)
            self.assertFalse(path.exists())
        self.assertEqual(callbacks, [])

    def test_nested_programmable_turn_cannot_execute_during_audit_or_save(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        callbacks = []

        class ProgrammableTurn(TurnReceipt):
            def __getattribute__(self, key):
                if key == "prompt_json":
                    callbacks.append("turn host callback")
                return super().__getattribute__(key)

        turn = valid.turns[0]
        forged = replace(valid, turns=(ProgrammableTurn(
            turn.prompt_json, turn.response_text,
            turn.public_feedback_json, turn.disposition),))
        self.assertFalse(audit_receipt(forged, public, factory))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            with self.assertRaises(TypeError):
                save_receipt_new(path, forged)
            self.assertFalse(path.exists())
        self.assertEqual(callbacks, [])

    def test_receipt_scalar_object_cannot_run_deepcopy_callback(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        callbacks = []

        class ProgrammableScalar:
            def __deepcopy__(self, memo):
                callbacks.append("deepcopy host callback")
                return "forged"

        forged = replace(valid, source_tree_sha256=ProgrammableScalar())
        self.assertFalse(audit_receipt(forged, public, factory))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            with self.assertRaises(TypeError):
                save_receipt_new(path, forged)
            self.assertFalse(path.exists())
        self.assertEqual(callbacks, [])

    def test_write_once_rejects_stale_self_digest_before_file_creation(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        stale = replace(valid, final_verdict="VERIFIED_FAILURE")
        self.assertNotEqual(valid.final_verdict, stale.final_verdict)
        self.assertFalse(audit_receipt(stale, public, factory))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tampered.json"
            with self.assertRaisesRegex(ValueError, "receipt digest"):
                save_receipt_new(path, stale)
            self.assertFalse(path.exists())

    def test_evaluator_error_cannot_be_audited_without_replaying_failure(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        receipt_fields = {field.name: getattr(valid, field.name)
                          for field in fields(RunReceipt)
                          if field.name != "receipt_sha256"}
        receipt_fields.update(
            turns=(replace(valid.turns[0], disposition="EVALUATOR_ERROR",
                           public_feedback_json=None),),
            final_verdict="EVALUATOR_ERROR", verified_escape=False)
        forged = _seal(**receipt_fields)
        # The action succeeds in the real examiner. A self-hash cannot turn it
        # into independently replayed evidence of an evaluator exception.
        self.assertNotEqual(forged.receipt_sha256, valid.receipt_sha256)
        self.assertFalse(audit_receipt(forged, public, factory))

    def test_model_call_error_requires_empty_response_on_audit(self):
        public, factory, _ = context()
        valid = run_candidate(public, factory, FixedActionAdapter(("wait",)))
        receipt_fields = {field.name: getattr(valid, field.name)
                          for field in fields(RunReceipt)
                          if field.name != "receipt_sha256"}
        receipt_fields.update(
            turns=(replace(valid.turns[0], disposition="MODEL_CALL_ERROR",
                           public_feedback_json=None),),
            final_verdict="MODEL_CALL_ERROR", verified_escape=False)
        forged = _seal(**receipt_fields)
        self.assertFalse(audit_receipt(forged, public, factory))

    def test_model_call_error_cannot_be_audited_without_replaying_failure(self):
        public, factory, _ = context()
        valid = run_candidate(
            public, factory, FixedActionAdapter(("wait",)), max_model_calls=1)
        self.assertTrue(audit_receipt(valid, public, factory))
        self.assertEqual(valid.turns[0].disposition, "ACTION")
        fields_without_digest = {
            field.name: getattr(valid, field.name)
            for field in fields(RunReceipt)
            if field.name != "receipt_sha256"
        }
        forged = _seal(**{
            **fields_without_digest,
            "turns": (replace(valid.turns[0], response_text="",
                              public_feedback_json=None,
                              disposition="MODEL_CALL_ERROR"),),
            "final_verdict": "MODEL_CALL_ERROR",
            "verified_escape": False,
        })
        # Model generated a valid ACTION, yet a digest-resealed exception
        # assertion must not count as independently replayed failure.
        self.assertFalse(audit_receipt(forged, public, factory))

    def test_real_model_call_exception_is_recorded_but_not_replay_verified(self):
        public, factory, _ = context()

        class FailingAdapter:
            model_id = "synthetic:raises-before-output"

            def generate(self, prompt, *, seed=None):
                raise RuntimeError("synthetic model-call exception")

        receipt = run_candidate(public, factory, FailingAdapter(),
                                authorize_local_adapter=True, max_model_calls=1)
        self.assertEqual(receipt.final_verdict, "MODEL_CALL_ERROR")
        self.assertEqual(receipt.turns[0].response_text, "")
        self.assertIsNone(receipt.turns[0].public_feedback_json)
        self.assertFalse(audit_receipt(receipt, public, factory))

    def test_rehashed_receipt_cannot_stop_early_with_unused_model_calls(self):
        public, factory, _ = context()
        legitimate = run_candidate(
            public, factory, FixedActionAdapter(("wait",)), max_model_calls=1)
        self.assertEqual(len(legitimate.turns), 1)
        self.assertEqual(legitimate.turns[0].disposition, "ACTION")
        self.assertTrue(audit_receipt(legitimate, public, factory))

        # Re-sign an otherwise real one-action trace to claim a larger budget.
        # The runner cannot stop after ACTION with calls still available.
        fields_without_digest = {
            field.name: getattr(legitimate, field.name)
            for field in fields(RunReceipt)
            if field.name != "receipt_sha256"
        }
        forged = _seal(**{**fields_without_digest, "max_model_calls": 32})
        self.assertNotEqual(forged.receipt_sha256, legitimate.receipt_sha256)
        self.assertFalse(audit_receipt(forged, public, factory))

    def test_rehashed_out_of_range_seed_fails_before_examiner_replay(self):
        public, factory, _ = context()
        legitimate = run_candidate(
            public, factory, FixedActionAdapter(("wait",)), seed=12,
            max_model_calls=1)
        self.assertTrue(audit_receipt(legitimate, public, factory))
        fields_without_digest = {
            field.name: getattr(legitimate, field.name)
            for field in fields(RunReceipt)
            if field.name != "receipt_sha256"
        }

        for impossible_seed in (-1, 2**31):
            with self.subTest(seed=impossible_seed):
                forged = _seal(**{**fields_without_digest, "seed": impossible_seed})
                calls = []

                def counted_factory():
                    calls.append(1)
                    return factory()

                self.assertFalse(audit_receipt(forged, public, counted_factory))
                self.assertEqual(calls, [])

    def test_rehashed_impossible_model_metadata_rejected_before_examiner_use(self):
        public, factory, _ = context()
        legitimate = run_candidate(
            public, factory, FixedActionAdapter(("wait",)),
            seed=12, max_model_calls=1)
        self.assertTrue(audit_receipt(legitimate, public, factory))
        original = {
            field.name: getattr(legitimate, field.name)
            for field in fields(RunReceipt)
            if field.name != "receipt_sha256"
        }
        impossible_values = (
            ("adapter_provenance", "INDEPENDENTLY_VERIFIED_NEURAL_MODEL"),
            ("adapter_provenance", ""),
            ("model_id_unattested", ""),
            ("model_id_unattested", "x" * 257),
        )
        for key, value in impossible_values:
            with self.subTest(field=key, value_length=len(value)):
                forged = _seal(**{**original, key: value})
                examiner_calls = []

                def counted_factory():
                    examiner_calls.append(1)
                    return factory()

                self.assertFalse(audit_receipt(forged, public, counted_factory))
                self.assertEqual(examiner_calls, [])

    def test_write_once_receipt_and_roundtrip_self_digest(self):
        public, factory, _ = context()
        receipt = run_candidate(public, factory, FixedActionAdapter(WINNING_SEED_ZERO))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            save_receipt_new(path, receipt)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["receipt_sha256"], receipt.receipt_sha256)
            with self.assertRaises(FileExistsError):
                save_receipt_new(path, receipt)


if __name__ == "__main__":
    unittest.main()
