"""Public-only candidate/model bridge tests against a real finite examiner."""

from dataclasses import replace
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
    audit_receipt,
    run_candidate,
    run_matched_pair,
    save_receipt_new,
    _public,
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
