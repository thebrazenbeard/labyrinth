"""Independent Wave17 receipt-source and process evidence falsifiers.

Tests are intentionally RED on exact PR161 source 7ecbaf9. Tests only;
they mutate deep-copied local fixed-child receipts, not live services.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import unittest

from labyrinth.switchback_circuit import make_case
from labyrinth.switchback_oracle import adjudicate_session, search, verify_receipt_replay
from scripts import switchback_batch_trial as batch


class Wave17ReceiptTrust(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = batch.BatchConfig((1,), (10,), 1, 1, .5, 3., 1024)
        cls.original = batch.run_batch(config, batch.AdapterSpec())
        if not batch.verify_batch_receipt(cls.original):
            raise AssertionError("original exact fixed-child receipt failed replay")
        if cls.original["cases"][0]["admission"] != "COMPLETED":
            raise AssertionError("positive fixed-child baseline was not completed")

    @staticmethod
    def _rehash_outer(receipt):
        receipt["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in receipt.items()
            if key != "batch_receipt_sha256"})

    def test_outer_source_commit_schema_must_reject_rehashed_string(self):
        forged = deepcopy(self.original)
        self.assertEqual(type(forged["source_commit"]), dict)
        forged["source_commit"] = "0" * 40
        self._rehash_outer(forged)
        self.assertFalse(batch.verify_batch_receipt(forged),
                         "outer Git provenance schema collapsed to an arbitrary string")

    def test_nested_trial_source_commit_schema_must_reject_rehashed_string(self):
        forged = deepcopy(self.original)
        trial = forged["cases"][0]["trial"]
        self.assertEqual(type(trial["source_commit"]), dict)
        trial["source_commit"] = "made-up-source"
        trial["receipt_sha256"] = batch._sha({
            key: value for key, value in trial.items()
            if key != "receipt_sha256"})
        self._rehash_outer(forged)
        self.assertFalse(batch.verify_batch_receipt(forged),
                         "nested source Git provenance schema was not revalidated")

    def test_nested_process_call_and_call_budget_must_be_exact_integers(self):
        """True must not pass as a claimed integer call count of one."""
        def mutate_call_count(trial):
            trial["process_calls"] = True

        def mutate_config_call_limit(trial):
            trial["configuration"]["max_model_calls"] = True
            trial["config_sha256"] = batch._sha(trial["configuration"])

        for mutation in (mutate_call_count, mutate_config_call_limit):
            with self.subTest(mutation=mutation.__name__):
                forged = deepcopy(self.original)
                trial = forged["cases"][0]["trial"]
                mutation(trial)
                trial["receipt_sha256"] = batch._sha({
                    key: value for key, value in trial.items()
                    if key != "receipt_sha256"})
                self._rehash_outer(forged)
                self.assertFalse(batch.verify_batch_receipt(forged),
                                 "bool was laundered into an integer process budget")

    def test_duplicate_key_wire_json_cannot_be_verified_as_parsed_child(self):
        """Wire hash correctness is not equivalent to JSON decision validity."""
        forged = deepcopy(self.original)
        trial = forged["cases"][0]["trial"]
        turn = trial["turns"][0]
        self.assertEqual(turn["decision"], {"kind": "ABSTAIN"})
        malformed = '{"kind":"ABSTAIN","kind":"PLAN","actions":[]}'
        turn["raw_response"] = malformed
        turn["response_sha256"] = sha256(malformed.encode("utf-8")).hexdigest()
        child = turn["process_receipts"][0]
        physical_wire = json.dumps({
            "schema": "labyrinth.minotaur-process.v1",
            "response": malformed,
            "child_pid": child["child_pid"],
            "parent_session_frame_visible": False,
        }, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
        child["response_sha256"] = sha256(physical_wire).hexdigest()
        child["response_bytes"] = len(physical_wire)
        trial["receipt_sha256"] = batch._sha({
            key: value for key, value in trial.items()
            if key != "receipt_sha256"})
        self._rehash_outer(forged)
        self.assertFalse(batch.verify_batch_receipt(forged),
                         "duplicate-key model JSON was accepted as parsed child output")

    def test_extra_process_receipt_and_undeclared_call_rejected(self):
        """Negative control: one budgeted turn cannot claim two child calls."""
        for mutation in ("extra_receipt", "extra_process_call"):
            with self.subTest(mutation=mutation):
                forged = deepcopy(self.original)
                trial = forged["cases"][0]["trial"]
                if mutation == "extra_receipt":
                    turn = trial["turns"][0]
                    turn["process_receipts"].append(deepcopy(
                        turn["process_receipts"][0]))
                else:
                    trial["process_calls"] = 2
                trial["receipt_sha256"] = batch._sha({
                    key: value for key, value in trial.items()
                    if key != "receipt_sha256"})
                self._rehash_outer(forged)
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_command_configuration_never_infers_unsafe_authority(self):
        """Negative control: simply mentioning an argv never grants launch."""
        with self.assertRaises(PermissionError):
            batch.AdapterSpec(mode="command", command_argv=("python",))
        with self.assertRaises(PermissionError):
            batch.AdapterSpec(mode="command", explicit_authorization=True)

    def test_raw_abstain_cannot_be_relabelled_a_winning_command_plan(self):
        """A valid oracle win is not evidence that the child proposed it."""
        config = batch.BatchConfig((0,), (10,), 1, 1, .5, 3., 1024)
        forged = batch.run_batch(config, batch.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(forged))
        row = forged["cases"][0]
        self.assertEqual(row["admission"], "COMPLETED")
        trial = row["trial"]
        turn = trial["turns"][0]
        self.assertEqual(turn["decision"], {"kind": "ABSTAIN"})
        self.assertIn("ABSTAIN", turn["raw_response"])

        path = search(make_case(0))["witness"]
        self.assertIsNotNone(path)
        proposal = {"kind": "PLAN", "actions": path}
        witness = adjudicate_session(make_case(0), [], proposal)
        self.assertTrue(witness["verified_success"])
        self.assertTrue(verify_receipt_replay(witness))

        # This is a *synthetic receipt mutation*, NOT execution of the argv.
        # Recast the original fixed-child raw ABSTAIN as a claimed command
        # process to avoid the intentionally special fixed-script consistency
        # check. All alleged child SHA/length values match the actual ABSTAIN
        # bytes; only the scored semantic decision has been falsely relabeled.
        spec = batch.AdapterSpec(mode="command",
                                 command_argv=("never-executed-unit-fixture",),
                                 explicit_authorization=True)
        forged["manifest"]["adapter"] = asdict(spec)
        forged["manifest_sha256"] = batch._sha(forged["manifest"])
        trial["mode"] = "command"
        trial["configuration"]["mode"] = "command"
        trial["config_sha256"] = batch._sha(trial["configuration"])
        trial["terminal_reason"] = "SUBMITTED"
        trial["source_receipt"] = witness
        trial["source_verdict"] = "VERIFIED_WIN"
        trial["verified_success"] = True
        trial["source_receipt_replay_valid"] = True
        trial["trial_binding_valid"] = True
        turn["decision"] = proposal
        child_receipt = turn["process_receipts"][0]
        child_receipt["parent_session_frame_visible"] = None
        child_receipt["request_sha256"] = turn["prompt_sha256"]
        raw_bytes = turn["raw_response"].encode("utf-8")
        child_receipt["response_sha256"] = sha256(raw_bytes).hexdigest()
        child_receipt["response_bytes"] = len(raw_bytes)
        trial["receipt_sha256"] = batch._sha({
            key: value for key, value in trial.items()
            if key != "receipt_sha256"})

        row["verified_win"] = True
        forged["aggregate"]["source_verified_wins"] = 1
        forged["aggregate"]["verified_fraction_planned"]["numerator"] = 1
        self._rehash_outer(forged)
        self.assertFalse(batch.verify_batch_receipt(forged),
                         "raw child ABSTAIN cannot receive a verified command WIN")


if __name__ == "__main__":
    unittest.main()
