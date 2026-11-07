"""Independent Wave16 adversarial scoring and receipt-trust controls.

Source baseline: PR157 integration 16f84839df869d2a89243d55baa212cf92527944.
No model, daemon, or host service is installed by these tests; fixed child
subprocesses are the only actual candidate execution.
"""

from copy import deepcopy
from hashlib import sha256
from importlib.util import find_spec
from unittest.mock import patch
import json
import unittest

from labyrinth.minotaur_process_adapter import ProcessModelAdapter, _ABSTAIN
from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth.switchback_oracle import search
from scripts import switchback_model_trial as trial


class SkepticModelEvidence(unittest.TestCase):
    @staticmethod
    def _fixed():
        return ProcessModelAdapter(mode="fixed", actions=(), generation_seed=0,
                                   timeout_seconds=1.0, max_output_bytes=1024)

    @staticmethod
    def _config(seed=0):
        return trial.TrialConfig(seed, 0, "fixed", 1, 1.0, 3.0, 1024)

    def test_single_shared_json_decoder_fault_cannot_invent_model_win(self):
        path = search(make_case(0))["witness"]
        self.assertIsNotNone(path)
        original = trial._read_json

        def corrupt_decoder(raw):
            if raw == _ABSTAIN.encode("utf-8"):
                return {"kind": "PLAN", "actions": path}
            return original(raw)

        with patch.object(trial, "_read_json", new=corrupt_decoder):
            report = trial.run_trial(self._config(0), self._fixed())

        # Actual fixed-process stdout was ABSTAIN, NOT the forged PLAN.
        self.assertEqual(report["turns"][0]["response_sha256"],
                         sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        self.assertFalse(report["verified_success"],
                         "both supposed independent wire decoders shared one corrupted function")

    def test_single_stdlib_json_decoder_fault_cannot_self_certify_false_win(self):
        """Independent wrappers may still share one mutable library function."""
        path = search(make_case(0))["witness"]
        original_loads = json.loads

        def broken_stdlib_decoder(raw, *args, **kwargs):
            if raw in (_ABSTAIN, _ABSTAIN.encode("utf-8")):
                return {"kind": "PLAN", "actions": path}
            return original_loads(raw, *args, **kwargs)

        with patch.object(json, "loads", new=broken_stdlib_decoder):
            report = trial.run_trial(self._config(0), self._fixed())
        self.assertEqual(report["turns"][0]["response_sha256"],
                         sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        self.assertFalse(report["verified_success"],
                         "both wire parsers delegated to a shared patched json.loads")

    def test_unkeyed_outer_checksum_cannot_authenticate_model_claims(self):
        """Negative control: a selfhash is reconstructible without a signer."""
        report = trial.run_trial(self._config(0), self._fixed())
        self.assertFalse(report["verified_success"])
        forged = deepcopy(report)
        forged["model_neural_identity_verified"] = True
        forged["model_weight_update_verified"] = True
        forged["generalization_qualified"] = True
        forged["source_verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["receipt_sha256"] = sha256(trial._canonical({
            key: value for key, value in forged.items()
            if key != "receipt_sha256"})).hexdigest()
        self.assertEqual(forged["receipt_sha256"], sha256(trial._canonical({
            key: value for key, value in forged.items()
            if key != "receipt_sha256"})).hexdigest())
        self.assertNotEqual(forged["source_receipt"]["verdict"], "VERIFIED_WIN")
        self.assertFalse(report["model_neural_identity_verified"])

    def test_public_child_view_hides_seed_but_host_report_exposes_it(self):
        """Negative control: host-private report must never become candidate input."""
        case = make_case(0)
        public = trial._public_request(ProbeSession(case).view(), 0)
        self.assertNotIn('"seed":', public)
        self.assertNotIn('"case_id":', public)
        report = trial.run_trial(self._config(0), self._fixed())
        self.assertEqual(report["case_seed_host_only"], 0)
        self.assertEqual(report["configuration"]["case_seed"], 0)
        self.assertFalse(report["model_neural_identity_verified"])

    @unittest.skipUnless(find_spec("scripts.switchback_batch_trial") is not None,
                         "Wave16 batch runner not yet present on this source head")
    def test_batch_reviewer_rejects_rehashed_completion_and_budget_spoofs(self):
        from scripts import switchback_batch_trial as batch

        config = batch.BatchConfig((0, 1), (10, 11), 1, 1, .5, 3.0, 1024)
        receipt = batch.run_batch(config, batch.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(receipt))
        self.assertEqual(receipt["cases"][1]["admission"],
                         "SKIPPED_GLOBAL_CALL_BUDGET")
        self.assertFalse(receipt["cases"][1]["trial_receipt_valid"])

        def forge_status(forged):
            forged["cases"][1]["admission"] = "COMPLETED"

        def forge_reserved_calls(forged):
            forged["aggregate"]["calls_reserved"] = 999

        def forge_skipped_count(forged):
            forged["aggregate"]["skipped_or_invalid"] = 0

        for mutation in (forge_status, forge_reserved_calls, forge_skipped_count):
            with self.subTest(mutation=mutation.__name__):
                tampered = deepcopy(receipt)
                mutation(tampered)
                tampered["batch_receipt_sha256"] = batch._sha({
                    key: value for key, value in tampered.items()
                    if key != "batch_receipt_sha256"})
                self.assertFalse(batch.verify_batch_receipt(tampered),
                                 "batch verifier accepted forged admission/coverage metric")

    @unittest.skipUnless(find_spec("scripts.switchback_batch_trial") is not None,
                         "Wave16 batch runner not yet present on this source head")
    def test_batch_reviewer_rejects_bool_spoofed_integer_fields(self):
        """A rehashed batch must not treat True as integer 1 in a receipt."""
        from scripts import switchback_batch_trial as batch

        config = batch.BatchConfig((1,), (10,), 1, 1, .5, 3.0, 1024)
        receipt = batch.run_batch(config, batch.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(receipt))

        def seed_as_bool(forged):
            forged["cases"][0]["case_seed"] = True

        def reserved_calls_as_bool(forged):
            forged["aggregate"]["calls_reserved"] = True

        def case_count_as_bool(forged):
            forged["aggregate"]["precommitted_case_count"] = True

        for mutation in (seed_as_bool, reserved_calls_as_bool, case_count_as_bool):
            with self.subTest(mutation=mutation.__name__):
                tampered = deepcopy(receipt)
                mutation(tampered)
                tampered["batch_receipt_sha256"] = batch._sha({
                    key: value for key, value in tampered.items()
                    if key != "batch_receipt_sha256"})
                self.assertFalse(batch.verify_batch_receipt(tampered),
                                 "bool/integer equality laundered noncanonical receipt schema")

    @unittest.skipUnless(find_spec("scripts.switchback_batch_trial") is not None,
                         "Wave16 batch runner not yet present on this source head")
    def test_batch_verifier_fails_closed_on_double_rehashed_wrong_manifest_type(self):
        """Checksum-valid malformed manifests must return False, never raise."""
        from scripts import switchback_batch_trial as batch

        config = batch.BatchConfig((1,), (10,), 1, 1, .5, 3.0, 1024)
        receipt = batch.run_batch(config, batch.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(receipt))

        for invalid_config in (None, [], "not-a-config"):
            with self.subTest(invalid_config=invalid_config):
                tampered = deepcopy(receipt)
                tampered["manifest"]["config"] = invalid_config
                tampered["manifest_sha256"] = batch._sha(tampered["manifest"])
                tampered["batch_receipt_sha256"] = batch._sha({
                    key: value for key, value in tampered.items()
                    if key != "batch_receipt_sha256"})
                self.assertFalse(batch.verify_batch_receipt(tampered),
                                 "batch verifier must fail closed on malformed nested manifest")

    @unittest.skipUnless(find_spec("scripts.switchback_batch_trial") is not None,
                         "Wave16 batch runner not yet present on this source head")
    def test_batch_rehash_cannot_launder_child_wire_or_receipt_evidence(self):
        """Nested rehashing must not erase physical child response provenance."""
        from scripts import switchback_batch_trial as batch

        config = batch.BatchConfig((1,), (10,), 1, 1, .5, 3.0, 1024)
        receipt = batch.run_batch(config, batch.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(receipt))
        self.assertEqual(receipt["cases"][0]["admission"], "COMPLETED")

        def remove_child_receipt(turn):
            turn["process_receipts"] = [None]

        def forge_raw_child_response(turn):
            turn["raw_response"] = "{}"

        def forge_child_response_digest(turn):
            turn["process_receipts"][0]["response_sha256"] = "0" * 64

        for mutation in (remove_child_receipt, forge_raw_child_response,
                         forge_child_response_digest):
            with self.subTest(mutation=mutation.__name__):
                tampered = deepcopy(receipt)
                trial_receipt = tampered["cases"][0]["trial"]
                mutation(trial_receipt["turns"][0])
                trial_receipt["receipt_sha256"] = batch._sha({
                    key: value for key, value in trial_receipt.items()
                    if key != "receipt_sha256"})
                tampered["batch_receipt_sha256"] = batch._sha({
                    key: value for key, value in tampered.items()
                    if key != "batch_receipt_sha256"})
                self.assertFalse(batch.verify_batch_receipt(tampered),
                                 "nested hashes cannot independently attest child execution")


if __name__ == "__main__":
    unittest.main()
