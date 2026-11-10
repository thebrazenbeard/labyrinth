"""Independent adversarial Novel Router child-report trust checks.

Only local fixed-fixture child processes are run. No actual neural model,
remote service, or privileged executable is launched.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import unittest

from scripts import novel_router_model_trial as runner


class NovelRouterProvenanceReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.genuine = runner.run_trial(
            runner.TrialConfig(0, 73, 1, 2.0, 3.0, 4096),
            runner.AdapterSpec())
        if not runner.verify_trial_receipt(cls.genuine):
            raise AssertionError("positive fixed-process receipt not independently replayable")

    def test_rehashed_malformed_source_binary_model_provenance_rejected(self):
        """Typed external provenance fields must not become arbitrary strings."""
        substitutions = (
            ("source_commit", "not-a-git-record"),
            ("executable", "not-an-executable-record"),
            ("source_files", {"runner": "fake"}),
            ("model_artifact", {"sha256": "faked"}),
        )
        for field, value in substitutions:
            with self.subTest(field=field):
                tampered = deepcopy(self.genuine)
                tampered[field] = value
                tampered["receipt_sha256"] = runner._sha({
                    key: item for key, item in tampered.items()
                    if key != "receipt_sha256"})
                self.assertFalse(runner.verify_trial_receipt(tampered),
                                 "malformed claimed source/model identity was replay accepted")

    def test_default_cli_does_not_dump_host_private_seed_and_raw_stdout(self):
        """The marker HOST_PRIVATE must affect the default output channel."""
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(runner.main([
                "--seed", "0", "--max-calls", "1", "--timeout", "2",
                "--total-timeout", "3"]), 0)
        printed = output.getvalue()
        for confidential in ('"case_seed"', '"raw_stdout_base64"'):
            with self.subTest(confidential=confidential):
                self.assertFalse(confidential in printed,
                                 "host-private field disclosed by default CLI stdout")


if __name__ == "__main__":
    unittest.main()
