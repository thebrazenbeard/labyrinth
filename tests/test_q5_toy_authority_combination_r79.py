"""Source screens and toy learning cannot compose into examination authority.

This is an application-level negative contract, not a host-custody test. Q5
signer material and the tiny model are controlled by this test process.
"""

from hashlib import sha256
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from labyrinth.journey import (
    DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    begin_journey,
    complete_maze,
)
from labyrinth.minotaur_exam_intake_guard_r36 import admit_development_exam
from labyrinth.minotaur_measured_toy_bridge_r72 import run_measured_toy_corridor
from labyrinth.q5_signed_data_witness_r72 import (
    SCHEMA as Q5_SCHEMA,
    SignedQ5SourceWitness,
    screen_q5_with_signed_source_witness,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.toy_backend import ToyLinearAdapter
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_journey import digest, exam_receipt, scope, training_receipt
from tests.test_q5_evaluation_preregistration_r68 import _fixture


_SIGNING_DOMAIN = b"labyrinth.q5-signed-preoutcome-source-witness.v1\0"


class Q5ToyAuthorityCombinationR79Tests(unittest.TestCase):
    @staticmethod
    def _source_screen():
        outcomes, precommit = _fixture()
        signer = Ed25519PrivateKey.generate()
        key = signer.public_key().public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw)
        event_id = "r79-source-toy-separation"
        body = {
            "schema": Q5_SCHEMA,
            "event_id": event_id,
            "precommit_sha256": precommit.precommit_sha256,
            "signer_public_key_sha256": sha256(key).hexdigest(),
            "declared_stage": "BEFORE_OUTCOMES",
        }
        signed_bytes = (json.dumps(body, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        return screen_q5_with_signed_source_witness(
            precommit=precommit,
            external_expected_precommit_sha256=precommit.precommit_sha256,
            witness=SignedQ5SourceWitness(
                signed_bytes, signer.sign(_SIGNING_DOMAIN + signed_bytes).hex()),
            pinned_signer_public_key=key,
            expected_event_id=event_id,
            consumed_event_sha256s=frozenset(),
            outcomes=outcomes,
        )

    @staticmethod
    def _measured_toy():
        declared = TrainingScopeSpec(
            objective_id="r79-source-toy-separation", domain="reasoning",
            skills=("arithmetic",), specificity=1, scale=2,
            max_cycles=1, seed=321,
        )
        learner = ToyLinearAdapter(seed=17)
        before = learner.model_hash
        report = run_measured_toy_corridor(
            declared, generate_maze_plan(declared, 0), learner,
            epochs=8, learning_rate=0.5,
        )
        return before, learner.model_hash, report

    @staticmethod
    def _development_exam_pass():
        initial = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(initial, training_receipt(initial, "updated"))
        return trained, admit_development_exam(
            trained, exam_receipt(trained, status="pass"))

    def test_valid_q5_and_real_toy_updates_plus_development_pass_still_require_private_qualification(self):
        source = self._source_screen()
        before, after, toy = self._measured_toy()
        _, exam = self._development_exam_pass()

        # This is deliberately the strongest source-only conjunction: good
        # signed-source statistics, actual parameter change and a development
        # exam PASS. None is independently authorized private qualification.
        self.assertIs(source["source_screen_pass"], True)
        self.assertIs(source["signed_witness_source_signature_verified"], True)
        self.assertNotEqual(before, after)
        self.assertEqual(toy["checkpoint_after_sha256"], after)
        self.assertGreater(toy["parameter_updates"], 0)
        self.assertEqual(exam.journey.phase,
                         DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)

        witness = source["witness_source_record"]
        self.assertIs(witness.external_chronology_verified, False)
        self.assertIs(witness.independent_custody_verified, False)
        self.assertIs(witness.external_nonreplay_verified, False)
        self.assertIs(witness.graduation_authorized, False)
        self.assertIs(source["independent_preoutcome_witness_custody_verified"], False)
        self.assertIs(source["graduation_authorized"], False)
        self.assertIs(toy["pretrained_improvement_established"], False)
        self.assertIs(toy["graduation_authorized"], False)
        self.assertTrue(toy["registry_transitions"])
        self.assertTrue(all(
            transition["authorizes_model_update"] is False
            for transition in toy["registry_transitions"]
        ))
        self.assertIs(exam.graduation_authorized, False)
        self.assertIs(exam.evaluator_host_independence_verified, False)
        self.assertIs(exam.neural_execution_verified, False)

    def test_q5_or_toy_report_is_not_a_typed_examination_receipt(self):
        source = self._source_screen()
        _, _, toy = self._measured_toy()
        trained, _ = self._development_exam_pass()

        for report in (source, toy):
            with self.subTest(schema=report.get("schema", "q5-source-screen")):
                with self.assertRaisesRegex(ValueError, "MinotaurExamReceipt"):
                    admit_development_exam(trained, report)


if __name__ == "__main__":
    unittest.main()
