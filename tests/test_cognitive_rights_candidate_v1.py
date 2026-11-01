"""Fail-closed provenance packaging for candidate cognitive TRAIN examples."""

import hashlib
import json
from dataclasses import asdict
from unittest import TestCase

from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.cognitive_rights_candidate_v1 import (
    CandidateAdmissionUnavailable,
    prepare_cognitive_rights_candidate,
    prepare_private_cognitive_steward_manifest,
    require_cognitive_rights_admission,
)
from labyrinth.journey import begin_journey
from tests.test_cognitive_objective_training_run import cognitive_contract


def _dataset(seed=1201):
    contract = cognitive_contract(seed=seed)
    return export_cognitive_train_jsonl(
        contract, begin_journey(contract.scope, checkpoint_sha256="1" * 64)
    )


def _candidate(data=None, **kwargs):
    if data is None:
        data = _dataset()
    params = {"source_repo": "thebrazenbeard/labyrinth",
              "source_commit": "a" * 40, "source_file_sha256": "b" * 64}
    params.update(kwargs)
    return prepare_cognitive_rights_candidate(data, **params)


def _private(data=None, **kwargs):
    if data is None:
        data = _dataset()
    params = {"source_repo": "thebrazenbeard/labyrinth",
              "source_commit": "a" * 40, "source_file_sha256": "b" * 64}
    params.update(kwargs)
    return prepare_private_cognitive_steward_manifest(data, **params)


class CognitiveRightsCandidateTests(TestCase):
    def test_public_dataclass_recursive_serialization_cannot_disclose_private_data(self):
        first = _candidate(_dataset(1201))
        second = _candidate(_dataset(1205))
        first_dict = asdict(first)
        second_dict = asdict(second)
        self.assertEqual(first_dict, second_dict)
        self.assertEqual(set(first_dict), {"manifest_bytes", "sha256", "case_count"})
        self.assertNotIn("_private_steward_manifest", first_dict)

    def test_original_train_only_digest_catalog_is_label_free_and_not_admitted(self):
        source = _dataset()
        candidate = _candidate(source)
        self.assertEqual(candidate.sha256, hashlib.sha256(candidate.manifest_bytes).hexdigest())
        self.assertFalse(candidate.authorizes_training)
        self.assertFalse(candidate.authorizes_host_execution)
        public = json.loads(candidate.manifest_bytes)
        self.assertEqual(public, {
            "schema": "labyrinth.cognitive-rights-public-status.v1",
            "status": "UNVERIFIED_REJECTED",
            "authorizes_training": False,
            "authorizes_host_execution": False,
        })
        private = json.loads(_private(source).private_bytes())
        self.assertEqual(private["schema"], "labyrinth.cognitive-rights-private-steward.v1")
        self.assertEqual(private["source"]["dataset_sha256"], source.sha256)
        self.assertEqual(private["source"]["producer_commit"], "a" * 40)
        self.assertEqual(private["source"]["producer_file_sha256"], "b" * 64)
        self.assertEqual(private["source"]["authorship_verified"], False)
        self.assertEqual(private["admission"]["status"], "UNVERIFIED_REJECTED")
        self.assertEqual(private["rights"]["permission"], "NOT_ESTABLISHED")
        self.assertIsNone(private["splits"]["dev"])
        self.assertEqual(private["splits"]["train"]["count"], 15)
        self.assertEqual(len(private["splits"]["train"]["rows"]), 15)
        for line in source.jsonl_bytes.splitlines():
            row = json.loads(line)
            self.assertNotIn(row["prompt"], candidate.manifest_bytes.decode())
            self.assertNotIn(row["case_id"], candidate.manifest_bytes.decode())
            self.assertNotIn('"target_choice"', candidate.manifest_bytes.decode())
            self.assertNotIn(row["source_binding_sha256"], candidate.manifest_bytes.decode())
            self.assertNotIn(hashlib.sha256(line + b"\n").hexdigest(),
                             candidate.manifest_bytes.decode())
        self.assertEqual(set(private["splits"]["train"]["rows"][0]),
                         {"case_id_sha256", "prompt_sha256", "semantic_sha256",
                          "source_binding_sha256", "row_sha256"})

    def test_public_status_is_identical_for_different_seeds_answers_and_provenance(self):
        first_dataset = _dataset(1201)
        second_dataset = _dataset(1205)
        self.assertNotEqual(
            json.loads(first_dataset.jsonl_bytes.splitlines()[0])["target_choice"],
            json.loads(second_dataset.jsonl_bytes.splitlines()[0])["target_choice"],
        )
        first = _candidate(first_dataset)
        second = _candidate(second_dataset, source_commit="c" * 40,
                            source_file_sha256="d" * 64,
                            exclusion_roster_sha256="e" * 64,
                            exclusion_semantic_sha256=("f" * 64,))
        self.assertEqual(first.manifest_bytes, second.manifest_bytes)
        self.assertEqual(first.sha256, second.sha256)
        private_first = _private(first_dataset)
        private_second = _private(second_dataset, source_commit="c" * 40,
                                  source_file_sha256="d" * 64,
                                  exclusion_roster_sha256="e" * 64,
                                  exclusion_semantic_sha256=("f" * 64,))
        self.assertNotEqual(private_first.private_bytes(), private_second.private_bytes())
        private_sha = hashlib.sha256(private_first.private_bytes()).hexdigest()
        self.assertNotIn(private_sha, repr(first))
        self.assertNotIn(private_sha, first.manifest_bytes.decode())
        private_first_row = json.loads(private_first.private_bytes())["splits"]["train"]["rows"][0]
        self.assertNotIn(private_first_row["source_binding_sha256"], repr(first))
        self.assertNotIn(private_first_row["source_binding_sha256"], repr(private_first))

    def test_mutated_bytes_duplicate_semantics_or_wrong_split_rejected(self):
        source = _dataset(1202)
        with self.assertRaises(ValueError):
            _candidate(source._replace(sha256="f" * 64))
        one_line = source.jsonl_bytes.splitlines(keepends=True)[0]
        with self.assertRaises(ValueError):
            _candidate(source._replace(jsonl_bytes=one_line,
                                      sha256=hashlib.sha256(one_line).hexdigest(),
                                      case_count=1))
        rows = [json.loads(line) for line in source.jsonl_bytes.splitlines()]
        rows[1]["semantic_sha256"] = rows[0]["semantic_sha256"]
        modified = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
        )
        with self.assertRaises(ValueError):
            _candidate(source._replace(jsonl_bytes=modified, sha256=hashlib.sha256(modified).hexdigest()))
        rows[1]["split"] = "train"
        rows[1]["semantic_sha256"] = "f" * 64
        modified = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
        )
        with self.assertRaises(ValueError):
            _candidate(source._replace(jsonl_bytes=modified, sha256=hashlib.sha256(modified).hexdigest()))
        rows[1]["semantic_sha256"] = json.loads(source.jsonl_bytes.splitlines()[1])["semantic_sha256"]
        rows[1]["split"] = "dev"
        modified = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
        )
        with self.assertRaises(ValueError):
            _candidate(source._replace(jsonl_bytes=modified, sha256=hashlib.sha256(modified).hexdigest()))

    def test_exclusion_overlap_is_rejected_and_roster_commitment_does_not_grant(self):
        source = _dataset(1203)
        row = json.loads(source.jsonl_bytes.splitlines()[0])
        with self.assertRaises(ValueError):
            _candidate(source, exclusion_semantic_sha256=(row["semantic_sha256"],),
                       exclusion_roster_sha256="c" * 64)
        candidate = _candidate(source, exclusion_semantic_sha256=("d" * 64,),
                               exclusion_roster_sha256="c" * 64)
        manifest = json.loads(_private(source, exclusion_semantic_sha256=("d" * 64,),
                                       exclusion_roster_sha256="c" * 64).private_bytes())
        self.assertEqual(manifest["protected_evaluation"]["roster_sha256"], "c" * 64)
        self.assertFalse(manifest["protected_evaluation"]["custodian_authenticated"])
        self.assertFalse(candidate.authorizes_training)

    def test_missing_or_self_claimed_steward_and_evaluator_are_never_training_grants(self):
        candidate = _candidate()
        with self.assertRaises(CandidateAdmissionUnavailable):
            require_cognitive_rights_admission(candidate)
        with self.assertRaises(CandidateAdmissionUnavailable):
            require_cognitive_rights_admission(candidate, steward_signature="fake-sig",
                                              evaluator_signature="fake-eval")
        with self.assertRaises(ValueError):
            _candidate(source_commit="not-a-commit")
