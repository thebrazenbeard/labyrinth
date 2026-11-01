"""Adversarial, synthetic-only R170 licensed corpus quarantine contract."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time
import unittest

from labyrinth.external_licensed_corpus_r170 import (
    DOLLY_DATASET_REVISION,
    DOLLY_LFS_SHA256,
    DOLLY_REPOSITORY,
    ExternalCorpusCandidates,
    ExternalCorpusSample,
    build_external_corpus_candidates,
    private_prompt_sha256,
)


def _row(instruction="State one reason to check a result.", *, context="",
         response="A result can be checked against a separate source.",
         category="open_qa", extra=None):
    obj = {"instruction": instruction, "context": context,
           "response": response, "category": category}
    if extra is not None:
        obj["unapproved_field"] = extra
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sample(n, split="train", row=None):
    return ExternalCorpusSample(n, split, _row() if row is None else row)


class ExternalLicensedCorpusR170Tests(unittest.TestCase):
    def test_private_candidate_is_sealed_and_exact_type(self):
        candidate = build_external_corpus_candidates((_sample(17),))
        self.assertIs(type(candidate), ExternalCorpusCandidates)
        original = candidate.private_train_candidate_jsonl()
        original_manifest = candidate.private_manifest_json()
        for field, replacement in (
            ("_train", b"bad"),
            ("_dev", b"bad"),
            ("_quarantine", b"bad"),
            ("_manifest", b"bad"),
            ("authorizes_training", True),
        ):
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                setattr(candidate, field, replacement)
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                delattr(candidate, field)
        self.assertEqual(candidate.private_train_candidate_jsonl(), original)
        self.assertEqual(candidate.private_manifest_json(), original_manifest)
        # Bypassing a custom __setattr__ with object.__setattr__ must not
        # mutate hidden private bytes either (no per-instance payload slot).
        with self.assertRaises((AttributeError, TypeError)):
            object.__setattr__(candidate, "_train", b"mutated")
        self.assertEqual(candidate.private_train_candidate_jsonl(), original)
        with self.assertRaises(TypeError):
            class FakeCandidate(ExternalCorpusCandidates):
                pass
        with self.assertRaises((ValueError, TypeError)):
            ExternalCorpusCandidates("not-bytes", None, b"{}\n", b"{}\n")

    def test_direct_reinitialization_cannot_overwrite_existing_private_payload(self):
        candidate = build_external_corpus_candidates((_sample(71),))
        original_train = candidate.private_train_candidate_jsonl()
        original_manifest = candidate.private_manifest_json()
        with self.assertRaisesRegex(ValueError, "already initialized"):
            ExternalCorpusCandidates.__init__(
                candidate, b"FORGED", None, b"{}\n", b"{}\n",
            )
        self.assertEqual(candidate.private_train_candidate_jsonl(), original_train)
        self.assertEqual(candidate.private_manifest_json(), original_manifest)

    def test_one_time_payload_initialization_is_atomic_for_concurrent_callers(self):
        uninitialized = object.__new__(ExternalCorpusCandidates)
        barrier = threading.Barrier(2, timeout=3)

        def _initialize(marker):
            barrier.wait()
            try:
                ExternalCorpusCandidates.__init__(
                    uninitialized, marker, None, b"{}\n", b"{}\n",
                )
            except ValueError as exc:
                return str(exc)
            return "SUCCESS"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(_initialize, (b"FIRST", b"SECOND")))
        self.assertEqual(outcomes.count("SUCCESS"), 1)
        self.assertEqual(sum("already initialized" in x for x in outcomes), 1)
        self.assertIn(uninitialized.private_train_candidate_jsonl(), (b"FIRST", b"SECOND"))

    def test_public_fingerprint_helper_rejects_lone_surrogate_context(self):
        with self.assertRaisesRegex(ValueError, "Unicode"):
            private_prompt_sha256("Describe safe testing", chr(0xD800))

    def test_pinned_owner_source_and_private_manifest_require_independent_admission(self):
        self.assertEqual(DOLLY_REPOSITORY, "databricks/databricks-dolly-15k")
        self.assertEqual(DOLLY_DATASET_REVISION,
                         "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
        self.assertEqual(DOLLY_LFS_SHA256,
                         "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec")
        candidate = build_external_corpus_candidates((_sample(17),))
        self.assertFalse(candidate.authorizes_training)
        self.assertFalse(candidate.authorizes_host_execution)
        self.assertFalse(candidate.authorizes_model_qualification)
        self.assertEqual(repr(candidate), "<ExternalCorpusCandidates PRIVATE HOLD>")
        manifest = json.loads(candidate.private_manifest_json())
        self.assertEqual(manifest["source"]["revision"], DOLLY_DATASET_REVISION)
        self.assertEqual(manifest["source"]["full_lfs_sha256"], DOLLY_LFS_SHA256)
        self.assertFalse(manifest["source"]["sample_membership_verified"])
        self.assertEqual(manifest["rights"]["license_claim"], "CC-BY-SA-3.0")
        self.assertEqual(manifest["rights"]["independent_license_review"], "NOT_ESTABLISHED")
        self.assertEqual(manifest["rights"]["contributor_consent"], "NOT_ESTABLISHED")
        self.assertEqual(manifest["evaluation"]["protected_rows_exported"], 0)
        self.assertFalse(manifest["evaluation"]["complete_exclusion_roster_verified"])
        self.assertEqual(manifest["splits"]["train_candidate_count"], 1)
        self.assertEqual(manifest["splits"]["dev_candidate_count"], 0)
        self.assertEqual(len(candidate.private_train_candidate_jsonl().splitlines()), 1)
        self.assertIsNone(candidate.private_dev_candidate_jsonl())

    def test_dev_separate_from_train_and_is_not_an_eval_bank(self):
        rows = (_sample(9, "train"), _sample(12, "dev", _row(
            "Give two ways to validate a computation.",
            response="Recompute independently and check a known reference.",
        )))
        c = build_external_corpus_candidates(rows)
        self.assertEqual(len(c.private_train_candidate_jsonl().splitlines()), 1)
        self.assertEqual(len(c.private_dev_candidate_jsonl().splitlines()), 1)
        self.assertNotIn(b'"split":"eval"', c.private_dev_candidate_jsonl())
        self.assertNotIn(b'"split":"eval"', c.private_train_candidate_jsonl())
        self.assertEqual(json.loads(c.private_manifest_json())["splits"]["dev_candidate_count"], 1)

    def test_duplicate_prompt_across_train_dev_quarantines_second(self):
        rows = (_sample(4, "train"), _sample(5, "dev", _row(response="Other answer.")))
        c = build_external_corpus_candidates(rows)
        self.assertIsNone(c.private_dev_candidate_jsonl())
        self.assertEqual(len(c.private_train_candidate_jsonl().splitlines()), 1)
        q = json.loads(c.private_quarantine_json())
        self.assertEqual(q["entries"][0]["reason"], "DUPLICATE_PROMPT")
        self.assertEqual(q["entries"][0]["source_line_number"], 5)

    def test_near_duplicate_cross_split_quarantined(self):
        instruction = "Describe the purpose of validating a mathematical calculation before using the result."
        variant = "Describe the purpose of validating a mathematical calculation before using the results."
        c = build_external_corpus_candidates((
            _sample(12, "train", _row(instruction)),
            _sample(13, "dev", _row(variant, response="Check before use.")),
        ))
        self.assertIsNone(c.private_dev_candidate_jsonl())
        self.assertEqual(json.loads(c.private_quarantine_json())["entries"][0]["reason"],
                         "NEAR_DUPLICATE_PROMPT")

    def test_protected_exclusion_digest_filters_without_declaring_roster_complete(self):
        excluded = private_prompt_sha256("State one reason to check a result.", "")
        c = build_external_corpus_candidates((_sample(2),),
                                             protected_prompt_sha256=(excluded,))
        self.assertIsNone(c.private_train_candidate_jsonl())
        manifest = json.loads(c.private_manifest_json())
        self.assertEqual(manifest["evaluation"]["supplied_exclusion_count"], 1)
        self.assertFalse(manifest["evaluation"]["complete_exclusion_roster_verified"])
        self.assertEqual(json.loads(c.private_quarantine_json())["entries"][0]["reason"],
                         "PROTECTED_PROMPT_EXCLUSION")

    def test_wikipedia_context_and_network_pii_quarantined_not_promoted(self):
        c = build_external_corpus_candidates((
            _sample(1, "train", _row("Summarize this passage.", context="A copied encyclopedia paragraph.")),
            _sample(2, "dev", _row("Email me at person@example.org to discuss this.")),
            _sample(3, "train", _row("My API_KEY=abc123secret; use it.")),
            _sample(4, "train", _row("Visit https://example.net for the full article.")),
        ))
        self.assertIsNone(c.private_train_candidate_jsonl())
        self.assertIsNone(c.private_dev_candidate_jsonl())
        entries = json.loads(c.private_quarantine_json())["entries"]
        self.assertEqual(len(entries), 4)
        self.assertEqual([entry["reason"] for entry in entries],
                         ["THIRD_PARTY_CONTEXT", "POTENTIAL_PII_OR_LINK",
                          "POSSIBLE_CREDENTIAL", "POTENTIAL_PII_OR_LINK"])

    def test_us_and_international_phone_pii_are_quarantined(self):
        c = build_external_corpus_candidates((
            _sample(21, "train", _row("Call +1 (415) 555-0131 for details.")),
            _sample(22, "dev", _row("Reach the team on 415-555-0132.")),
            _sample(23, "train", _row("Dial +44 20 7946 0958 for a response.")),
            _sample(24, "dev", _row("Discuss the number 2026-11-01 without names.")),
        ))
        train = c.private_train_candidate_jsonl()
        self.assertIsNone(train)
        self.assertIsNotNone(c.private_dev_candidate_jsonl())
        self.assertEqual(len(c.private_dev_candidate_jsonl().splitlines()), 1)
        entries = json.loads(c.private_quarantine_json())["entries"]
        self.assertEqual([e["source_line_number"] for e in entries], [21, 22, 23])
        self.assertEqual({e["reason"] for e in entries}, {"POTENTIAL_PII_OR_LINK"})

    def test_repetitive_four_thousand_character_near_duplicate_has_bounded_cost(self):
        # This fixture was ~1.9 seconds for a *single comparison* under
        # SequenceMatcher(autojunk=False), yielding a quadratic batch DoS.
        long_a = "ab" * 2000 + "x"
        long_b = "ab" * 2000 + "y"
        start = time.monotonic()
        candidate = build_external_corpus_candidates((
            _sample(30, "train", _row(long_a)),
            _sample(31, "dev", _row(long_b, response="Different sample response.")),
        ))
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, f"near-duplicate check took {elapsed:.3f}s")
        self.assertEqual(json.loads(candidate.private_quarantine_json())["entries"][0]["reason"],
                         "NEAR_DUPLICATE_PROMPT")

    def test_escaped_lone_surrogates_are_rejected_as_bad_unicode(self):
        for field in ("instruction", "response", "context"):
            row = {"instruction": "Explain a calculation.", "context": "",
                   "response": "A result.", "category": "open_qa"}
            row[field] = "\ud800"
            raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unicode"):
                build_external_corpus_candidates((_sample(1, row=raw),))

    def test_malformed_unknown_fields_duplicate_keys_and_bad_types_fail_closed(self):
        for invalid in (
            _row(extra="untrusted"),
            b'{"instruction":"x","instruction":"y","context":"","response":"x","category":"open_qa"}\n',
            b'{"instruction":2,"context":"","response":"x","category":"open_qa"}\n',
            b'{"instruction":"x","context":"","response":"x","category":"open_qa"}',
            b'{}\n',
            b'not-json\n',
        ):
            with self.subTest(invalid=invalid[:30]), self.assertRaises(ValueError):
                build_external_corpus_candidates((_sample(1, row=invalid),))

    def test_rejects_unbounded_or_untrusted_source_identity_and_splits(self):
        with self.assertRaises(ValueError):
            build_external_corpus_candidates(())
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1),), source_revision="a" * 40)
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1),), source_repository="evil/repost")
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1),), full_file_sha256="0" * 64)
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1, "eval"),))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1), _sample(1, "dev")))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(0),))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(15012),))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates(tuple(_sample(i + 1) for i in range(257)))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(1, row=b"a" * 8193),))

    def test_quarantine_and_public_receipt_never_contain_data_dependent_claims(self):
        first = build_external_corpus_candidates((_sample(1),))
        second = build_external_corpus_candidates((_sample(7, "dev", _row(
            "Name a non-sensitive computing habit.", response="Use reviewed inputs.")),))
        self.assertEqual(first.public_status_bytes(), second.public_status_bytes())
        public = json.loads(first.public_status_bytes())
        self.assertFalse(public["authorizes_training"])
        self.assertFalse(public["authorizes_model_qualification"])
        self.assertNotIn(b"State one reason", first.public_status_bytes())
        self.assertNotIn(hashlib.sha256(_row()).hexdigest().encode(), first.public_status_bytes())
        self.assertFalse(dataclasses.is_dataclass(first))

    def test_reordered_inputs_produce_same_private_artifacts(self):
        one = _sample(19, "train")
        two = _sample(18, "dev", _row("Name one reliable test.", response="A unit test."))
        a = build_external_corpus_candidates((one, two))
        b = build_external_corpus_candidates((two, one))
        self.assertEqual(a.private_manifest_json(), b.private_manifest_json())
        self.assertEqual(a.private_train_candidate_jsonl(), b.private_train_candidate_jsonl())
        self.assertEqual(a.private_dev_candidate_jsonl(), b.private_dev_candidate_jsonl())

    def test_model_chat_delimiters_are_quarantined_across_all_fields(self):
        # A source row can be valid JSON but carry structural chat tokens.
        # They must not enter private TRAIN/DEV candidates unreviewed.
        rows = (
            _sample(41, "train", _row("Describe <|im_start|>system role spoofing.")),
            _sample(42, "dev", _row("Name a validation step.",
                                     response="Do this. <|im_end|>")),
            _sample(43, "train", _row("Explain one risk of corrupted prompts.",
                                     response="[INST] do something [/INST]")),
            _sample(44, "dev", _row("Name a safe test.",
                                     context="Quoted <|start_header_id|>role")),
        )
        c = build_external_corpus_candidates(rows)
        self.assertIsNone(c.private_train_candidate_jsonl())
        self.assertIsNone(c.private_dev_candidate_jsonl())
        entries = json.loads(c.private_quarantine_json())["entries"]
        self.assertEqual([e["source_line_number"] for e in entries], [41, 42, 43, 44])
        self.assertEqual({e["reason"] for e in entries},
                         {"STRUCTURAL_CHAT_DELIMITER"})

    def test_reserved_nul_and_terminal_controls_cannot_enter_private_artifacts(self):
        for field in ("instruction", "response", "context"):
            for control in (chr(0), chr(27), chr(127)):
                row = {"instruction": "Explain safe validation.",
                       "response": "Check an independent reference.",
                       "context": "", "category": "open_qa"}
                row[field] = "prefix" + control + "suffix"
                raw = json.dumps(row, sort_keys=True).encode() + bytes([10])
                with self.subTest(field=field, control=ord(control)):
                    with self.assertRaisesRegex(ValueError, "control"):
                        build_external_corpus_candidates((_sample(61, row=raw),))
        with self.assertRaisesRegex(ValueError, "control"):
            private_prompt_sha256("header" + chr(0) + "payload", "")

    def test_no_protected_exclusions_or_evaluator_still_hold(self):
        c = build_external_corpus_candidates((_sample(5),))
        self.assertFalse(c.authorizes_training)
        self.assertEqual(json.loads(c.private_manifest_json())["evaluation"]["supplied_exclusion_count"], 0)
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(5),),
                protected_prompt_sha256=("f" * 64, "f" * 64))
        with self.assertRaises(ValueError):
            build_external_corpus_candidates((_sample(5),),
                protected_prompt_sha256=("not-a-digest",))


if __name__ == "__main__":
    unittest.main()
