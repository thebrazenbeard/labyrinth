"""R170 model-neutral private corpus assembly; no training/evaluation grant."""

import hashlib
import json
import time
import unittest
from unittest.mock import patch

from labyrinth import private_cognitive_corpus_r170 as corpus_module

from labyrinth.approved_cognitive_empathy_train_r90 import (
    PROVIDER_ID, RUBRIC_ID, _FOCUS, _source_binding_digest,
    semantic_sha256_from_prompt,
)
from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl
from labyrinth.journey_challenges import PreparedTrainingRoom
from labyrinth.journey import begin_journey, complete_maze
from labyrinth.maze import Challenge
from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
from tests.test_cognitive_objective_training_run import (
    cognitive_contract, development_fail_for_reroll,
)

from labyrinth.private_cognitive_corpus_r170 import (
    DEV_ROW_SCHEMA, CorpusSource,
    build_private_cognitive_corpus as _real_build_private_cognitive_corpus,
)


def _cycle_exports():
    contract = cognitive_contract(seed=304)
    first = begin_journey(contract.scope, checkpoint_sha256="4" * 64)
    train0 = export_cognitive_train_jsonl(contract, first)
    run0 = CognitiveObjectiveTrainingRun(contract, first)
    for _ in range(15):
        run0.run_training_chamber(lambda view: "A")
    waiting = complete_maze(first, run0.self_reported_completion_receipt(
        checkpoint_after_sha256="5" * 64,
    ))
    second = development_fail_for_reroll(waiting)
    train1 = export_cognitive_train_jsonl(contract, second)
    return train0, train1


def _sources():
    return (
        CorpusSource(cycle_index=0, repository="thebrazenbeard/labyrinth",
                     source_commit="a" * 40, source_file_sha256="b" * 64),
        CorpusSource(cycle_index=1, repository="thebrazenbeard/labyrinth",
                     source_commit="a" * 40, source_file_sha256="b" * 64),
    )


def build_private_cognitive_corpus(*args, source_contract=None, **kwargs):
    """Fixture calls always pin the original source ObjectiveContract."""
    contract = cognitive_contract(seed=304) if source_contract is None else source_contract
    return _real_build_private_cognitive_corpus(
        *args, source_contract=contract, **kwargs,
    )


def _dev_row(train_prompt=None, *, target="B", split="dev", case_id="dev-cognitive-new-01"):
    if train_prompt is None:
        prompt = (
            f"Case: {case_id}\n"
            "At a shared workshop, a participant asks to switch seats. No one has supplied a complete account of that person's reasons.\n"
            "Which interpretation best preserves uncertainty about the person's perspective?\n"
            "Options:\n"
            "A. They may be seeking a quieter space without disliking anyone.\n"
            "B. They want to undermine the instructor.\n"
            "C. They are secretly planning to abandon the group.\n"
            "Respond with only A, B, or C."
        )
    else:
        prompt = train_prompt.replace("Case: train-cognitive-", "Case: dev-cognitive-", 1)
        case_id = prompt.splitlines()[0].removeprefix("Case: ")
    row = {"schema": DEV_ROW_SCHEMA, "split": split, "case_id": case_id,
           "prompt": prompt, "target_choice": target,
           "provenance_id": "operator-dev-candidate-unverified"}
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


class PrivateCognitiveCorpusR170Tests(unittest.TestCase):
    def test_original_source_contract_is_required_not_guessed_from_digests(self):
        train0, _ = _cycle_exports()
        with self.assertRaisesRegex(ValueError, "approved ObjectiveContract required"):
            _real_build_private_cognitive_corpus((train0,), (_sources()[0],))
        with self.assertRaisesRegex(ValueError, "source contract"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],),
                source_contract=cognitive_contract(seed=313),
            )

    def test_swapped_genuine_gold_rebound_to_wrong_room_is_rejected(self):
        train0, _ = _cycle_exports()
        lines = train0.jsonl_bytes.splitlines(keepends=True)
        originals = [json.loads(line) for line in lines[:2]]

        def bind(row, focus):
            challenge = Challenge(
                id=row["case_id"], kind="synthetic_cognitive_perspective",
                tier=1, split="train", prompt=row["prompt"],
                answer=row["target_choice"],
                metadata=(("provider_id", PROVIDER_ID),
                          ("training_rubric", RUBRIC_ID),
                          ("focus_tag", focus),
                          ("objective_contract_sha256", row["objective_contract_sha256"]),
                          ("semantic_scenario_sha256", row["semantic_sha256"])),
            )
            return _source_binding_digest(
                PreparedTrainingRoom(row["room_id"], "cognitive_empathy", challenge),
                row["semantic_sha256"], PROVIDER_ID, RUBRIC_ID,
            )

        for position in (0, 1):
            base, donor = originals[position], originals[1 - position]
            new_row = dict(base)
            new_row["prompt"] = donor["prompt"].replace(
                f"Case: {donor['case_id']}\n", f"Case: {base['case_id']}\n", 1,
            )
            new_row["target_choice"] = donor["target_choice"]
            new_row["prompt_sha256"] = hashlib.sha256(new_row["prompt"].encode()).hexdigest()
            new_row["semantic_sha256"] = semantic_sha256_from_prompt(new_row["prompt"])
            focus = next(x for x in _FOCUS if bind(base, x) == base["source_binding_sha256"])
            new_row["source_binding_sha256"] = bind(new_row, focus)
            lines[position] = json.dumps(new_row, sort_keys=True,
                                         separators=(",", ":")).encode() + b"\n"
        payload = b"".join(lines)
        forged = train0._replace(jsonl_bytes=payload,
                                 sha256=hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError, "deterministic TRAIN roster"):
            build_private_cognitive_corpus((forged,), (_sources()[0],))

    def test_escaped_lone_surrogate_in_unadmitted_dev_is_rejected(self):
        train0, _ = _cycle_exports()
        row = json.loads(_dev_row())
        row["prompt"] = row["prompt"].replace("switch seats", "switch seats\ud800")
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaisesRegex(ValueError, "Unicode scalar"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],), dev_candidate_jsonl=payload,
            )

    def test_lexical_near_duplicate_avoids_quadratic_sequence_matching(self):
        first = "A team participant may need to consider private circumstances. " * 64
        second = first.replace("private circumstances", "personal circumstances", 1)
        with patch.object(corpus_module, "SequenceMatcher", create=True,
                          side_effect=AssertionError("quadratic comparator forbidden")):
            start = time.perf_counter()
            for _ in range(64):
                self.assertTrue(corpus_module._near_duplicate(first, second))
            self.assertLess(time.perf_counter() - start, 2.0)

    def test_oversized_dev_scenario_cannot_force_large_pairwise_comparisons(self):
        train0, _ = _cycle_exports()
        row = json.loads(_dev_row())
        lines = row["prompt"].splitlines()
        lines[1] = "A person revises their view. " * 40
        row["prompt"] = "\n".join(lines)
        candidate = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaisesRegex(ValueError, "lexical comparison bound"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],), dev_candidate_jsonl=candidate,
            )

    def test_forged_train_gold_with_recomputed_complete_source_binding_rejected(self):
        train0, _ = _cycle_exports()
        lines = train0.jsonl_bytes.splitlines(keepends=True)
        source = json.loads(lines[0])

        def binding(row, tag):
            challenge = Challenge(
                id=row["case_id"], kind="synthetic_cognitive_perspective",
                tier=1, split="train", prompt=row["prompt"],
                answer=row["target_choice"],
                metadata=(("provider_id", PROVIDER_ID),
                          ("training_rubric", RUBRIC_ID),
                          ("focus_tag", tag),
                          ("objective_contract_sha256", row["objective_contract_sha256"]),
                          ("semantic_scenario_sha256", row["semantic_sha256"])),
            )
            return _source_binding_digest(
                PreparedTrainingRoom(row["room_id"], "cognitive_empathy", challenge),
                row["semantic_sha256"], PROVIDER_ID, RUBRIC_ID,
            )

        original_tag = next(tag for tag in _FOCUS
                            if binding(source, tag) == source["source_binding_sha256"])
        source["target_choice"] = next(x for x in "ABC" if x != source["target_choice"])
        source["source_binding_sha256"] = binding(source, original_tag)
        lines[0] = json.dumps(source, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        payload = b"".join(lines)
        forged = train0._replace(jsonl_bytes=payload,
                                 sha256=hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError,
                                    "TRAIN gold/source binding|deterministic TRAIN roster"):
            build_private_cognitive_corpus((forged,), (_sources()[0],))

    def test_tampered_train_label_cannot_launder_digest_and_stale_binding(self):
        train0, _ = _cycle_exports()
        lines = train0.jsonl_bytes.splitlines(keepends=True)
        tampered = json.loads(lines[0])
        tampered["target_choice"] = next(choice for choice in "ABC"
                                         if choice != tampered["target_choice"])
        lines[0] = json.dumps(tampered, sort_keys=True,
                              separators=(",", ":")).encode() + b"\n"
        mutated_bytes = b"".join(lines)
        forged = train0._replace(
            jsonl_bytes=mutated_bytes,
            sha256=hashlib.sha256(mutated_bytes).hexdigest(),
        )
        with self.assertRaisesRegex(ValueError,
                                    "source binding|TRAIN gold|deterministic TRAIN roster"):
            build_private_cognitive_corpus((forged,), (_sources()[0],))

    def test_private_candidate_rejects_ordinary_mutation_and_reinitialization(self):
        train0, _ = _cycle_exports()
        candidate = build_private_cognitive_corpus((train0,), (_sources()[0],))
        original_train = candidate.private_train_jsonl()
        original_manifest = candidate.private_manifest_json()
        with self.assertRaises((AttributeError, TypeError)):
            candidate._train = b"replaced-TRAIN"
        with self.assertRaises((AttributeError, TypeError)):
            candidate._manifest = b"replaced-MANIFEST"
        with self.assertRaises((AttributeError, TypeError)):
            candidate.__init__(b"forged", b"forged", b"forged")
        self.assertEqual(candidate.private_train_jsonl(), original_train)
        self.assertEqual(candidate.private_manifest_json(), original_manifest)

    def test_semantic_paraphrase_gap_is_explicit_review_hold(self):
        train0, _ = _cycle_exports()
        prompt = json.loads(train0.jsonl_bytes.splitlines()[0])["prompt"]
        original_scenario = prompt.splitlines()[1]
        variant = prompt.replace(
            original_scenario,
            "Someone skips an optional team gathering. Their reasons have not been explained to the group.",
            1,
        )
        candidate = build_private_cognitive_corpus(
            (train0,), (_sources()[0],), dev_candidate_jsonl=_dev_row(variant),
        )
        manifest = json.loads(candidate.private_manifest_json())
        self.assertEqual(manifest["sources"][0]["external_source_verification"],
                         "NOT_ESTABLISHED")
        self.assertEqual(manifest["splits"]["dev"]["semantic_equivalence_review"],
                         "REQUIRED_UNVERIFIED")
        self.assertEqual(manifest["protected_eval"]["semantic_overlap_review"],
                         "REQUIRED_UNVERIFIED")
        self.assertFalse(candidate.authorizes_model_qualification)

    def test_fifteen_train_only_is_quarantined_with_constant_public_receipt(self):
        train0, _ = _cycle_exports()
        candidate = build_private_cognitive_corpus((train0,), (_sources()[0],))
        self.assertFalse(candidate.authorizes_training)
        self.assertFalse(candidate.authorizes_model_qualification)
        self.assertFalse(candidate.authorizes_host_execution)
        self.assertEqual(candidate.private_train_jsonl(), train0.jsonl_bytes)
        self.assertIsNone(candidate.private_dev_jsonl())
        manifest = json.loads(candidate.private_manifest_json())
        self.assertEqual(manifest["splits"]["train"]["count"], 15)
        self.assertIsNone(manifest["splits"]["dev"])
        self.assertEqual(manifest["protected_eval"]["rows_exported"], 0)
        self.assertFalse(manifest["protected_eval"]["custodian_authenticated"])
        self.assertEqual(manifest["rights"]["status"], "UNISSUED")
        for row in manifest["splits"]["train"]["rows"]:
            self.assertEqual(row["source_repository"], "thebrazenbeard/labyrinth")
            self.assertEqual(row["source_commit"], "a" * 40)
            self.assertEqual(row["source_file_sha256"], "b" * 64)
            self.assertEqual(row["license_status"], "UNVERIFIED")
            self.assertEqual(row["consent_status"], "UNVERIFIED")
        self.assertEqual(repr(candidate), "<PrivateCognitiveCorpusCandidate PRIVATE HOLD>")
        self.assertNotIn(b'"target_choice"', candidate.public_status_bytes())
        self.assertNotIn(train0.sha256.encode(), candidate.public_status_bytes())

    def test_two_train_cycles_are_30_unique_rows_and_order_independent(self):
        train0, train1 = _cycle_exports()
        expected_sources = _sources()
        first = build_private_cognitive_corpus((train1, train0), expected_sources[::-1])
        second = build_private_cognitive_corpus((train0, train1), expected_sources)
        self.assertEqual(first.private_train_jsonl(), second.private_train_jsonl())
        self.assertEqual(first.private_manifest_json(), second.private_manifest_json())
        rows = [json.loads(line) for line in first.private_train_jsonl().splitlines()]
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["semantic_sha256"] for r in rows}), 30)
        self.assertEqual([r["cycle_index"] for r in rows], [0] * 15 + [1] * 15)
        self.assertFalse(first.authorizes_training)

    def test_provided_dev_is_candidate_only_and_is_never_eval(self):
        train0, _ = _cycle_exports()
        candidate = build_private_cognitive_corpus(
            (train0,), (_sources()[0],), dev_candidate_jsonl=_dev_row(),
            protected_roster_sha256="c" * 64,
        )
        manifest = json.loads(candidate.private_manifest_json())
        self.assertEqual(manifest["splits"]["dev"]["count"], 1)
        self.assertEqual(manifest["splits"]["dev"]["rows"][0]["source_provenance_id"],
                         "operator-dev-candidate-unverified")
        self.assertEqual(manifest["splits"]["dev"]["rows"][0]["license_status"],
                         "UNVERIFIED")
        self.assertEqual(manifest["splits"]["dev"]["rows"][0]["consent_status"],
                         "UNVERIFIED")
        self.assertEqual(manifest["protected_eval"]["rows_exported"], 0)
        self.assertFalse(manifest["protected_eval"]["custodian_authenticated"])
        self.assertEqual(candidate.private_dev_jsonl(), _dev_row())
        self.assertFalse(candidate.authorizes_model_qualification)

    def test_dev_train_identity_and_near_duplicate_overlap_rejected(self):
        train0, _ = _cycle_exports()
        prompt = json.loads(train0.jsonl_bytes.splitlines()[0])["prompt"]
        with self.assertRaisesRegex(ValueError, "TRAIN/DEV duplicate"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],), dev_candidate_jsonl=_dev_row(prompt),
            )
        near = _dev_row(prompt).replace(b". No one has supplied", b"! No one has supplied", 1)
        with self.assertRaisesRegex(ValueError, "TRAIN/DEV near-duplicate"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],), dev_candidate_jsonl=near,
            )

    def test_rejects_protected_overlap_and_malformed_dev_split(self):
        train0, _ = _cycle_exports()
        semantic = json.loads(train0.jsonl_bytes.splitlines()[0])["semantic_sha256"]
        with self.assertRaisesRegex(ValueError, "protected exclusion"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],),
                protected_semantic_sha256=(semantic,),
                protected_roster_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "DEV"):
            build_private_cognitive_corpus(
                (train0,), (_sources()[0],), dev_candidate_jsonl=_dev_row(split="train"),
            )

    def test_invalid_source_and_cycle_replay_rejected(self):
        train0, _ = _cycle_exports()
        with self.assertRaises(ValueError):
            build_private_cognitive_corpus((train0, train0), _sources())
        with self.assertRaises(ValueError):
            build_private_cognitive_corpus((train0,), (_sources()[1],))
        with self.assertRaises(ValueError):
            build_private_cognitive_corpus((train0,), (
                CorpusSource(0, "unknown", "a" * 40, "b" * 64),
            ))

    def test_public_status_cannot_be_label_or_seed_oracle(self):
        train0, _ = _cycle_exports()
        other_contract = cognitive_contract(seed=313)
        other = export_cognitive_train_jsonl(
            other_contract,
            begin_journey(other_contract.scope, checkpoint_sha256="6" * 64),
        )
        first = build_private_cognitive_corpus((train0,), (_sources()[0],))
        second = build_private_cognitive_corpus(
            (other,), (_sources()[0],), source_contract=other_contract,
        )
        self.assertEqual(first.public_status_bytes(), second.public_status_bytes())
        self.assertNotEqual(first.private_train_jsonl(), second.private_train_jsonl())
        self.assertNotIn(hashlib.sha256(first.private_train_jsonl()).hexdigest().encode(),
                         first.public_status_bytes())


if __name__ == "__main__":
    unittest.main()
