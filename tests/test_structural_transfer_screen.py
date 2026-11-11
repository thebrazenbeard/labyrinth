"""Hostile structural-transfer prescreen, never a neural graduation claim."""

from dataclasses import replace
from hashlib import sha256
import hmac
import unittest

from labyrinth.structural_transfer_screen import (
    CaseCommitment, CheckpointLineage, TransferReport, TrialOutcome,
    assess, canonical_report, precommit,
)
from labyrinth.minotaur_public_exam import run_public_exam
from labyrinth.exact_cover_tribunal import Scenario, fingerprint


KEY = b"test-only-evaluator-key-not-a-secret" + b"+" * 20


def digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def lineage(*, updated=True, reloaded=True):
    after = digest("checkpoint:adapted") if updated else digest("checkpoint:before")
    return CheckpointLineage(
        base_sha256=digest("base"),
        candidate_before_sha256=digest("checkpoint:before"),
        candidate_after_sha256=after,
        candidate_reload_sha256=after if reloaded else digest("different-reload"),
        zero_update_sha256=digest("control:zero"),
        corrupted_label_sha256=digest("control:corrupted"),
        optimizer_steps=8 if updated else 0,
    )


def cases(*, heldout_mechanics=8, transfer_mechanics=8, repeat_paraphrases=1):
    entries = [CaseCommitment("train-1", "train", digest("mechanic:train"),
                              digest("prompt:train"))]
    for split, count in (("heldout", heldout_mechanics),
                         ("transfer", transfer_mechanics)):
        for index in range(count):
            for paraphrase in range(repeat_paraphrases):
                entries.append(CaseCommitment(
                    f"{split}-{index}-{paraphrase}", split,
                    digest(f"mechanic:{split}:{index}"),
                    digest(f"prompt:{split}:{index}:{paraphrase}")))
    return tuple(entries)


def signed_result(manifest, *, memorizer=False, candidate=True,
                  drop_one=False, source_public=False):
    rows = tuple(TrialOutcome(
        case_id=entry.case_id,
        candidate_correct=candidate,
        candidate_valid=True,
        base_correct=False,
        zero_correct=False,
        corrupted_correct=False,
        memorizer_correct=memorizer,
    ) for entry in manifest.cases if entry.split != "train")
    if drop_one:
        rows = rows[:-1]
    report = TransferReport(manifest.sha256, rows)
    signature = hmac.new(KEY,
                         b"labyrinth.structural-transfer-screen.v1\x00" + canonical_report(report),
                         "sha256").hexdigest()
    return report, signature


class StructuralTransferScreenTests(unittest.TestCase):
    def manifest(self, **options):
        source_public = options.pop("source_public", False)
        return precommit(cases(**options), lineage(),
                         evaluator_sha256=digest("evaluator:v1"),
                         dataset_sha256=digest("new-blinded-split:v1"),
                         source_public=source_public)

    def test_eight_public_paraphrases_do_not_make_eight_mechanics(self):
        # A zero-update seed/variant lookup can score 8/8 while seeing just
        # one mechanic. Treating these as eight independent units is invalid.
        with self.assertRaisesRegex(ValueError, "independent mechanics"):
            self.manifest(heldout_mechanics=1, transfer_mechanics=1,
                          repeat_paraphrases=8, source_public=True)

    def test_actual_public_exam_seed_variants_are_not_blinded_transfer(self):
        """Exercise real prior public CLI mechanics, not invented test labels."""
        snapshots = (run_public_exam(seed=17, variant=0),
                     run_public_exam(seed=987, variant=31),
                     run_public_exam(seed=23, variant=4))
        signatures = []
        records = []
        for split, receipt in zip(("train", "heldout", "transfer"), snapshots):
            self.assertEqual(receipt["claim_scope"]["neural_weight_updates"], 0)
            self.assertFalse(receipt["claim_scope"]["structural_transfer_tested"])
            normalized = []
            for name, case in receipt["families"]["constant_exam"]["cases"].items():
                normalized.append(("constant:" + name, case["verdict"],
                                   case["semantic_family_id"]))
            for name, case in receipt["families"]["veiled_fork"]["cases"].items():
                normalized.append(("fork:" + name, case["verdict"],
                                   case["rules_fingerprint"]))
            signatures.append(tuple(normalized))
            for name, _, family in normalized:
                records.append(CaseCommitment(f"{split}-{name.replace(':', '-')}", split,
                                              family, digest(f"{split}:{name}:prompt")))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])
        self.assertEqual(len(signatures[0]), 8)
        with self.assertRaisesRegex(ValueError, "split overlap"):
            precommit(tuple(records), lineage(), evaluator_sha256=digest("eval"),
                      dataset_sha256=digest("dataset"), source_public=True)

    def test_training_family_cannot_reappear_under_new_prompt_hash(self):
        records = list(cases())
        records[1] = replace(records[1], mechanic_sha256=records[0].mechanic_sha256)
        with self.assertRaisesRegex(ValueError, "split overlap"):
            precommit(tuple(records), lineage(), evaluator_sha256=digest("evaluator"),
                      dataset_sha256=digest("dataset"), source_public=False)

    def test_transfer_family_cannot_reuse_heldout_family(self):
        records = list(cases())
        first_heldout = next(x for x in records if x.split == "heldout")
        transfer_index = next(i for i, x in enumerate(records) if x.split == "transfer")
        records[transfer_index] = replace(records[transfer_index],
                                          mechanic_sha256=first_heldout.mechanic_sha256)
        with self.assertRaisesRegex(ValueError, "split overlap"):
            precommit(tuple(records), lineage(), evaluator_sha256=digest("evaluator"),
                      dataset_sha256=digest("dataset"), source_public=False)

    def test_relabelled_isomorphic_cover_graph_is_only_declared_distinct(self):
        # These are the SAME exact-cover incidence mechanics under a vertex
        # bijection and reordered crew slots. A provenance fingerprint of raw
        # names/order is not a semantic identity or a blinded split proof.
        first = Scenario(("A", "B", "C", "D", "E", "F"),
                         (("A", "B", "C"), ("D", "E", "F")), 2)
        relabel = Scenario(("Q", "R", "S", "U", "V", "W"),
                           (("U", "V", "W"), ("Q", "R", "S")), 2)
        self.assertNotEqual(fingerprint(first), fingerprint(relabel))
        rename = dict(zip(first.sites, relabel.sites))
        self.assertEqual({frozenset(rename[site] for site in crew)
                          for crew in first.crews},
                         {frozenset(crew) for crew in relabel.crews})
        records = list(cases())
        heldout_index = next(i for i, c in enumerate(records) if c.split == "heldout")
        transfer_index = next(i for i, c in enumerate(records) if c.split == "transfer")
        records[heldout_index] = replace(records[heldout_index],
                                         mechanic_sha256=fingerprint(first))
        records[transfer_index] = replace(records[transfer_index],
                                          mechanic_sha256=fingerprint(relabel))
        manifest = precommit(tuple(records), lineage(),
                             evaluator_sha256=digest("evaluator"),
                             dataset_sha256=digest("dataset"), source_public=False)
        report, signature = signed_result(manifest)
        # The screen cannot deduce graph isomorphism from opaque SHA strings;
        # its most favorable verdict remains PENDING physical custody.
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "PENDING_INDEPENDENT_CUSTODY")
        self.assertFalse(result.neural_learning_verified)
        self.assertFalse(result.private_exam_qualified)

    def test_zero_update_checkpoint_cannot_pass_training_transfer_screen(self):
        manifest = precommit(cases(), lineage(updated=False),
                             evaluator_sha256=digest("evaluator"),
                             dataset_sha256=digest("dataset"), source_public=False)
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("parameter update", " ".join(result.reasons))

    def test_reload_must_identify_same_checkpoint(self):
        manifest = precommit(cases(), lineage(reloaded=False),
                             evaluator_sha256=digest("evaluator"),
                             dataset_sha256=digest("dataset"), source_public=False)
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("reload", " ".join(result.reasons))

    def test_identical_checkpoint_cannot_claim_gain_over_control(self):
        # A source-signed report may *say* the candidate improves over a
        # control even when both identify exactly the same checkpoint bytes.
        # That claim is self-contradictory evidence, regardless of HMAC.
        for control in ("base", "zero_update", "corrupted_label"):
            with self.subTest(control=control):
                stated_lineage = lineage()
                same_bytes = getattr(stated_lineage, control + "_sha256")
                contradictory = replace(
                    stated_lineage, candidate_after_sha256=same_bytes,
                    candidate_reload_sha256=same_bytes)
                manifest = precommit(cases(), contradictory,
                                     evaluator_sha256=digest("evaluator"),
                                     dataset_sha256=digest("dataset"),
                                     source_public=False)
                report, signature = signed_result(manifest)
                result = assess(manifest, report, signature_hex=signature,
                                independent_key=KEY)
                self.assertEqual(result.disposition, "BLOCKED")
                self.assertIn("identical", " ".join(result.reasons))

    def test_source_public_receipt_never_qualifies_private_transfer(self):
        manifest = self.manifest(source_public=True)
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("source-public", " ".join(result.reasons))

    def test_perfect_memorizer_control_blocks_equal_candidate(self):
        manifest = self.manifest()
        report, signature = signed_result(manifest, memorizer=True)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("memorizer", " ".join(result.reasons))

    def test_all_matched_controls_need_demonstrated_gain(self):
        manifest = self.manifest()
        report, _ = signed_result(manifest)
        for control in ("base", "zero", "corrupted", "memorizer"):
            with self.subTest(control=control):
                key = control + "_correct"
                equal_control = replace(report, rows=tuple(
                    replace(row, **{key: True}) for row in report.rows))
                signature = hmac.new(
                    KEY, b"labyrinth.structural-transfer-screen.v1\x00"
                    + canonical_report(equal_control), "sha256").hexdigest()
                result = assess(manifest, equal_control,
                                signature_hex=signature, independent_key=KEY)
                self.assertEqual(result.disposition, "BLOCKED")
                self.assertIn(control, " ".join(result.reasons))

    def test_unsigned_or_altered_receipt_fails_closed(self):
        manifest = self.manifest()
        report, signature = signed_result(manifest)
        altered = replace(report, rows=(replace(report.rows[0], candidate_correct=False),
                                        *report.rows[1:]))
        for invalid_report, invalid_sig in ((report, "0" * 64),
                                            (altered, signature)):
            with self.subTest(report=invalid_report == report):
                result = assess(manifest, invalid_report, signature_hex=invalid_sig,
                                independent_key=KEY)
                self.assertEqual(result.disposition, "BLOCKED")

    def test_exact_precommitted_roster_prevents_cherrypicking(self):
        manifest = self.manifest()
        report, signature = signed_result(manifest, drop_one=True)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")
        self.assertIn("roster", " ".join(result.reasons))

    def test_synthetic_success_is_only_pending_external_custody(self):
        manifest = self.manifest()
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "PENDING_INDEPENDENT_CUSTODY")
        self.assertEqual(result.independent_families, (8, 8))
        self.assertFalse(result.neural_learning_verified)
        self.assertFalse(result.private_exam_qualified)

    def test_paraphrases_count_once_per_mechanic(self):
        manifest = self.manifest(repeat_paraphrases=3)
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.independent_families, (8, 8))
        self.assertEqual(result.disposition, "PENDING_INDEPENDENT_CUSTODY")

    def test_bad_python_types_rejected_before_comparison(self):
        class DeceptiveString(str):
            def __eq__(self, other):
                return True

        records = list(cases())
        records[1] = replace(records[1], split=DeceptiveString("heldout"))
        with self.assertRaises(ValueError):
            precommit(tuple(records), lineage(), evaluator_sha256=digest("evaluator"),
                      dataset_sha256=digest("dataset"), source_public=False)

    def test_precommit_snapshots_source_dataclasses_before_mutation(self):
        records = list(cases())
        original = records[1]
        manifest = precommit(tuple(records), lineage(),
                             evaluator_sha256=digest("evaluator"),
                             dataset_sha256=digest("dataset"), source_public=False)
        object.__setattr__(original, "mechanic_sha256", digest("forged-after-commit"))
        self.assertNotEqual(original.mechanic_sha256, manifest.cases[1].mechanic_sha256)
        report, signature = signed_result(manifest)
        result = assess(manifest, report, signature_hex=signature, independent_key=KEY)
        self.assertEqual(result.disposition, "PENDING_INDEPENDENT_CUSTODY")


if __name__ == "__main__":
    unittest.main()
