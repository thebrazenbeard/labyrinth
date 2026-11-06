"""Mock-only local Vera GGUF conversation quality: no neural/human proof."""

from dataclasses import replace
from pathlib import Path
import unittest

from labyrinth.gguf_entrance_fingerprint_v1 import capture_gguf_entrance
from labyrinth.local_full_state_probe_v1 import public_probe_suite_sha256
from labyrinth.vera_gguf_quality_acceptance_v1 import (
    PUBLIC_VERA_CASES, GGUFSubjectPin, HumanRating, VeraQualityError,
    compare_blinded_human_ratings, lexical_diagnostics, make_blind_review_packet,
    public_vera_prompts, run_gguf_quality_probe,
)
from tests.test_gguf_entrance_fingerprint_v1 import _gguf
from tests.test_local_full_state_probe_v1 import FullStateLocalProbeTests


class VeraGGUFQualityAcceptanceTests(unittest.TestCase):
    def setUp(self):
        fixture = FullStateLocalProbeTests("test_matched_public_probes_rehash_all_subjects_without_gold_leak")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.activation_calls = []
        self.operator_secret = b"operator-private-fixture-secret-32bytes!"
        self.pins = []
        for name in ("parent", "control", "left", "right"):
            folder = self.root / ("gguf_" + name)
            folder.mkdir()
            path = folder / "local_gguf_fixture.gguf"
            marker = ("w0" if name in ("parent", "control") else name)
            gguf_bytes = _gguf([
                ("general.architecture", 8, "qwen3"),
                ("general.name", 8, "fictional-" + marker),
            ])
            if name in ("left", "right"):
                gguf_bytes = gguf_bytes[:-1] + (b"L" if name == "left" else b"R")
            path.write_bytes(gguf_bytes)
            receipt = capture_gguf_entrance(base_path=path)
            self.pins.append(GGUFSubjectPin(name, receipt, receipt["manifest_sha256"]))
        self.pins = tuple(self.pins)
        self.prompts = public_vera_prompts()
        self.suite_sha = public_probe_suite_sha256(self.prompts)

    def activate(self, request):
        self.activation_calls.append(request)
        return "OPERATOR_REPORTS_ACTIVATED"

    def run_quality(self, **extra):
        arguments = dict(
            population=self.fixture.generation.population,
            targets=self.fixture.targets,
            gguf_pins=self.pins,
            expected_public_suite_sha256=self.suite_sha,
            activate_gguf=self.activate,
            transport=self.fixture.fake,
        )
        arguments.update(extra)
        return run_gguf_quality_probe(**arguments)

    def _review_fixture(self, packet, *, left=2, right=4, parent=3, control=3):
        values = {"left": left, "right": right, "parent": parent, "control": control}
        blind = {case.blind_subject for case in packet.items}
        from labyrinth.vera_gguf_quality_acceptance_v1 import _alias, _DIMENSIONS
        aliases = {_alias(name, self.operator_secret): value
                   for name, value in values.items()}
        self.assertEqual(set(aliases), blind)
        return tuple(HumanRating(
            reviewer_id=reviewer, blind_subject=item.blind_subject,
            case_id=item.case_id,
            dimensions=tuple((dim, aliases[item.blind_subject]) for dim in _DIMENSIONS),
            explanation="Source-only fictional reviewer rubric fixture; not actual human judgment.",
        ) for reviewer in ("reviewer_a", "reviewer_b") for item in packet.items)

    def test_public_gguf_matched_conversation_run_and_blind_human_quality_screen(self):
        run = self.run_quality()
        self.assertFalse(run.model_load_proven)
        self.assertFalse(run.neural_improvement_proven)
        self.assertEqual(run.public_probe.classification,
                         "PUBLIC_LOCALHOST_OUTPUT_UNVERIFIED_LOADED_MODEL")
        self.assertEqual(len(run.public_probe.subjects), 4)
        self.assertEqual(len(self.activation_calls), 4)
        self.assertEqual(len(self.fixture.calls), 4 * len(PUBLIC_VERA_CASES))
        self.assertEqual(run.gguf_subjects[0][1], run.gguf_subjects[1][1])
        self.assertNotEqual(run.gguf_subjects[2][1], run.gguf_subjects[3][1])
        for activation in self.activation_calls:
            self.assertTrue(activation.gguf_base_path.is_absolute())
            self.assertEqual(len(activation.gguf_base_sha256), 64)
        for _endpoint, payload, _timeout in self.fixture.calls:
            self.assertEqual(payload["temperature"], 0)
            self.assertFalse(set(payload) & {
                "expected", "rubric", "gold", "examiner", "score", "human_rating",
                "private_case", "manifest", "reference_claims"})
        flags = lexical_diagnostics(run)
        self.assertTrue(any(x.indicator == "vera_name_absent" and x.flag_present
                            for x in flags))
        packet = make_blind_review_packet(run, operator_secret=self.operator_secret)
        self.assertTrue(all(item.blind_subject.startswith("anon_")
                            and item.blind_subject not in ("parent", "control", "left", "right")
                            for item in packet.items))
        self.assertNotIn("full_state_sha256", repr(packet))
        ratings = self._review_fixture(packet)
        comparison = compare_blinded_human_ratings(
            run, packet, ratings, operator_secret=self.operator_secret,
            trainer_ids=("fixture_trainer",))
        self.assertEqual(comparison.accepted_subjects, ("right",))
        self.assertEqual(comparison.rejected_subjects, ("left",))
        self.assertEqual(comparison.reviewer_count, 2)
        self.assertEqual(comparison.classification,
                         "PROVISIONAL_CALLER_REVIEWED_DEVELOPMENT_ONLY")
        self.assertFalse(comparison.independent_human_authenticated)
        self.assertFalse(comparison.model_identity_proven)
        self.assertFalse(comparison.neural_improvement_proven)
        self.assertFalse(comparison.graduation_authorized)
        with self.assertRaises(VeraQualityError):
            replace(run, model_load_proven=True)
        with self.assertRaises(VeraQualityError):
            replace(comparison, graduation_authorized=True)

    def test_rejects_gguf_control_mismatch_before_transport(self):
        different = list(self.pins)
        different[1] = replace(different[1], entrance=different[2].entrance,
                               trusted_manifest_sha256=different[2].trusted_manifest_sha256)
        with self.assertRaisesRegex(VeraQualityError, "unchanged GGUF control"):
            self.run_quality(gguf_pins=tuple(different))
        self.assertEqual(self.fixture.calls, [])
        self.assertEqual(self.activation_calls, [])

    def test_rejects_gguf_content_tamper_and_missing_public_suite_pin(self):
        with self.assertRaisesRegex(VeraQualityError, "precommitment"):
            self.run_quality(expected_public_suite_sha256="0"*64)
        self.assertEqual(self.fixture.calls, [])
        candidate_file = Path(self.pins[2].entrance["base"]["path"])
        original = candidate_file.read_bytes()
        candidate_file.write_bytes(original[:-1] + b"x")
        with self.assertRaises(ValueError):
            self.run_quality()
        self.assertEqual(self.fixture.calls, [])

    def test_gguf_metadata_only_candidate_edit_is_not_tensor_mutation(self):
        left = self.pins[2]
        path = Path(left.entrance["base"]["path"])
        # Keep the candidate's distinct GGUF name/header, but make its
        # actual tensor payload identical to the W0/control fixture.
        path.write_bytes(path.read_bytes()[:-1] + b"F")
        receipt = capture_gguf_entrance(base_path=path)
        altered = list(self.pins)
        altered[2] = GGUFSubjectPin("left", receipt, receipt["manifest_sha256"])
        with self.assertRaisesRegex(VeraQualityError, "tensor payload"):
            self.run_quality(gguf_pins=tuple(altered))
        self.assertEqual(self.fixture.calls, [])

    def test_ratings_must_be_blinded_complete_distinct_human_and_not_self_grade(self):
        run = self.run_quality()
        packet = make_blind_review_packet(run, operator_secret=self.operator_secret)
        ratings = self._review_fixture(packet)
        with self.assertRaisesRegex(VeraQualityError, "every reviewer"):
            compare_blinded_human_ratings(
                run, packet, ratings[:-1], operator_secret=self.operator_secret)
        with self.assertRaisesRegex(VeraQualityError, "non-trainer"):
            compare_blinded_human_ratings(
                run, packet, tuple(replace(x, reviewer_id="fixture_trainer")
                                   if x.reviewer_id == "reviewer_a" else x for x in ratings),
                operator_secret=self.operator_secret,
                trainer_ids=("fixture_trainer",))
        with self.assertRaisesRegex(VeraQualityError, "forged"):
            compare_blinded_human_ratings(
                run, packet, (replace(ratings[0], producer_role="TRAINER"), *ratings[1:]),
                operator_secret=self.operator_secret)
        with self.assertRaisesRegex(VeraQualityError, "review packet"):
            compare_blinded_human_ratings(
                run, replace(packet, public_suite_sha256="0"*64), ratings,
                operator_secret=self.operator_secret)

    def test_zero_reviewer_is_pending_and_one_chosen_reviewer_is_provisional(self):
        run = self.run_quality()
        flags = lexical_diagnostics(run)
        self.assertTrue(flags)  # Diagnostics need no reviewer or scoring authority.
        packet = make_blind_review_packet(run, operator_secret=self.operator_secret)
        not_reviewed = compare_blinded_human_ratings(
            run, packet, (), operator_secret=self.operator_secret)
        self.assertEqual(not_reviewed.reviewer_count, 0)
        self.assertEqual(not_reviewed.classification,
                         "NO_HUMAN_REVIEW_YET_LEXICAL_ONLY")
        self.assertEqual(not_reviewed.accepted_subjects, ())
        self.assertEqual(not_reviewed.rejected_subjects, ())
        self.assertFalse(not_reviewed.independent_human_authenticated)

        operator_chosen = tuple(
            rating for rating in self._review_fixture(packet)
            if rating.reviewer_id == "reviewer_a")
        one = compare_blinded_human_ratings(
            run, packet, operator_chosen, operator_secret=self.operator_secret,
            trainer_ids=("fixture_trainer",))
        self.assertEqual(one.reviewer_count, 1)
        self.assertEqual(one.accepted_subjects, ("right",))
        self.assertEqual(one.rejected_subjects, ("left",))
        self.assertEqual(one.classification,
                         "PROVISIONAL_CALLER_REVIEWED_DEVELOPMENT_ONLY")
        self.assertFalse(one.independent_human_authenticated)
        self.assertFalse(one.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
