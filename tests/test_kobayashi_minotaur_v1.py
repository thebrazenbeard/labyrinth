"""Kobayashi/Minotaur source tests: no actual independent model or judge."""

from dataclasses import FrozenInstanceError, replace
from os import link as hardlink
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim,
    DialogueFinalityError,
    EVIDENCE_CLASS,
    EvidenceAssessment,
    KobayashiCorpus,
    KobayashiError,
    MinotaurDialogue,
    ReasonedTurnFeedback,
    RebuttalCase,
)
from labyrinth import kobayashi_minotaur_v1 as minotaur
from labyrinth.kobayashi_canon_v1 import KobayashiWorld, KobayashiAction


def _h(n: int) -> str:
    return f"{n:064x}"


def _claim(
    *,
    scenario: str = _h(1),
    world: str = _h(2),
    action: str = _h(3),
    text: str = "I turn the emergency valve because the pressure gauge is rising.",
) -> BoundClaim:
    return BoundClaim(scenario, world, action, text)


def _case(
    claim: BoundClaim,
    *,
    category: str = "HIDDEN_COUPLING",
    finding: str = "REFUTED",
    evidence: str = _h(5),
    exact_claim: bool = False,
) -> RebuttalCase:
    return RebuttalCase(
        claim.scenario_sha256,
        claim.world_sha256,
        claim.action_sha256,
        claim.claim_sha256 if exact_claim else None,
        category,
        finding,
        evidence,
        "The public safety argument overlooks a precommitted coupling. "
        "Explain what visible evidence would falsify your premise.",
    )


def _assessment(
    claim: BoundClaim,
    case: RebuttalCase | None,
    transcript_hash: str,
    *,
    check: str = "CONFIRMED",
    evidence_sha256: str | None = None,
    novelty_specificity: str = "UNSPECIFIED",
    semantic_scope: str = "CLAIM_REFUTED",
    **rebind: object,
) -> EvidenceAssessment:
    values = dict(
        scenario_sha256=claim.scenario_sha256,
        world_sha256=claim.world_sha256,
        claim_sha256=claim.claim_sha256,
        case_sha256=case.case_sha256 if case else None,
        dialogue_sha256=transcript_hash,
        check=check,
        evidence_sha256=evidence_sha256 if evidence_sha256 is not None else (case.evidence_sha256 if case else None),
        public_explanation="Host source evidence was checked for the exact bound claim.",
        reviewer_id="mock-source-only-minotaur",
        novelty_specificity=novelty_specificity,
        semantic_scope=semantic_scope,
    )
    values.update(rebind)
    return EvidenceAssessment(**values)


def _two_rounds(dialogue: MinotaurDialogue, *, evidence: tuple[str, ...] = ()) -> None:
    dialogue.submit_argument("The mechanism would isolate heat based on the publicly visible gauge.", evidence)
    dialogue.submit_argument("If the gauge lies, a separate continuity measurement would falsify my hypothesis.", evidence)


class KobayashiMinotaurTests(unittest.TestCase):
    def test_known_action_refuted_only_with_exact_host_verified_case_evidence(self) -> None:
        claim = _claim()
        case = _case(claim)
        corpus = KobayashiCorpus("Kobayashi-Maru-2026", (case,))
        calls = []

        def host_check(c, known, arguments, transcript):
            calls.append((c.claim_sha256, known.case_sha256, len(arguments)))
            return _assessment(c, known, transcript)

        dialogue = MinotaurDialogue(corpus, claim, evidence_checker=host_check)
        reply = dialogue.submit_argument("The pressure indicator suggests a cooling route.")
        self.assertEqual((reply.round_number, reply.challenge_category), (1, "HIDDEN_COUPLING"))
        self.assertNotIn("solution", reply.public_challenge.lower())
        self.assertIsNone(dialogue.final)
        with self.assertRaisesRegex(KobayashiError, "at least two"):
            dialogue.finalize()
        dialogue.submit_argument("I would cross-check a second pressure indicator.")
        final = dialogue.finalize()
        self.assertEqual(final.status, "REFUTED")
        self.assertEqual(final.evidence_class, EVIDENCE_CLASS)
        self.assertEqual(final.case_sha256, case.case_sha256)
        self.assertEqual(final.claim_sha256, claim.claim_sha256)
        self.assertEqual(len(final.review_sha256), 64)
        self.assertIs(dialogue.finalize(), final)
        self.assertEqual(calls, [(claim.claim_sha256, case.case_sha256, 2)])
        with self.assertRaises(DialogueFinalityError):
            dialogue.submit_argument("One more objection")

    def test_equivalent_wording_same_physical_action_matches_known_case(self) -> None:
        claim = _claim()
        case = _case(claim)
        corpus = KobayashiCorpus("canonical", (case,))
        paraphrase = _claim(text="I rotate the emergency valve; fire pressure might fall.")
        self.assertNotEqual(paraphrase.claim_sha256, claim.claim_sha256)
        self.assertIs(corpus.lookup(paraphrase), case)
        dialogue = MinotaurDialogue(
            corpus, paraphrase,
            evidence_checker=lambda c, known, args, transcript: _assessment(c, known, transcript),
        )
        _two_rounds(dialogue)
        self.assertEqual(dialogue.finalize().status, "REFUTED")

    def test_no_false_match_for_different_world_scenario_or_action(self) -> None:
        claim = _claim()
        corpus = KobayashiCorpus("canonical", (_case(claim),))
        for candidate in (
            _claim(scenario=_h(77)),
            _claim(world=_h(78)),
            _claim(action=_h(79)),
        ):
            with self.subTest(candidate=candidate.claim_sha256):
                self.assertIsNone(corpus.lookup(candidate))
        exact_only = KobayashiCorpus("canonical-exact", (_case(claim, exact_claim=True),))
        self.assertIsNone(exact_only.lookup(_claim(text="I rotate the emergency valve.")))

    def test_known_mechanism_reused_across_cosmetic_scenes_without_world_wildcard(self) -> None:
        a = KobayashiWorld(variant_seed=0)
        b = KobayashiWorld(variant_seed=1)
        self.assertEqual(a.graph_sha256, b.graph_sha256)
        self.assertNotEqual(a.observe().sha256, b.observe().sha256)
        action = KobayashiAction("turn", "cobalt_valve")
        first = _claim(scenario=a.observe().sha256, world=a.graph_sha256,
                       action=action.sha256, text="I turn the cobalt valve.")
        second = _claim(scenario=b.observe().sha256, world=b.graph_sha256,
                        action=action.sha256, text="I rotate the cobalt valve.")
        canon_case = RebuttalCase(
            scenario_sha256=None, world_sha256=first.world_sha256,
            action_sha256=first.action_sha256, claim_sha256=None,
            category="HIDDEN_COUPLING", finding="REFUTED",
            evidence_sha256=_h(505),
            public_challenge="What public evidence excludes downstream pressure failure?",
        )
        corpus = KobayashiCorpus("stable-world-rebuttal", (canon_case,))
        self.assertIs(corpus.lookup(first), canon_case)
        self.assertIs(corpus.lookup(second), canon_case)
        self.assertIsNone(corpus.lookup(_claim(
            scenario=second.scenario_sha256, world=_h(600),
            action=second.action_sha256,
        )))

    def test_valid_counterexample_is_not_precommitted_as_failure(self) -> None:
        claim = _claim()
        with self.assertRaises(KobayashiError):
            _case(claim, category="VALID_COUNTEREXAMPLE", finding="REFUTED")
        case = _case(claim, category="VALID_COUNTEREXAMPLE", finding="SURVIVES")
        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(
                c, known, transcript, semantic_scope="CLAIM_SUPPORTED",
            ),
        )
        _two_rounds(dialogue)
        self.assertEqual(dialogue.finalize().status, "SURVIVES")

    def test_same_action_opposite_predictions_are_not_equally_refuted(self) -> None:
        overconfident = _claim(text="I turn cobalt valve, guaranteed to save everyone.")
        accurate = _claim(text=(
            "I turn cobalt valve, but it will breach chamber pressure and cause "
            "catastrophic failure; there is no verified way to save this station."
        ))
        self.assertEqual(overconfident.action_sha256, accurate.action_sha256)
        corpus = KobayashiCorpus("canonical", (_case(overconfident),))
        self.assertIsNotNone(corpus.lookup(accurate))

        # A reviewer that ONLY confirms physical failure has not assessed the
        # meaning of either sentence; both must be epistemically UNKNOWN.
        for claim in (overconfident, accurate):
            with self.subTest(unassessed=claim.claim_sha256):
                dialogue = MinotaurDialogue(
                    corpus, claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, semantic_scope="UNASSESSED",
                    ),
                )
                _two_rounds(dialogue)
                self.assertEqual(dialogue.finalize().status, "UNKNOWN")

        for claim, meaning, expected in (
            (overconfident, "CLAIM_REFUTED", "REFUTED"),
            (accurate, "CLAIM_SUPPORTED", "SURVIVES"),
        ):
            with self.subTest(semantic=meaning):
                dialogue = MinotaurDialogue(
                    corpus, claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, semantic_scope=meaning,
                    ),
                )
                _two_rounds(dialogue)
                self.assertEqual(dialogue.finalize().status, expected)

    def test_exact_precommit_cannot_override_conflicting_semantic_review(self) -> None:
        """An answer key and reviewer can disagree: neither auto-wins."""
        claim = _claim()
        for finding, semantic_review in (
            ("REFUTED", "CLAIM_SUPPORTED"),
            ("SURVIVES", "CLAIM_REFUTED"),
            ("REFUTED", "UNASSESSED"),
        ):
            with self.subTest(finding=finding, semantic_review=semantic_review):
                case = _case(claim, finding=finding, exact_claim=True)
                dialogue = MinotaurDialogue(
                    KobayashiCorpus("exact-claim", (case,)), claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, semantic_scope=semantic_review,
                    ),
                )
                _two_rounds(dialogue)
                final = dialogue.finalize()
                self.assertEqual(final.status, "UNKNOWN")

    def test_exact_precommit_with_matching_review_retains_supported_verdict(self) -> None:
        claim = _claim()
        for finding, semantic_review in (
            ("REFUTED", "CLAIM_REFUTED"),
            ("SURVIVES", "CLAIM_SUPPORTED"),
        ):
            with self.subTest(finding=finding):
                case = _case(claim, finding=finding, exact_claim=True)
                dialogue = MinotaurDialogue(
                    KobayashiCorpus("exact-claim", (case,)), claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, semantic_scope=semantic_review,
                    ),
                )
                _two_rounds(dialogue)
                self.assertEqual(dialogue.finalize().status, finding)

    def test_unrecognized_ungrounded_hashes_never_promote_novelty(self) -> None:
        claim = _claim()
        corpus = KobayashiCorpus("no-prior-case", ())
        dialogue = MinotaurDialogue(
            corpus, claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(
                c, known, transcript, check="INDETERMINATE",
                novelty_specificity="SPECIFIC_FALSIFIABLE",
            ),
        )
        _two_rounds(dialogue, evidence=(_h(100),))
        final = dialogue.finalize()
        self.assertEqual(final.status, "UNKNOWN")
        self.assertIsNone(final.case_sha256)

    def test_unrecognized_novel_candidate_requires_confirmed_specific_evidence(self) -> None:
        claim = _claim()
        corpus = KobayashiCorpus("no-prior-case", ())

        def checker(c, known, args, transcript):
            return _assessment(
                c, known, transcript, check="CONFIRMED",
                evidence_sha256=_h(100),
                novelty_specificity="SPECIFIC_FALSIFIABLE",
            )

        dialogue = MinotaurDialogue(corpus, claim, evidence_checker=checker)
        _two_rounds(dialogue, evidence=(_h(100),))
        self.assertEqual(dialogue.finalize().status, "NOVEL_CANDIDATE")
        self.assertEqual(corpus.cases, ())  # no self-approved corpus learning

    def test_fake_unrecognized_confirmation_with_foreign_hash_is_unknown(self) -> None:
        claim = _claim()
        dialogue = MinotaurDialogue(
            KobayashiCorpus("blank", ()), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(
                c, known, transcript, check="CONFIRMED",
                evidence_sha256=_h(1234),
                novelty_specificity="SPECIFIC_FALSIFIABLE",
            ),
        )
        _two_rounds(dialogue, evidence=(_h(100),))
        self.assertEqual(dialogue.finalize().status, "UNKNOWN")

    def test_known_refutation_without_valid_evidence_is_unknown(self) -> None:
        claim = _claim()
        case = _case(claim)
        for check, foreign_evidence in (("INDETERMINATE", None), ("CONFIRMED", _h(999))):
            with self.subTest(check=check):
                dialogue = MinotaurDialogue(
                    KobayashiCorpus("canonical", (case,)), claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, check=check,
                        evidence_sha256=foreign_evidence,
                    ),
                )
                _two_rounds(dialogue)
                self.assertEqual(dialogue.finalize().status, "UNKNOWN")

    def test_known_case_contradiction_needs_verified_counterevidence(self) -> None:
        claim = _claim()
        case = _case(claim)
        corpus = KobayashiCorpus("canonical", (case,))

        def evidence_checker(c, known, args, transcript):
            return _assessment(
                c, known, transcript, check="CONTRADICTED", evidence_sha256=_h(300),
                novelty_specificity="SPECIFIC_FALSIFIABLE",
            )

        unsupported = MinotaurDialogue(corpus, claim, evidence_checker=evidence_checker)
        _two_rounds(unsupported)
        self.assertEqual(unsupported.finalize().status, "UNKNOWN")
        evidenced = MinotaurDialogue(corpus, claim, evidence_checker=evidence_checker)
        _two_rounds(evidenced, evidence=(_h(300),))
        self.assertEqual(evidenced.finalize().status, "NOVEL_CANDIDATE")

    def test_appeal_round_limits_idempotency_and_tamper_detection(self) -> None:
        claim = _claim()
        case = _case(claim)
        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(c, known, transcript),
            max_rounds=2,
        )
        first = dialogue.submit_argument("First mechanistic reason")
        self.assertIs(dialogue.submit_argument("First mechanistic reason"), first)
        self.assertEqual(len(dialogue.arguments), 1)
        second = dialogue.submit_argument("Second independent falsifier")
        self.assertIs(dialogue.submit_argument("First mechanistic reason"), first)
        self.assertEqual(len(dialogue.arguments), 2)
        self.assertEqual((first.round_number, second.round_number), (1, 2))
        with self.assertRaisesRegex(KobayashiError, "limit exceeded"):
            dialogue.submit_argument("Third argument")
        final = dialogue.finalize()
        final.verify()
        with self.assertRaises(FrozenInstanceError):
            final.status = "SURVIVES"
        object.__setattr__(final, "status", "SURVIVES")
        with self.assertRaisesRegex(KobayashiError, "seal mismatch"):
            final.verify()
        with self.assertRaisesRegex(KobayashiError, "seal mismatch"):
            dialogue.finalize()

    def test_reviewer_assessment_tamper_prevents_final_approval(self) -> None:
        claim = _claim()
        case = _case(claim)

        def fake_reviewer(c, known, args, transcript):
            assessment = _assessment(c, known, transcript)
            object.__setattr__(assessment, "evidence_sha256", _h(999))
            return assessment

        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=fake_reviewer,
        )
        _two_rounds(dialogue)
        with self.assertRaisesRegex(KobayashiError, "seal mismatch"):
            dialogue.finalize()
        self.assertIsNone(dialogue.final)

    def test_minotaur_replies_are_part_of_sealed_transcript_and_verified(self) -> None:
        claim = _claim()
        case = _case(claim)
        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(c, known, transcript),
        )
        _two_rounds(dialogue)
        original = dialogue.dialogue_sha256
        dialogue._replies[0] = replace(
            dialogue._replies[0], public_challenge="Minotaur secretly says there is a winning answer.",
        )
        self.assertNotEqual(dialogue.dialogue_sha256, original)
        with self.assertRaisesRegex(KobayashiError, "challenge altered"):
            dialogue.finalize()

    def test_reentrant_checker_cannot_add_arguments_after_snapshot(self) -> None:
        claim = _claim()
        case = _case(claim)
        holding = {}

        def hostile_checker(c, known, args, transcript):
            with self.assertRaisesRegex(KobayashiError, "reentrant"):
                holding["dialogue"].submit_argument("Malicious third turn during finalization")
            with self.assertRaisesRegex(KobayashiError, "reentrant"):
                holding["dialogue"].finalize()
            return _assessment(c, known, transcript)

        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=hostile_checker, max_rounds=3,
        )
        holding["dialogue"] = dialogue
        _two_rounds(dialogue)
        original = dialogue.dialogue_sha256
        self.assertEqual(dialogue.finalize().dialogue_sha256, original)
        self.assertEqual(len(dialogue.arguments), 2)

    def test_host_turn_reviewer_can_offer_bound_argument_sensitive_challenges(self) -> None:
        claim = _claim()
        case = _case(claim)

        def turn_review(c, known, args, prefix_digest):
            return ReasonedTurnFeedback(
                claim_sha256=c.claim_sha256,
                argument_sha256=args[-1].argument_sha256,
                dialogue_sha256=prefix_digest,
                reason_code="REQUEST_FALSIFIER",
                public_challenge="You argued: " + args[-1].text[:120] + " What observation would refute it?",
                reviewer_id="host-source-turn-reviewer",
            )

        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(c, known, transcript),
            turn_reviewer=turn_review,
        )
        first = dialogue.submit_argument("Heat must stop within five minutes")
        second = dialogue.submit_argument("Pressure can be checked from the gauge")
        self.assertIn("five minutes", first.public_challenge)
        self.assertIn("pressure", second.public_challenge.lower())
        self.assertNotEqual(first.feedback_sha256, second.feedback_sha256)
        self.assertEqual(dialogue.finalize().status, "REFUTED")

    def test_host_checker_cannot_review_different_scenario_or_transcript(self) -> None:
        claim = _claim()
        case = _case(claim)
        for mismatched in ({"world_sha256": _h(900)}, {"scenario_sha256": _h(901)},
                           {"claim_sha256": _h(902)}, {"case_sha256": _h(903)},
                           {"dialogue_sha256": _h(904)}):
            with self.subTest(mismatched=mismatched):
                dialogue = MinotaurDialogue(
                    KobayashiCorpus("canonical", (case,)), claim,
                    evidence_checker=lambda c, known, args, transcript: _assessment(
                        c, known, transcript, **mismatched,
                    ),
                )
                _two_rounds(dialogue)
                with self.assertRaisesRegex(KobayashiError, "did not bind"):
                    dialogue.finalize()
                self.assertIsNone(dialogue.final)

    def test_participant_cannot_supply_minotaur_verdict_or_checker(self) -> None:
        claim = _claim()
        case = _case(claim)
        with self.assertRaises(KobayashiError):
            MinotaurDialogue(KobayashiCorpus("canonical", (case,)), claim, evidence_checker=None)
        dialogue = MinotaurDialogue(
            KobayashiCorpus("canonical", (case,)), claim,
            evidence_checker=lambda c, known, args, transcript: _assessment(c, known, transcript),
        )
        _two_rounds(dialogue)
        with self.assertRaises(TypeError):
            dialogue.finalize("SURVIVES")
        self.assertEqual(dialogue.finalize().status, "REFUTED")

    def test_corpus_persistent_immutable_revisions_and_replay(self) -> None:
        original = _claim()
        first_case = _case(original)
        first = KobayashiCorpus("canonical", (first_case,))
        next_case = _case(_claim(action=_h(456)), category="PHYSICAL_CONTRADICTION")
        revision = first.append_revision((next_case,))
        self.assertEqual(revision.parent_corpus_sha256, first.corpus_sha256)
        self.assertEqual(len(first.cases), 1)
        self.assertEqual(len(revision.cases), 2)
        self.assertNotEqual(revision.corpus_sha256, first.corpus_sha256)
        self.assertEqual(first.public_commitment()["corpus_sha256"], first.corpus_sha256)
        self.assertNotIn("cases", first.public_commitment())
        with self.assertRaises(KobayashiError):
            first.append_revision((first_case,))  # cannot rewrite known claim

        # New review of a known intervention is possible only as a new immutable
        # superseding evidence case, with the exact prior case hash committed.
        corrected = RebuttalCase(
            scenario_sha256=first_case.scenario_sha256,
            world_sha256=first_case.world_sha256,
            action_sha256=first_case.action_sha256,
            claim_sha256=None,
            category="INSUFFICIENT_EVIDENCE", finding="REFUTED",
            evidence_sha256=_h(602),
            public_challenge="Newly reviewed evidence questions prior certainty.",
            supersedes_case_sha256=first_case.case_sha256,
        )
        later_revision = first.append_revision((corrected,))
        self.assertIs(later_revision.lookup(original), corrected)
        self.assertEqual(later_revision.cases[0], first_case)
        self.assertEqual(later_revision.parent_corpus_sha256, first.corpus_sha256)

        exact_only_superseding = RebuttalCase(
            first_case.scenario_sha256, first_case.world_sha256,
            first_case.action_sha256, first_case.claim_sha256 or original.claim_sha256,
            "PHYSICAL_CONTRADICTION", "REFUTED", _h(603),
            "This newer evidence is specific to the original exact statement.",
            supersedes_case_sha256=first_case.case_sha256,
        )
        narrowed = first.append_revision((exact_only_superseding,))
        self.assertIs(narrowed.lookup(original), exact_only_superseding)
        self.assertIsNone(narrowed.lookup(_claim(text="The same action in other words.")))

        with TemporaryDirectory() as directory:
            first_file = Path(directory) / "first.json"
            second_file = Path(directory) / "second.json"
            first.persist(first_file)
            first.persist(first_file)  # exact replay is safe idempotent no-op
            self.assertEqual(KobayashiCorpus.load(first_file).corpus_sha256, first.corpus_sha256)
            with self.assertRaises(KobayashiError):
                revision.persist(first_file)  # cannot replace committed artifact
            revision.persist(second_file)
            self.assertEqual(KobayashiCorpus.load(second_file).parent_corpus_sha256,
                             first.corpus_sha256)
            tampered = second_file.read_text(encoding="ascii").replace(
                "PHYSICAL_CONTRADICTION", "INSUFFICIENT_EVIDENCE",
            )
            second_file.write_text(tampered, encoding="ascii")
            with self.assertRaisesRegex(KobayashiError, "digest mismatch"):
                KobayashiCorpus.load(second_file)
            symlink = Path(directory) / "shortcut.json"
            symlink.symlink_to(first_file)
            with self.assertRaises((OSError, KobayashiError)):
                KobayashiCorpus.load(symlink)

    def test_corpus_publish_failure_does_not_expose_partial_canonical_file(self) -> None:
        corpus = KobayashiCorpus("canonical", (_case(_claim()),))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "corpus.json"

            # Serializing to the sibling temp must finish before the canonical
            # pathname can be published. An I/O failure leaves no false canon.
            with patch("labyrinth.kobayashi_minotaur_v1.os.fsync",
                       side_effect=OSError("injected fsync failure")):
                with self.assertRaisesRegex(OSError, "injected fsync failure"):
                    corpus.persist(destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

            # If atomic create-once naming is unavailable, do not fall back to
            # a potentially replacing rename or partially written final path.
            with patch("labyrinth.kobayashi_minotaur_v1.os.link",
                       side_effect=PermissionError("no hardlink support")):
                with self.assertRaisesRegex(KobayashiError, "unsupported"):
                    corpus.persist(destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

            corpus.persist(destination)
            canonical_bytes = destination.read_bytes()
            self.assertEqual(KobayashiCorpus.load(destination).corpus_sha256,
                             corpus.corpus_sha256)
            corpus.persist(destination)  # exact same immutable artifact replays
            self.assertEqual(destination.read_bytes(), canonical_bytes)
            self.assertEqual(list(Path(directory).iterdir()), [destination])

            later = corpus.append_revision((_case(_claim(action=_h(888))),))
            with self.assertRaisesRegex(KobayashiError, "another commitment"):
                later.persist(destination)
            self.assertEqual(destination.read_bytes(), canonical_bytes)

    def test_corpus_loader_rejects_ambiguous_duplicate_json_keys(self) -> None:
        corpus = KobayashiCorpus("canonical", (_case(_claim()),))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "corpus.json"
            corpus.persist(destination)
            original = destination.read_text(encoding="ascii")
            ambiguous = original.replace(
                '"corpus_id": "canonical",',
                '"corpus_id": "attacker",\n  "corpus_id": "canonical",', 1,
            )
            self.assertNotEqual(ambiguous, original)
            destination.write_text(ambiguous, encoding="ascii")
            with self.assertRaisesRegex(KobayashiError, "duplicate"):
                KobayashiCorpus.load(destination)

    def test_corpus_loader_rejects_symlinks_without_os_nofollow(self) -> None:
        """Windows-like platforms lack O_NOFOLLOW; preflight still matters."""
        with TemporaryDirectory() as directory:
            canonical = Path(directory) / "corpus.json"
            KobayashiCorpus("canonical", (_case(_claim()),)).persist(canonical)
            shortcut = Path(directory) / "shortcut.json"
            shortcut.symlink_to(canonical)
            with patch.object(minotaur.os, "O_NOFOLLOW", 0, create=True):
                with self.assertRaisesRegex(KobayashiError, "symlink"):
                    KobayashiCorpus.load(shortcut)

    def test_corpus_loader_limits_untrusted_file_size_before_json_parsing(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "oversized.json"
            with destination.open("wb") as output:
                output.truncate(16 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(KobayashiError, "byte limit"):
                KobayashiCorpus.load(destination)

    def test_corpus_concurrent_publisher_wins_without_overwrite(self) -> None:
        first = KobayashiCorpus("canonical", (_case(_claim()),))
        competing = first.append_revision((_case(_claim(action=_h(889))),))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "corpus.json"
            competing_path = Path(directory) / "competing.json"
            competing.persist(competing_path)
            competing_bytes = competing_path.read_bytes()

            def competing_publication(source: str, target: Path) -> None:
                # Simulate another process linking its fully written revision
                # between first's existence probe and its atomic link.
                hardlink(competing_path, target)
                raise FileExistsError("concurrent publication won")

            with patch("labyrinth.kobayashi_minotaur_v1.os.link",
                       side_effect=competing_publication):
                with self.assertRaisesRegex(KobayashiError, "another commitment"):
                    first.persist(destination)
            self.assertEqual(destination.read_bytes(), competing_bytes)
            self.assertFalse(any(Path(directory).glob(".kobayashi-corpus-*.tmp")))

            destination.unlink()

            def identical_publication(source: str, target: Path) -> None:
                hardlink(source, target)
                raise FileExistsError("concurrent publication won")

            with patch("labyrinth.kobayashi_minotaur_v1.os.link",
                       side_effect=identical_publication):
                first.persist(destination)
            self.assertEqual(KobayashiCorpus.load(destination).corpus_sha256,
                             first.corpus_sha256)
            self.assertFalse(any(Path(directory).glob(".kobayashi-corpus-*.tmp")))

    def test_bad_case_or_corpus_claim_is_never_silent(self) -> None:
        claim = _claim()
        case = _case(claim)
        with self.assertRaises(KobayashiError):
            RebuttalCase(claim.scenario_sha256, claim.world_sha256, claim.action_sha256,
                         None, "UNLISTED_TRAP", "REFUTED", _h(5), "bad category")
        with self.assertRaises(KobayashiError):
            KobayashiCorpus("duplicate", (case, case))
        corpus = KobayashiCorpus("canonical", (case,))
        object.__setattr__(case, "finding", "SURVIVES")
        with self.assertRaisesRegex(KobayashiError, "seal mismatch"):
            corpus.lookup(claim)


if __name__ == "__main__":
    unittest.main()
