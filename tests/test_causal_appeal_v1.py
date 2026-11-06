"""Source-only Minotaur appeals.  All reviewer callbacks here are test mocks."""

from dataclasses import FrozenInstanceError
from copy import deepcopy
from hashlib import sha256
import json
import re
import unittest

from labyrinth.causal_appeal_v1 import (
    AppealError,
    AppealFinalityError,
    AppealPetition,
    EVIDENCE_CLASS,
    IndependentReview,
    MinotaurAppealCourt,
    SealedRuling,
)
from labyrinth.causal_reality_v1 import CausalBlueprint, CausalWorld, MechanismAction
from labyrinth.causal_model_episode_v1 import CausalEpisodeRun, run_causal_episode
from labyrinth.local_model_action_adapter_v1 import LocalModelActionConfig


def _h(n: int) -> str:
    return f"{n:064x}"


def _initial(*, consequence: str = "FAIL", reasoning: str = "SUPPORTED") -> SealedRuling:
    return SealedRuling(
        world_sha256=_h(1), action_sha256=_h(2), event_sha256=_h(3),
        public_observation_sha256=_h(4), public_rationale_sha256=_h(5),
        decision_time_utc="2026-11-06T16:45:00Z",
        consequence=consequence, reasoning=reasoning,
        reason_code="KNOWN_CATASTROPHIC_OUTCOME",
    )


def _petition(initial: SealedRuling, *, ground: str = "SCORING_ERROR", arg: str = "Scorer misread event") -> AppealPetition:
    return AppealPetition(initial.seal, ground, arg, (_h(10),))


def _review(
    initial: SealedRuling,
    petition: AppealPetition,
    *,
    defect: str = "CONFIRMED",
    consequence_verified: bool | None = True,
    reasoning_verified: str = "SUPPORTED",
    **rebindings: object,
) -> IndependentReview:
    values = dict(
        original_seal=initial.seal,
        petition_digest=petition.digest,
        world_sha256=initial.world_sha256,
        action_sha256=initial.action_sha256,
        event_sha256=initial.event_sha256,
        public_observation_sha256=initial.public_observation_sha256,
        public_rationale_sha256=initial.public_rationale_sha256,
        decision_time_utc=initial.decision_time_utc,
        ground=petition.ground,
        defect=defect,
        consequence_verified=consequence_verified,
        reasoning_verified=reasoning_verified,
        review_evidence_sha256=_h(11),
        reviewer_id="mock-host-reviewer-source-test",
    )
    values.update(rebindings)
    return IndependentReview(**values)


class MinotaurAppealTests(unittest.TestCase):
    def test_replay_episode_receipt_seals_real_source_outcome_not_model_claim(self) -> None:
        blueprint = CausalBlueprint.from_seed(23, "solvable")
        mechanism = blueprint.actuators[0]
        verb = {"valve": "rotate", "lever": "pull", "cable": "cut", "brace": "push"}[mechanism.form]

        class FakeTransport:
            def complete(self, config: object, request: dict[str, object]) -> dict[str, object]:
                return {
                    "choices": [{
                        "message": {"content": f"I {verb} the {mechanism.label.replace('_', ' ')} once, because the public continuity indicator suggests cooling."},
                        "finish_reason": "stop",
                    }],
                }

        run = run_causal_episode(
            LocalModelActionConfig(port=1234, model="fake-for-source-test", json_mode=False),
            FakeTransport(), seed=23, challenge_class="solvable",
        )
        sealed = SealedRuling.from_causal_episode(run, decision_time_utc="2026-11-06T17:30:00Z")
        self.assertEqual(sealed.consequence, "PASS")
        self.assertEqual(sealed.reasoning, "INDETERMINATE")
        self.assertEqual(sealed.event_sha256, run.public_receipt["source_event"]["event_sha256"])
        self.assertEqual(sealed.action_sha256, run.protected_evaluation["full_source_event"]["action_sha256"])
        self.assertEqual(sealed.public_observation_sha256,
                         run.public_receipt["public_observation_sha256"])
        self.assertEqual(sealed.public_rationale_sha256,
                         run.public_receipt["model_response_sha256"])

        tampered_protected = deepcopy(run.protected_evaluation)
        tampered_protected["full_source_event"]["observed_outcome"] = "CATASTROPHIC_FAILURE"
        forged = CausalEpisodeRun(run.public_receipt, tampered_protected)
        with self.assertRaisesRegex(AppealError, "cannot be replayed"):
            SealedRuling.from_causal_episode(forged, decision_time_utc="2026-11-06T17:30:00Z")

        tampered_public = deepcopy(run.public_receipt)
        tampered_public["source_event"]["observed_outcome"] = "CATASTROPHIC_FAILURE"
        forged = CausalEpisodeRun(tampered_public, run.protected_evaluation)
        with self.assertRaisesRegex(AppealError, "receipt digest inconsistent"):
            SealedRuling.from_causal_episode(forged, decision_time_utc="2026-11-06T17:30:00Z")

        # Even if attacker recomputes public/protected receipt hashes, the
        # original raw utterance must be reparsed, not trusted as a mapping.
        def source_hash(value: object) -> str:
            return sha256(json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")).hexdigest()

        tampered_public = deepcopy(run.public_receipt)
        tampered_protected = deepcopy(run.protected_evaluation)
        tampered_public["proposal_and_rationale_as_submitted"] = "I pull a lever that does not exist."
        tampered_public["model_response_sha256"] = source_hash(
            tampered_public["proposal_and_rationale_as_submitted"]
        )
        tampered_public.pop("receipt_sha256")
        tampered_public["receipt_sha256"] = source_hash(tampered_public)
        tampered_protected["public_receipt_sha256"] = tampered_public["receipt_sha256"]
        forged = CausalEpisodeRun(tampered_public, tampered_protected)
        with self.assertRaisesRegex(AppealError, "raw model proposal"):
            SealedRuling.from_causal_episode(forged, decision_time_utc="2026-11-06T17:30:00Z")

    def test_verified_scorer_error_changes_evaluation_not_world_event(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        calls = []

        def host_reviewer(sealed: SealedRuling, appeal: AppealPetition) -> IndependentReview:
            calls.append(appeal.digest)
            return _review(sealed, appeal)

        court = MinotaurAppealCourt(initial, reviewer=host_reviewer)
        final = court.submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("CORRECTED", "PASS", "SUPPORTED"))
        self.assertEqual(final.event_sha256, initial.event_sha256)
        self.assertEqual(initial.consequence, "FAIL")
        self.assertEqual(final.original_seal, initial.seal)
        self.assertEqual(final.evidence_class, EVIDENCE_CLASS)
        self.assertIs(court.submit(petition), final)
        self.assertEqual(calls, [petition.digest])

    def test_catastrophic_outcome_does_not_invalidate_supported_reasoning(self) -> None:
        initial = _initial(consequence="FAIL", reasoning="SUPPORTED")
        petition = _petition(initial)
        final = MinotaurAppealCourt(
            initial,
            reviewer=lambda sealed, appeal: _review(
                sealed, appeal, defect="REFUTED", consequence_verified=False,
                reasoning_verified="SUPPORTED",
            ),
        ).submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("UPHELD", "FAIL", "SUPPORTED"))

    def test_confirmed_misparse_does_not_retroactively_fix_event(self) -> None:
        initial = _initial()
        petition = _petition(initial, ground="MISPARSED_ACTION")
        final = MinotaurAppealCourt(
            initial, reviewer=lambda sealed, appeal: _review(sealed, appeal),
        ).submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("UNRESOLVED", "FAIL", "SUPPORTED"))
        self.assertEqual(final.event_sha256, initial.event_sha256)
        self.assertEqual(final.explanation_code, "CAUSAL_OUTCOME_NOT_REWRITABLE")

    def test_other_confirmed_uncertainty_is_unknown(self) -> None:
        for ground in ("MECHANICS_CONTRADICTION", "INSUFFICIENT_EVIDENCE"):
            with self.subTest(ground=ground):
                initial = _initial()
                petition = _petition(initial, ground=ground)
                final = MinotaurAppealCourt(
                    initial,
                    reviewer=lambda sealed, appeal: _review(
                        sealed, appeal, consequence_verified=None,
                        reasoning_verified="INDETERMINATE",
                    ),
                ).submit(petition)
                self.assertEqual((final.state, final.consequence, final.reasoning),
                                 ("UNRESOLVED", "FAIL", "INDETERMINATE"))

    def test_scoring_error_without_verifiable_outcome_cannot_award_pass(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        final = MinotaurAppealCourt(
            initial,
            reviewer=lambda sealed, appeal: _review(
                sealed, appeal, consequence_verified=None,
                reasoning_verified="INDETERMINATE",
            ),
        ).submit(petition)
        self.assertEqual((final.state, final.consequence), ("UNRESOLVED", "FAIL"))

    def test_reasoning_only_correction_preserves_realized_failure(self) -> None:
        initial = _initial(consequence="FAIL", reasoning="INDETERMINATE")
        petition = _petition(initial)
        final = MinotaurAppealCourt(
            initial,
            reviewer=lambda sealed, appeal: _review(
                sealed, appeal, consequence_verified=None,
                reasoning_verified="SUPPORTED",
            ),
        ).submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("CORRECTED", "FAIL", "SUPPORTED"))
        self.assertEqual(final.event_sha256, initial.event_sha256)

    def test_indeterminate_review_remains_neutral(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        final = MinotaurAppealCourt(
            initial,
            reviewer=lambda sealed, appeal: _review(
                sealed, appeal, defect="INDETERMINATE",
                consequence_verified=True, reasoning_verified="SUPPORTED",
            ),
        ).submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("UNRESOLVED", "FAIL", "SUPPORTED"))

    def test_one_final_appeal_and_idempotent_replay(self) -> None:
        initial = _initial()
        first = _petition(initial)
        court = MinotaurAppealCourt(initial, reviewer=lambda sealed, appeal: _review(sealed, appeal))
        final = court.submit(first)
        self.assertIs(court.submit(first), final)
        with self.assertRaises(AppealFinalityError):
            court.submit(_petition(initial, arg="This is a different claim"))

    def test_final_ruling_seal_and_replay_integrity(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        court = MinotaurAppealCourt(initial, reviewer=lambda sealed, appeal: _review(sealed, appeal))
        final = court.submit(petition)
        final.verify()
        object.__setattr__(final, "consequence", "UNKNOWN")
        with self.assertRaisesRegex(AppealError, "integrity"):
            final.verify()
        with self.assertRaisesRegex(AppealError, "integrity"):
            court.submit(petition)

    def test_exact_world_action_event_observation_rationale_time_ground_binding(self) -> None:
        for changed in (
            {"world_sha256": _h(20)},
            {"action_sha256": _h(20)},
            {"event_sha256": _h(20)},
            {"public_observation_sha256": _h(20)},
            {"public_rationale_sha256": _h(20)},
            {"decision_time_utc": "2026-11-06T16:46:00Z"},
            {"ground": "MISPARSED_ACTION"},
            {"petition_digest": _h(20)},
            {"original_seal": _h(20)},
        ):
            with self.subTest(changed=changed):
                initial = _initial()
                petition = _petition(initial)
                court = MinotaurAppealCourt(
                    initial, reviewer=lambda sealed, appeal: _review(sealed, appeal, **changed),
                )
                with self.assertRaisesRegex(AppealError, "not bound"):
                    court.submit(petition)
                self.assertIsNone(court.final)

    def test_participant_has_no_review_submission_or_self_selected_verdict(self) -> None:
        initial = _initial()
        with self.assertRaises(TypeError):
            AppealPetition(initial.seal, "SCORING_ERROR", "x", outcome="PASS")
        with self.assertRaisesRegex(AppealError, "host-configured"):
            MinotaurAppealCourt(initial, reviewer=None)
        petition = _petition(initial)
        forged = _review(initial, petition, consequence_verified=True)
        court = MinotaurAppealCourt(
            initial,
            reviewer=lambda sealed, appeal: _review(sealed, appeal, defect="REFUTED"),
        )
        with self.assertRaises(TypeError):
            court.submit(petition, forged)
        self.assertEqual(court.submit(petition).consequence, "FAIL")

    def test_frozen_objects_and_integrity_prevent_tampering(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        calls = []
        court = MinotaurAppealCourt(initial, reviewer=lambda sealed, appeal: calls.append(1))
        with self.assertRaises(FrozenInstanceError):
            initial.consequence = "PASS"
        object.__setattr__(petition, "argument", "Changed after sealing")
        with self.assertRaisesRegex(AppealError, "integrity"):
            court.submit(petition)
        self.assertEqual(calls, [])
        object.__setattr__(initial, "consequence", "PASS")
        with self.assertRaisesRegex(AppealError, "integrity"):
            court.submit(_petition(_initial()))

    def test_review_integrity_prevents_unsealed_forged_outcome(self) -> None:
        initial = _initial()
        petition = _petition(initial)

        def forged(sealed: SealedRuling, appeal: AppealPetition) -> IndependentReview:
            review = _review(sealed, appeal)
            object.__setattr__(review, "consequence_verified", False)
            return review

        court = MinotaurAppealCourt(initial, reviewer=forged)
        with self.assertRaisesRegex(AppealError, "integrity"):
            court.submit(petition)
        self.assertIsNone(court.final)

    def test_closed_grounds_and_bounded_claims(self) -> None:
        for invalid_ground in ("A", "PICK_BEST", "", "SCORING_ERROR_AND_MORE"):
            with self.subTest(ground=invalid_ground):
                with self.assertRaises(AppealError):
                    _petition(_initial(), ground=invalid_ground)
        with self.assertRaises(AppealError):
            _petition(_initial(), arg="x" * 2049)
        with self.assertRaises(AppealError):
            AppealPetition(_initial().seal, "SCORING_ERROR", "claim", (_h(1),) * 9)
        with self.assertRaises(AppealError):
            AppealPetition(_initial().seal, "SCORING_ERROR", "claim", (_h(1), _h(1)))

    def test_cannot_claim_independently_attested_from_source_only(self) -> None:
        initial = _initial()
        petition = _petition(initial)
        with self.assertRaisesRegex(AppealError, "attested"):
            _review(initial, petition, evidence_class="INDEPENDENTLY_ATTESTED")

    def test_end_to_end_real_causal_world_model_petition_host_minotaur(self) -> None:
        world = CausalWorld(CausalBlueprint.from_seed(23, "solvable"))
        observation = world.observe()
        # Only the public safe-loop clue and visible label define the model's
        # chosen manipulation here; no private answer key is read by the model.
        match = re.search(r"associates (\w+) with the isolated coolant loop", observation.public_clues[0])
        self.assertIsNotNone(match)
        target = match.group(1)
        verb = {"valve": "turn", "lever": "pull", "cable": "cut", "brace": "push"}[target.split("_")[-1]]
        action = MechanismAction(verb, target)
        rationale = f"The public continuity clue links {target} to an isolated coolant loop."
        event = world.step(action)
        self.assertEqual(event.observed_outcome, "STATION_STABILIZED")
        initial = SealedRuling(
            world_sha256=world.blueprint_sha256,
            action_sha256=event.action_sha256,
            event_sha256=event.event_sha256,
            public_observation_sha256=observation.sha256,
            public_rationale_sha256=sha256(rationale.encode("utf-8")).hexdigest(),
            decision_time_utc="2026-11-06T17:00:00Z",
            consequence="FAIL",  # deliberately erroneous initial evaluator grade
            reasoning="INDETERMINATE",
            reason_code="INITIAL_SCORER_ERROR_UNDER_TEST",
        )
        petition = AppealPetition(
            initial.seal,
            "SCORING_ERROR",
            "The event reports STATION_STABILIZED, while the consequence was marked FAIL.",
            (event.event_sha256,),
        )

        def host_minotaur_review(sealed: SealedRuling, appeal: AppealPetition) -> IndependentReview:
            event_readback = world.result
            verdict = world.private_verdict()
            self.assertEqual(event_readback.event_sha256, sealed.event_sha256)
            self.assertEqual(appeal.public_evidence_sha256, (event_readback.event_sha256,))
            return _review(
                sealed, appeal,
                defect="CONFIRMED",
                consequence_verified=(verdict.outcome == "STATION_STABILIZED"),
                reasoning_verified="SUPPORTED",
                review_evidence_sha256=verdict.oracle_certificate_sha256,
            )

        final = MinotaurAppealCourt(initial, reviewer=host_minotaur_review).submit(petition)
        self.assertEqual((final.state, final.consequence, final.reasoning),
                         ("CORRECTED", "PASS", "SUPPORTED"))
        self.assertEqual(final.event_sha256, world.result.event_sha256)
        self.assertEqual(world.result.observed_outcome, "STATION_STABILIZED")


if __name__ == "__main__":
    unittest.main()
