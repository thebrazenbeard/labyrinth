"""Host-owned, source-only maze-completion obligation and replay tests."""

from queue import Queue
from threading import Barrier, Thread
import unittest

from labyrinth.kobayashi_completion_gate_v1 import (
    CompletionGateError,
    KobayashiCompletionGate,
    KobayashiCompletionLedger,
)
from labyrinth.kobayashi_encounter_v1 import KobayashiMazeLifecycle
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from tests.test_kobayashi_encounter_v1 import _maze, _trigger, _resume_to_atrium


EVIDENCE = "a" * 64


def _open_episode(ledger, maze_id="maze-A", instance="1" * 64):
    lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
    gate = KobayashiCompletionGate(ledger, maze_id, instance, lifecycle)
    _trigger(lifecycle)
    raw = "I turn the cobalt valve once."
    result = lifecycle.submit_proposal(raw)
    observed = lifecycle._challenge.observe()
    claim = BoundClaim(observed.sha256, lifecycle._challenge.graph_sha256,
                       result["action_sha256"], raw)

    def checker(bound, case, arguments, transcript):
        return EvidenceAssessment(
            bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
            None, transcript, "INDETERMINATE", None,
            "No independently verified physical counterexample.", "host-test-reviewer",
        )

    dialogue = MinotaurDialogue(
        KobayashiCorpus("source-only-host-corpus", ()), claim,
        evidence_checker=checker, min_rounds=2, max_rounds=2,
    )
    lifecycle.bind_minotaur(dialogue)
    return gate, lifecycle, dialogue


def _finish(lifecycle, with_evidence=True):
    lifecycle.submit_argument("My first falsifiable explanation.",
                              (EVIDENCE,) if with_evidence else ())
    lifecycle.submit_argument("I recognize the evidence may contradict my model.")
    lifecycle.finalize_appeal()
    _resume_to_atrium(lifecycle)


class KobayashiCompletionGateTests(unittest.TestCase):
    def test_parallel_same_instance_acceptance_has_one_winner_without_deadlock(self):
        ledger = KobayashiCompletionLedger()
        first, first_lifecycle, first_dialogue = _open_episode(
            ledger, "racing-A", "9" * 64)
        second, second_lifecycle, second_dialogue = _open_episode(
            ledger, "racing-B", "9" * 64)
        _finish(first_lifecycle)
        _finish(second_lifecycle)
        start = Barrier(3)
        results = Queue()

        def attempt(gate, dialogue):
            try:
                start.wait(timeout=3)
                results.put(("accepted", gate.commit(dialogue, evidence_refs=(EVIDENCE,))))
            except CompletionGateError as exc:
                results.put(("rejected", str(exc)))
            except BaseException as exc:
                results.put(("unexpected", repr(exc)))

        workers = [Thread(target=attempt, args=(first, first_dialogue)),
                   Thread(target=attempt, args=(second, second_dialogue))]
        for worker in workers:
            worker.start()
        start.wait(timeout=3)
        for worker in workers:
            worker.join(timeout=4)
            self.assertFalse(worker.is_alive(), "completion ledger concurrent call deadlocked")
        outcomes = [results.get_nowait() for _ in range(2)]
        self.assertCountEqual([row[0] for row in outcomes], ["accepted", "rejected"])
        self.assertIn("instance replay", next(row[1] for row in outcomes
                                               if row[0] == "rejected"))

    def test_source_completion_requires_final_appeal_and_full_maze(self):
        ledger = KobayashiCompletionLedger()
        gate, lifecycle, dialogue = _open_episode(ledger)
        with self.assertRaisesRegex(CompletionGateError, "final"):
            gate.commit(dialogue, evidence_refs=(EVIDENCE,))
        lifecycle.submit_argument("My first falsifiable explanation.", (EVIDENCE,))
        lifecycle.submit_argument("I recognize the evidence may contradict my model.")
        lifecycle.finalize_appeal()
        with self.assertRaisesRegex(CompletionGateError, "maze.*completed"):
            gate.commit(dialogue, evidence_refs=(EVIDENCE,))
        _resume_to_atrium(lifecycle)
        receipt = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
        self.assertTrue(receipt.maze_completed_source)
        self.assertEqual(receipt.host_acceptance, "SOURCE_ONLY_UNATTESTED")
        self.assertEqual(receipt.argument_count, 2)
        self.assertEqual(receipt.evidence_refs, (EVIDENCE,))
        receipt.verify()

    def test_local_catastrophe_is_not_global_maze_failure(self):
        ledger = KobayashiCompletionLedger()
        gate, lifecycle, dialogue = _open_episode(ledger)
        self.assertEqual(lifecycle.snapshot()["local_encounter_status"], "LOCAL_LOSS")
        _finish(lifecycle)
        receipt = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
        self.assertEqual(receipt.local_encounter_status, "LOCAL_LOSS")
        self.assertFalse(receipt.maze_failed_due_to_local_loss)
        self.assertTrue(receipt.maze_completed_source)
        self.assertNotEqual(receipt.verdict_status, "SURVIVES")

    def test_identical_verdict_allowed_in_distinct_legitimate_instances(self):
        ledger = KobayashiCompletionLedger()
        first, lifecycle, dialogue = _open_episode(ledger, "maze-A", "1" * 64)
        second, other_lifecycle, other_dialogue = _open_episode(
            ledger, "maze-B", "2" * 64)
        _finish(lifecycle)
        _finish(other_lifecycle)
        self.assertEqual(dialogue.final.verdict_sha256,
                         other_dialogue.final.verdict_sha256)
        receipt = first.commit(dialogue, evidence_refs=(EVIDENCE,))
        self.assertIs(first.commit(dialogue, evidence_refs=(EVIDENCE,)), receipt)
        second_receipt = second.commit(other_dialogue, evidence_refs=(EVIDENCE,))
        self.assertEqual(second_receipt.verdict_sha256, receipt.verdict_sha256)
        self.assertNotEqual(second_receipt.receipt_sha256, receipt.receipt_sha256)
        self.assertIs(ledger.lookup("maze-A", "1" * 64), receipt)
        self.assertIs(ledger.lookup("maze-B", "2" * 64), second_receipt)

    def test_reused_instance_nonce_and_live_lifecycle_are_rejected(self):
        ledger = KobayashiCompletionLedger()
        first, lifecycle, dialogue = _open_episode(ledger, "maze-A", "1" * 64)
        second, second_lifecycle, second_dialogue = _open_episode(
            ledger, "maze-B", "1" * 64)
        _finish(lifecycle)
        _finish(second_lifecycle)
        first.commit(dialogue, evidence_refs=(EVIDENCE,))
        with self.assertRaisesRegex(CompletionGateError, "instance replay"):
            second.commit(second_dialogue, evidence_refs=(EVIDENCE,))

        # Two host gate objects may open on one raw maze before the proposal,
        # but may not each claim its one run under different maze identities.
        shared = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        gate_a = KobayashiCompletionGate(ledger, "shared-A", "4" * 64, shared)
        gate_b = KobayashiCompletionGate(ledger, "shared-B", "5" * 64, shared)
        _trigger(shared)
        raw = "I turn the cobalt valve once."
        attempted = shared.submit_proposal(raw)
        observation = shared._challenge.observe()
        claim = BoundClaim(observation.sha256, shared._challenge.graph_sha256,
                           attempted["action_sha256"], raw)
        dialogue_shared = MinotaurDialogue(
            KobayashiCorpus("source-only-host-corpus", ()), claim,
            evidence_checker=lambda bound, case, args, digest: EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, digest, "INDETERMINATE", None,
                "No independently verified physical counterexample.", "host-test-reviewer"),
            min_rounds=2, max_rounds=2,
        )
        shared.bind_minotaur(dialogue_shared)
        _finish(shared)
        gate_a.commit(dialogue_shared, evidence_refs=(EVIDENCE,))
        with self.assertRaisesRegex(CompletionGateError, "lifecycle replay"):
            gate_b.commit(dialogue_shared, evidence_refs=(EVIDENCE,))

    def test_missing_or_extra_evidence_references_rejected(self):
        gate, lifecycle, dialogue = _open_episode(KobayashiCompletionLedger())
        _finish(lifecycle)
        with self.assertRaisesRegex(CompletionGateError, "evidence"):
            gate.commit(dialogue, evidence_refs=())
        with self.assertRaisesRegex(CompletionGateError, "duplicate"):
            gate.commit(dialogue, evidence_refs=(EVIDENCE, EVIDENCE))
        receipt = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
        self.assertEqual(receipt.evidence_refs, (EVIDENCE,))

    def test_mutated_final_verdict_and_substituted_dialogue_rejected(self):
        gate, lifecycle, dialogue = _open_episode(KobayashiCompletionLedger())
        _finish(lifecycle)
        fake = MinotaurDialogue(
            KobayashiCorpus("unbound", ()), dialogue.claim,
            evidence_checker=lambda *_: None, min_rounds=2, max_rounds=2)
        with self.assertRaisesRegex(CompletionGateError, "bound"):
            gate.commit(fake, evidence_refs=(EVIDENCE,))

        verdict = dialogue.final
        object.__setattr__(verdict, "status", "SURVIVES")
        with self.assertRaisesRegex(CompletionGateError, "verdict.*seal"):
            gate.commit(dialogue, evidence_refs=(EVIDENCE,))

    def test_obligation_must_open_before_proposal_and_requires_strict_mode(self):
        ledger = KobayashiCompletionLedger()
        ordinary = KobayashiMazeLifecycle(_maze())
        with self.assertRaisesRegex(CompletionGateError, "strict"):
            KobayashiCompletionGate(ledger, "ordinary", "3" * 64, ordinary)
        strict = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        _trigger(strict)
        strict.submit_proposal("I turn the cobalt valve once.")
        with self.assertRaisesRegex(CompletionGateError, "before.*proposal"):
            KobayashiCompletionGate(ledger, "late", "4" * 64, strict)


if __name__ == "__main__":
    unittest.main()
