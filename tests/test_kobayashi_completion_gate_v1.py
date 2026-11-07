"""Host-owned, source-only maze-completion obligation and replay tests."""

from queue import Queue
from threading import Barrier, Thread
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from labyrinth.kobayashi_completion_gate_v1 import (
    CompletionGateError,
    KobayashiCompletionGate,
    KobayashiCompletionLedger,
    SourceOnlyCompletionReceipt,
)
from labyrinth.kobayashi_encounter_v1 import KobayashiMazeLifecycle
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from tests.test_kobayashi_encounter_v1 import _maze, _trigger, _resume_to_atrium


EVIDENCE = "a" * 64


def _open_episode(ledger, maze_id="maze-A", instance="1" * 64, *,
                  reserve=True, start=True):
    lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
    gate = KobayashiCompletionGate(ledger, maze_id, instance, lifecycle)
    if reserve:
        gate.reserve()
    return gate, lifecycle, _bind_dialogue(lifecycle) if start else None


def _bind_dialogue(lifecycle):
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
    return dialogue


def _finish(lifecycle, with_evidence=True):
    lifecycle.submit_argument("My first falsifiable explanation.",
                              (EVIDENCE,) if with_evidence else ())
    lifecycle.submit_argument("I recognize the evidence may contradict my model.")
    lifecycle.finalize_appeal()
    _resume_to_atrium(lifecycle)


class KobayashiCompletionGateTests(unittest.TestCase):
    def test_unstarted_lifecycle_cannot_insert_forged_receipt_directly(self):
        lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        receipt = SourceOnlyCompletionReceipt(
            maze_id="forged", scenario_instance_sha256="1" * 64,
            maze_world_sha256="a" * 64, maze_state_sha256="a" * 64,
            scenario_sha256="a" * 64, scenario_world_sha256="a" * 64,
            proposal_sha256="a" * 64, action_sha256="a" * 64,
            claim_sha256="a" * 64, corpus_sha256="a" * 64,
            dialogue_sha256="a" * 64, review_sha256="a" * 64,
            verdict_sha256="a" * 64, verdict_status="SURVIVES",
            argument_count=2, evidence_refs=(),
            local_encounter_status="UNRESOLVED",
            lifecycle_receipt_sha256="a" * 64,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            with self.assertRaisesRegex(CompletionGateError, "gate|validated"):
                ledger.accept(receipt, lifecycle)
            self.assertIsNone(ledger.lookup("forged", "1" * 64))
            ledger.close()
            reopened = KobayashiCompletionLedger(storage_path=path)
            self.assertIsNone(reopened.lookup("forged", "1" * 64))
            reopened.close()

    def test_pending_reservation_survives_restart_and_blocks_new_execution(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(ledger, "pending-A", "5" * 64)
            self.assertEqual(ledger.admission_status("pending-A", "5" * 64), "PENDING")
            ledger.close()  # Simulate interrupted process: no completion receipt.
            recovered = KobayashiCompletionLedger(storage_path=path)
            self.assertEqual(recovered.admission_status("pending-A", "5" * 64), "PENDING")
            self.assertIsNone(recovered.lookup("pending-A", "5" * 64))
            other = KobayashiCompletionGate(
                recovered, "pending-A", "5" * 64,
                KobayashiMazeLifecycle(_maze(), require_final_appeal=True))
            with self.assertRaisesRegex(CompletionGateError, "pending|reserved"):
                other.reserve()
            recovered.close()

    def test_sqlite_independent_processes_contend_before_execution(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            # Both subprocesses have independent Python memory and DB handles.
            # The loser must not proceed to the side-effect marker.
            probe = """
import sys
from labyrinth.kobayashi_completion_gate_v1 import (
    CompletionGateError, KobayashiCompletionLedger, KobayashiCompletionGate)
from labyrinth.kobayashi_encounter_v1 import KobayashiMazeLifecycle
from tests.test_kobayashi_encounter_v1 import _maze
ledger = KobayashiCompletionLedger(storage_path=sys.argv[1])
gate = KobayashiCompletionGate(ledger, sys.argv[2], '6' * 64,
                              KobayashiMazeLifecycle(_maze(), require_final_appeal=True))
try:
    gate.reserve()
    print('RESERVED', flush=True)
except CompletionGateError:
    print('REJECTED', flush=True)
ledger.close()
"""
            processes = [subprocess.Popen(
                [sys.executable, "-c", probe, str(path), f"race-{index}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for index in range(2)]
            results = [process.communicate(timeout=10) for process in processes]
            self.assertEqual([process.returncode for process in processes], [0, 0], results)
            self.assertCountEqual([stdout.strip() for stdout, _ in results],
                                  ["RESERVED", "REJECTED"])
            reopened = KobayashiCompletionLedger(storage_path=path)
            self.assertEqual(reopened.admission_status("race-0", "6" * 64), "PENDING")
            reopened.close()

    def test_reservation_commit_failure_rolls_back_but_postcommit_unknown_blocks(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(
                ledger, "fail-reserve", "8" * 64, reserve=False, start=False)
            with patch.object(ledger, "_sqlite_commit", side_effect=RuntimeError("precommit")):
                with self.assertRaisesRegex(RuntimeError, "precommit"):
                    gate.reserve()
            self.assertEqual(ledger.admission_status("fail-reserve", "8" * 64),
                             "UNCLAIMED")
            # Failure after the SQLite COMMIT cannot be treated as an absent
            # reservation. A retry would duplicate the interrupted execution.
            with patch.object(ledger, "_read_pending_sqlite", side_effect=[
                None, CompletionGateError("postcommit readback uncertain")]):
                with self.assertRaisesRegex(CompletionGateError, "readback uncertain"):
                    gate.reserve()
            self.assertEqual(ledger.admission_status("fail-reserve", "8" * 64),
                             "PENDING")
            with self.assertRaisesRegex(CompletionGateError, "pending"):
                gate.reserve()
            ledger.close()

    def test_gate_cannot_reserve_retroactively_after_model_world_actions(self):
        ledger = KobayashiCompletionLedger()
        gate, lifecycle, dialogue = _open_episode(ledger, reserve=False)
        with self.assertRaisesRegex(CompletionGateError, "before any"):
            gate.reserve()
        self.assertEqual(ledger.admission_status("maze-A", "1" * 64), "UNCLAIMED")
        _finish(lifecycle)
        with self.assertRaisesRegex(CompletionGateError, "pre-execution"):
            gate.commit(dialogue, evidence_refs=(EVIDENCE,))

    def test_gate_cannot_reserve_twice_to_restart_execution(self):
        ledger = KobayashiCompletionLedger()
        gate, lifecycle, dialogue = _open_episode(ledger)
        with self.assertRaisesRegex(CompletionGateError, "already reserved"):
            gate.reserve()
        self.assertEqual(ledger.admission_status("maze-A", "1" * 64), "PENDING")

    def test_sqlite_separate_connection_race_has_one_committed_nonce(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            first = KobayashiCompletionLedger(storage_path=path)
            second = KobayashiCompletionLedger(storage_path=path)
            left, left_lifecycle, left_dialogue = _open_episode(
                first, "race-A", "7" * 64, reserve=False, start=False)
            right, right_lifecycle, right_dialogue = _open_episode(
                second, "race-B", "7" * 64, reserve=False, start=False)
            barrier = Barrier(3)
            results = Queue()

            def attempt(gate):
                try:
                    barrier.wait(timeout=3)
                    gate.reserve()
                    results.put(("reserved", gate))
                except CompletionGateError as exc:
                    results.put(("rejected", str(exc)))
                except BaseException as exc:
                    results.put(("unexpected", repr(exc)))

            workers = [Thread(target=attempt, args=(left,)),
                       Thread(target=attempt, args=(right,))]
            for worker in workers:
                worker.start()
            barrier.wait(timeout=3)
            for worker in workers:
                worker.join(timeout=4)
                self.assertFalse(worker.is_alive(), "two SQLite connections deadlocked")
            outcomes = [results.get_nowait() for _ in range(2)]
            self.assertCountEqual([entry[0] for entry in outcomes], ["reserved", "rejected"])
            winner = next(value for kind, value in outcomes if kind == "reserved")
            self.assertIn("pending", next(value for kind, value in outcomes
                                                  if kind == "rejected"))
            if winner is left:
                left_dialogue = _bind_dialogue(left_lifecycle)
                _finish(left_lifecycle)
                accepted = left.commit(left_dialogue, evidence_refs=(EVIDENCE,))
            else:
                right_dialogue = _bind_dialogue(right_lifecycle)
                _finish(right_lifecycle)
                accepted = right.commit(right_dialogue, evidence_refs=(EVIDENCE,))
            reopened = KobayashiCompletionLedger(storage_path=path)
            self.assertEqual(reopened.lookup(accepted.maze_id, "7" * 64), accepted)
            first.close()
            second.close()
            reopened.close()

    def test_sqlite_receipt_reopens_in_separate_process_without_lifecycle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(ledger)
            _finish(lifecycle)
            saved = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
            self.assertEqual(gate.commit(dialogue, evidence_refs=(EVIDENCE,)), saved)
            ledger.close()
            with self.assertRaisesRegex(CompletionGateError, "closed"):
                ledger.lookup("maze-A", "1" * 64)
            with self.assertRaisesRegex(CompletionGateError, "closed"):
                gate.commit(dialogue, evidence_refs=(EVIDENCE,))

            # Fresh interpreter: there is no original Python lifecycle to consult.
            probe = subprocess.run([sys.executable, "-c", """
import sys
from labyrinth.kobayashi_completion_gate_v1 import KobayashiCompletionLedger
ledger = KobayashiCompletionLedger(storage_path=sys.argv[1])
receipt = ledger.lookup('maze-A', '1' * 64)
assert receipt is not None
receipt.verify()
print(receipt.receipt_sha256)
ledger.close()
""", str(path)], text=True, capture_output=True, timeout=10, check=True)
            self.assertEqual(probe.stdout.strip(), saved.receipt_sha256)
            reopened = KobayashiCompletionLedger(storage_path=path)
            self.assertEqual(reopened.lookup("maze-A", "1" * 64), saved)
            with self.assertRaisesRegex(CompletionGateError, "validated"):
                reopened.accept(saved, lifecycle)
            reopened.close()

    def test_sqlite_cross_process_nonce_collision_and_repeat_exam(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            first = KobayashiCompletionLedger(storage_path=path)
            competing = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(first, "maze-A", "1" * 64)
            _finish(lifecycle)
            saved = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
            with self.assertRaisesRegex(CompletionGateError, "committed"):
                _open_episode(competing, "maze-B", "1" * 64)
            with self.assertRaisesRegex(CompletionGateError, "committed"):
                _open_episode(competing, "maze-A", "1" * 64)
            self.assertEqual(competing.lookup("maze-A", "1" * 64), saved)
            distinct, distinct_lifecycle, distinct_dialogue = _open_episode(
                competing, "maze-B", "2" * 64)
            _finish(distinct_lifecycle)
            second = distinct.commit(distinct_dialogue, evidence_refs=(EVIDENCE,))
            self.assertEqual(second.verdict_sha256, saved.verdict_sha256)
            self.assertNotEqual(second.receipt_sha256, saved.receipt_sha256)
            first.close()
            competing.close()

    def test_sqlite_failed_commit_rolls_back_without_memory_acceptance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(ledger)
            _finish(lifecycle)
            with patch.object(ledger, "_sqlite_commit", side_effect=RuntimeError("failpoint")):
                with self.assertRaisesRegex(RuntimeError, "failpoint"):
                    gate.commit(dialogue, evidence_refs=(EVIDENCE,))
            self.assertEqual(ledger.admission_status("maze-A", "1" * 64), "PENDING")
            self.assertIsNone(ledger.lookup("maze-A", "1" * 64))
            reopened = KobayashiCompletionLedger(storage_path=path)
            self.assertIsNone(reopened.lookup("maze-A", "1" * 64))
            recovered = gate.commit(dialogue, evidence_refs=(EVIDENCE,))
            self.assertEqual(reopened.lookup("maze-A", "1" * 64), recovered)
            ledger.close()
            reopened.close()

    def test_sqlite_symlink_and_modified_receipt_fail_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "real.sqlite3"
            ledger = KobayashiCompletionLedger(storage_path=path)
            gate, lifecycle, dialogue = _open_episode(ledger)
            _finish(lifecycle)
            gate.commit(dialogue, evidence_refs=(EVIDENCE,))
            ledger.close()
            link = Path(directory) / "alias.sqlite3"
            link.symlink_to(path)
            with self.assertRaisesRegex(CompletionGateError, "symlink"):
                KobayashiCompletionLedger(storage_path=link)
            # A self-writable DB has no external authenticity: detect accidental
            # mutation only, not an attacker who recomputes the source digest.
            raw = sqlite3.connect(path)
            row = raw.execute("SELECT receipt_json FROM accepted_receipts").fetchone()
            payload = json.loads(row[0])
            payload["verdict_status"] = "SURVIVES"
            raw.execute("UPDATE accepted_receipts SET receipt_json=?", (json.dumps(payload),))
            raw.commit()
            raw.close()
            reopened = KobayashiCompletionLedger(storage_path=path)
            with self.assertRaisesRegex(CompletionGateError, "seal|tamper|mismatch"):
                reopened.lookup("maze-A", "1" * 64)
            reopened.close()

    def test_parallel_same_instance_acceptance_has_one_winner_without_deadlock(self):
        ledger = KobayashiCompletionLedger()
        first, first_lifecycle, first_dialogue = _open_episode(
            ledger, "racing-A", "9" * 64, reserve=False, start=False)
        second, second_lifecycle, second_dialogue = _open_episode(
            ledger, "racing-B", "9" * 64, reserve=False, start=False)
        start = Barrier(3)
        results = Queue()

        def attempt(gate):
            try:
                start.wait(timeout=3)
                gate.reserve()
                results.put(("reserved", gate))
            except CompletionGateError as exc:
                results.put(("rejected", str(exc)))
            except BaseException as exc:
                results.put(("unexpected", repr(exc)))

        workers = [Thread(target=attempt, args=(first,)),
                   Thread(target=attempt, args=(second,))]
        for worker in workers:
            worker.start()
        start.wait(timeout=3)
        for worker in workers:
            worker.join(timeout=4)
            self.assertFalse(worker.is_alive(), "completion ledger concurrent call deadlocked")
        outcomes = [results.get_nowait() for _ in range(2)]
        self.assertCountEqual([row[0] for row in outcomes], ["reserved", "rejected"])
        self.assertIn("pending", next(row[1] for row in outcomes
                                               if row[0] == "rejected"))
        winner = next(row[1] for row in outcomes if row[0] == "reserved")
        if winner is first:
            first_dialogue = _bind_dialogue(first_lifecycle)
            _finish(first_lifecycle)
            first.commit(first_dialogue, evidence_refs=(EVIDENCE,))
        else:
            second_dialogue = _bind_dialogue(second_lifecycle)
            _finish(second_lifecycle)
            second.commit(second_dialogue, evidence_refs=(EVIDENCE,))

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
        with self.assertRaisesRegex(CompletionGateError, "pending"):
            _open_episode(ledger, "maze-B", "1" * 64)
        _finish(lifecycle)
        first.commit(dialogue, evidence_refs=(EVIDENCE,))
        with self.assertRaisesRegex(CompletionGateError, "committed"):
            _open_episode(ledger, "maze-B", "1" * 64)

        # Two host gate objects may open on one raw maze before the proposal,
        # but may not each claim its one run under different maze identities.
        shared = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        gate_a = KobayashiCompletionGate(ledger, "shared-A", "4" * 64, shared)
        gate_b = KobayashiCompletionGate(ledger, "shared-B", "5" * 64, shared)
        gate_a.reserve()
        gate_b.reserve()
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
