# Minotaur Wave 70 — fail closed on unauditable error receipts (SOURCE ONLY)

Date: 2026-11-15. Branch: `fix/minotaur-error-replay-wave70-20261115`.
Base: Labyrinth Draft PR #251 commit `c821ffe5f674e170d4f58b69d3c69e2bf38f25ed`.
Status: DRAFT/HOLD. No main merge, runtime qualification, model graduation or training.

## Reproduced RED on exact parent source

The `audit_receipt` loop returned True for an inert, *correctly recomputed self-hash* describing `EVALUATOR_ERROR` even though its recorded ACTION was valid and succeeds in the actual examiner. The error branch accepted the claim without replaying the purported exception. Another correctly rehashed `MODEL_CALL_ERROR` record still contained nonempty candidate response bytes, contradicting the production emission path.

Two real regression tests failed before the fix, both `AssertionError: True is not false`, process `RED_EXIT=1`:
- `test_evaluator_error_cannot_be_audited_without_replaying_failure`
- `test_model_call_error_requires_empty_response_on_audit`

## Minimal repair

`labyrinth/minotaur_candidate_bridge.py` now rejects `EVALUATOR_ERROR` on audit: receipt contains no independently replayable evaluator-error identity, so fail closed instead of endorsing an untested exception. `MODEL_CALL_ERROR` still has the existing structural audit path but only with the empty response its runner actually writes. A real subprocess `EVALUATOR_ERROR` fixture is additionally asserted unauditable.

No evaluator simulation, model implementation, credential boundary, storage schema, signer, permissions, or workflow execution changed.

## Local tests and evidence classes

Focused GREEN: 2/2 after repair.
Windows Python bridge + subprocess + examiner-private probe: 54/54 PASS. `python -m compileall -q labyrinth` PASS; `git diff --check` PASS.
Lappy WSL Python bridge: 34/34 PASS normally, 34/34 PASS with `-O`. Subprocess suite status recorded separately at final check.
These are local TEST results, not independently authenticated runtime/model evidence.

## Hostile challenge and outstanding holds

> A self-hashed receipt is self-assertion, not a signer attestation. Denying an unverifiable error receipt prevents one false audit result but cannot prevent an attacker from regenerating a plausible transcript and hash. `MODEL_CALL_ERROR` still cannot prove a real model was called; it only enforces its recorded shape.

The two known hard blockers stay open:
- Issue #146: same-principal subprocess code can access known examiner-private file paths; genuine independent custody needs OS identity/security work and negative real-host probes.
- Issue #147: distinct IDs may reuse the same semantic mechanism; family independence, signer trust, control equivalence and authenticated inference are unproven.

No private-file isolation proof, hosted CI, independent reviewer, actual model inference/training, deployment, protected merge or graduation is claimed.
