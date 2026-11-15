# Minotaur Wave 71: reject premature ACTION receipt truncation (source-only)

## Threat and reproduction
An attacker controlling a self-hashed candidate receipt can restate a legitimate
one-call ACTION as a 32-call evaluation, recompute the unkeyed digest, and
convince the old auditor that the trace completed. The actual runner would
continue while the examiner remains active and another model call is permitted.

Parent: `6c3c60999431246d83eb67af70f90a514c7c2203` (draft PR #252).
Worktree: `fix/minotaur-audit-premature-stop-wave71-20261115`.

## RED / GREEN
- Test: `test_rehashed_receipt_cannot_stop_early_with_unused_model_calls`.
- Legitimate `max_model_calls=1` ACTION receipt replays successfully.
- Rehashed same trace with `max_model_calls=32`: before fix,
  `AssertionError: True is not false`; 1 failure; `RED_EXIT=1`.
- Auditor now rejects a final ACTION on a nonterminal case when
  `len(receipt.turns) < call_limit`. Valid terminal and call-limit exits remain.
- Same focused test: 1/1 PASS; `GREEN_EXIT=0`.

## Bounded test evidence on Lappy
- Windows: bridge + subprocess + private-read: 55/55 PASS.
- Windows: `python -m compileall -q labyrinth`: exit 0.
- Lappy WSL: bridge + subprocess: 49/49 PASS.
- Lappy WSL optimized Python (-O): bridge 35/35 PASS.
- `git diff --check`: exit 0.

## Hostile boundary
This verifies source-level trace completeness under a freshly replayed examiner,
not model provenance or neural training. Anyone who controls the receipt can
recompute its plain SHA-256. Same-user private case file access (issue #146),
mechanism-overlap/false heldout diversity (issue #147), and independent
examiner signer/custody remain outside this fix. DRAFT/HOLD; no graduation.
