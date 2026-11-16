# Minotaur Wave 75 — replay metadata vocabulary (source only)

## Exact source and scope
- Repo: `thebrazenbeard/labyrinth`.
- Parent: `25e77e451b4355c77f525f5f9dfe6494d7d7a10b`, draft PR #256.
- Isolated Lappy worktree: `D:\VERA\worktrees\labyrinth-minotaur-provenance-wave75-20261116`.
- Source/test change only; no training, deployment or examiner-custody claim.

## RED: runner-impossible metadata passed replay
- Genuine one-ACTION `wait` receipt with seed 12 and max calls 1 passed audit.
- Recompute the unkeyed SHA digest after changing `adapter_provenance` to
  `INDEPENDENTLY_VERIFIED_NEURAL_MODEL` or the empty string.
- Likewise change `model_id_unattested` to empty or 257 characters.
- Before fix all four altered receipts were accepted (`audit_receipt=True`);
  four focused subtests failed, `RED_EXIT=1`.
- Runner only issues 3 exact provenance labels and model IDs of length 1–256.

## GREEN: validate claims before examiner construction
- Auditor now rejects undefined provenance labels and model-ID lengths
  outside 1–256; a valid receipt retains replay behavior.
- The focused same regression passed, `GREEN_EXIT=0`.
- Windows bridge+subprocess+private-read tests **60/60 PASS**.
- WSL bridge+subprocess tests **54/54 PASS**.
- WSL optimized `-O` bridge tests **40/40 PASS**.
- **154 targeted test executions**, Python `compileall` and `git diff --check` exit 0.

## Limits and hostile challenge
- An allowed provenance label is still self-reported, not authenticated.
- Receipt digest is unkeyed and can be recomputed; provenance/source
  weights, examiner OS-principal isolation #146 and held-out custody #147
  remain unqualified. A green test is not model learning or graduation.
