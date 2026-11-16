# Minotaur Wave 76 — no escape credit without a model turn

## Source custody
- Repo: `thebrazenbeard/labyrinth`, stacked draft only.
- Parent: `e2eea5b9318c0c779242471f3a21396be26bc963` (draft PR #257).
- Isolated host worktree: `D:\VERA\worktrees\labyrinth-minotaur-inference-credit-wave76-20261116`.
- Source-only change: bridge runner, replay auditor, and regression test.

## Hostile RED
- Examiner returns `VERIFIED_WIN`, `finished=True`, `escape_verified=True`
  in its INITIAL public view, before any candidate response.
- Existing runner issued `verified_escape=True` with zero model calls,
  and matched-pair results inherited that unearned win.
- Focused regression: `AssertionError: True is not false`,
  `RED_EXIT=1`, showing false candidate escape credit.

## Minimal GREEN and limits
- Runner requires at least one recorded model turn before escape credit.
- Replay auditor enforces the same invariant against digest-resealed false wins.
- Zero-turn `final_verdict=VERIFIED_WIN` remains a factual examiner status,
  but is NOT a model-earned `verified_escape`.
- Focused identical regression `GREEN_EXIT=0` (zero adapter calls, inert
  terminal trace, receipt replay, rehashed forged claim denied, pair both no-win).
- Windows bridge/subprocess/private-read/graduation 97/97 PASS.
- WSL normal bridge/subprocess/graduation 91/91 PASS.
- WSL optimized bridge/graduation 71/71 PASS.
- Total 259 targeted test executions; compileall and git diff --check PASS.

## Explicit hold
> Nonempty turns are conditional trace evidence, not authentication of actual
> neural inference. Self-rehashed receipts, examiner privilege isolation #146,
> independently custodied semantic case mechanics #147 and neural learning
> remain unqualified. No merge, deployment, model training or graduation.
