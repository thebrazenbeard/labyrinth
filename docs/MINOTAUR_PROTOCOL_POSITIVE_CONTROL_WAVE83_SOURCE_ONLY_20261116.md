# Minotaur Wave 83 — Source protocol positive control and witness scheduling (2026-11-16)

## Source, qualification and status
- Canonical `thebrazenbeard/labyrinth`; isolated clean-parent source `D:\VERA\worktrees\labyrinth-minotaur-local-trial-wave82-20261116`, exact parent `e8a6e0efecda904004a6ed8c767772100037a8c2` (PR #264, open draft, hosted CPU and Python success). New worktree `D:\VERA\worktrees\labyrinth-minotaur-positive-control-wave83-20261116`, branch `feat/minotaur-positive-control-wave83-20261116`.
- **SOURCE-ONLY PUBLIC DEVELOPMENT.** No additional neural inference, training, model-weight modification, held-out independence, qualified security custody, grading promotion, host install or service effects.
- Model-origin attestation, learning qualification, independent case-mechanism custody and comparison qualification remain FALSE. #146 (same-user examiner private custody) and #147 (semantic-case provenance) remain OPEN. Real Qwen source-only negative trials from PR #264 are unmodified.

## RED → GREEN hostile defects
1. A model trial could fail even where a known solution exists; previous code calculated a source oracle verdict but never executed its witness via the candidate protocol. RED: `KeyError: 'protocol_positive_control'` for a winning-case positive trial and for under-budget/unsat negative cases. GREEN: source-derived fixed scripted witness is run through the actual `run_candidate` and independently audited. With public seed 7 unsealed and 10 actions available, this source-controlled script produces `VERIFIED_WIN`, `verified_escape=True`, `audit_replay=True`, and exactly ten ACTION turns. Neither outcome is attributed to the model.
2. Source oracle witness was computed *before* first model child was called. RED: monkeypatched search observed zero HTTP model calls and assertion `0 != 1`. GREEN: oracle deferred until model child returned.
3. Initial single-case fix left prior-case witness in source-namespace memory during a later model subprocess. RED: in two-case public-identical but private-different trial, search observed only 2 HTTP calls before the next candidate instead of 4 (`2 != 4`). GREEN: the runner performs **all candidate trials first**, then computes any oracle witnesses, executes positive controls, and runs fresh zero-update negative controls. The host still retains private case data, so this is a narrower *witness exposure* improvement, not an OS-principal isolation proof.
4. Under-budget known solutions yield `WITNESS_EXCEEDS_BUDGET`; declared sealed unsatisfiable cases yield `PROVEN_UNSAT_NO_WITNESS`. The positive control cannot grant a neural win and never becomes part of the model's prompt.

## Reconstructible self-test
- New public-source artifact: `research/MINOTAUR_WAVE83_PUBLIC_SOURCE_PROTOCOL_POSITIVE_SEED7.json` (966 bytes; SHA-256 `ebad461546bcc85e0f42302ab8500819ce9cc5579a4b20f42dbf7026c737147e`). Reproduced by invoking existing `search(make_case(7, sealed=False))` and executing its 10-action witness with `FixedActionAdapter` via `run_candidate`, checking `audit_receipt`; separately source-verified sealed seed7 `PROVEN_UNSAT_UNDER_DECLARED_RULES`. The file is immutable-on-write `O_EXCL` local evidence and intentionally lists the publicly known development witness—not a protected private test case.
- Its status fields `model_inference=false`, `candidate_is_neural_model=false`, `learning_qualified=false`, `independent_custody_qualified=false` are material. No Qwen response was generated in Wave83.
- Revised runner Python file SHA-256 (before final commit): `33EE83E11F606B5458F608AF2D41A862D0EB8E770B1A694E204E5D790B4FC02D`.

## Measured source validation
- Starting parent 54/54 targeted tests PASS.
- New regressions RED as above; exact focused GREEN 3/3 and later revised source 9/9.
- Windows Python 3.14 focused suite: **132/132 PASS**; WSL Python normal **126/126 PASS**; WSL optimized `python3 -O` **88/88 PASS**, total **346 targeted test executions**.
- Python `compileall` exit 0; `git diff --check` exit 0; source-focused and hostile tests cover no-output/failed model, local-only loopback HTTP, exact public case binding, multi-case witness ordering, no model credit, script fresh-state, and no evidence overwrite.
- No full-repository test suite or independent hostile model review is claimed, and these tests cannot prove independent principal/process-file custody, semantic learning, or model-origin authenticity.

## Host and next qualification
- Fresh pre-run Lappy readback: Ollama had **no models loaded**; RTX 3050 reported **3761 MiB free / 4096 MiB total**, GPU utilization 0%. Wave83 did not load/unload models or touch another worker's process.
- Genuine native specialist subagent dispatch was unavailable; all source and adversarial tests executed in this single worker, not independent review.
- Review as stacked DRAFT/HOLD on PR #264. Do not merge/promote graduation without separate OS-principal hostile read probe (#146), independent semantic case-mechanism authority and neural origin attestation (#147).