# Minotaur Wave 82 — Bounded local model trial, controls, and falsification (2026-11-16)

## Custody and scope
- Canonical source: `thebrazenbeard/labyrinth`; stacked on draft PR #263 at `6c1920f7684a4f8dd2a66debdedd9b7b638f3adc`.
- Isolated Lappy worktree: `D:\VERA\worktrees\labyrinth-minotaur-local-trial-wave82-20261116`, branch `feat/minotaur-local-trial-wave82-20261116`.
- **Exact source for all three real model runs:** `01982f4bcd6215627ec24a9a6b2fcdc8525c9229` (before the later host-oracle reporting patch).
- New `labyrinth/minotaur_local_trial.py` supports explicit bounded local inference via `python -m labyrinth.minotaur_local_trial --model qwen3.5:0.8b --seeds 7 --sealed false --max-calls 16 --timeout-seconds 30 --authorize-local-inference --output <new-path>`. It uses existing guarded subprocess child, loopback-only HTTP, real host-run CandidateSession, and **fresh** no-update fixed-action control per case.
- Reports mark model identity, comparison, learning and examiner custody as **UNATTESTED / NOT QUALIFIED**; response/self-hash do not prove who executed the model.
- Reports save exclusively (O_EXCL) and reject pre-existing destination/symlink; maximum 16 candidate calls per experiment; no deletion or overwrite.

## Hostile RED → GREEN source findings
1. Original trial runner mistakenly reused a stateful fixed script across cases. RED: second control emitted `CLAIM` immediately instead of two `ACTION` steps. Fix: construct a new control per candidate case; identical two-variant test GREEN.
2. The first two-variant trial used eight calls per case. Source oracle `search(make_case(7, sealed=False))` says `VERIFIED_WIN`, shortest witness **10 actions**, 55 expansions; therefore eight calls could never prove a win. The `sealed=True` case returns `PROVEN_UNSAT_UNDER_DECLARED_RULES`, 40 expansions. Both have **identical initial public views** and SHA256 `6865197622753af95ec29ae4737cd95c80d0ecf0823200cf84f84d88a5c28c94`.
3. New host-side `source_development_oracle` annotation reports verdict, known witness length or null, and whether the call budget can fit it; **never provided to the candidate or used to credit escape**. RED: two `KeyError: source_development_oracle` test failures; GREEN: five focused tests. This annotation was added **after** the three real runs and is intentionally absent from their untouched evidence files.

## Actual local synthetic-development inference (no held-out examination)
- Model runtime label: `qwen3.5:0.8b`, Ollama-reported ID prefix `de63045f2975`; actual weights/source remain unverified. Seed **7**, child `think=False`, explicit local adapter authority. $0 additional compute charges.
- `research/MINOTAUR_LOCAL_DEV_TRIAL_WAVE82_SEED7.json` (original eight-call calibration): file SHA256 `f9fc4b4acbac4c7c95192579a6c23133eb6b765438d846ecd731a98e1415a953`. Both variants: model responses `ACTION forward`, then `CLAIM ABSTAIN UNKNOWN`; 2 turns, verdict UNKNOWN, escape false, audit true. Control: eight forward actions, no escape. **Cannot support a no-win skill estimate because budget < known 10-action solution.**
- `research/MINOTAUR_LOCAL_DEV_TRIAL_WAVE82_SEED7_FALSE_16CALLS.json`: SHA256 `235781dfec6f35b82d3cb3d946d1168e3c5e497336cbff64ffb026fc751a876e`. Unsealed, 16-call allowance, model same 2-turn forward/abstain, UNKNOWN, no escape, replay true. Control used all 16 forwards, VERIFIED_FAILURE, no escape, replay true.
- `research/MINOTAUR_LOCAL_DEV_TRIAL_WAVE82_SEED7_TRUE_16CALLS.json`: SHA256 `7a6ef5018092c235af0a313bc287482b67e0d6f0f4d7423783ae6a590854c23a`. Sealed, 16-call allowance, same 2-turn model response, UNKNOWN/no escape/replay true; 16-turn forward control VERIFIED_FAILURE/no escape/replay true.
- Model trace falsifies claims that this model solved this source-development task. It **does not** prove general impossibility, neural learning, a correctly reasoned unsat diagnosis, or trustworthiness of model-origin labels.

## Resource and negative integrity findings
- Initial eight-call trial started while Qwen 0.8B was already loaded. My enclosing shell incorrectly issued `ollama stop qwen3.5:0.8b` afterward despite no ownership proof; it may have disturbed another worker. This is a documented operational mistake, not a successful restoration of pretrial runtime state.
- Before corrected 16-call trials, `qwen3:4b-instruct` and `qwen3.5:0.8b` were both loaded, GPU free **148 MiB**, utilization **79%**. Run latency is confounded by competing work; neither model was explicitly unloaded or restarted during those corrected trials. Afterward `ollama ps` showed none loaded, without an explicit stop command.
- The corrected two-case command continued despite reporting heavy GPU contention before its first case. That was a scheduling mistake; no further GPU experiments ran after it finished. No independent hidden case, process principal deny, model provenance attestation, neural training, graduation, shared runtime installation, protected merge, or subagent dispatch. No qualified head-to-head performance claim.
- #146 examiner-private custody and #147 independent semantic mechanism/case provenance remain OPEN. A self-hash and a same-user Job Object cannot resolve them.

## Verification
- First runner commit `01982f4bcd6215627ec24a9a6b2fcdc8525c9229`, pushed and remote SHA matched. Fresh-source Windows **127**, WSL **121**, WSL optimized **83** tests passed (331) before host-oracle patch.
- After calibration annotation: Windows **128/128**, WSL **122/122**, optimized WSL **84/84** passed = **334 targeted test executions**, Python compileall exit 0, `git diff --check` exit 0. Tests validate public-only prompts, child HTTP calls, remote rejection, failed-call fail-closed, no evidence overwrite, independent fresh controls, underbudget annotation, and no model-win credit from host oracle.
- These are targeted TEST/SOURCE and local synthetic inference receipts, not independent security or semantic qualification.