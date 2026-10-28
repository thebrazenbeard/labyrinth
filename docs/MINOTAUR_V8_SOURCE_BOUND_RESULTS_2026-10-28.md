# Minotaur V8 — fresh specialists, independent public examination, cross-prime evidence

**Date:** 2026-10-28. **Status: RESEARCH ONLY; DO_NOT_PROMOTE.** This is a source-first follow-up stacked on Labyrinth Draft PR #45, exact initial head `c05a56f5f2431a5af1241b445908b5ea3433074f`. No model installed, no release, no production Minotaur graduation.

## Fresh authenticated specialist work

Four separate real Lappy Codex CLI sessions, not native ChatGPT child agents, operated in isolated Git worktrees through connected **Executor**: recurrent architecture `01a1833a-b429-72b1-9915-d2a6044b6f54`; public examiner `01a1833a-e948-7db1-8ac7-0ffbd1b8245c`; multi-modulus transfer model `01a1833d-d86e-7431-86b5-d6e2fd53e702`; initially read-only hostile reviewer `01a1833c-7ead-7473-bdba-242d70a66a5e`. The original reviewer was blocked by Windows runner-pipe startup, so it was rerun in a strictly instructed source-read-only fallback session `01a1833f-c25e-78e0-993f-8eea4259db71`. These agents did not share source worktrees or grant graduation authority.

The three coding specialists committed real, separate source and test changes: recurrent digit-state source `6639be3` plus split-correction `803651c`; fail-closed public scorer `200a459`; multi-modulus network `8c44cde`. The lead independently tested each, cherry-picked to an isolated integration branch, and wrote/committed the cross-component examiner runner before running its formal public examination.

## Recurrent binary-digit architecture

`labyrinth/minotaur_v8_structure.py` uses a learned GRU over seven binary digits of each operand and modulus. Explicit priors: a digit stream, operand symmetry and scalar residual prediction. The model forward contains **no product, modulo-product lookup, hardcoded multiplication procedure, or output class table**; multiplication labels occur in offline supervision.

Actual clean committed source `803651c56da1b6841e1269218ff61801b45a12c7`, two random initializations, 476 optimizer steps per seed: output 53 absent from all training: **8/94 and 10/94**; unseen input operand 53: **2/193 both**; entirely untrained modulus101: **104/10,201 and 89/10,201**. Shuffled-label training did not produce the missing-output gains. The architecture therefore has *some public split-dependent signal*, but does not qualify as transferable arithmetic.

Evidence: `docs/evidence/MINOTAUR_V8_DIGIT_STATE_2026-10-28.json`, SHA-256 `00223debafe96a0b8bffb15a53c700bafe5bf57c3b72affd7ea9985740004779`.

## Independent public examiner — stronger negative result

`labyrinth/minotaur_v8_examiner.py` is pure Python, recomputes labels independently, refuses training/test overlap and reversals, checks custody and prediction hashes, scores **one unordered pair once**, and records constant/copy/train-majority/frozen controls. Its `modulus101` challenge is limited to **398 canonical pairs containing residues 97–100** to avoid duplicate pair keys across moduli; it is *not* an all-10,201-pair assessment. This public scorer cannot sign a private qualification.

The lead's `labyrinth/minotaur_v8_qualification.py` trained the same GRU on the examiner's independently chosen training members, submitted predictions **without labels**, and scored them via `score_submission`. Actual clean committed source `ea1fe4b7248df46ea8e3da2156ba2016c1e851e2`; 476 real AdamW updates per seed. On the independently defined output-53 challenge, accuracy fell to **0/48 for both seeds**, compared with the earlier 8/94 and 10/94 on a different training split. Independent ordinary holdout: **6/461 and 9/461**; input-53 holdout **2/97 both**; modulus101-only-new-residue subset **8/398 and 5/398**. All source/test families are PUBLIC. This contradicts any claim of robust cross-split missing-label generalization.

Evidence: `docs/evidence/MINOTAUR_V8_PUBLIC_EXAM_2026-10-28.json`, SHA-256 `fe926e3f40f111a62e51431e7b3fedef1d7a5470df119c308a7761c387b7fdc2`.

## Multi-prime transfer candidate

`labyrinth/minotaur_v8_transfer.py` is a shared modulus-conditioned neural predictor, trained on moduli **83, 89 and 97**; prime **101 is entirely evaluation-only**. This is *not* the original V7 single-prime training protocol. The model has no learned (x,y)-answer table and does not compute multiplication in its forward pass; trained product targets are offline only. Mod97 training excludes operand and output residue 53.

The independently rerun source-committed study at `876d198c922b10d572148ec6ba9a7cb53c5c3a82`, 774 optimizer updates per seed, yielded: heldout mod101 **313/10,201 (3.07%)** and **225/10,201 (2.21%)**; absent operand53 mod97 **4/193 both seeds**; absent output53 mod97 **2/94 and 4/94**. It does not demonstrate cross-modulus extrapolation.

Evidence: `docs/evidence/MINOTAUR_V8_CROSS_PRIME_2026-10-28.json`, SHA-256 `9165d7e9ab7caa98721a5542de90f869d0ec5e94507b3b293b31d2b01dc98a04`.

## Independent source review and scientific boundaries

The hostile reviewer audited V6/V7 source, grouped-score limitations, V7 Fourier/symmetry priors, hardcoded modulo reduction in V6, and public-family leakage; no successful hidden examiner was claimed. It noted that V7's recorded benchmark working-file SHA differed from current Windows checkout SHA. The lead verified this is **CRLF vs LF only**: the committed Git source blob at `c52d562` hashes exactly to the recorded benchmark SHA `681c84661d51693681d3aa73fe3bb4d55af8f95a16bfdbb3de418761727f3f1b`, and normalizing the current Windows worktree's line endings reproduces identical bytes. Thus this is not a source-content mismatch.

All V8 outcomes above are from **public, researcher-visible, finite datasets**. Even green CI cannot establish independent rule discovery, arbitrary prime extrapolation, or trustworthy autonomous graduation. Missing tests include sealed independent unseen-prime examinations, algorithm-family transfer, longer compositions, and stronger controls against public-test adaptation. The appropriate next objective is learning a reusable latent algorithm with private post-freeze examination rather than scaling these weak regressors.

Final test, Linux CI and exact-head Git claims must be supported by live readback, not inferred from this report.
