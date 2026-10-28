# Minotaur V7 all-output-residue public transfer stress — 2026-10-28

**Status:** RESEARCH ONLY / DO_NOT_PROMOTE / NOT_ESTABLISHED. Stacked on Labyrinth Draft PR #45 at source base `025db1ca2bc379dbfb9926bc024429995618a6a0`. No model installed, no graduation, no private examiner credential, no production branch merge.

## Exact source and experimental method

`labyrinth/minotaur_v7_all_residues.py` and its test-first suite are a bounded public benchmark of the *existing* 2,641-parameter V7 symmetric numeric-output neural model. No model weights or architecture changed. For every residue r in 0..96, build a fresh train/dev/test over **all** modulo-97 multiplication combinations, but **remove all target==r examples** from train **and** dev. Hold out all target==r cases for test (unseen output r). Reversal group pairs do not cross partitions. The benchmark source pre-existed in the pinned base. Two predetermined random initialization seeds (20261028 and 20261029); no model selection from test.

For each seed and each residue a new model is randomly initialized and trained with real CPU PyTorch AdamW. Inputs are only numeric operands and modulus; targets are passed to offline SmoothL1 loss and never the inference model. After training, round the scalar prediction into 0..96. Compute train-only majority-constant control (accuracy zero by construction on a withheld label), same-seed frozen-weight control, parameter-update indicator and per-label source split SHA digests. Only aggregate arithmetic result correctness was retained; no per-case frozen prediction trace or external/private blind examination.

The TDD tests initially failed with `ModuleNotFoundError`, then passed after implementation. The optional Torch CI job explicitly executes the new tests while the dependency-free core remains runnable.

## Actual complete 97-output experiments, two frozen seeds

| Source-frozen experiment | 6 epochs | 30 epochs |
| --- | ---: | ---: |
| Executable source commit | `b3f6cbf04f045c71428d928692f98f24f8055a98` | `ef9dfc5ce264cad8f5ed2c5a3f182280429cecda` |
| New learned models | 194 | 194 |
| Total actual AdamW updates | 5,820 | 29,100 |
| Correct predictions on withheld-output test | 142 / 18,818 | 243 / 18,818 |
| Macro accuracy, mean over 97 labels × 2 seeds | **0.762%** | **1.305%** |
| Frozen random control, same study family | 1.069% | 1.069% |
| Labels with *any* correct case | 7 / 97 | 43 / 97 |
| Labels perfect under both seeds | **0 / 97** | **0 / 97** |
| Optimizer timeouts | 0 | 0 |

Six-epoch saved report: `docs/evidence/MINOTAUR_V7_ALL_97_OUTPUTS_2026-10-28.json`, SHA256 `ca0e4590bd5fcf61b90b18bebd5d2688dd01ff85a0a8e1bf00df615e1fe744d3`. Thirty-epoch report: `docs/evidence/MINOTAUR_V7_ALL_97_OUTPUTS_30EPOCHS_2026-10-28.json`, SHA256 `fe2dd84ced807f4101c15faec01f193b28e32a6c7cc9ec29fb18cc0b531b77a4`. Both source claims reflect the exact **clean** commit before each experiment. Reports contain config, all 194 run rows, per-label dataset SHA256, frozen/train-only baselines and updates.

The result is **negative**: a numeric-output head avoids the strictly unseen-softmax-class limitation but does not learn reliable output-class-extrapolative modulo multiplication. The measured 1.305% mean should not be turned into a hypothesis win; no IID confidence inference from paired reversals is claimed.

## Scope and next evidence gate

The newer independent V8 line in Draft PR #47 also failed its source-frozen public independent missing-label and modulus-101 examination. No result from that line was trained or promoted here. The next meaningful research goal requires a reusable arithmetic latent algorithm and a genuinely external post-freeze examiner, rather than repeated post-test tuning of a weak scalar MLP. Running 97 output labels on two fixed seeds is broader than the original label-53-only test, but still researcher-visible and does not establish cross-prime extrapolation, longer program reasoning, or private qualification.

Source, CI, installation and effectiveness must remain separately qualified. This branch is proposed as an isolated draft review, **not** an automatically mergeable model upgrade.
