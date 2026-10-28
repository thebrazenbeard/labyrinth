# V7 numeric multiplication study (2026-10-28)

**Research only; NO model promotion or graduation.** Source branch based on Labyrinth V6 Draft PR #44 exact head `c53c901bdd4138cb6ad3bb13bb654e10a02df98e`.

Two authenticated fresh Codex CLI specialists were started through connected **Executor Lappy** in isolated worktrees: numeric neural architecture session `01a182e3-3e8d-7732-b3a4-b65bfdc6ab67` and independent challenge data session `01a182e3-7389-7383-9358-1206ab6fe586`. Both reached a Codex account usage limit before finishing. Each left useful standalone **test scaffolding but no qualified implementation or source commit**; the BT2 integrator corrected the benchmark test's false duplicate-group rejection, then independently implemented the source, reran tests, and reviewed it. No work is falsely credited as a completed agent deliverable.

`labyrinth/minotaur_v7_benchmark.py` defines pure-Python, SHA-bound public paired train/dev/test splits over full modulo-97; all 97 output-class-absent challenges; union of absent input/output label 53; explicit zero, copy, random and train-only majority baselines; and evaluation-only modulus 101. No training code imports or consumes output targets as neural input. Canonical (a,b)/(b,a) pairs remain together; their rows are not independent statistical units.

`labyrinth/minotaur_v7_numeric.py` is a randomly initialized, shared, symmetric PyTorch numeric-output regressor: it uses normalized operands and Fourier features, learned operand towers and a scalar output head. Its *structural symmetry and Fourier periodicity are supplied priors*, not independently discovered arithmetic laws. No numeric product, modulo product, output class embedding, trained pair lookup, modular log table or double-and-add algorithm occurs in forward. The offline teacher generates labels for optimizer targets **only**. All training pairs with input value 53 or target value 53 are removed, plus a separately grouped 10% holdout of remaining pairs. Evaluation includes absent-output 53, absent-input 53, grouped holdout and all modulus-101 pairs (zero mod-101 training). Test metrics and frozen/constant controls remain distinct. The heldout label 53 may occur as the target for input-53 tests, but no positive label 53 occurs in training.

The primary acceptance criterion is demonstrated successful prediction of absent-output label 53 and quantitative transfer to modulus 101 under multiple seeds, with a private external examiner still required for promotion. A source-first negative result is valuable; do not describe passing test code as learned competence. No weights are to be deployed as a result of this study.

**All claims of exact-source training, tests and hosted CI must be filled only after live readback.**


## Actual source-frozen two-seed experiment

The implementation and TDD tests were committed at `c52d56225e43612add05f181a5323522e7a6c627` **before training**. Executor on Lappy then ran two genuine CPU PyTorch random-initialized experiments, 30 epochs and 510 AdamW updates per seed. Both neural parameter SHA digests changed. The model used train-only labels and never saw output residue 53 or input value 53 in train. The external modulus-101 dataset was evaluation only. No checkpoint was promoted.

| Public challenge | Seed 20261028 | Seed 20261029 |
| --- | ---: | ---: |
| Independent paired heldout within permitted training values | 26/910 (2.86%) | 16/910 (1.76%) |
| Label 53 entirely absent from training | **0/94 (0%)** | **0/94 (0%)** |
| Operand value 53 absent from training | 2/193 (1.04%) | 4/193 (2.07%) |
| Modulus 101, entirely untrained | 202/10201 (1.98%) | 192/10201 (1.88%) |

The numeric regression head avoids the per-output-class positive-example trap in principle, but these results prove it **still fails** the intended arithmetic generalization. This is not a successful neural multiplication model. The output-label-53 test excludes two cases that belong to the separately scored withheld-operand-53 group; therefore it contains 94, not 96, directed rows.

Evidence JSON: `docs/evidence/MINOTAUR_V7_NUMERIC_2026-10-28.json` SHA-256 `34e731a1c8699166eb78add7380053c30107c4ee587b7a61ea0ae22b64c6e27b`. The frozen executable commit, exact module and benchmark SHA-256, model/optimizer counts and all per-challenge measurements are in the evidence file. **DO_NOT_PROMOTE**; no external private qualification, no model install, no trainee graduation.

The strongest next hypothesis is explicit intermediate arithmetic representation learning with independently masked transfer tasks and a separate private examiner. Merely enlarging this symmetric regression MLP has no demonstrated path to modular multiplication; the architecture must be evaluated for rule learning, not retuned against this public holdout.

## Cross-platform evidence auditor correction (source-bound)

The original training artifact was written on Lappy with Windows CRLF line endings and was committed by Git with LF-normalized bytes. The original Windows SHA-256 remains recorded as historical producer evidence; the auditor now verifies an additional **pinned Git-canonical LF SHA-256** for the report and both source modules, while requiring the report's original producer hashes to match exactly. This prevents an LF-only Linux checkout from falsely failing authentication without relaxing content-change detection. The red-green test exercises both CRLF and LF forms, and actual Linux CI still must be read back at the exact new commit. This change does **not** alter the neural training report, its failure measurements, or the DO_NOT_PROMOTE decision.
