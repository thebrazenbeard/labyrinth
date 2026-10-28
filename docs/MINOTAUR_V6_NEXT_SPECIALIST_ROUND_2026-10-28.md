# Minotaur V6 next engineering round — source, specialists, tests, negative findings

**Subject:** source branch stacked on Labyrinth Draft PR #37, base `1f249b81e65ad992124585e4594f193c3318cfcb`. All models are **research-only / DO_NOT_PROMOTE**. No production runtime changes, no signed private examiner, no trainee release.

## Fresh real Codex specialists

Six authenticated, distinct **Codex CLI** specialist sessions were launched through connected Executor on Lappy in separate Git worktrees: spectral architecture `01a182b7-7479-7e03-9b50-4693e64b0aad`; learned group `01a182b7-ac7a-7960-88ab-e72d13f8a641`; holdout generation `01a182b7-e65b-7a10-9f8c-8afdaa4533c9`; independent examiner `01a182b8-28c4-7d71-9e90-b1a2eb6ad89e`; red team `01a182b8-5fa1-7a22-93ac-3aedf1972a6b`; CI `01a182b8-b0ea-78b3-8a78-df160318d3a1`. The six initial Windows Codex runner sessions failed with `runner pipe-in` connection timeouts and **none produced qualified patches**.

Two narrowly scoped retries subsequently obtained real process execution: independent read-only reviewer session `01a182bf-295b-73b0-930d-e5807ebe147c` and separate learned-group model session `01a182bf-ae07-72a1-bc7f-6a7f2d9908d2`. The latter implemented, tested and committed `3425b750d79ff2d178ef1f2c03b426f17fbbaf57`, which the integrator verified independently and cherry-picked as `40695a05869d00211f4d3202c2f00212ac85ccd6`. The reviewer read source and actual report hashes; its independent findings below are sourced to the exact branch. These are **Codex CLI sessions on Lappy**, not native ChatGPT `spawn_agent` children.

## Spectral model — corrected, 10-seed source-bound findings

The original spectral model has 37,537 trainable parameters and a shared symmetric embedding/decoder; no numeric product is computed by its inference forward. But a low-rank factorization can still implicitly complete a finite multiplication table. Pair-grouping prevents train/test reversal leakage; the old aggregate counted both directions of a non-diagonal pair as separate rows. This matters for independent sample counts and confidence intervals.

The lead implemented a test-first **canonical-pair audit** in `labyrinth/minotaur_v6_spectral.py`: it rejects missing reverse directions/duplicate rows, recomputes multiplication labels using the separately implemented `minotaur_v6_mul_eval`, stores sorted prediction SHA-256, counts **each unordered pair once**, and explicitly refuses IID confidence claims. The tests were observed failing before implementation, then passed.

The protocol and ten-seed harness were committed first at `f095f7b4b7a491b856666ff398b1dd14d9fe3eb6`, producing the original directed-row report with SHA-256 `4e5b831f791ad12f056c3a42d10a43b62459a7d3bf0e821040701fd3ec4d186e`. After the group auditor and group model were committed at `40695a05869d00211f4d3202c2f00212ac85ccd6`, a **fresh full ten-seed training** was rerun and independently audited. Exact evidence: `docs/evidence/MINOTAUR_V6_TEN_SEED_GROUP_AUDITED_2026-10-28.json`, SHA-256 `367e1426336c4e0895d7d1cbd6816dff9bf92b234f8541480e50a1480c5edbaf`.

Each of ten predeclared seeds trained 160 epochs and 1,280 real CPU AdamW updates. Each train split fitted at 100%; heldout canonical-pair results: mean **78.16%**, median **79.39%**, worst **67.09%**, best **85.28%**. **Zero of ten achieved 95%.** Even with a paired/unordered correction the model fails the 95%-in-9/10 gate. No seed was chosen as a winner or deployed. Group-level prediction SHA and independent label replay are present; source/CI/semantic qualification remain separate.

## Fresh group-prototype candidate — negative experiment

The independent specialist contributed `labyrinth/minotaur_v6_group.py` and five tests. It has **4,656 learned parameters**: residue embedding + output prototypes, with **explicitly supplied latent sum prior** `theta[x]+theta[y]`, not a learned group axiom, arithmetic oracle or pair lookup in forward.

Lead reran all three tests under the exact clean committed source `858bb6ffb963614b88b0949d3ed0a353134fb745`: each tested with independent seeds 20261028 and 20261029 and source-bound parameters/controls.

| Public challenge | Seed 20261028 | Seed 20261029 |
| --- | ---: | ---: |
| Reversal-paired 80/20 heldout | 1.27% (1,885 cases) | 1.81% (1,877) |
| Withheld input residue 53 | 2.07% (193 cases) | 1.04% (193) |
| Absent output label 53 | **0% (96 cases)** | **0% (96)** |

Training accuracy itself was about 3–5%; this candidate fails even fitting the train distribution reliably, and offers **no demonstrated new multiplication ability**. Evidence: `docs/evidence/MINOTAUR_V6_GROUP_ALTERNATIVE_2026-10-28.json`, SHA-256 `8d2605babdbeafcae3d93a981019e653a2cfe7bdc96bb84dd60900e17c05b407`. Do not call this a successful group learner.

## Independent reviewer and next falsifiable gate

The reviewer confirmed the two initial spectral JSON files were source-hash-consistent but were produced by a wrapper, not solely by the tracked `run_experiment`; source/provenance reports must not hide that distinction. It traced spectral symmetry and reversal pairing, highlighted the explicit `x*y` engineered basis in `minotaur_algebraic_rules.py` (an algorithm control, not discovery), and recommended frozen per-case prediction and pair-level scoring, then all-residue absent-output stress tests. Our new audit directly addresses pair-level counting and source-bound prediction hashes; **not** all suggested controls have been implemented. A 10-seed × 97-residue absent-output test, pair-cluster uncertainty, full new-modulus 101 transfer, and genuinely private independent examiner remain OPEN.

**Final state must be claimed only after exact-head local and GitHub CI readbacks.** No ordinary Git green check establishes learned general arithmetic rules or safety of trainee graduation.
