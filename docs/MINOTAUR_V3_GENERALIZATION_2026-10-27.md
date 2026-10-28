# Minotaur V3 — generalization experiment and hostile review (2026-10-27)

**Status: NOT_ESTABLISHED / DO_NOT_PROMOTE.** This is actual CPU neural training on PUBLIC synthetic questions, not a privately custodied qualification, model release, or deployment.

## Real specialist execution

Five independent authenticated Codex CLI sessions, each in its own Git worktree and based on an exact reviewed source commit (these are not native ChatGPT `spawn_agent` tools):

- Combinatorial program curriculum: `01a17fed-ea7b-7e30-b1a1-3af78553c9f9`.
- Two-seed, four-arm scratch neural ablation: `01a17fee-2049-7491-b417-e70caaf8dbd3`.
- Independent printed-program evaluator: `01a17fee-528f-7760-80be-42e001f5c0d8`.
- Read-only hostile scientific/code reviewer: `01a17fee-87a0-7c82-a1fc-ec04fe47cb7d`.
- Source-first combinatorial neural experiment: `01a17ff4-f86c-7823-8283-b7b0c74369e6`.

The fifth agent encountered linked-worktree Git-metadata permission and Torch-execution restrictions, and explicitly reported no model results. The lead independently repaired two test-proven source defects, verified the agent-authored trainer, committed it, and executed the actual neural runs through Lappy's accessible CPU Torch interpreter. No agent result is credited for execution it could not perform.

## Engineering improvements

`labyrinth.minotaur_combinatorial_curriculum` samples varying 1–6-step register programs from many canonical operation sequences, including both registers, literal/register operands, varied ADD/SUB/MUL compositions, and multiple observed-state corruption families. Both training and holdout include paired correct/incorrect traces, but canonical operation-sequence membership is disjoint across train/dev/transfer. Synthetic training labels are outside model-visible tokens. A newly discovered no-change mutation bug at larger counts was repaired by selecting only real state transitions and retrying entirely idle cases. The repair has a regression case at actual training scale.

`labyrinth.minotaur_independent_eval` uses a separately implemented public replayer, refusing invalid claimed labels and malformed data before scoring; it distinguishes exact earliest-error accuracy, false approval/rejection, abstention/coverage, and trivial baselines. It recognizes only explicitly enumerated known public preambles. This prevents self-reported labels from silently setting the denominator, but is not an independently owned sealed private exam.

`labyrinth.minotaur_v3_ablation` tests four actual scratch learning regimes × two random seeds against the old synthetic families. None passes the reserved transfer gate. An independent hostile audit identified source-provenance defects in the earlier V2 report, misleading use of binary error-detection for first-error localization, predictable fault placement, four-template structural pseudoreplication, and length-transfer embeddings that never received training gradients. See `docs/MINOTAUR_GENERALIZATION_AUDIT_2026-10-27.md`.

## New source-first combinatorial neural experiment

**Executable experiment source commit:** `c5ae51b91bbff301428d632a50281bb18f505812` (module, dataset fix, evaluator, and tests already committed before execution). Reproducibility output: `docs/evidence/MINOTAUR_COMBINATORIAL_TRAINING_2026-10-27.json`, SHA-256 `b31756b7269c983bc12ad029023afed6e6a38f522bc4c6c4db17809889fe0e64`. The report explicitly records `source_tree_clean_at_run: true`. An independent repeat of the complete two-seed execution matched split commitments, negative controls, predictions and trained parameter digests; runtime timings varied and were excluded from the deterministic comparison.

- **Train:** 128 cases, 64 unique canonical sequences, 64 correct / 64 incorrect.
- **Dev:** 64 cases, 32 distinct sequences, 32 correct / 32 incorrect.
- **Structural transfer:** 64 cases, 32 sequences disjoint from train and dev, 32 correct / 32 incorrect.
- Randomly initialized Transformer V2, 32 width, one layer, two heads for correctness/error position; 24 epochs and **192 real AdamW steps per random seed**. Invalid error-position logits are masked to negative infinity. No weights saved or promoted. Execution uses actual CPU PyTorch, not an adapter to a pretrained model.
- Control arms: frozen random, always approve, and always reject step 1, scored by the same independent public evaluator. The trivial always-approve control achieves 50% exact first-error accuracy on balanced data, illustrating why 35–38% does not constitute a gain.

| Seed | Dev exact first error | Transfer exact first error | Transfer false approvals (of 32 wrong) | Transfer false rejections (of 32 correct) |
| --- | ---: | ---: | ---: | ---: |
| 20261027 | 22/64 (34.4%) | 23/64 (35.9%) | 19/32 (59.4%) | 14/32 (43.8%) |
| 20261028 | 17/64 (26.6%) | 24/64 (37.5%) | 18/32 (56.3%) | 13/32 (40.6%) |

The neural model underperforms trivial approval in exact score and still approves a majority of incorrect transfer traces. More exposure to varied structured tasks did not establish learned arithmetic state verification.

## Qualification boundaries and next falsifiable hypothesis

Software tests, saved-checkpoint compatibility, and mathematical competence are **different subjects**. No independent private evaluator, signer custody, authenticated external examination, or graduation effect was established. The previous V1 symbolic checker stays domain-only and authoritative for its register grammar; the scratch V2/V3 neural experiments remain diagnostic-only.

**Next falsifiable experiment:** replace the global categorical-embedding classifier with a causal, per-instruction state-transition learner. Train it to predict the next register state (with a loss on intermediate states, not just the final correct/error class), then separately compare whether detecting the *first* mismatch generalizes to structurally held-out operation compositions and renamed operators. Pre-register family-level performance and selective false-approval bounds, compare to frozen/always-approve/reject and exact symbolic controls, and permit no graduation or model promotion absent independently custodied blind test evidence.

This source package documents failed learning attempts and instrumented improvements honestly. No measured general-purpose reasoning improvement is claimed.
