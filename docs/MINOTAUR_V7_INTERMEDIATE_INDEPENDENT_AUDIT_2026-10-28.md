# Minotaur V7 — independently replayed intermediate neural arithmetic probe

**Qualification: NOT_ESTABLISHED / DO_NOT_PROMOTE.** Source is Labyrinth PR #45 head `c05a56f5f2431a5af1241b445908b5ea3433074f`, with V7 research extensions isolated on a new feature branch. No production model, trainee graduation authority, or main merge.

## Fresh specialists and source provenance

Two actual, separate authenticated Lappy Codex CLI sessions were dispatched through connected Executor using independent Git worktrees: intermediate architecture specialist `01a18356-6f59-7ac1-bcec-00fa3a3d7f92`, and hostile public evaluation specialist `01a18356-bacf-7df0-8ba5-fbdb8d5118e5`. The first committed source `80b0352134f6b1232d3641c1524f1480fcd1e45f`, and the second `7a7dd998ee61c2895393be996af45c6d46f570a9`. The lead independently ran their tests, cherry-picked as `9982a472d4d99ab48f9ce54b175a951eb4228bb7` / `e5c8f60a03937b6bd1d2f2dbec2f3217ee8d77a2`, and independently integrated the replay capture interface with red/green tests. No specialist was represented as a native ChatGPT spawned agent.

## Distinct hypothesis and controls

The new **3,587-parameter** symmetric PyTorch network predicts three learned quantities: modular remainder, quotient, and normalized integer product. It learns from three **offline label channels** including product-scale supervision; *that is extra supervision*, not a self-discovered algorithm. Its inference forward does **not** calculate `x*y`, invoke a modulo oracle, use residue-output class embeddings, a pairwise multiplication lookup, or execute an algorithmic long-multiplication procedure. Engineered normalized inputs, fixed Fourier coordinate features, and exchange symmetry are explicit priors.

The training split excludes every example containing residue 53 as either input or output, and separately excludes 10% of remaining unordered operand groups. No modulo-101 examples enter training. Both fixed-seed runs use 30 genuine CPU epochs, **510 AdamW steps**, and changed trainable weight digests. The model never uses test labels to optimize.

## Exact-source measurements, independently rechecked

Measured clean-source model + independent replay runner commit: `fee62ef8080777c18e7d0cd5ff6a960d419b27a7`. The pure-Python `minotaur_v7_blindish_eval` module independently regenerated its own labels and challenge memberships, without importing the model or benchmark; the replay checks exact train membership, every score, abstentions, and a complete directed prediction record.

| Public synthetic challenge | Seed 20261028 | Seed 20261029 |
| --- | ---: | ---: |
| Reversal-grouped unseen operand combinations | 12/910 (1.32%) | 26/910 (2.86%) |
| Missing output label 53, **no positive train examples** | 2/94 (2.13%) | 4/94 (4.26%) |
| Missing input operand 53 | 2/193 (1.04%) | 10/193 (5.18%) |
| Zero-training transfer to modulus 101 | 140/10,201 (1.37%) | 146/10,201 (1.43%) |

**None demonstrates transferable multiplication.** The all-zero, train-derived majority, copy, and abstention baselines remain public controls; the review is not under independent private examination.

## Reproducible evidence and threat model

- Model-only frozen commit `e5c8f60a03937b6bd1d2f2dbec2f3217ee8d77a2`: `docs/evidence/MINOTAUR_V7_INTERMEDIATE_2026-10-28.json`, SHA-256 `205004e1962d0ee20c12fcadb5d1b3d800777e19dd2c9465bdef136a10c86d42`.
- Source-frozen full independent replay `fee62ef8080777c18e7d0cd5ff6a960d419b27a7`: `docs/evidence/MINOTAUR_V7_INTERMEDIATE_INDEPENDENT_REPLAY_2026-10-28.json`, SHA-256 `6ea7ffcf5e5963878e815ff08c6436a0632a15c2d2f831a2dc1861c157c89291`.
- Complete input-plus-model-output snapshot, **no gold target labels**: `docs/evidence/MINOTAUR_V7_INTERMEDIATE_PREDICTIONS_2026-10-28.json.gz`, SHA-256 `8b10d7f77988f994999268405e3eb7a4a66d6887b60418e78f77ba0b337463d7` (66,133 bytes); uncompressed canonical prediction JSON SHA-256 `e98b7c0fcd80e79fefd615f7478d097fc8af67be6a676fa7bc98de36ca179649`.
- All first-error / graduation assertions are **NOT_ESTABLISHED**. A public seeded deterministic audit, even a fully replayed one, is not an independent blind private examiner. These paired reversals are not IID independent tests.
- The V7 numeric and intermediate models both fail missing-output class 53 and zero-training modulus-101 transfer. Merely increasing the number of training epochs or tuning against public heldout sets does not demonstrate acquisition of a transferable operation.

## Research recommendation

Explore [Neural Arithmetic Units](https://openreview.net/pdf?id=H1gNOeHKPS) and [Neural Power Units](https://papers.nips.cc/paper_files/paper/2020/hash/48e59000d7dfcf6c1d96ce4a603ed738-Abstract.html) as carefully *disclosed architectural priors*, rather than presenting their built-in arithmetic components as independent rule discovery. Design a truly held-out family of moduli, and predeclare a masked-remainder test before any new training. Preserve the source, raw predictions, split digests and failed seeds. **No promotion or merge is warranted by this V7 model.**
