# Minotaur diagnostic and process-claim pretrained experiments (2026-10-27)

**Evidence:** Three CPU training runs on Lappy using the same already-local Open-Reasoner-Zero-0.5B pretrained model and tokenizer. All were **2 optimizer updates** with frozen / zero-update / corrupted-label controls, new opt-in output folders, source-tree binding, unchanged base weights and real LoRA tensor updates. These small publicly generated cases are *development-only*; no neural guardian was qualified or installed in production.

The mechanisms tested are (1) logit-margin and abstention reporting on ordinary correctness claims; (2) balanced, paired first-error-*claim verification* training on trace programs; and (3) direct two-class Yes/No cross-entropy rather than full-vocabulary token CE. Item (2) is **not** a model that generates a first-error location unaided.

## Evidence subjects

| Subject | Exact source commit | Report SHA-256 |
|---|---|---|
| Original verdict loss + margin diagnostic | `92d2dbcd60c327431645a619e93fa1afe68133f6` | `9ea57782b7e506772637d0836d24af07fe667f93ec9042e16324381b2b758b61` |
| Process first-error-claim supervision | `9b1dff8ba8c1451db0121ca4a37d6f8155475f8b` | `da46e08be7d31045024b0281c1fa3074a1604f5789ed0dc7b1ab07401266d1f7` |
| Direct binary-logit cross-entropy | `903087a5da7ce823e120ebc8243076478208bdca` | `a79f1c994d00f148c83421cb068c20922fe5fbfb8943699d040916cceb390b98` |

Base weights: SHA-256 `f3fabf5933c996348e19cc65f2eaf4ceb4a036cb9981d2da1142c2e8b4020938`. Local outputs remain on Lappy in respectively `D:/VERA/experiments/minotaur-critic-margin-v3-20261027`, `minotaur-process-claim-v3-20261027`, and `minotaur-binary-loss-v3-20261027`. No binary model assets are checked into Git.

## Observed results

| Experiment | Frozen dev | Trained dev | Frozen transfer | Trained transfer | Candidate false approvals |
|---|---:|---:|---:|---:|---|
| Ordinary claim / causal-token CE | 5/8 | 4/8 | 2/4 | 2/4 | Dev 4/4; transfer 2/2 |
| Process first-error claim / causal-token CE | 8/16 | 8/16 | 6/8 | 3/8 | All incorrect claims approved |
| Ordinary claim / direct binary-logit CE | 5/8 | 4/8 | 2/4 | 2/4 | Zero: **all correct claims also rejected** |

Candidate ordinary-claim mean Yes-minus-No margin was **+3.234** on development with causal-token CE versus frozen **+0.152**. Direct binary CE changed the same candidate mean to **-3.236**. Both objectives shift the global decision prior dramatically; neither improves discrimination at this budget.## Reload and decision integrity

All trained adapters changed LoRA tensors while zero-update controls preserved the initial digest. Fresh-process adapter reloads matched the saved LoRA tensor digest and discrete prediction hashes. The new detailed margin summaries differed by roughly 1-8e-6 on CPU. The initial exact-dictionary reload therefore **failed**; a regression-tested bound now tolerates at most 1e-4 **only** in five scalar margin summaries. Predictions, confusion counts, per-threshold abstention/risk, data binding, weights and run conditions must still match exactly. Later fresh reloads exited successfully, but raw `same_predictions_and_metrics` remained false due to descriptive float drift.

The predeclared selective thresholds 0.0, 0.5, 1.0, and 2.0 failed to induce abstention on the saturated positive-margin model. Do **not** adjust thresholds on the same reported development cases and call the result calibrated generalization. A low false-approval percentage from a reject-all critic is vacuous without positive-class accuracy and evaluation coverage.

**Qualification:** NOT_ESTABLISHED for all three subjects. No independent final exam, credible causal neural improvement, pretrained semantic-generalization claim, production route replacement, or live graduation effect.

## External research and implemented scope

- [ProcessBench (ACL 2025)](https://aclanthology.org/2025.acl-long.50/) asks for earliest-error detection. This branch trains a *proposed-location judge* on synthetic register traces only, not an official ProcessBench baseline.
- [Rewarding Progress (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/98711dea460bdefe0e651ca23ec98ba2-Abstract-Conference.html) distinguishes genuine process progress from outcome-only rewards. Its progress-value method is **not implemented** here.
- [Generative Verifiers (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html) suggests richer verification training objectives; no generative verifier claim is made.
- [Microsoft Universal Verifier (2026)](https://www.microsoft.com/en-us/research/publication/the-art-of-building-verifiers-for-computer-use-agents/) motivates separating process from outcome failures and using explicit rubrics. The private examination / signer custody remains external and unimplemented.
- [Hard2Verify (ACL 2026)](https://aclanthology.org/2026.acl-long.1031/) underscores the challenge of difficult human-annotated first-error assessment; synthetic register tasks are not a substitute.

## Next falsifiers and integration ceiling

1. Develop within-task **pairwise ranking** or other objectives that cancel global Yes/No prior; compare with equally resourced frozen, shuffled-label and zero-update controls, and an independent positive/negative calibration set.
2. Prove non-vacuous abstention at a predeclared, privately calibrated false-approval-risk ceiling, with enough independent structural examples; don't optimize on public heldout.
3. Build/qualify an independent evaluator with genuinely private oracle, bounded signer custody and independently attested checkpoint execution. Existing HMAC evidence merely authenticates the signer; it does not make that evaluator honest or independent.
4. Preserve separation of guardian advice from the atomic, trusted graduation custody. Do not merge/deploy this draft as an autonomous exit decision-maker.
5. Windows regression suite on combined source at `903087a5da7ce823e120ebc8243076478208bdca`: **190/190 Python unittest PASS**, `git diff --check` clean. This qualifies source tests, **not** model effect.
