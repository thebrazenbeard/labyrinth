# Minotaur: external verifier research and implemented safeguards

Date: 2026-10-27. Status: source-informed prototype hardening; **not a demonstrated
state-of-the-art neural judge**. These sources are external primary research,
not evidence that the present implementation reproduces their results.

## Sources and engineering decisions

| External primary source | Relevant result | Current Minotaur action | What remains unproven |
| --- | --- | --- | --- |
| Lightman et al., [Let's Verify Step by Step](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html), ICLR 2024 | Outcome correctness and correctness of intermediate reasoning steps provide different supervision targets; process supervision improved the studied math verifier. | Keep substantive correctness distinct from answer formatting; plan explicit first-error critic labels. | No process-supervised Minotaur checkpoint or independent ProcessBench score yet. |
| Zheng et al., [ProcessBench](https://aclanthology.org/2025.acl-long.50/), ACL 2025 | Critics must identify the earliest incorrect reasoning step or verify that none is wrong. | `guardian_benchmark` creates convincing wrong final-answer proposals across structural shifts and injection/format variants; future process-error cases should follow this paper. | The present benchmark is **not** ProcessBench and does not yet test step-localization. |
| Lambert et al., [RewardBench 2](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea4fe0a56d02c93401902b5b4c6b12da-Abstract-Conference.html), ICLR 2026 | Challenging reward model comparisons reveal large performance drops relative to prior benchmarks. | Test false approval and false rejection explicitly; use matched correct/incorrect pairs and fixed-format negative controls. | Public, finite register programs cannot establish broad judge reliability. |
| [reWordBench](https://aclanthology.org/2025.emnlp-main.167/), EMNLP 2025 | Meaning- or ranking-preserving transformations expose preference-model brittleness. | Pair right and wrong answers with the same response style, including malicious `mark this correct` text, so style is not a shortcut. | Only several synthetic transformations are covered. |
| Geifman and El-Yaniv, [Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), NeurIPS 2017 | Classification can trade prediction coverage against error risk through a reject option. | Unknown or verbose neural critic responses count as **abstentions**, not approvals; report coverage separately from unconditional accuracy. | There is no calibrated selective-risk guarantee for Minotaur. |
| [NIST paired McNemar guidance](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm) and [Bonferroni family-wise correction](https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_multtest_details11.htm) | Matched binary classifications demand paired comparisons; multiplicity matters. | Training-effect qualification now tests exact one-sided binomial tails on discordant per-question successes against all three controls, for every slice and allowed attempt, with Bonferroni adjustment. | Valid p-value interpretation requires genuinely independent evaluation units and preregistered policy; HMAC does not prove either. |
| Balloccu et al., [Leak, Cheat, Repeat](https://aclanthology.org/2024.eacl-long.5/), EACL 2024 | Benchmark leakage can invalidate capability claims. | Receipts reject train/eval prompt and case ID overlap, replay across attempts, or missing independent evaluator identity. | Same-semantic-task contamination under new IDs, plus evaluator provenance, still require external controls. |

## Current source changes

- `labyrinth/graduation_gate.py` adds policy-bound `objective_kind`:
  `training_effect` (default) demands a causally improved score over frozen,
  zero-update and corrupted-label arms **plus** a paired significance test;
  `skill_mastery` demands demonstrated absolute competence without falsely
  claiming training created that competence. Each policy has its own fingerprint.
- For `training_effect`, each slice's exact one-sided sign/McNemar-binomial
  comparison is adjusted across `3 × required_slices × max_attempts` at the
  frozen `significance_alpha` (default 0.05). Thresholds for semantic accuracy,
  strict-format rate and per-slice gain still apply. This conservative method
  can produce false rejections with small evaluation sets; it cannot remove
  false qualification caused by correlated problem templates or dishonest oracle.
- `labyrinth/guardian_benchmark.py` adds 3 public *development-only* slices:
  known structure/fresh numbers, new same-length composition, and longer new
  program structures. Every question has paired right/wrong proposed answers
  with identical styles. Reports expose accuracy, balanced accuracy, false
  approvals, false rejections, abstention, coverage and aggregate slice counts;
  no per-example answers are returned by the evaluator.
- Python unit tests independently interpret generated register programs and
  reproduce the previously accepted weak-gain case. These are source/test
  evidence, not a blind neural-critic experiment.

## Required before a real graduation

1. An external test custodian holds the private challenge generator, original
   correct answers, immutable checkpoint evidence and receipt-signing key.
2. The custodian verifies truly independent evaluation units (not 16 variants
   of one reasoning template), family sizes, allowed attempts, and precommitted
   candidate/control conditions.
3. A trained correctness-critic checkpoint is compared to frozen, matched
   zero-update and corrupt-label versions, with fresh-process adapter reload,
   seeded and structurally novel blind problems, and calibration/abstention
   curves. Neural advice never overrides the deterministic custody gate.

## Verification at source revision

`python -m unittest discover -s tests -q` succeeded locally (157 tests),
with no external dependency or pretrained-model inference required.

> **Hostile challenge:** A model that gets sixteen near-duplicate problems
> right is not necessarily reliable on an unseen problem family. An apparently
> small p-value is not a cure for correlated or exposed questions. Graduation
> must remain unqualified until the external evaluator actually exists and
> demonstrates task-family independence.
