# 2026-10-27 pretrained source-priority LoRA experiment

**Disposition: TASK-SPECIFIC POSITIVE LEARNING SIGNAL; NO MINOTAUR GRADUATION.**
This record documents an executed 96-step, original-format pretrained neural
LoRA adaptation that improved numeric source-priority selection over matched
negative controls on predeclared public validation cases.

## Exact experimental identity

- Labyrinth training source: a29aeb1c3729bf2b2da22b88a787df3a2bd2f69d, not any later GitHub main.
- Donor: HuggingFaceTB/SmolLM2-135M-Instruct, Hugging Face revision
  12fd25f77366fa6b3b4b768ec3050bf629380bac; original model.safetensors SHA-256:
  5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c.
- Frozen plan SHA-256: fb72fe9e6312b2bfddda4a416f9a797b76e1afa004f9b323031872e911529491.
- Training seed 74413, tier 3 source-priority retrieval,
  32 synthetic training cases, 16
  development cases, 96 CPU training optimizer steps
  for correct-label and corrupted-label arms, plus equal-visit/equal-step
  learning-rate-zero control.
- Answer-only SFT: constant ANSWER: scaffold is placed in the prompt;
  supervised gradient targets only numeric answer tokens and EOS.
- LoRA rank 4, alpha 8,
  supervised learning rate 0.0002. Source selection uses
  three source reports, arbitrary five-digit values, a stated authority
  ordering and an untrusted lower-priority quoted note.
- Final public evaluation was fixed in advance at seed
  92747, 32 cases,
  with frozen evaluator SHA-256
  02c370343baa36d135d7e0dfbf24040d0c885a13f0345f191dfb042d2ffe199d.
  Public deterministic generators are not private, blind independent exams.

## Neural update and retention

Trainable LoRA parameter value SHA-256 before training:
18f1c4e8c338f563efe434acf401dd842a2af474060cb197090a596b23b1b392.

Value SHA-256 after correct-label training:
56e5e6a5fbde4f39704b37dec4e2771206fc1ffc3c87751aca03483bcfa523e9.

These values differ. The separate adapter was saved, and a new
Python process reported an identical LoRA tensor digest and **all candidate
development trial records reproduced exactly**. The separate process was
observed via different PID; this is not trusted independent process attestation.
Frozen base safetensors and metadata were rechecked; production Vera runtime
was not touched.

## Development set — 16 cases

| Arm | Semantic correct | Numeric suffix parseable |
|---|---:|---:|
| baseline | 0/16 | 0/16 |
| sham | 0/16 | 0/16 |
| corrupted | 1/16 | 16/16 |
| candidate | 13/16 | 16/16 |

## Predeclared reserved public final — 32 cases

| Arm | Semantic correct | Numeric suffix parseable |
|---|---:|---:|
| baseline | 0/32 | 0/32 |
| sham | 0/32 | 0/32 |
| corrupted | 2/32 | 32/32 |
| candidate | 29/32 | 32/32 |

Candidate versus corrupted-label control on the *same* 32 cases:
27 candidate-only correct,
0 corrupted-only correct,
and accuracy difference +0.8438.
Candidate versus frozen and zero-LR sham: 29 candidate-only correct
and zero controls-only correct in either comparison.

**Interpretation:** The candidate learned a reproducible task-specific
source-selection behavior on this finite public grammar, far beyond its
negative controls. This is not evidence of general or autonomous reasoning
competence. The frozen and sham controls did not produce strict numeric
responses, so the corrupted-label control (2/32 with 32 parseable responses)
is the more informative comparator for semantic label dependence.

## Independent checks and precise limitations

- Training report SHA-256: 7334747fa323090fd7822767531cdfc46145a5459ef5ef9485837573601157ad
- Fresh-process reload SHA-256: 9ffee9b8df2ce57d8c4f77a9794c42f4c23d8fdcd38cc526a1da5737b808066c
- Reserved final report SHA-256: aea985f1ed5bb07a94a612df49fbda36b39fd5ca99ae3834f701382c668496c7
- Labyrinth development internal audit: PASS,
  64/64 records rechecked.
- Separate read-only final-trial consistency audit: PASS,
  128/128 records rescored after regenerating
  public answer keys and verifying order, IDs, and output hashes.
- The run source predates PR #26's additional frozen-parameter and nonfinite-
  gradient protections. This run **must not** be said to have passed those
  later safeguards. Source-bound state differs from build/current main.
- The benchmark is a **public finite synthetic template** and may reward
  memorizing template syntax or simple source-value extraction. No private
  independently operated evaluator, signed decision, transfer challenge,
  or externally authenticated model-execution custody was present.
- Neither the internal consistency auditors nor these SHA-256 hashes
  authenticate the actual neural execution independently of the operator.
- A small one-seed study cannot establish robust generalization across
  tasks, architectures or hardware.
- Minotaur graduation remains **DENIED**, candidate **UNINSTALLED**,
  no authorization for Vera production inference or autonomous adaptation.

Local evidence (not committed with weights) resides on Lappy at
C:/Temp/bt2-nextsemantic-20261027. This repository record binds observed
claims and code subjects but does not itself activate any system.


## Exploratory priority counterfactuals ? limitation revealed

After the developmental and final experiment scores were obtained, we
generated eight distinct public triplets under a frozen secondary probe plan
(seed 96581, SHA-256
03d2e262f7f87ca468ce98a35997520bc54c0ed80063c0e9454acfac2b7da75d).
This probe used source commit 0fb412646800a444e3ec6ac585de8557fb7ce9ab
after the training source and was **not part of the preregistered final**.
The saved candidate adapter was reloaded and its LoRA tensor digest verified.

Each triplet contains the original report, a swapped top-two authority order
where the correct answer must change, and reversed report-row order where the
correct answer must not change. Of 24 recorded responses the candidate got
16 individually correct, broken down as original 4/8, authority swaps 6/8,
and row reorders 6/8. It got all three correct for just **3/8 triplets**.

The separate read-only output auditor PASS checked all 24 reported trial
IDs, full response previews, hashes, evaluator answers, and aggregate paired
scores. Probe report SHA-256:
2fa54af74b1ce09274d0946b4119a45f1b3d72462e09c978488ad5b138845b49.

The result identifies a real weakness in **counterfactual consistency**,
even though the original 32-case heldout score was 29/32. It is a
post-training exploratory stress test, not an independent blind sample or
a calibrated transfer benchmark. No graduation or production activation
follows from either result.
