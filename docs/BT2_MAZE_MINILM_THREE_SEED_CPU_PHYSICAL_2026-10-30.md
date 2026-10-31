# BT2 Maze-conditioned pretrained MiniLM — three-seed physical development replication

**Date:** 2026-10-30 America/Indiana/Indianapolis.
**Evidence:** isolated Lappy CPU model execution and physical checkpoint
file readbacks. These are repeated **public-development** experiments, not a
private final exam or independent learning qualification.

## Exact setup

- Each execution used `scripts/qualify_maze_conditioned_minilm_cpu.py`,
  host-matched source SHA-256
  `28902026e959dd0d9b3604a6d25426e1894f7b6a05bf94f3101968c452537944`.
- Model: offline cached pretrained
  `sentence-transformers/all-MiniLM-L6-v2`; reported full frozen-base
  tensor digest
  `001fba634a5244f2d2e4dc914b9c4f72fcdd41fc36a281f9ce4040dd20de4525`
  in each run. This is **self-reported model-origin identity**, not
  independently attested publisher/model-loader custody.
- Each seed used a distinct authentic two-cycle Maze: 12 generated TRAIN
  cases per cycle, 24 distinct room IDs, case seeds, exposure identities
  and prompt texts, with a 6/6 first-cycle room allocation and 9/3
  operator-declared weak-negative-skill second-cycle allocation.
- All arms received the same generated case visits within a seed.
  Correct-label and inverted-label heads/LoRA ran 16 AdamW updates
  per cycle. The zero-update arm performed matched forward/backward passes
  but did **not** call `optimizer.step`. No randomized-label or
  zero-learning-rate optimizer-step control was physically executed.
- Existing other-host processes and unrelated training output were left
  alone. Seed-specific inputs and runs were created in the isolated
  `C:\Temp\bt2-minilm-q3-one-534e644-20261030\resume_lab` directory
  after checking host identity, free memory, active Python trainers,
  target nonexistence and dry-run data validity.

## Physical results on repeated *public* evaluation fixtures

| Scope seed | Correct dev | Zero dev | Inverted dev | Correct public transfer | Zero transfer | Inverted transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261030 | **10/10** | 5/10 | 0/10 | **8/10** | 6/10 | 3/10 |
| 20261031 | **9/10** | 6/10 | 1/10 | **7/10** | 5/10 | 3/10 |
| 20261032 | **9/10** | 5/10 | 1/10 | **7/10** | 5/10 | 3/10 |

Exact physical `report.json` SHA-256 digests (independently recomputed
on Lappy):

- Seed 20261030, `outputs\mazeconditioned16`:
  `b23dc9fac4a97494d272909221246e2ee08eaf30ae63013cc4aa24025a321e21`
- Seed 20261031, `outputs\mazeconditioned_seed31`:
  `c042baf9fbd3303899fb87096706847a5fc1cd00bce970842123b1f7d31c6f82`
- Seed 20261032, `outputs\mazeconditioned_seed32`:
  `3df5b9d75f301123bd35c2bd922d0c97c74a743b4c0c1b9795a8096ddacec05c`

Each reported successful regeneration-stage trainable tensor continuity,
frozen-base stability, and final fresh-instance saved adapter/head reload
with identical individual public predictions. A separate host-side
PowerShell readback recomputed all **12 adapter/head checkpoint file SHA-256**
digests for each run: **36/36 matched the report commitments**. The
zero-update trainable digest remained unchanged; the two updated arms'
trainable tensors changed. The reports forbid graduation and mark protected
final examination `NOT_RUN`.

## Interpretation and withheld qualifications

**Observed:** Correct-label training beat both neural controls on each
fixed public development and transfer fixture for all three seeds after
16 updates/cycle. This is positive evidence of pretrained adapter/head
learning and persistence in the bounded binary-polarity development task.

**Not established:** The same ten public development and ten public transfer
cases were reused across all three seeds, so `30` model answers per split
are **not thirty independent exam cases**. The 16-update setting was chosen
after a weaker eight-update run. The public fixtures have lexical/family
overlap with training, the weakness hint is operator-declared rather than
measured by the Minotaur, and the inverted and zero-update arms are not the
full preregistered four-arm control protocol. No external evaluator,
independently sealed semantic holdout, prior-skill forgetting check,
protected final examination, or host-owned atomic recovery sink was used.

Thus Q3/Q4 have narrow physical support; Q5 has a repeated **public** positive
control comparison but overall independent qualification remains UNKNOWN.
Q6 transfer/retention UNKNOWN, Q7 NOT RUN, Q8 FAIL/HOLD, Q9 merge HOLD.
No merge or graduation is authorized by these results.
