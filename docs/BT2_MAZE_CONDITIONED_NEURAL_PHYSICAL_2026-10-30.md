# BT2 Maze-conditioned MiniLM — physical development experiment (2026-10-30)

This records an opt-in **physical CPU pretrained-model training execution**
and a negative matched-control outcome. It is not evidence of private final
examination, demonstrated general learning transfer, closed-loop Minotaur
adaptation, safe crash recovery, or graduation authorization.

## Provenance and boundary

- Executor target: `Lappy` / Windows; authority root `D:\VERA` (not modified).
  Existing portal, Blotter, and REPL processes were not stopped or changed.
- The isolated work directory was
  `C:\Temp\bt2-minilm-q3-one-534e644-20261030\resume_lab`.
  New experiment output under `outputs\mazeconditioned01` did not exist
  before launch. Preflight found zero matching competing trainers and
  approximately 8.57 million KiB free physical memory.
- Source `scripts/qualify_maze_conditioned_minilm_cpu.py` SHA-256
  `28902026e959dd0d9b3604a6d25426e1894f7b6a05bf94f3101968c452537944`.
  Dependency source `labyrinth/regenerative_maze.py` SHA-256
  `91318fb7a09cf76a4aa98126434bbc80b74bfa739f409225e02207a3ff596bb8`;
  `labyrinth/training_scope.py` SHA-256
  `25fe9635b125848a9b0a8f2ac0f4613c282778c3c7afdf97dca01d678ad9b4d2`.
  All three host file hashes were read back and matched local source.
- Public model: cached pretrained
  `sentence-transformers/all-MiniLM-L6-v2`. Pretrained base-weight
  SHA-256 over named raw model tensors was
  `001fba634a5244f2d2e4dc914b9c4f72fcdd41fc36a281f9ce4040dd20de4525`.
  This is self-reported byte identity, **not publisher or external
  model-loader attestation**.
- Command: `py -m scripts.qualify_maze_conditioned_minilm_cpu
  --output-parent <isolated outputs> --run-name mazeconditioned01
  --weak-cycle-one negative_polarity --steps-per-cycle 8 --execute`.
  The exact command's dry-run first passed without training or output creation.
  The weakness hint was **operator-declared, not a measured Minotaur response**.

## Generated curriculum and physical learning

The real `TrainingScopeSpec` and `generate_maze_plan` yielded two training
cycles of 12 training cases, all uniquely bound to room IDs, case seeds,
model-visible prompt texts, and exposure IDs. The first maze used six
positive and six negative polarity rooms; the regenerated maze used three
positive and nine negative rooms.

Three arms processed identical 24 generated training cases in the same order:

| Arm | Dev after cycle 1 | Public transfer after cycle 1 | Optimizer steps per cycle |
| --- | ---: | ---: | ---: |
| Correct labels | **5/10** | **6/10** | 8 |
| Zero update | **5/10** | **6/10** | 0 |
| Inverted labels | **5/10** | **5/10** | 8 |

Correct-label and inverted-label trainable LoRA/head digest values changed
in each cycle. Zero-update digest values remained unchanged. The first
checkpoint's trainable digest matched the second cycle's reloaded initial
digest; frozen PEFT weights did not change. Each arm's final checkpoint was
loaded into a **fresh pretrained-model instance**, followed by exact saved
weight/hash and per-case prediction equivalence checks.

The Lappy `report.json` SHA-256 was independently recomputed as
`95edcd620d03aebab5e63dfd5e447dbf0b42fd65715e97ec2ea3366dc29fb421`.
The experiment script's recorded SHA matched its Lappy source. The file-hash
readback compared the six LoRA adapter and six classifier-head checkpoint
files against their report commitments: **12/12 matched**. The report states
`private_final_exam=NOT_RUN`, `crash_recovery=NOT_TESTED`,
`graduation_authorized=false`.

## Falsification and qualification

The direct experiment established that real pretrained neural adapter/head
parameters were updated, loaded again after Maze regeneration, and persisted
to disk with a successful final fresh-instance readback.

**It did not establish a learning advantage**: the correct-label model
matched the zero-update control on both the exposed final development and
public transfer sets. Small public sets, lexical overlap, operator-provided
weakness, and absent independent model/exam custody further limit inference.
An unrelated earlier taskmaster experiment also lost its apparent learning
advantage on adversarial quantitative cases (see
`BT2_NEURAL_TASKMASTER_PHYSICAL_2026-10-30.md`).

No private final exam was released. No graduation or merger is authorized by
this evidence. Narrow physical weight-change/persistence observations can
support Q3/Q4; Q5 failed to demonstrate a positive advantage in this run,
Q6 remains UNKNOWN for independent retention/transfer, Q7 NOT RUN,
Q8 full FAIL/HOLD, and Q9 HOLD.

## Follow-up fixed-seed sixteen-step ablation

To test whether the first run was underpowered, a fresh, isolated experiment
used the **same source checksum, model ID, generated Maze, learner seed,
training cases, operator-declared weakness, and three control arms**, with
only `--steps-per-cycle 16` instead of 8 and the new destination
`outputs\mazeconditioned16`. Lappy preflight again reported zero
conflicting trainers, a nonexistent target directory, and approximately
8.64 million KiB free RAM. The completed report independently hashed to
`b23dc9fac4a97494d272909221246e2ee08eaf30ae63013cc4aa24025a321e21`.
The same frozen pretrained base SHA-256
`001fba634a5244f2d2e4dc914b9c4f72fcdd41fc36a281f9ce4040dd20de4525`
was recorded, along with a fresh-instance final readback in each arm.

| Arm | Final public development | Final public transfer | Optimizer updates/cycle |
| --- | ---: | ---: | ---: |
| Correct labels | **10/10** | **8/10** | 16 |
| Zero update | **5/10** | **6/10** | 0 |
| Inverted labels | **0/10** | **3/10** | 16 |

All **12/12 saved adapter/head checkpoint file hashes** independently matched
the report again. Correct/inverted arms changed trainable values in both
cycles; zero-update remained unchanged; cycle-to-cycle trainable continuity
and frozen weight identity matched for all three arms.

This establishes a positive **matched-exposure public development advantage**
after 16 steps. It does not retroactively change the eight-step negative
result. The experiment was selected after observing that result, its public
evaluation prompts are exposed and lexically related to training, and it uses
one seed with a non-random inverted-label control. An independent preregistered
multiseed examination remains necessary before any generalization, retention,
or graduation claim. Q5 can be described as a **physical public-control
comparison PASS for this exact 16-step case only**, not a fully independently
qualified gate. Q6/Q7/Q8/Q9 remain unqualified.
