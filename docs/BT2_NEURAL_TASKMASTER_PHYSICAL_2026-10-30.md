# BT2 Neural Minotaur — physical pretrained-feature development run (2026-10-30)

This is a **bounded physical taskmaster-learning observation**, not an
independently examined or general neural Minotaur qualification.

## Exact execution and provenance

- Host: `LAPPY` (Windows; Python 3.14.5, CPU-only PyTorch available). The
  pretrained cached offline `sentence-transformers/all-MiniLM-L6-v2` encoder
  supplies frozen text features; a separate 4-class PyTorch linear policy head
  is trainable. The script does not use Lappy's NVIDIA GPU.
- Source: `scripts/qualify_minotaur_neural_taskmaster_cpu.py` SHA-256
  `46d9ed8d6a43a586a25e3a3e34607871822a441ff88df9098b577e4e23345fe8`.
  Host-file SHA-256 matched exactly before the run. Dry-run preflight had no
  write effect. The host reported ~7.83 GiB free RAM and no matching active
  Python experiments before launch.
- Isolated destination:
  `C:\Temp\bt2-minilm-q3-one-534e644-20261030\resume_lab\outputs\taskmaster01`.
  The directory did not exist before the opt-in `--execute` launch, and no
  unrelated training directory or host service was altered.
- Report `report.json` SHA-256 independently retrieved from Lappy:
  `9ec20b074fd1c13b8b40504cee2e53dc002bc85208dc9dd5538d731b06d99fec`.
- The source test module SHA-256
  `ac54463e47487a8fcec410f14243ae5f3b10940bbcc5766e31ff060a02e725ec`
  also matched on Lappy; `py -m unittest
  tests.test_qualify_minotaur_neural_taskmaster_cpu -q` returned **7 tests,
  0 skipped, PASS**, exercising the real CPU PyTorch head-update control test.
- Three arms: correct-label, zero-update, balanced corrupted-label. Each saw
  the same 16 exposed synthetic training prompts per cycle, the same visit
  order and the same initially seeded head parameters. Two distinct training
  cycles used 32 forward/backward steps per cycle; updated arms had 32 AdamW
  optimizer steps and the zero-update arm had none. A 16-item public held-out
  and a 16-item public alternative-template set were scored after cycle two.

## Measured development result

| Arm | Held-out | Alternative-template transfer | Optimizer updates/cycle |
| --- | ---: | ---: | ---: |
| Correct labels | 12/16 (75%) | 9/16 (56.25%) | 32 |
| Zero update | 4/16 (25%) | 3/16 (18.75%) | 0 |
| Corrupted labels | 1/16 (6.25%) | 3/16 (18.75%) | 32 |

All arms started with identical head-value SHA-256
`2424f3245492d8f53b0b9e645ca259cbfa39eea8e1028886161c6ace743f4292`.
The correct head changed to
`deb222e4789c276532316acafeeea436d0734663238252c60cb196eed1daca6d`
after cycle 0 and to
`50c43a28b16a27933ed40d3d115f646fea31d90031fd3daa38b56d74b86663dc`
after cycle 1. The zero-update head retained its initial digest across both
cycles. The corrupted head changed to
`b60a7aafc22af92d6a1885794d25cc21393776913937cb681e897dc47aea2709`
at the end. The runner loads each saved `cycle0.safetensors` head into the
new training cycle and rejects a reload mismatch. Six arm checkpoint files
exist in the isolated run directory; the script's report includes individual
file SHA-256 commitments.

These results establish actual trainable head-weight change, persistence
across regenerated public training cycles, and positive matched-input
development comparisons. They do **not** demonstrate a trainable general
Minotaur language model, authenticated independent transfer, or an external
graduation examination.

## Hostile evidence boundary

An independent source reviewer built a *post-hoc* four-rule keyword classifier
using obvious cues in the exposed taskmaster fixture; it correctly classified
all 64 public prompts across training and development sets **without neural
learning**. That is not a preregistered matched control, but it falsifies a
claim that high scores here require novel semantic reasoning. The correct
arm's transfer score was only 9/16, not a broad-generalization qualification.

The zero-update arm matched forward/backward visits but intentionally skipped
optimizer steps; this is not identical optimizer computation. The pretrained
encoder was frozen and locally cached; publisher model-byte provenance and
which executable read the weights were not independently attested. The
experiment contains no private exam, host-enforced trainer/evaluator isolation,
signed journal recovery or neural taskmaster deployment. The report explicitly
sets `graduation_authorized=false`, `independent_exam=NOT_RUN`, and
`model_execution_authenticated=false`.

Q3/Q4 may be marked **PASS only for this narrow pretrained-feature head
update and persistence observation**. Q5/Q6 remain UNKNOWN for independent
control/transfer/retention, Q7 NOT RUN, Q8 full FAIL/HOLD, and Q9 HOLD.
