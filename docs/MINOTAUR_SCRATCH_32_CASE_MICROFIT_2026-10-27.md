# Minotaur scratch 32-case microfit diagnostic (2026-10-27)

**Verdict:** the local scratch Transformer and AdamW path can memorize the
declared 32-case fixture. This is a positive optimizer diagnostic and **not**
evidence of held-out neural competence. Qualification remains
`NOT_ESTABLISHED`.

## Audit findings

The committed v1 run used 384 training cases and 96 optimizer steps. The later
negative experiment described in `MINOTAUR_GUARDED_TRACE_EVIDENCE_2026-10-27.md`
used 512 training cases and 384 optimizer steps. Its held-out dev accuracy fell
to 29.2% and longer-program accuracy to 30.2%.

The original trainer cannot establish why that happened:

- It records only dev/transfer metrics before and after training. It does not
  record training loss, training accuracy, per-step loss, gradient norms, or
  clipping frequency. A working optimizer, failed optimization, and overfit
  shortcut learning therefore were not distinguishable from its report.
- Cross entropy uses PyTorch's mean reduction and gradients are clipped to L2
  norm 1.0, but neither choice is written to the report. Mean reduction makes
  gradient scale depend on per-batch averaging, while each final partial batch
  would still receive a full optimizer step. The reported 384- and 512-case
  runs were divisible by batch size 32, so a partial batch does not explain
  those two measurements.
- The tokenizer has disjoint special, operation, register, operand-kind, and
  numeric token ranges, and its maximum numeric token is 112 within vocabulary
  size 128. It does not put the gold label in the input. It does, however,
  hard-code a 96-token tokenizer limit separately from `config.max_tokens`, and
  it accepts arbitrary printed instruction numbers because the regex checks
  operation order but not question-side step numbering. Changing this encoding
  would invalidate the existing sealed checkpoint, so this diagnostic leaves
  it unchanged and tests the exact current path.
- Case IDs are disjoint across the three seeded datasets in `run_scratch`.
  However, both training and development combine `fresh_numbers` and
  `new_composition`; the generator calls the latter `matched_length_ood`.
  Consequently the old "dev" score is a new-seed matched-family score, not a
  preserved composition boundary. Only `longer_program` is held out by slice.

The bounded evidence supports a narrower explanation: the update machinery is
capable of fitting this data, while more updates did not improve held-out
behavior. Overfitting or shortcut learning is therefore plausible. The missing
extended checkpoint and missing training curves prevent claiming it as the
unique demonstrated cause of the 384-step regression.

## Diagnostic design and criteria

`run_microfit` always starts a new width-32, one-layer Transformer from seed
411. It never loads pretrained or existing weights. It selects 16
`fresh_numbers` programs from seed 4404 and uses each program's paired correct
and faulty trace: 32 cases total, 16 label 0 and 16 nonzero labels. These same
cases are optimized and measured deliberately; there is no held-out claim.

AdamW uses learning rate 0.003, zero weight decay, whole-fixture batches, and
explicit mean loss over 32 cases. Pre-clip gradient L2 norms are recorded; the
diagnostic clip threshold is 5.0. It may take at most 384 optimizer steps and
stops when all 32 labels are correct and mean loss is at most 0.05. Success also
requires changed parameter bytes and finite, strictly positive gradient norms
at every update.

The output is separate from `models/minotaur_scratch_v1`:
`docs/evidence/minotaur_microfit_32_20261027/`. No existing checkpoint was read
or modified.

## Test-first and measured results

The new behavioral test was run before implementation and failed with one
error: `ImportError: cannot import name 'run_microfit'`. After implementation:

| Measure | Before | After |
|---|---:|---:|
| Correct / 32 | 3 | 32 |
| Accuracy | 0.09375 | 1.0 |
| Mean cross entropy | 1.9917196035 | 0.0478830636 |

The stop criterion was reached after 68 real optimizer updates. Pre-clip
gradient norms were all finite and positive: first 2.8811402, last 0.1361455,
minimum 0.0759037, maximum 2.8811402. No step reached the L2 clip threshold of
5.0. The initial and trained complete-parameter digests differ:

- initial: `f81c8232437824e071a39dc35e5fc6a554dc6550ed7fc92d6c00fe02633b14e2`
- trained: `8be96b7d96e8557e0f66b6f47c4c32a3da2043584e5531e5e2cd2339af27eecc`
- output Safetensors: `a0af11f2e84152d007d0df83f1a235549908e48d215cf0cdf9d86e0578973f2d`

## Exact commands

The local preinstalled CPU-capable environment was used; no network or install
command was invoked.

```powershell
& 'D:\Python\Envs\vera-training-20261014\Scripts\python.exe' -m unittest tests.test_minotaur_microfit -v
& 'D:\Python\Envs\vera-training-20261014\Scripts\python.exe' -m labyrinth.minotaur_scratch --microfit-output docs/evidence/minotaur_microfit_32_20261027
& 'D:\Python\Envs\vera-training-20261014\Scripts\python.exe' -m unittest tests.test_minotaur_scratch tests.test_minotaur_microfit -v
& 'D:\Python\Envs\vera-training-20261014\Scripts\python.exe' -m unittest discover -s tests
```

The focused result was 5/5 passing in 11.539 seconds. The full repository result
was 217/217 passing in 64.518 seconds. The standalone microfit command completed
well below its 90-second CPU budget.

## Next experiments

1. Re-run the 96-step and 384-step configurations from identical initial
   weights while logging train/dev loss, accuracy, unclipped gradient norms,
   clipping counts, and per-slice confusion matrices.
2. Restore `new_composition` as a true slice holdout and compare early stopping
   selected only on a separate matched-family development set.
3. Add a controlled tokenizer revision with validated instruction numbering
   and one explicit configuration-owned token budget; compare against the
   current encoding from the same seeds without touching v1 artifacts.
