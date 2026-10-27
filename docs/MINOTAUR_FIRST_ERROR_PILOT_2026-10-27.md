# Minotaur first-error step critic — opt-in pilot

Status: **experimental, NOT_ESTABLISHED**. This is a genuine trainable
pretrained causal-LM LoRA *path*, not a trained/released neural critic and not
an official ProcessBench evaluation. Nothing is wired into production routing.

## Task and data

`labyrinth/minotaur_first_error_pilot.py` generates deterministic register
programs using the existing `program_maze` generator. An independent interpreter
replays the printed instructions, checks the source's final A register, and
verifies the earliest incorrect displayed state. Every program has a clean
trace (`first_error=None`, equivalent to ProcessBench `-1`) and a trace with
one intentionally incorrect state at a reproducible, varying **one-based**
step. All later states are oracle-correct, so the earliest discrepancy is
unambiguous; it is **not** a demonstration of realistic cascading mistakes.

The critic receives the instruction semantics, initial A/B, entire program,
and a proposed numbered execution **prefix**. It answers `Yes` if every shown
state is correct and `No` otherwise. It never receives the source final-answer
request, source answer, trace variant, injection index, or oracle label in its
input. At evaluation, the earliest prefix classified `No` becomes
`ERROR_STEP:N`; no rejection is `ALL_CORRECT`. This gives a ProcessBench-style
first-error output but not ProcessBench provenance or capability claims.

Four separate deterministic populations:

| Population | Seed | Register structure profile | Default programs / traces |
| --- | ---: | --- | ---: |
| Train | 1357 | `train` | 24 / 48 |
| Dev (fresh literals) | 2468 | `train` | 8 / 16 |
| Composition transfer | 3579 | `matched_length_ood` | 8 / 16 |
| Length + composition transfer | 4680 | `structural_holdout` | 8 / 16 |

IDs and problem text cannot overlap across populations; both transfer profiles
must have no instruction-structure fingerprint shared with train. Generator,
splits, prompts, and labels are **public**, not blind. Reserve new seeds and
an independent evaluator before drawing broader conclusions.

## Controls and metrics

One optimizer step consumes one previously unseen train-program pair: a clean
prefix and corrupted prefix at the identical injected index. Each step thus
contains exactly one positive and one negative target. Control arms share
the selected examples, order, optimizer-step count, initial LoRA parameter
values, base checkpoint, prompt lengths and forward/backward budget:

- `frozen`: pretrained base inference only, no LoRA training;
- `zero_update`: identical paired losses/backprop and AdamW steps, but zeroed
  gradients, verified unchanged parameter digest;
- `corrupted`: same updates with all Yes/No training targets exchanged;
- `candidate`: genuine AdamW/PEFT LoRA updates, verified changed digest.

All arms report exact first-error accuracy, error-class and clean-class
accuracies, their harmonic mean, false-approval and false-rejection rates,
and a digest of predictions. The candidate adapter is saved as safetensors.
Training records retain selected example IDs, losses, update counts, and
initial/control parameter digests. A **separate process** must invoke `reload`
and match the candidate's predictions and metrics exactly after verifying
base and adapter bytes, LoRA tensor values, runtime versions, source identity,
and different process ID. This is persistence evidence, not a blind exam.

## Offline use

Requires already-installed torch, transformers, peft and a pre-existing local
safetensors causal LM.

Run from the repository root. No dependencies or checkpoints are downloaded.
The existing HF preflight caps local weight files at 1 GiB, rejects unsafe
checkpoint layouts, checks CPU/GPU headroom, and rejects existing output paths.
Defaults bound 24 programs, 16 paired optimizer steps, 512 tokens, LoRA rank 4,
and CPU threads to at most four. For a smaller trial, use `--steps 2`.

```powershell
py -3 -m unittest tests.test_minotaur_first_error_pilot -v
py -3 -m labyrinth.minotaur_first_error_pilot train --model-path C:\local\small-model --output-dir .\first-error-run --steps 2 --device cpu
# In a fresh Python process:
py -3 -m labyrinth.minotaur_first_error_pilot reload --run-dir .\first-error-run --report .\first-error-reload.json
```

Use a new, absent output directory for every run. Failures deliberately do not
overwrite or silently resume a partial run. Do not use either report as
production authorization. The main threats to validity are synthetic single-
line perturbations, public finite templates, small and non-blind splits, one
seed, and a likely severe gap from realistic mathematical reasoning traces.

## Execution record in this branch

The dependency-free unit tests ran; they exercise deterministic dataset
generation, oracle and structural separation, masking/negative balance, metric
definitions and bounds. No pretrained model was trained, evaluated, saved or
reloaded in this environment: `torch`, `transformers`, `peft`, and `safetensors`
were absent from the available Python interpreter. A real neural outcome
requires the explicit offline commands above on a machine with a pinned local
checkpoint and installed dependencies, then independently blinded evidence.
