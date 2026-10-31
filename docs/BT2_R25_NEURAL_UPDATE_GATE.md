# R25 neural weight-update gate (source-only)

The R22 two-cycle CPU checkpoint smoke verifies interrupted AdamW continuation
against an uninterrupted run. Parity by itself does not exclude the vacuous
case where neither trajectory changes trainable weights.

`scripts/qualify_v2_weight_updates_cpu_r25.py` adds an independent, opt-in
physical observation of **trainable parameter bytes**, the supposed frozen
base's parameter bytes, and AdamW first/second moment tensors after each
generated Maze cycle. It rejects unchanged trainable weights, altered frozen
weights, non-advancing moments, invalid optimizer step counters, nonfinite
weights or moments, and missing/nonfinite trainable gradients at each step.

Run the no-effect preflight:

```shell
python3 -m scripts.qualify_v2_weight_updates_cpu_r25
```

Run the actual ephemeral CPU probe only in an authorized environment with
PyTorch available:

```shell
python3 -m scripts.qualify_v2_weight_updates_cpu_r25 --execute
```

Exit status `3` and `SKIPPED_NO_PHYSICAL_EXECUTION` means no physical neural
execution took place. That is *not* a qualified pass. A passing R25 physical
run observes real changes to randomly initialized toy-network tensors but
**does not load pretrained MiniLM**, prove task improvement, independently
custody evidence, qualify the R22 snapshot in the same run, prove antirollback,
or authorize graduation. A full pretrained-LoRA resume probe is separately
implemented in `scripts/qualify_minilm_resume_cpu.py` and has its own model
cache and dependency requirements.

Source-only gates:

```shell
python3 -m unittest tests.test_qualify_v2_weight_updates_cpu_r25 -v
```

The two conditional physical tests skip when PyTorch is unavailable. All
claims must name exact runtime, source head, and execution status.
