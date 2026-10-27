# Minotaur — pretrained correctness-critic LoRA pilot

Scope: **optional, offline and advisory-only**. It trains a pretrained causal
language model to judge whether another model's proposed integer answer is
mathematically correct. This is distinct from the older maze-answering Minotaur
adapter. It never makes graduation decisions or replaces Vera's model.

External rationale: [*Let's Verify Step by Step*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html)
studies outcome versus process supervision; [*ProcessBench*](https://aclanthology.org/2025.acl-long.50/)
shows that recognizing the earliest erroneous step is a nontrivial verifier
task; and [*Generative Verifiers*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html)
investigates a richer model-based verification direction. This pilot implements
only **binary outcome correctness supervision**, not those full systems.

## Research question

Can actual pretrained LoRA updates improve Minotaur's balanced true/false
judgments and reduce false approvals on unseen generated arithmetic/operation
chains? Do the gains transfer to structurally different register programs?

## Reproducible local execution

Requires an **already-local** Hugging Face causal-LM checkpoint directory
containing safetensors weights and compatible tokenizer/config files, total
weights no larger than 1 GiB. No model is downloaded; no package is installed.
The optional current environment must already have PyTorch, Transformers and
PEFT. Use a short trusted path under a new research-only output directory.

```bash
python -m labyrinth.minotaur_critic_pilot train \
  --model-path /absolute/path/to/pretrained-hf-checkpoint \
  --output-dir /absolute/path/to/new/minotaur-critic-pilot \
  --steps 24 --device cpu

# Run only in an entirely NEW Python process after the train process exits.
python -m labyrinth.minotaur_critic_pilot reload \
  --run-dir /absolute/path/to/new/minotaur-critic-pilot \
  --report /absolute/path/to/new/minotaur-critic-reload.json
```

The immutable default protocol has 48 tier-1 and tier-2 training *tasks*,
each with paired right/wrong proposed answers. It selects **24 distinct tasks**,
one answer apiece and balanced labels, for 24 supervised AdamW updates. The
loss is masked to a single next-token Yes or No verdict. No eval answer or
reference solution enters the prompt at inference time. The model must contain
both Yes and No as unambiguous single tokens or the pilot fails closed.

All arms use the same pretrained checkpoint and evaluation prompts:

1. `frozen`: no adapter, no gradient updates.
2. `candidate`: correct Yes/No supervision and real PEFT updates.
3. `zero_update`: identical candidate LoRA initialization, same prompts and
   forward/backward/optimizer steps, **zeroed gradients and zero weight decay**;
   adapter digest must remain identical to initialization.
4. `corrupted`: the same initialization and budget as candidate, with labels
   inverted to test whether mere exposure and format learning explain changes.

The development slice reserves tier-1 and tier-2 problems from a different
generator seed and split. The transfer slice reserves tier-4 register programs.
Both compare all four arms on exactly the same cases. Every result records
accuracy, balanced accuracy, false approvals, false rejections, per-task-kind
metrics, and a prediction digest. The candidate adapter and base checkpoint
digests are recorded. Separate-process reload verifies exact LoRA tensor
values and identical held-out/transfer predictions. A same-process or
prediction-mismatching reload now fails with a nonzero exit, and the report
records Python, platform, Torch, Transformers, PEFT and deterministic-algorithm
settings; a changed reload runtime is rejected. These are self-reported
reproducibility metadata, **not independently authenticated evidence**.

## Claims and limits

- **Mechanism verified** only after a run proves real parameter changes and
  fresh-process adapter persistence. Source tests alone are not enough.
- **Behavioral change observed** only if the same-case candidate predictions
  differ from frozen and controls under fixed conditions.
- **Reasoning improvement supported** only if held-out correctness beats all
  controls with adequate statistical power and no detectable leakage.
- **Generalization supported** only with truly independent, structurally novel
  problems beyond this finite and fully public generator.

This protocol is deliberately bounded to 24 steps and tiny public tasks. It
cannot support general claims even if observed transfer is positive. Before
graduation, a separate evaluator must maintain privately custodied tests and
sign evidence under a pinned acceptance policy. The neural critic does not
possess the credential that can graduate a trainee.

### Known environment blocker

The current scratch Python lacks the optional Torch runtime. Lappy has
CPU-only Torch/Transformers/PEFT but **no verified small pretrained HF model
directory** in the checked canonical model location. Training and reload were
therefore not executed by merely adding this code. A suitable local checkpoint
must be confirmed before calling the above commands. Do not silently download
large model weights onto Lappy's capacity-constrained disk.
