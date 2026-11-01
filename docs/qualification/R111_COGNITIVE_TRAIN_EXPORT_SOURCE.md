# R111 operator-only cognitive TRAIN dataset export (source only)

`labyrinth.cognitive_train_dataset_export.export_cognitive_train_jsonl` is an
explicit, in-memory export for the existing **approved synthetic cognitive
perspective TRAIN provider**. It does not generate new curriculum cases or
examination items.

```python
from labyrinth.cognitive_train_dataset_export import export_cognitive_train_jsonl

# `contract` and `journey` must already be valid, approved objects in TRAINING.
dataset = export_cognitive_train_jsonl(
    contract, journey,
    forbidden_semantic_sha256=operator_reserved_semantic_hashes,
)
assert dataset.sha256  # local content checksum, not an independent signature
# dataset.jsonl_bytes: operator-only supervised TRAIN examples containing labels
```

The function returns a detached, immutable `CognitiveTrainDataset` tuple with
canonical newline-delimited JSON bytes, a SHA-256 digest, cycle, objective,
contract and provider identifiers, and count. Nothing is written to disk or
sent over the network. Each row includes the approved **TRAIN-only** prompt,
its `A`/`B`/`C` target choice, exact case/prompt/semantic source identities,
and a local source-binding digest. This is **training data with gold labels**.
It is not an answerless learner view and **must not** be shown to an untrusted
learner or used as an independent protected examination set.

The source exporter reuses the cognitive objective runtime's journey/contract
validation and regenerated, approved private room roster. Incorrect phases,
scope/provider/rubric substitutions, mismatched prior-cycle attempts, blocked
case/prompt/semantic identities, historical aliases, and malformed exclusions
are rejected. The corpus contains at most 15 cases per bounded cycle. The
approved 30 scenarios split into two nonoverlapping 15-case banks; a third
full cycle fails closed. Export content is bounded to 256 KiB.

## Trust and physical boundaries

- **Operator-only is a usage contract, not an authenticated API role.** The
  importable Python function does not check caller identity or isolate its
  labeled output. Any process permitted to import the module and supply an
  approved contract can request TRAIN gold. It must not be exposed through
  the public CLI, trainee callbacks, a network endpoint, or an examination
  path. Process separation and host-owned permission enforcement are absent.
- The bytes are **not** a model update, LoRA adapter, GGUF, optimizer result,
  Qwen training job, or evidence of an improved neural checkpoint.
- The digest and binding metadata establish deterministic **local source
  consistency only**, not host-owned signatures, rollback-resistant replay
  authority, or independently attested dataset provenance.
- A caller in the same process can inspect private source objects, forge
  arbitrary new `CognitiveTrainDataset` tuples, and choose its own output
  file. The exported label must therefore be handled by a trusted trainer,
  isolated from any independent examiner.
- R102's separate, in-memory Q6 arithmetic/program exposure CAS does **not**
  protect this cognitive dataset or the cognitive prompt runtime.
- Host/GPU lease, stable Executor route, restricted evaluation data, exact
  checkpoint and optimizer continuity, real model-weight updates, evaluator
  independence, Q5–Q9 gate evidence and Minotaur graduation remain **HOLD**.
- No cloud services, paid compute, physical machine actions or deployments
  are triggered; the project's hard spending boundary is **$0**.

The operator may pass `dataset.jsonl_bytes` only to a separately approved
trainer after authenticating data access, host lease, and evaluation
partition. A future supervised fine-tuning adapter must mask loss for the
prompt tokens and supervise only the target-choice response tokens; it must
not print gold labels or reserve evaluation content to public logs. This file
deliberately does not implement that physical handoff.

## Verification

Run `python -m unittest -v tests.test_cognitive_train_dataset_export` for
determinism, exact 15-case labels, operator source checks, nonoverlap across
two cycles, immutable detached output, wrong phase/scope/provider, malformed
semantic exclusions, and forged source gold. The integrator must add this
test module to the repository's mandatory no-skip CI list before publication.
