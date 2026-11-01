# R107 — Offline cognitive-empathy TRAIN demo (source-only)

This runnable example exercises the approved `cognitive_empathy` synthetic
perspective-choice TRAIN provider and its objective-training runtime without
hardware, network services, paid inference, model weights, or private test data.
It is a **usability demonstration**, not a qualification or deployment receipt.

From the repository root with Python 3.10+:

```bash
python -m scripts.experiments.cognitive_train_demo
python -m scripts.experiments.cognitive_train_demo --seed 202 --mock-policy all-b
python -m scripts.experiments.cognitive_train_demo --seed 107 --mock-policy rotate-abc --json
python -m unittest tests.test_cognitive_train_demo -v
```

The default command displays **two cycles of 15 trainee-visible prompts**.
Each prompt presents a synthetic vignette and three interpretations, but does
not identify the correct choice. The mock learner selects `A`, `B`, `C`, or
rotates through `ABC` according to a declared deterministic policy; it never
inspects hidden answer keys. Only per-cycle aggregate practice counts and
source commitments appear in the terminal receipt. No per-room gold answer
or correct-letter feedback is printed. `--json` omits the vignette text and
prints one deterministic machine-readable source-only receipt instead.

The first cycle's 15-room completion uses **fabricated, labeled mock**
checkpoint hashes to exercise `complete_maze`. A deliberately fabricated,
zero-score *development* diagnostic then triggers regeneration into a second,
semantically non-reused 15-room cycle. The diagnostic is not based on a
private/held-out test set and is not evidence of any real exam, failure, or
skill improvement. The second cycle ends with a source-only training handoff;
it does not attempt to certify mastery or enter a real final exam.

The receipt explicitly records:

- The objective, seed, deterministic mock policy, and two aggregate cycles.
- Source `room_id`, training `case_id`, and public prompt SHA-256 commitments.
- The fact that all checkpoint digests are synthetic mock identifiers and all
  completion receipts are self-reports, not independently witnessed events.
- `q6_host_authorized=false`, `physical_training_performed=false`,
  `real_model_checkpoint_observed=false`, `independent_exam_performed=false`,
  and `mastery_or_graduation_authorized=false`.

The demonstration executes entirely in one Python process. Its mock learner
is controlled local code, *not* a sandboxed untrusted model or remotely
authenticated trainer. A malicious same-process learner could attempt to
inspect Python memory, so the prompt-only interface is API hygiene, not an
isolation guarantee. The runnable source and five smoke tests intentionally
avoid any network, GPU, credential, or paid-model path.

The existing `python -m labyrinth` commands and their behavior are untouched.
The cognitive demo is a separately named module in `scripts/experiments`.

**Qualification boundary:** this source-only example does not authenticate
host-owned Q6 exposure leases, protect against rollback of independently
stored state, witness physical checkpoint updates, execute a genuine Qwen
training run, establish learning gains or retention, administer a protected
exam, or authorize Minotaur graduation. Any such claims require separate
trusted infrastructure and independently verified evidence.
