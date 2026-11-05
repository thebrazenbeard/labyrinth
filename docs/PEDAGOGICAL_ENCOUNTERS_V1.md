# Pedagogical encounters: the first causal world-to-learning bridge

**Status:** DEVELOPMENT_ONLY source prototype. No demonstrated neural training,
identity formation, autonomous evolution, externally validated generalization
or private graduation. This is one small prerequisite for Patrick's larger
living, mythic Labyrinth design.

## Problem actually corrected

Previously the operator generated a TRAIN challenge *before* asking the scene
model to propose a world. Branching locations and prose could not influence
which task the learner experienced. A changed map was essentially a cosmetic
overlay over the same authorized training sequence.

The operator now previews the **TRAIN-safe assigned room descriptor without
materializing a challenge**. The scene model receives the objective, skill,
cycle, public room seed and public feedback, but no TRAIN prompt/answer and no
Minotaur heldout/private material.

A `WorldProposal` attaches distinct `PedagogicalEncounter` descriptors to
each direct branch out of `threshold`. These are strictly finite choices:

- a source-approved arithmetic TRAIN tactic: `practice`, `contrast`, `transfer`;
- a difficulty adjustment in `[-1, 0, +1]`, clamped within the original
  authorized level 1–5 range.

Distinct branches must remain different in at least one approved pedagogical
dimension (**task tactic/modality or effective difficulty**) **after** the
difficulty clamp for every legal source baseline 1–5. For example, the
superficially different `practice/0` and `practice/+1` choices are refused
because both become `practice/5` at a saturated difficulty-5 room.

Crucial distinction: `practice` versus `transfer` can exercise the **same
underlying arithmetic atom** (e.g. `8+4`) in symbolic and situated prose.
That is real representation-transfer exposure, not proof of a new mathematical
skill, novel computation, weight-space improvement or independent generalization.
An eventual evaluator must explicitly measure the intended generalization;
different prompt hashes by themselves do not establish it.

The actual path the learner takes determines which descriptor reaches
`TrainingCorridorRunner.next_training_prompt(route=...)`. The runner still
owns assignment, numeric challenge generation, deterministic fresh-case redraw,
answer authority, exact TRAIN case registration, non-duplication and feedback.
The model cannot author a prompt, answer, exam split, score, model checkpoint,
graduation verdict or executable callback through its encounter proposal.
In the living default, model worlds with no consequences are rejected rather
than being silently accepted as an "evolutionary" experience. A caller may
explicitly set `require_pedagogical_encounters=False` to replay older cosmetic
worlds, but that is intentionally not the default mode.

When the learner reaches `chamber`, the source generates the challenge with
the chosen bounded tactic/difficulty. An admission failure quarantines the
in-process episode as `INDETERMINATE`; it does not silently retry a possibly
partial TRAIN registry effect. Existing physical training failures remain
quarantined as well. This is not crash-durable process recovery.

The world fingerprint commits to the graph plus annotated pedagogical routes
when encounters are present. Each successful answerless Ariadne TRAIN event
now also hash-links the actual chosen branch node, source-resolved tactic,
bounded difficulty and public TRAIN prompt hash, in addition to the checkpoint
and policy transition. Existing route-less V1 journals still verify, but do
not retroactively acquire that historical provenance; an unsigned full-journal
rewrite remains possible until external anchoring is implemented.

After scored TRAIN, Daedalus's next `SceneRequest`
receives a bounded rolling 16-outcome history of tactic, task difficulty and
correctness, **across failed Minotaur cycles and maze regeneration**. Minotaur
receives the same safe public development feedback. No prompt, answer, private
case or source gold enters that history. Ariadne separately retains its own
integrity chain; the rolling contextual window is not the full recovery log.
These aggregates describe TRAIN observations, **not** behavioral improvement
or fresh-process checkpoint attribution.

Minotaur exam pressure has a separate source/operator-owned gate:
`SagaSession` requires an explicit `ExamCoveragePolicy(minimum_difficulty=...)`
chosen for its arithmetic learning objective, **not** inferred from training
`specificity` (which describes narrowness). The model may propose harder exams,
but a proposal below that independently frozen owner floor is raised to the
floor before source-only heldout/transfer case generation. The local examiner
also rejects any request whose effective difficulty lies below its supplied
minimum. Both heldout and transfer coverage remain required. This is still
a DEVELOPMENT_ONLY in-process gate, never an independent private qualification.

## Falsifiable source tests

Two otherwise identical entrance states, deterministic Python `SceneDirector`
stubs implementing the model-shaped proposal interface, objective, seed and
policy take different learner choices at `threshold`:

- `left` → archive → source-produced `contrast` challenge;
- `right` → forge → source-produced `transfer` challenge.

The cases have distinct actual prompt hashes, source registry admissions and
arithmetic task structures. Both TRAIN hooks write distinct local checkpoint
files, though the stand-in learner deliberately remains incorrect in both
counterfactuals. Their subsequent Daedalus/Minotaur public feedback contexts
show the selected tactics, and a deterministic test Minotaur pressure selector
receives different inputs and chooses different development difficulty.
No neural scene model is actually invoked by these source tests.
Invalid tactic names, gold-like strings, missing/duplicate branches, out-of-
range difficulty, absent default encounters and genuinely equivalent routes
fail before training. A hostile regression shows a weak single-digit-only
predictor could pass an easy arithmetic development exam while failing the
explicit harder objective; model-selected easy pressure cannot override the
owner's difficulty floor. A second hostile regression confirms actual TRAIN
feedback remains visible to Daedalus after a failed Minotaur and regeneration.

Run:

`python -m unittest -q tests.test_model_saga_runtime tests.test_journey_policy_runner`

## What remains before the actual epic

This is still an arithmetic-only, in-process prototype. The model cannot yet
choose among a rich set of **operator-authorized cognitive/creative/social
experiences** or carry rich world-state consequences across restarts. There is
no authentic Qwen/LoRA training adapter in this path and no checkpoint-reloaded
causal evaluation of learning. The persistent world, evolutionary candidate
population, general-domain encounter registry and source-owned lineage
selection still need end-to-end integration. This contribution establishes
one narrower fact: **world choices now causally alter approved training
exposure** rather than merely decorating a predetermined problem.
