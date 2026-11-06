# Evolutionary Saga Runner V1 — mechanical integration, not model evolution

**Status:** runnable source-only DEVELOPMENT prototype. No real neural model,
LoRA training, final graduation, autonomous selection, crash recovery or
behavioral generalization is established by this module.

## What is actually wired

`run_saga_trial()` drives an actual `SagaSession` through a validated world,
real source-owned TRAIN materialization, operator `train_supervised` calls,
physical checkpoint hash readback, score aggregation and Minotaur development
cycles. The old Python `choose_threshold_action` callback is operator-installed
scripted navigation, **not** proof that an LLM agent navigated the world.

`run_theseus_saga_trial()` is the alternative action boundary for
`theseus_agent_loop_v1.drive_theseus`: the action-policy interface receives
public observations and legal actions, and invalid responses halt without
fallback. It requires that separate module to be integrated first. Even a
valid model-action receipt does not prove its provider is a pretrained model.

`run_mechanism_campaign()` composes two or more generations, each with at least
two named candidate sessions. It compares the **actual physical entrance
weights and optimizer digests** of each sibling against the selected parent's
current bytes, refuses aliasing an entrance path, rehashes the parent before
and after each sibling run, and validates all selected checkpoints at the end.
The caller's explicit owner-supplied selector may choose only a real candidate
with source `developmental_status == PASS`. Rejected siblings remain in each
generation's evidence. The selected Wt artifact becomes the next generation's
physical parent. There is no in-process candidate merging or silent weight
reset to W0.

`fresh_process_checkpoint_control()` starts a **real separate Python process**
to rehash W0 and optional optimizer from disk. It checks the returned digest
and child PID against a parent readback, using a minimal child environment and
no prompt, answer, expected score or private examination data. This is
*physical checkpoint readback*, **not a matched fresh-process behavioral
evaluation**; a model with the same bytes can still behave differently under
different runtimes, loading configurations or sampling choices. A meaningful
experimental population requires frozen probes/splits, a real neural loader,
independent evaluator and W0-vs-candidate-vs-shuffled controls.

All emitted evidence is marked `MECHANISM_ONLY` / `UNQUALIFIED`. A synthetic
trainer that writes arbitrary bytes and a test examiner that scores by cycle
may produce a development PASS but **never** qualify `NEURAL_UPDATE_OBSERVED`,
independently demonstrated skill gain or graduation. The acceptance tests
deliberately use those falsifying fixtures.

## Non-composed boundaries (fail closed)

1. `evolutionary_population_v1`: only the operator-supplied selector seam is
   presently available to this runner. A production selector must own signed
   lineage, immutable full-model/base/tokenizer/adapter/optimizer/RNG manifest,
   fresh reserved fitness evaluation and candidate provenance; this runner's
   digest comparison does not replace that authority.
2. `conversation_objective_v1`: `SagaSession` currently hard-refuses scopes
   beyond `skills == ('arithmetic',)`. This runner **does not** falsely route
   conversational examples through the arithmetic task generator. A separate
   source-owned conversation TRAIN/eval provider or generalized SagaSession
   needs explicit implementation and tests before conversation adaptation is
   live. Do not present test prose as a conversation training result.
3. `trainingday_adapter_v1`: this runner cannot convert a mock file write or
   a `WORKER_SELF_REPORTED_UNVERIFIED` dispatch receipt into a neural update.
   Only a physically read-back, checkpoint-reloaded model-training adapter
   can establish that effect.
4. `ariadne_recovery_v1`: the existing saga enters `INDETERMINATE` in memory
   before a potentially partial training side effect. There is no fsynced
   pre-effect intent here and therefore **no crash-safe automatic restart**.
   Recovery requires separate custody proof and an explicit no-retry decision.

These adapters may be added without granting an untrusted scene, Theseus
navigation model, Minotaur director or worker receipt authority over TRAIN
gold, private exams, parent checkpoint custody or candidate promotion.

## Local mechanical acceptance

`python -m unittest -q tests.test_evolutionary_saga_runner_v1`

The source-only fixture exercises two generations (four siblings), discarded
failed candidates, selected checkpoint physical ancestry, an actual child
process W0 hash control, malformed navigation refusal, mismatched sibling
checkpoint rejection, and refusal to promote development-failed candidates.
