# Theseus: learner-driven choices in the living Labyrinth

**Status:** Development-only source and tests. An action interface exists; no
large language model has yet been shown driving the episode. A neural update
must be proved separately from the checkpoint bytes and from this controller.

`labyrinth.theseus_agent_loop_v1.drive_theseus` changes who chooses the path:
the engine offers a public observation and exact legal actions, and a
model-compatible `TheseusActionPolicy.choose_action` responds with one line,
`ACTION: left` (or another listed action). The same engine handles the actual
source-owned TRAIN prompt/admission, model prediction, trusted update callback,
Minotaur examination and Ariadne evidence.

The policy can see the public scene, legal actions, generation, decision index
and its last eight public **event names**. Its `maze_cycle_index` is a world
regeneration counter, *not* a verified population generation or new model.
It cannot see TRAIN gold, candidate
grading cases, a private examination seed, or an examiner scoring callback.
It cannot declare a verdict. An invalid action response **stops** without
automatically choosing a route. Failed effects bubble up as indeterminate and
require recovery; no silent retry of a possibly applied SGD update occurs.

The controller reports four distinctions explicitly:

- `DEVELOPMENT_EPISODE_ENDED`: the existing SagaSession terminated, potentially
  after its *local* development criteria. Never implies promotion.
- `HALTED_DECISION_BUDGET`: the caller stopped before the episode ended.
- `HALTED_SOURCE_ACTION_BUDGET`: the source-owned action budget expired.
- `HALTED_INVALID_POLICY_OUTPUT` / `HALTED_INDETERMINATE`: cannot proceed.

Each action has a receipt binding an observation SHA, chosen action, exact
before/after checkpoint SHA, source effect event and maze cycle before/after.
The action trace itself is local self-reported material, not an external
immutable witness. Use `SagaSession.save_receipt_new` for its separate Ariadne
hash-linked history; neither stream is formal graduation.

### Real-model integration gap

An operator still needs to attach a truly inference-capable Vera model with
two distinct operations: (1) generating authorized action lines in response to
the public story and (2) predicting source-owned TRAIN prompts. Neither an
action script nor a fake byte-producing TRAIN callback is evidence of neural
learning. A dedicated actual trainer, independent fresh-process W0/Wt
inference and controlled matched candidate comparisons remain mandatory.

Reproduce source-level tests:

```bash
python -m unittest -q tests.test_theseus_agent_loop_v1
```
