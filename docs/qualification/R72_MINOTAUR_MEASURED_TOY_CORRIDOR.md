# R72 — Measured toy learner / adaptive Minotaur corridor

Status: **SOURCE-ONLY REAL TOY PARAMETER UPDATE; PRETRAINED LEARNING AND GRADUATION HOLD**.

`run_measured_toy_corridor(scope, generate_maze_plan(scope, 0), learner)`
connects two previously separate, runnable implementations:

1. `TrainingCorridorRunner` dynamically materializes TRAIN-only arithmetic
   chambers, registers exact training case/prompt commitments, feeds only
   `TraineeRoomView` to inference, and updates the bounded Minotaur policy
   from each learner response.
2. The operator-trusted bridge privately accumulates those admitted TRAIN
   `Challenge` objects. It refuses non-practice tactics, non-arithmetic cases,
   malformed scope/plan, and zero learning rate *before* starting toy updates.
3. The existing `ToyLinearAdapter.train` performs real four-parameter
   normalized-SGD updates over that exact admitted batch. The bridge
   independently rereads `learner.model_hash` before and after and rejects an
   unchanged or inconsistent optimizer report. The report represents this as
   a **separate post-corridor optimizer event**; all scored chamber receipts
   refer to the earlier, unchanged pre-training checkpoint.

The runner now exposes separate `registry_transitions` for non-learning TRAIN
inventory admissions. The bridge checks the predecessor/successor hashes of
each transition between adjacent scored `ObservedTrainingChamber` receipts,
closing the previously unexplained policy-hash gap between rooms. The first
TRAIN case bootstraps the policy and is not a registry-extension receipt.

Because the `ToyLinearAdapter` API requires at least two training cases, the
toy weight update occurs **after** all chambers in the episode. Minotaur's
adaptation within that episode responds to pre-update inference feedback; it
does not yet respond to intermediate weight improvements. Scores must not be
attributed to the post-optimizer checkpoint merely because the optimizer ran
later. This test uses no private final exam. The policy statistics are not the
learner's weights.

Verification:

```bash
python3 -m unittest -q tests.test_minotaur_measured_toy_bridge_r72
python3 -m unittest -q tests.test_minotaur_measured_toy_bridge_r72 \
  tests.test_journey_policy_runner tests.test_journey_toy \
  tests.test_minotaur_policy_learning tests.test_toy_backend
```

This is **not** an HF/LoRA bridge. `hf_pilot` trains its own independently
generated scoped challenge catalog and does not accept `TrainingCorridorRunner`
as an optimizer exposure source. Without a typed, validated adapter from
the corridor's exact case ID, prompt digest, supervision, and optimizer
exposure cursor into a real HF training step, matching a final checkpoint
digest alone could link unrelated training and must not be called integration.

The source-level hash is not independently hosted custody, proof of
generalization, structural transfer, Q5–Q9 qualification, or authority to
merge, deploy, or graduate a model. All Python callbacks and optimizer code
run in the same trusted interpreter. Model provenance, host authority and
rollback resistance require separately governed, externally witnessed checks.
