# Labyrinth

Labyrinth is an offline-first, model-agnostic **reasoning challenge and evaluation harness**. It generates seeded tasks, scores strictly formatted answers, and compares a frozen baseline with three in-context conditioning controls.

It is a working **prototype**, not a weight trainer or evidence that a model has learned persistently. A scored training trajectory and its demonstrated examples can be saved and reloaded in another process. That is *persistent prompt memory*, not a model update. Any claim of improved reasoning must come from independently reserved held-out results and suitable statistical replication.

## Run it

Requires Python 3.10+; core functionality uses the standard library.

```bash
python -m labyrinth run --seed 42 --count 3 --tiers 1,2,3 --adapter replay
python -m labyrinth experiment --seed 42 --train-count 3 --eval-count 3 --artifact-out training.json --report evaluation.json
python -m labyrinth experiment --seed 42 --eval-count 3 --artifact-in training.json --report rerun.json
python -m unittest discover -s tests -v
```

`--count`, `--train-count`, and `--eval-count` mean tasks **per tier**. Reports are JSON; existing paths are never overwritten. Without a specified model, `replay:constant-zero` is a deterministic negative-control fixture that outputs `ANSWER: 0` for every question. It does not solve problems or learn. To get nontrivial scores offline, pass `--replay-file mapping.json`, where the JSON object maps exact public prompt text to fixed answer strings. Missing fixture keys fail closed.

For a local user-operated Ollama installation:

```bash
python -m labyrinth run --adapter ollama --model YOUR_INSTALLED_MODEL --seed 42
python -m labyrinth experiment --adapter ollama --model YOUR_INSTALLED_MODEL --seed 42 --report local-model.json
```

The Ollama adapter is opt-in. It targets only a local loopback address by default (`http://127.0.0.1:11434`), with disabled proxies and redirects, a bounded response, and a timeout. It makes no shell calls using model output. The CLI provides `--allow-remote-ollama` for deliberately trusted non-loopback endpoints; use it only under your own network security policy. There is no default download, service installation, remote inference, or change to a Vera runtime.

## What the experiment measures

For each run, the generator produces separate `train` and `eval` splits using a seeded, source-controlled generator. A scored training trajectory records the adapter's response followed by evaluator feedback. Its prompts and correct examples become a saved in-context demonstration artifact. Evaluation then uses the **same held-out challenges, in the same order and with the same per-challenge generation seeds**, across four independent adapter instances:

| Arm | Information supplied before the held-out problem |
| --- | --- |
| Baseline | None |
| Candidate | Correct demonstrations from the training split |
| Placebo | A neutral passage matched to candidate **character** length (not token count) |
| Shuffled | The same training demonstrations, but answers cyclically shifted |

Tier 1 uses arithmetic, tier 2 uses multi-operation reasoning, and tier 3 tests stated evidence priority against contradictory data and a quoted prompt-injection attempt. Tiers are nominal categories, **not empirically calibrated difficulty levels**. The curriculum module separately supports deterministic progression/feedback using only training examples. It cannot elevate held-out test scores into the curriculum state.

The evaluator accepts only a complete `ANSWER: <integer>` line. Reports distinguish *answer accuracy* from *output-format adherence* and include challenge commitments, seeds, model identity, generation options when available, report schema, source-tree hash, and per-trial prompt/response hashes. Correct answers are never placed in public evaluation prompts or reported as explicit answer keys.

Training artifacts contain answer keys because their purpose is to preserve demonstrations. On load they must match the current canonical training generator outputs and source-tree hash; a self-computed checksum alone is not treated as authenticity. Experiment runs reject train/eval challenge ID or normalized prompt overlap. Do not tune against the same held-out seed repeatedly and then call it independent evaluation; reserve new seeds for final qualification.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

Tests cover seeded determinism, exact scoring, injection resistance, adapter boundaries, curriculum replay, artifact tampering, separate control-arm adapter instances, bounded untrusted output, and fresh-process artifact reload.

Important limitations:

- The replay baseline has no reasoning ability. Accuracy changes from a real model must be measured, not inferred from the harness.
- In-context demonstrations are **not** parameter or optimizer updates, and merely reloading a prompt artifact does not establish persistent model learning.
- Same-seed requests do not guarantee identical output from every runtime. Model filenames, quantization, model-weight hashes, and runtime builds must be recorded separately if strict model identity matters; the harness reports missing weight hashes as unknown.
- The task generator is public and finite. Source/domain overfitting, memorization, stochastic runtime variance, and benchmark gaming remain concerns. For strong claims, reserve unseen generator seeds or distinct problem families, replicate trials, and compare against external benchmarks.
- Controls have unequal tokenization; the placebo is character-length matched only. Shuffled labels may coincide when two answers share a value; reports count these collisions.
- A fresh Python adapter instance does not prove that an external inference server is stateless.
- Source-tree hashes and training-artifact checksums identify content but **do not authenticate** a repository, trainer, or model creator.

## Code organization

- `labyrinth/maze.py`: seeded challenge generator; evaluator-only answers
- `labyrinth/evaluation.py`: strict, side-effect-free scoring
- `labyrinth/adapters.py`: replay and local Ollama interface
- `labyrinth/curriculum.py`: training-only progression state
- `labyrinth/runner.py`: scored runs, artifacts, four-arm controls, JSON provenance
- `labyrinth/cli.py`: `run` and `experiment` commands

Labyrinth deliberately does not install itself into Vera, execute model responses, mutate Git repositories, or claim any externally deployed training effect.
