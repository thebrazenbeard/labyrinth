# Labyrinth

Labyrinth is an offline-first **reasoning challenge, training-lifecycle, and evaluation harness**. It generates seeded tasks, scores strictly formatted answers, compares in-context controls, and includes a deliberately small CPU-only parameter-learning demonstration.

It is a working **prototype**, not yet a language-model weight trainer or evidence of generalized reasoning improvement. Prompt demonstrations are *persistent prompt memory*, not a model update. The separate toy backend really updates four numeric coefficients and reloads them, but its hand-engineered arithmetic hypothesis class is not an LLM.

## Run it

Requires Python 3.10+; core functionality uses the standard library.

```bash
python -m labyrinth run --seed 42 --count 3 --tiers 1,2,3 --adapter replay
python -m labyrinth run --seed 42 --count 3 --tiers 4 --adapter replay
python -m labyrinth experiment --seed 42 --train-count 3 --eval-count 3 --artifact-out training.json --report evaluation.json
python -m labyrinth experiment --seed 42 --eval-count 3 --artifact-in training.json --report rerun.json
python -m labyrinth train-toy --seed 0 --train-count 64 --eval-count 64 --checkpoint-out toy-checkpoint.json --report toy-comparison.json
python -m labyrinth toy-eval --seed 0 --eval-count 64 --checkpoint toy-checkpoint.json --report toy-reloaded.json
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
| Shuffled | The same demonstrations with a full, label-multiset-preserving derangement |

Tier 1 uses arithmetic, tier 2 uses multi-operation reasoning, and tier 3 tests evidence priority against contradictory data and a quoted prompt-injection attempt. Tier 4 introduces a bounded two-register program language: training structures have 3–4 instructions, while structural holdouts have 5–6 and different operation compositions. An independent test interpreter checks the generated answers. Tiers are nominal categories, **not empirically calibrated difficulty levels**. The curriculum module separately supports deterministic progression/feedback using only training examples; the standard `experiment` command does not yet train from this adaptive curriculum.

The evaluator accepts only a complete `ANSWER: <integer>` line. Reports distinguish *answer accuracy* from *output-format adherence* and include challenge commitments, seeds, model identity, generation options when available, report schema, source-tree hash, and per-trial prompt/response hashes. Correct answers are never placed in public evaluation prompts or reported as explicit answer keys.

Training artifacts contain answer keys because their purpose is to preserve demonstrations. On load they must match the current canonical training generator outputs and source-tree hash; a self-computed checksum alone is not treated as authenticity. Experiment runs reject train/eval challenge ID or normalized prompt overlap. The shuffled control now **fails closed** when duplicate answer values prevent a complete label derangement, rather than silently becoming identical to the candidate. The scored model responses in this command's training trajectory are recorded but do **not** causally change its oracle-generated demonstration prompts; this is an in-context experiment, not a feedback-driven learner. Do not tune against the same held-out seed repeatedly and then call it independent evaluation; reserve new seeds for final qualification.

## Actual parameter-update demonstration (toy backend)

`train-toy` trains a four-parameter linear arithmetic policy on tier-1 training cases using normalized SGD. It then compares four instances initialized from the same seed: no update, correct supervised updates, zero-update case visits, and equal-visit supervision with every target value deranged. All four see the identical held-out cases without demonstrations. The command can save the trained checkpoint, and `toy-eval` reloads it in a separate process without training.

The seed-0 fixture with 64 training and 64 held-out cases yields **0/64 baseline, 64/64 trained, 0/64 zero-update, and 0/64 corrupted-label control**. These numbers establish only that this tiny, explicitly engineered parameterization learned a two-operator numerical mapping and retained it across processes. They are **not** evidence of neural-language-model fine-tuning, transfer to tier 4, or general reasoning improvement. Checkpoints preserve bounded coefficients and an integrity digest; neither proves the claimed training history authentic.

The next distinct research gate is a CPU-capable, optional supervised fine-tuning backend for an explicitly supplied local causal LM (for example a parameter-efficient adapter using Transformers/PEFT), with base/tokenizer hashes, optimizer receipts, and fresh-process held-out evaluation. No such LLM training backend or local model installation is claimed here.

See the [research landscape and staged local-training gate](docs/REASONING_TRAINING_LANDSCAPE_2026-10-26.md), its [evidence log and independent review](docs/RESEARCH_LOG_2026-10-26.md), the [experimental qualification protocol](docs/EXPERIMENT_PROTOCOL.md) for learning/transfer controls, and the [independent challenge record](docs/INDEPENDENT_CHALLENGES_2026-10-26.md) for reproduced prototype failures.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

Tests cover seeded determinism, independent register-program interpretation, exact scoring, injection resistance, adapter boundaries, curriculum replay, artifact tampering, valid shuffled controls, separate control-arm adapter instances, bounded untrusted output, toy parameter updates, and fresh-process artifact reload.

Important limitations:

- The replay baseline has no reasoning ability. Accuracy changes from a real model must be measured, not inferred from the harness.
- In-context demonstrations are **not** parameter or optimizer updates, and merely reloading a prompt artifact does not establish persistent model learning.
- Same-seed requests do not guarantee identical output from every runtime. Model filenames, quantization, model-weight hashes, and runtime builds must be recorded separately if strict model identity matters; the harness reports missing weight hashes as unknown.
- The task generator is public and finite. Source/domain overfitting, memorization, stochastic runtime variance, and benchmark gaming remain concerns. For strong claims, reserve unseen generator seeds or distinct problem families, replicate trials, and compare against external benchmarks.
- Controls have unequal tokenization; the placebo is character-length matched only. Some answer multisets cannot be fully deranged, so such comparisons are rejected rather than qualified.
- A fresh Python adapter instance does not prove that an external inference server is stateless.
- Source-tree hashes and training-artifact checksums identify content but **do not authenticate** a repository, trainer, or model creator.

## Code organization

- `labyrinth/maze.py`: seeded challenge generator; evaluator-only answers
- `labyrinth/program_maze.py`: structurally reserved register-program challenge family
- `labyrinth/evaluation.py`: strict, side-effect-free scoring
- `labyrinth/adapters.py`: replay and local Ollama interface
- `labyrinth/curriculum.py`: training-only progression state
- `labyrinth/runner.py`: scored runs, artifacts, four-arm controls, JSON provenance
- `labyrinth/toy_backend.py`: explicitly limited CPU parameter learning and checkpoint reload
- `labyrinth/cli.py`: `run`, `experiment`, `train-toy`, and `toy-eval` commands

Labyrinth deliberately does not install itself into Vera, execute model responses, mutate Git repositories, or claim any externally deployed training effect.
