# Labyrinth

Labyrinth is an offline-first **reasoning challenge, training-lifecycle, and evaluation harness**. It generates seeded tasks, scores strictly formatted answers, compares in-context controls, and includes a deliberately small CPU-only parameter-learning demonstration.

It is a working **prototype**, not evidence of generalized reasoning improvement. Prompt demonstrations are *persistent prompt memory*, not a model update. The separate toy backend really updates four numeric coefficients and reloads them, but its hand-engineered arithmetic hypothesis class is not an LLM. An **opt-in, dependency-isolated HF/PEFT LoRA pilot** now provides a real neural-weight-update code path. A bounded CPU smoke run on Lappy demonstrated numerical adapter changes and fresh-process reload with a tiny, **randomly initialized** HF model; it did **not** demonstrate improved reasoning.

## Run it

Requires Python 3.10+; core functionality uses the standard library.

```bash
python -m labyrinth run --seed 42 --count 3 --tiers 1,2,3 --adapter replay
python -m labyrinth run --seed 42 --count 3 --tiers 4 --adapter replay
python -m labyrinth experiment --seed 42 --train-count 3 --eval-count 3 --artifact-out training.json --report evaluation.json
python -m labyrinth experiment --seed 42 --eval-count 3 --artifact-in training.json --report rerun.json
python -m labyrinth train-toy --seed 0 --train-count 64 --eval-count 64 --checkpoint-out toy-checkpoint.json --report toy-comparison.json
python -m labyrinth toy-eval --seed 0 --eval-count 64 --checkpoint toy-checkpoint.json --report toy-reloaded.json
python -m labyrinth journey-plan --scope examples/scopes/empathy_declaration.json --cycle 0 --report planned-maze.json
python -m labyrinth journey-plan --objective-contract examples/objectives/empathy_observable_declaration.json
python -m labyrinth journey-plan --scope examples/scopes/program_specificity5_declaration.json --materialize-training
python -m labyrinth journey-toy --scope examples/scopes/arithmetic_toy_declaration.json --epochs-by-cycle 4,4,4 --report toy-journey.json
python -m labyrinth journey-toy-controls --scope examples/scopes/arithmetic_toy_declaration.json --epochs-by-cycle 4,4,4
python -m unittest discover -s tests -v
```

`--count`, `--train-count`, and `--eval-count` mean tasks **per tier**. Reports are JSON; existing paths are never overwritten. CLI-supplied replay JSON and saved demonstration artifacts are limited to **8 MiB per input file** before parsing, including files that change during the read. Without a specified model, `replay:constant-zero` is a deterministic negative-control fixture that outputs `ANSWER: 0` for every question. It does not solve problems or learn. To get nontrivial scores offline, pass `--replay-file mapping.json`, where the JSON object maps exact public prompt text to fixed answer strings. Missing fixture keys fail closed.

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

### Audit a saved report offline

```bash
python -m labyrinth run --seed 42 --count 3 --tiers 1,2,3 --report maze.json
python -m labyrinth audit-report --input maze.json
# For a four-arm experiment, provide the original demonstration artifact:
python -m labyrinth audit-report --input evaluation.json --artifact training.json --report audit.json
```

The auditor re-generates public challenges, rechecks the answer commitment, identifiers, seeds, prompt hashes, complete response hashes, scores, and aggregated accuracy/format measures. It also verifies the training-artifact and control metadata when provided. CLI audit inputs reject duplicate JSON keys, nonfinite JSON numbers and files over 8 MiB; audit output paths are write-once. An exit code of **0** means `PASS` (all available internal checks), and **2** means `FAIL` or `INCOMPLETE`. In particular, four-arm reports without the original training artifact are `INCOMPLETE`, as are reports containing responses longer than the stored 4,096-character preview. Source-revision mismatch fails closed.

**Critical evidence limit:** `PASS` means only *internal consistency against the current public generator and scorer*. The self-reported model identity and unsigned output are not authenticated; a third party can fabricate a completely consistent report without running a model. This command does **not** establish trained-model improvement, independent runtime execution, or blind evaluation.

Training artifacts contain answer keys because their purpose is to preserve demonstrations. On load they must match the current canonical training generator outputs and source-tree hash; a self-computed checksum alone is not treated as authenticity. Experiment runs reject train/eval challenge ID or normalized prompt overlap. The shuffled control now **fails closed** when duplicate answer values prevent a complete label derangement, rather than silently becoming identical to the candidate. The scored model responses in this command's training trajectory are recorded but do **not** causally change its oracle-generated demonstration prompts; this is an in-context experiment, not a feedback-driven learner. Do not tune against the same held-out seed repeatedly and then call it independent evaluation; reserve new seeds for final qualification.

## Actual parameter-update demonstration (toy backend)

`train-toy` trains a four-parameter linear arithmetic policy on tier-1 training cases using normalized SGD. It then compares four instances initialized from the same seed: no update, correct supervised updates, zero-update case visits, and equal-visit supervision with every target value deranged. All four see the identical held-out cases without demonstrations. The command can save the trained checkpoint, and `toy-eval` reloads it in a separate process without training.

The seed-0 fixture with 64 training and 64 held-out cases yields **0/64 baseline, 64/64 trained, 0/64 zero-update, and 0/64 corrupted-label control**. These numbers establish only that this tiny, explicitly engineered parameterization learned a two-operator numerical mapping and retained it across processes. They are **not** evidence of neural-language-model fine-tuning, transfer to tier 4, or general reasoning improvement. Checkpoints preserve bounded coefficients and an integrity digest; neither proves the claimed training history authentic.

## Regenerative, objective-conditioned Labyrinth (development prototype)

Patrick's complete intended learner/maze/Minotaur behavior, including the empathy example and what it means to retain learning when a maze resets, is preserved in [REGENERATIVE_LABYRINTH_VISION_V1.md](docs/REGENERATIVE_LABYRINTH_VISION_V1.md). The architecture is **declare → generate → train → development-examine → regenerate or provisionally pass**. The independent, signed private graduation gate remains a separate authority.

`journey-plan` parses a bounded, immutable JSON declaration with an objective, domain, specific skills, specificity, scale, a maximum cycle budget, and seed. It deterministically schedules training chambers for each skill. Failed-skill feedback can increase the next cycle's chamber allocation, and each regeneration changes room identities and seeds. A declaration is data; it is not evaluated as code. `--objective-contract` accepts an alternative declaration describing the intention, per-skill focus tags, observable behavioral targets, provisional rubric identifiers, thresholds and forbidden claims. These are not executable evaluators. `journey-plan` previews a maze but never trains a model.

Opt-in `journey-plan --materialize-training` yields only trainee-visible training prompts, **not answers or private exams**. For an arithmetic-only scope a deterministic bounded Minotaur routing policy selects actual difficulty and *practice/contrast/word-problem* tactics. An operator-trusted adapter may propose alternative choices through `route_training_rooms`; two distinct adapter implementations have been tested, but this is not a trained general-purpose Minotaur. For a `program_reasoning` scope the approved training-only source restricts actual program structure by specificity: from mixed 3–4 step programs at level 1 to cross-register programs with modular wraparound on both registers at level 5. Other built-in skills retain ordinary training providers, and empathy remains unsupported for executable training. `--materialize-training` will reject unsupported skills.

`journey-toy` demonstrates the bounded end-to-end control loop **with actual changes to a four-parameter arithmetic learner across maze resets**, fresh seeded development cases, a frozen zero-update baseline, and per-cycle checkpoint identities. `journey-toy-controls` runs fixed-cycle baseline, correct-label, zero-update and corrupted-label arms from identical starting values, with identical case visits and matched development problems. On one deterministic fixture the correct-label arm improves while controls do not; this is a narrow arithmetic result, not a model-generalization claim. Both commands accept only an arithmetic-only declaration and never produce pretrained-model checkpoints. Development scores and model hashes are local self-reports, not independent attestation. An exhausted budget is an expected, valid outcome, not a graduation failure to conceal. All three commands create output reports exclusively: existing report files are not overwritten.

The optional `TrainingCorridorRunner` now actually generates one **arithmetic TRAIN** chamber at a time after the previous response has been scored. An `AdaptiveWalkState` shifts training emphasis after feedback; the persistent `MinotaurTrainingPolicy` learns bounded room difficulty/tactic preferences from approved TRAIN outcomes. A learner callback sees only a prompt-only `TraineeRoomView`. Every case must match a preapproved exact room/case/skill/tactic/difficulty/prompt commitment. The policy is bound to the entire declared scope fingerprint, and only a validated `JourneyState` containing prior train/development-exam seals can start a regenerated corridor. Policy registry or checkpoint-hash changes alone are **not** evidence of neural or semantic learning. This is an in-repo **tabular taskmasker**, not Minotaur's trained general-purpose neural model; empathy has no executable provider yet.

`journey_qualification.py` provides a separate *identity-only* handoff from an internal developmental Minotaur pass to the existing signed private graduation gate. It does not expose private questions, provision an evaluator, or turn a trainer's self-reported progress into certified mastery. Independent evaluator custody and actual transfer performance must still be established.

The `Minotaur` developmental examiner operates on untrusted score receipts and enforces per-skill, per-split coverage, paired improvement, absolute performance, and nonreused evidence commitments. Its optional model suggestion interface validates proposed tactics and difficulty, and the **opt-in arithmetic routing generator consumes those suggestions** to change training problems; it is **not** an autonomous trained Minotaur model and is not used by the toy training journey. Operator-installed Python adapters execute with caller permissions (no sandbox). Unsupported domains such as empathy can be declared and planned, but are not trainable until a validated challenge provider, rubric, learning backend, and adversarial evaluation are supplied.

**Qualification gap:** structural specificity is implemented only for one program-reasoning family; it does not yet apply to arbitrary domains or individual intra-maze feedback. There is no checkpoint-resumed HF LoRA/MeZO path across journeys, no trained general Minotaur weights, no evidence of broad transfer or retained old competencies, and no independent private graduation result. The source implementation is a testable development slice toward the full vision, **not completion of that vision**.

## Optional real language-model LoRA pilot (unqualified runtime)

The `train-hf` and `eval-hf` commands operate on an **already-local**, small Hugging Face causal-LM checkpoint in safetensors format. Unlike `experiment` and `train-toy`, this path loads optional PyTorch, Transformers and PEFT dependencies and attempts a genuine LoRA supervised update of **train-split answer tokens only**. It records trainable tensor changes, saves a separate adapter, compares same-base zero-update and corrupted-label controls, and supports reload/evaluation in a separately started Python process. Reports now count strict response **format validity separately from answer accuracy**, so becoming better at the required answer syntax cannot masquerade as improved mathematics.

```bash
# Optional packages must already exist in your chosen isolated Python environment.
# The model directory must contain local config/tokenizer/safetensors files, <= 1 GiB weights.
python -m labyrinth train-hf --model-path LOCAL_HF_MODEL_DIR --output-dir NEW_EMPTY_PARENT/RUN_001 --device cpu --steps 2
# Opt-in cycle 0 under a predeclared, arithmetic-only objective:
python -m labyrinth train-hf --model-path LOCAL_HF_MODEL_DIR --output-dir NEW_EMPTY_PARENT/SCOPED_0 --device cpu --steps 8 --tiers 1 --scope-json examples/scopes/arithmetic_toy_declaration.json --cycle-index 0
# Invoke separately to qualify persistence; report path must not exist.
python -m labyrinth eval-hf --run-dir NEW_EMPTY_PARENT/RUN_001 --report NEW_EMPTY_PARENT/reload.json
# Offline check of a saved HF pilot report's internally reproducible fields:
python -m labyrinth audit-hf-report --input NEW_EMPTY_PARENT/RUN_001/pilot_report.json
```

This path never downloads weights, installs packages, runs arbitrary model text, or replaces the base checkpoint. It rejects existing output directories and checkpoints exceeding its strict 1 GiB no-quantization limit; therefore the existing ~6.15 GB 3B-class local checkpoint on Lappy is **not eligible** for this initial pilot. Its default two steps consume only two of eight default generated training examples. The public finite mazes are a mechanism smoke test, **not** a qualified learning effect. New runs from the v3 pilot use a **zero-learning-rate, same-exposure placebo**
that executes the same number of forward/backward and optimizer steps and the
same deterministic tier-balanced training-case order. The source verifies its
LoRA tensors **and persistent model buffers** are unchanged, failing closed
rather than claiming matched initial conditions if stateful model buffers
change during any arm. This is better matched for training operations and
example visits but is *not* proof of identical kernel timing, private test
independence, or semantic learning. Historical v2 reports retain the original
unmatched sham meaning. The opt-in --answer-only mode moves the fixed
ANSWER: scaffold into the prompt, where it receives no supervised loss.
Only answer tokens and EOS receive targets. In that mode format_valid
means the model emitted a parseable numeric suffix after an operator-provided
scaffold, not that the model independently learned output formatting. This
separates numeric correctness from the trivial constant format token but does
not make public finite arithmetic templates an independent reasoning exam.

The new opt-in scoped HF path accepts an immutable training declaration and materializes a deterministic TRAIN-only maze. After a first cycle, its `--predecessor-run-dir`, `--predecessor-report-sha256`, `--predecessor-lora-sha256`, `--resume-authority-ref`, `--cycle-index` and same `--scope-json` flags ask to continue **the verified prior LoRA tensor values**, with newly generated public training cases. The predecessor must be a locally preserved, matching v3 report and adapter; mismatched base/source/scope or skipped cycle indices fail closed. A previous unscoped v3 report needs an explicit `--allow-legacy-unscoped-predecessor` at cycle one, recorded as an operator-selected transfer, **not** evidence of prior same-objective learning. A scoped successor now **fails closed by default** at the `run_pilot` API and CLI: the signed and atomically reserved development-FAIL path is `run_exam_bound_successor`, while `--unauthenticated-research-resume` must be **explicitly** requested for an unsigned mechanical experiment and records `UNAUTHENTICATED_RESEARCH_ONLY`. The two-cycle `scripts/qualify_hf_regenerative.py` is such a research-only mechanical probe, **not** a Minotaur-authorized successor. The source-level, in-process guard does not defend against arbitrary trainer-code execution, prove external evaluator key custody, or durably prevent host event replay. The pilot records cumulative public training/diagnostic case commitments; it does **not** establish custody of independent Minotaur exam items. Each sham, candidate, and corrupted-label arm starts from identical retained LoRA values, but optimizer moments are *not* resumed. These interfaces have source-level/mock tests; no new multi-cycle pretrained host execution or qualified improvement is claimed.

The opt-in `labyrinth.hf_development_event_journal` gives an operator-owned
SQLite/WAL reservation callback for the separately signed development-FAIL
successor entry point (`run_exam_bound_successor`): initialize a journal once
with `initialize_development_event_journal`, then provide a callback that calls
`reserve_development_event(path, event_id, state_key=host_secret)`. A successful
reservation is atomically committed **before** successor training begins;
subsequent calls, including after a simulated post-reservation crash, reject
the same event. `inspect_development_event` and `reconcile_development_event`
record at most one bounded **operator observation**, never a verified neural
result, event release, or graduation. Host filesystem custody, off-trainee
state-MAC key storage, whole-database rollback prevention, and real crash/host
recovery still require external qualification. This does not automatically
guard the separately exposed experimental `hf_pilot` entry point.

`audit-hf-report` is dependency-free and re-generates the declared train/eval case IDs, held-out prompt commitment, control metadata, training visit order, and reported evaluation metrics. It verifies file/tensor **digest receipt syntax**, but does not open or authenticate the checkpoint files. For responses whose entire text is recoverable from the stored 128-character preview, it independently rechecks the response digest and answer score. Longer responses whose hashes differ from their preview yield `INCOMPLETE`, never an unqualified `PASS`. Inconsistent records yield `FAIL`; the command exits nonzero for both. Source fingerprints must match the auditor checkout, so older reports should be audited at their source revision. All receipts and model-origin claims are self-reported: even `PASS` is **not proof** that training actually occurred or improved reasoning. The original run's full generated text was not always retained; future run schemas should save auditable full text in a separately controlled evidence artifact.

**Executed Lappy CPU smoke (exact source commit `6f7b2924d7f6178623c9d014dece3762890bcf82`, 2026-10-26):** locally generated random tiny GPT-2 model, two PEFT LoRA optimizer updates, changed trainable tensor SHA-256, saved adapter, independent Python process reload and verified equality of the trained/reloaded tensor-value SHA-256. Both processes exited successfully. Held-out scores were **baseline 0/2, trained 0/2, untrained-adapter 0/2, corrupted supervision 0/2**. This is direct evidence of a working neural adapter training/persistence *mechanism*, not improved reasoning, pretrained adaptation or GPU training. See the [exact evidence record](docs/RESEARCH_LOG_2026-10-26.md) and [hostile review](docs/HF_PILOT_HOSTILE_REVIEW_2026-10-26.md). Independent structural/paraphrase holdouts are still needed before higher claims.

**Follow-up on a genuine already-local pretrained checkpoint:** The same exact code revision also loaded `Open-Reasoner-Zero-0.5B` (Qwen2 architecture) fully offline on Lappy's CPU, performed **one genuine LoRA optimizer update**, saved the adapter and reloaded **numerically identical trained tensor values** in a second Python process (both processes exit 0). Accuracy remained **0/2 in every arm**. Baseline and untrained-adapter outputs were both invalid format on each case; the candidate emitted one parseable-but-incorrect answer and one invalid-format answer. This proves a bounded pretrained-model adaptation/checkpoint *mechanism*, not reasoning improvement. Source/model hashes, exact parameters and control results are in [E-017](docs/RESEARCH_LOG_2026-10-26.md). Before any capability claim: precommit a larger blind family, measure format adherence separately, compare equal-budget controls and replicate training seeds.

For an entirely offline **mechanism smoke test only**, `scripts/make_tiny_hf_fixture.py --output-dir NEW_MODEL_DIR` can create a very small randomly initialized causal-LM model and tokenizer using optional installed Torch/Transformers/Tokenizers. Its weights have no pretrained reasoning ability; a successful LoRA update on this fixture proves plumbing, not improvement to an existing capable model. Keep model and run outputs out of Git.

The next gate is executing this optional path in an isolated environment with independently measured resource use, immutable model/tokenizer identity and a matched repeated-seed experiment. Do not infer from the code or changed adapter bytes that a pretrained language model became better at reasoning.

See the [research landscape and staged local-training gate](docs/REASONING_TRAINING_LANDSCAPE_2026-10-26.md), its [evidence log and independent review](docs/RESEARCH_LOG_2026-10-26.md), the [HF pilot hostile review](docs/HF_PILOT_HOSTILE_REVIEW_2026-10-26.md), the [experimental qualification protocol](docs/EXPERIMENT_PROTOCOL.md) for learning/transfer controls, and the [independent challenge record](docs/INDEPENDENT_CHALLENGES_2026-10-26.md) for reproduced prototype failures.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

Tests cover seeded determinism, independent register-program interpretation, exact scoring, injection resistance, adapter boundaries, curriculum replay, artifact tampering, bounded input JSON and untrusted output, saved-report scoring/provenance self-consistency checks, valid shuffled controls, separate control-arm adapter instances, toy parameter updates, fresh-process artifact reload, and optional HF pilot preflight, train/eval split separation, label derangement and prompt-loss masking. The HF unit tests exercise the **contract**, not an actual backward pass.

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
- `labyrinth/report_audit.py`: offline report consistency verifier; neither provenance authentication nor a learning claim
- `labyrinth/toy_backend.py`: explicitly limited CPU parameter learning and checkpoint reload
- `labyrinth/training_scope.py`: frozen, bounded declarations; no executable objective content
- `labyrinth/objective_contract.py`: per-skill human-authored intent, focus and evaluation targets; no executable rubrics
- `labyrinth/regenerative_maze.py`: deterministic per-cycle maze scheduling and weak-skill rebalancing
- `labyrinth/journey_challenges.py`: strict, approved training-only challenge providers
- `labyrinth/semantic_chambers.py`: bounded actual structural specificity for program training challenges
- `labyrinth/journey.py`: regenerative state machine and checkpoint/holdout separation
- `labyrinth/minotaur_exam.py`: development mastery decision, never final graduation
- `labyrinth/minotaur_model.py`: bounded operator-trusted taskmaster suggestion interface
- `labyrinth/taskmaster_routing.py`: opt-in arithmetic room generation driven by validated taskmaster suggestions
- `labyrinth/journey_toy.py`: arithmetic toy-model journey with retained parameter updates
- `labyrinth/journey_controls.py`: matched-exposure toy control arms across regenerated cycles
- `labyrinth/adaptive_corridors.py`: TRAIN-only per-chamber feedback and prior development-exam exclusion
- `labyrinth/minotaur_policy_learning.py`: persistent approved-TRAIN tabular taskmaster with exact-scope binding
- `labyrinth/journey_policy_runner.py`: one-by-one arithmetic room generation from learned taskmaster feedback
- `labyrinth/journey_qualification.py`: public identity handoff to independent signed graduation custody
- `labyrinth/hf_pilot.py`: opt-in local HF/PEFT LoRA SFT and same-checkpoint controls; no default ML dependency
- `labyrinth/hf_resume_bridge.py`: source- and tensor-pinned trainable PEFT continuation between scoped cycles
- `labyrinth/hf_report_audit.py`: dependency-free self-consistency audit of HF pilot receipts and available predictions
- `labyrinth/cli.py`: `run`, `experiment`, `audit-report`, `train-toy`, `toy-eval`, `train-hf`, and `eval-hf` commands

Labyrinth deliberately does not install itself into Vera, execute model responses, mutate Git repositories, or claim any externally deployed training effect.

### Expanded but bounded semantic research budget

The pretrained LoRA pilot allows up to 128 steps (previously 32) with a
hard cap of 64 total training examples and at most 1 GiB of local safetensors
weights. The matched v3 auditor reconstructs the full tier-balanced sequence
for extended runs; historical v2 reports retain the original 32-step bound.
This changes only permitted experimental duration, **not** the standard for
graduation, independent evaluation, or model deployment. A higher optimizer
step count is not itself evidence of semantic learning.


The enhanced guarded training loop now value-hashes all frozen neural
parameters and persistent buffers across *training and inference*, not just
trainable LoRA parameters. A finite loss accompanied by a nonfinite gradient
norm is rejected before optimizer stepping. These checks detect mutations
at their measurement boundaries; they do not authenticate a hostile runtime
or prevent all concurrent in-memory modification races.
