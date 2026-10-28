# Labyrinth

Labyrinth is an offline-first **reasoning challenge, training-lifecycle, and evaluation harness**. It generates seeded tasks, scores strictly formatted answers, compares in-context controls, and includes a deliberately small CPU-only parameter-learning demonstration.

It is a working **prototype**, not evidence of generalized reasoning improvement. Prompt demonstrations are *persistent prompt memory*, not a model update. The separate toy backend really updates four numeric coefficients and reloads them, but its hand-engineered arithmetic hypothesis class is not an LLM. An **opt-in, dependency-isolated HF/PEFT LoRA pilot** now provides a real neural-weight-update code path. A bounded CPU smoke run on Lappy demonstrated numerical adapter changes and fresh-process reload with a tiny, **randomly initialized** HF model; it did **not** demonstrate improved reasoning.

## New from-scratch Minotaur neural checkpoint (experimental)

Unlike the existing pretrained LoRA studies, \`labyrinth.minotaur_scratch\`
constructs **new random neural weights** for a 114,887-parameter Transformer
first-error verifier. CPU training actually updated **all** of those weights;
the distinct, versioned 451.5 KiB Safetensors checkpoint lives at
[\`models/minotaur_scratch_v1\`](models/minotaur_scratch_v1) with a JSON
training report and independent fresh-process reload.

**Important:** this is an original narrow *neural process verifier*, not a
large language model pretrained from scratch, not a GGUF, and not an effective
Labyrinth gatekeeper. Development accuracy was **43/96** and transfer **20/48**;
false approvals were **39/48** and **22/24** respectively. No graduation,
deployment, or semantic-generalization claim follows.

Details, hashes, source binding and CLI reload instructions:
[\`docs/MINOTAUR_SCRATCH_NEURAL_V1_2026-10-27.md\`](docs/MINOTAUR_SCRATCH_NEURAL_V1_2026-10-27.md).
Core Labyrinth still runs without PyTorch; scratch training and reload require
optional local CPU PyTorch and \`safetensors\` installed. The checkpoint in
this draft is **experimental evidence**, not a production model download.

### Guarded Minotaur v1 ? bounded register-trace advice

\`labyrinth.minotaur_guarded\` connects the **actual scratch checkpoint**
to a second, independent modulo-97 register-program interpreter. A wrong
neural approval can never overrule the independent checker: disagreement,
unknown grammar, malformed inputs, and invalid checkpoint identity cause
**ABSTAIN**, not a pass. This is a deliberately small domain-specific
neuro-symbolic system, **not general reasoning or neural mastery**.

To try it locally, supply a JSON file with exactly \`question\` (the printed
register program) and \`steps\` (its observed trace, as an array of strings):

\`\`\`bash
python -m labyrinth.minotaur_guarded --checkpoint models/minotaur_scratch_v1 --input-json trace.json
\`\`\`

The checkpoint is never downloaded; only the existing local CPU PyTorch and
Safetensors libraries are needed. Output includes \`graduation_authority: none\`
and \`qualification: NOT_ESTABLISHED\`. The independent executable interpreter
does the authoritative arithmetic checking **within its tiny public grammar**.
The neural model may contribute agreement but has no final-exam signing key,
permission to release trainees, or access to production inference routing.
See \`docs/MINOTAUR_GUARDED_TRACE_EVIDENCE_2026-10-27.md\` for all results
and limitations.

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

For tier-four *development* transfer tests, `labyrinth.transfer_slices.generate_transfer_slices(seed, count)` exposes three disjoint, replayable slices: fresh numeric cases on the four known 3–4-operation training structures (`fresh_numbers`), four new 3–4-operation structures (`new_composition`), and four new 5–6-operation structures (`longer_program`). The optional `profile` keyword on `generate_program_challenges` permits explicit profile selection while leaving existing three-argument behavior and generated prompts unchanged. `transfer_commitment` gives public-question/structure hashes without answer keys. These are finite, public generator templates; scores should be grouped by structure fingerprint and **cannot** independently establish broad generalization. The new slices are not yet part of the `train-hf` CLI's evaluation: integration must preserve the same frozen prompts, decoding and arm controls before comparing any model scores.

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

## Optional real language-model LoRA pilot (unqualified runtime)

The `train-hf` and `eval-hf` commands operate on an **already-local**, small Hugging Face causal-LM checkpoint in safetensors format. Unlike `experiment` and `train-toy`, this path loads optional PyTorch, Transformers and PEFT dependencies and attempts a genuine LoRA supervised update of **train-split answer tokens only**. It records trainable tensor changes, saves a separate adapter, compares same-base zero-update and corrupted-label controls, and supports reload/evaluation in a separately started Python process. Reports now count strict response **format validity separately from answer accuracy**, so becoming better at the required answer syntax cannot masquerade as improved mathematics.

```bash
# Optional packages must already exist in your chosen isolated Python environment.
# The model directory must contain local config/tokenizer/safetensors files, <= 1 GiB weights.
python -m labyrinth train-hf --model-path LOCAL_HF_MODEL_DIR --output-dir NEW_EMPTY_PARENT/RUN_001 --device cpu --steps 2
# Invoke separately to qualify persistence; report path must not exist.
python -m labyrinth eval-hf --run-dir NEW_EMPTY_PARENT/RUN_001 --report NEW_EMPTY_PARENT/reload.json
# Offline check of a saved HF pilot report's internally reproducible fields:
python -m labyrinth audit-hf-report --input NEW_EMPTY_PARENT/RUN_001/pilot_report.json
```

This path never downloads weights, installs packages, runs arbitrary model text, or replaces the base checkpoint. It rejects existing output directories and checkpoints exceeding its strict 1 GiB no-quantization limit; therefore the existing ~6.15 GB 3B-class local checkpoint on Lappy is **not eligible** for this initial pilot. Its default two steps consume only two of eight default generated training examples. The public finite mazes are a mechanism smoke test, **not** a qualified learning effect. In the v3 pilot the unchanged-adapter sham now performs the same forward/backward optimizer steps and consumes the same training tokens as the candidate, at zero learning rate, with an exact LoRA tensor-digest equality check. Equal compute does **not** imply an independently blinded semantic assessment.

The separate `.github/workflows/minotaur-pretrained-cpu-pilot.yml` runs a **real pretrained CPU experiment**, triggered by changes to the pilot workflow/code or manually. It stages `HuggingFaceTB/SmolLM2-135M-Instruct` from one immutable revision, validates the known safetensors SHA-256, then disables network model access before invoking the local `train-hf` and fresh-process `eval-hf` commands. Its 32 optimizer visits per sham/candidate/corrupted-label arm use 16 train problems and 16 distinct public development holdouts; the baseline, zero-learning-rate compute-matched sham, learned candidate, and deranged-label control are scored on the same prompts. A dependency-free audit and `scripts/ci_pilot_verdict.py` reject missing updates or non-reproducible adapter reloads. Successful execution produces a seven-day evidence artifact containing the adapter and audit receipts. **Even with measured semantic gains, this job always returns `EVIDENCE_ONLY_DO_NOT_PROMOTE`:** public templates, one seed, and self-attested receipts cannot replace an independently held private examiner or establish general reasoning.

`audit-hf-report` is dependency-free and re-generates the declared train/eval case IDs, held-out prompt commitment, control metadata, training visit order, and reported evaluation metrics. It verifies file/tensor **digest receipt syntax**, but does not open or authenticate the checkpoint files. For responses whose entire text is recoverable from the stored 128-character preview, it independently rechecks the response digest and answer score. Longer responses whose hashes differ from their preview yield `INCOMPLETE`, never an unqualified `PASS`. Inconsistent records yield `FAIL`; the command exits nonzero for both. Source fingerprints must match the auditor checkout, so older reports should be audited at their source revision. All receipts and model-origin claims are self-reported: even `PASS` is **not proof** that training actually occurred or improved reasoning. The original run's full generated text was not always retained; future run schemas should save auditable full text in a separately controlled evidence artifact.

**Executed Lappy CPU smoke (exact source commit `6f7b2924d7f6178623c9d014dece3762890bcf82`, 2026-10-26):** locally generated random tiny GPT-2 model, two PEFT LoRA optimizer updates, changed trainable tensor SHA-256, saved adapter, independent Python process reload and verified equality of the trained/reloaded tensor-value SHA-256. Both processes exited successfully. Held-out scores were **baseline 0/2, trained 0/2, untrained-adapter 0/2, corrupted supervision 0/2**. This is direct evidence of a working neural adapter training/persistence *mechanism*, not improved reasoning, pretrained adaptation or GPU training. See the [exact evidence record](docs/RESEARCH_LOG_2026-10-26.md) and [hostile review](docs/HF_PILOT_HOSTILE_REVIEW_2026-10-26.md). Independent structural/paraphrase holdouts are still needed before higher claims.

**Follow-up on a genuine already-local pretrained checkpoint:** The same exact code revision also loaded `Open-Reasoner-Zero-0.5B` (Qwen2 architecture) fully offline on Lappy's CPU, performed **one genuine LoRA optimizer update**, saved the adapter and reloaded **numerically identical trained tensor values** in a second Python process (both processes exit 0). Accuracy remained **0/2 in every arm**. Baseline and untrained-adapter outputs were both invalid format on each case; the candidate emitted one parseable-but-incorrect answer and one invalid-format answer. This proves a bounded pretrained-model adaptation/checkpoint *mechanism*, not reasoning improvement. Source/model hashes, exact parameters and control results are in [E-017](docs/RESEARCH_LOG_2026-10-26.md). Before any capability claim: precommit a larger blind family, measure format adherence separately, compare equal-budget controls and replicate training seeds.

For an entirely offline **mechanism smoke test only**, `scripts/make_tiny_hf_fixture.py --output-dir NEW_MODEL_DIR` can create a very small randomly initialized causal-LM model and tokenizer using optional installed Torch/Transformers/Tokenizers. Its weights have no pretrained reasoning ability; a successful LoRA update on this fixture proves plumbing, not improvement to an existing capable model. Keep model and run outputs out of Git.

### Minotaur's guardian gate and adversarial development benchmark

`labyrinth.graduation_gate` holds the *qualification rule*, not the private
exam. A separately trusted evaluator must sign a complete, pinned receipt;
Minotaur cannot grade its own certification. `objective_kind="training_effect"`
(the default) now requires statistically supported paired semantic gains over
the frozen pretrained, zero-update and corrupted-label controls, across every
held-out slice. The exact one-sided paired comparison applies Bonferroni
correction across all arms, slices and allowed attempts, in addition to
predeclared accuracy, format and gain thresholds. `"skill_mastery"` is an
explicit alternative for competence when the pretrained model might already
know the skill; it requires a one-sided exact binomial result above the
predeclared accuracy floor and **does not** establish a causal training
improvement. Both paths require held-out and transfer slices with at least 16
cases each and cannot lower their semantic accuracy, format or gain floors to
zero. The choice and significance level are fingerprint-bound to persistent
gate state. All significance statements require genuinely independent cases;
public repeated program patterns cannot supply independent final evidence.
Production receipts must also carry the signed
`score_provenance: "independent_private_exam.v1"` attestation. This is a
fail-closed interface boundary: Minotaur's bounded/public advisory scores have
no qualifying provenance and cannot be silently recast as graduation evidence.
It is not a substitute for custody—the external signer must actually own the
private oracle and credential; software cannot turn a fabricated key or false
assertion into independent examination.

`labyrinth.graduation_state` adds an **opt-in, local custodian-owned** SQLite
WAL store for that trusted `GateState`. `initialize_campaign` creates a new,
exclusive policy-bound campaign; `decide_campaign` serializes signed decisions
with `BEGIN IMMEDIATE`, authenticates stored state with a separate HMAC key,
and durably consumes attempts before reporting an exit. A second process
cannot spend a failed signed receipt again. The caller must keep both keys
outside trainee control and secure/backup the database directory. This module
does **not** run checkpoints, generate genuinely private exam cases, attest
independent structural problem families, or protect against privileged rollback
of the entire database. It is not installed into production or a credential
manager, and does not make Minotaur independently qualified.

`labyrinth.guardian_benchmark.generate_guardian_cases` creates public
development cases with three register-program structure profiles, correct and
convincing incorrect trainee answers in matched styles, plus injected claims
that a judge should mark answers correct. Its process counterpart,
`generate_process_cases`, creates paired valid/faulty execution traces and
asks the guardian to identify the **first incorrect step**, as in the
ProcessBench research problem. The associated evaluators report false
approvals, false rejections, abstentions, coverage and slice metrics; neither
grants qualification. Public answers must **not** be used for a blind final
exam. See the [outside-source research mapping and evidence limits](docs/MINOTAUR_GUARDIAN_EXTERNAL_RESEARCH_2026-10-27.md).

For a real pretrained **correctness-critic** rather than a maze-answering
adapter, `labyrinth.minotaur_critic_pilot` provides a strictly optional,
offline PEFT/LoRA experiment. It trains Yes/No judgments of another model's
proposed answers against matched frozen, zero-update, and corrupted-label
controls, measures false approvals on identical development and structurally
reserved transfer cases, and requires an unchanged adapter with identical
predictions after **fresh-process** reload. A same-process or mismatching
reload fails closed. A **real pretrained CPU pilot** subsequently ran two
candidate LoRA optimizer updates, verified an unchanged zero-update control,
and reloaded the adapter in a separate process. However, development accuracy
fell from **5/8 to 4/8** and transfer remained **2/4**: the mechanism works,
but reasoning improvement and generalization are NOT ESTABLISHED. See the
[critic execution protocol](docs/MINOTAUR_CORRECTNESS_CRITIC_PILOT_2026-10-27.md)
and [live Lappy run evidence E-018](docs/MINOTAUR_CRITIC_LAPPY_RUN_2026-10-27.md).
A [separate E-021 alternate-objective experiment](docs/MINOTAUR_DIAGNOSIS_AND_PROCESS_PILOTS_2026-10-27.md)
recorded real pretrained logit-margin, first-error *claim* and binary-loss
CPU runs. Its experimental implementation lives in a [separate research
branch](https://github.com/thebrazenbeard/labyrinth/tree/bt2/minotaur-diagnostics-and-windows-20261027),
**not** in this Minotaur draft or production. These runs also failed to
demonstrate useful verifier discrimination or safe graduation.
The follow-up v2 pairs right/wrong answers within every update, but its real
two- and eight-update runs **still** approved every incorrect development and
transfer answer. See [E-019](docs/MINOTAUR_CRITIC_PAIRED_LAPPY_RUN_2026-10-27.md).

`labyrinth.processbench_adapter` can score a reviewer on caller-supplied,
strictly validated **Qwen/ProcessBench** JSONL records, with correct one-based
first-error translation, class-balanced metrics and false-approval rates.
It also reports whether judgments on caller-asserted logically equivalent
reformulations remain consistent **and correct**. It does not fetch or bundle
the official benchmark, and invented fixture tests are not a real benchmark
score. See the [external benchmark adapter provenance and protocol](docs/PROCESSBENCH_EXTERNAL_ADAPTER.md).

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
- `labyrinth/hf_pilot.py`: opt-in local HF/PEFT LoRA SFT and same-checkpoint controls; no default ML dependency
- `labyrinth/hf_report_audit.py`: dependency-free self-consistency audit of HF pilot receipts and available predictions
- `labyrinth/cli.py`: `run`, `experiment`, `audit-report`, `train-toy`, `toy-eval`, `train-hf`, and `eval-hf` commands

Labyrinth deliberately does not install itself into Vera, execute model responses, mutate Git repositories, or claim any externally deployed training effect.
