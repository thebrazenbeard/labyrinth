# Pretrained neural trace-SFT experiments — evidence report

Status: **real neural training and checkpoint reload PASS; reasoning transfer FAIL
on the evaluated development sets.** This is not a final holdout qualification.

## Method and provenance

- Source lineage: hardened Labyrinth `e717f5eab139ae9226bf08a793a794177732c001`,
  post-hardening Draft PR #4, then stacked Draft PR #6.
- Models loaded offline, `trust_remote_code=False`, from local Hugging Face
  safetensors and tokenizers. All runs used CPU PyTorch 2.8.0, Transformers
  4.57.1, AdamW, learning rate 0.0001, seed recorded in each receipt, four CPU
  threads, no prompt demonstrations.
- Only the *last existing pretrained transformer block* was unfrozen. AdamW
  updated neural weights, not a four-parameter toy; sham arms called the
  optimizer without a gradient update. Whole-model checkpoints and receipts
  were saved and independently reloaded from fresh Python processes.
- No evaluator-private answers or seeds were copied to the training host.
  The actual public prompts, raw predictions, frozen arm-to-checkpoint
  identity maps and private oracles are retained separately in evaluator custody.
- Score is strict last-line `ANSWER: <integer>`, with separate format, full
  trace and representation statistics. Trained trace targets now explicitly
  request `STATE N: A=x, B=y` followed by an `ANSWER` line; evaluation
  prompts still request a final answer, so these are transfer tests rather
  than demonstrations of trace imitation under a trace-request prompt.

## 135M model: completed failures

SmolLM2-135M-Instruct: 134,515,008 pretrained parameters, 3,540,096 trainable,
32 optimizer calls/case visits per arm. Base-model SHA-256:
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Train fixture SHA-256:
`1ddde0f4831617e53e6280ef78578ea018f732cfa5fb15ad8d34290cec841c84`.

| Arm | Dev1 exact /16 | Dev1 parseable /16 | Dev2 exact /24 | Dev2 parseable /24 |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 0 | 0 | 0 | 0 |
| Answer-only SFT | 0 | 11 | 0 | 9 |
| Trace SFT (original conflicting prompt in Dev1) / corrected in Dev2 | 0 | 5 | 0 | 9 |
| Zero-update sham | 0 | 0 | 0 | 0 |
| Corrupted trace | 0 | 6 | 0 | 9 |

All Dev1/Dev2 arms had zero correct full intermediate-state traces.
**Dev1 and Dev2 trace-SFT are distinct runs; this table is not a
within-checkpoint longitudinal comparison.** Source evaluations were
independently scored against precommitted arm identities.

Dev2 public prompt SHA-256:
`5fbbbf96f2e4d9a222dc69a4a4d1573dca4da03883af44e9f8e179df1d617c48`.
Dev2 independent five-arm aggregate score SHA-256:
`5ff5f6c82d496c724c69e7eb88cab70fb741be4dfce4ac5f86fdde54897af88a`.
The 24 Dev2 cases comprise eight easy one-/two-step, eight structurally novel
three-/four-step, and eight longer/representation-shifted programs.

## 360M model: longer real neural pilot

SmolLM2-360M-Instruct: 361,821,120 pretrained parameters, 9,832,320
trainable. Base-model SHA-256:
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
Deterministic train-only 128-case seed-1807 corpus:
`a18d0a9dcf9e7c3a8f4803e3582315aa30c31964edd67b1b9559a7d0981c6ad0`.
The corrected prompt/task contract and full-tokenization-prefix alignment
were checked for all 128 cases before training.

| Arm | Gradient steps | Trainable layer after SHA-256 | Checkpoint SHA-256 | Reload |
| --- | ---: | --- | --- | --- |
| Correct trace | 128 | `d97633c17749b09797f1924d530ffef007bb657c302a125775ddff1e079fc51a` | `337e9cbdb99a977e38da5471e4816a345ca06cc6c66eb808a45abbbaf0db1068` | PASS |
| Sham | 0 | `22d5a228c18ed0361e0efe7fbb61871b0bf71b5d7b3d16a314dbaa4794d55d3f` | `d387d4323e5314a3d8f37b4ca628776a16a5963169eb775ed70a201bee5a7544` | PASS |
| Corrupted trace | 128 | `71c3ceb79ec0ab550ea263bfbabec00a88f916acb259fa13203c7ff14fce3c84` | `c6804196e6ec48d83f95ad631971cc6cf79d21b88dab1ce87a67d23f2fcca957` | PASS |

The unmodified base and sham share the same last-block SHA-256
`22d5a228c18ed0361e0efe7fbb61871b0bf71b5d7b3d16a314dbaa4794d55d3f`.
Correct-trace training covered 7,145 supervised tokens in 153.923s;
sham 7,145 tokens / 138.185s, corrupted 7,139 tokens / 154.851s.
These are case-/step-matched, **not exactly token-/compute-matched** arms.

The same declared Dev2 24 public prompts and greedy 96-token budget yielded:

| Arm | Exact final answers /24 | Parseable final answers /24 | Correct full traces |
| --- | ---: | ---: | ---: |
| Frozen base (exploratory baseline) | 0 | 0 | 0 |
| Correct trace SFT | 0 | 1 (wrong) | 0 |
| Corrupted trace SFT | 0 | 0 | 0 |

The independent evaluator found 56 syntactically valid STATE rows from
the correctly trained model and 61 from corrupted training across raw
outputs, but **zero matching a gold intermediate state** in either arm.
There were no correct answers in the easy, structural, or long/shift slices.
The sham checkpoint had verified zero neural changes but no separate
96-token inference arm; do not claim a four-arm 360M prediction comparison.
The baseline was generated before its arm-map precommit, and Dev2 had been
used in an earlier 135M analysis, so this is **development evidence only**.

The three-arm 360M checkpoint identity map SHA-256
`23f715eb29f2340497ed6b8610759a1ebe352582f099471dcb9ead8e79a4aada`
was appended to the authenticated Blotter ledger before trained and
corrupted predictions. The baseline's earlier timing was explicitly disclosed.

## Falsifiers and remaining gates

> Correct training labels reduced loss and changed millions of pretrained
> weights, but no scored novel register program was solved. The data do not
> distinguish emergent reasoning from response-pattern imitation.

1. The train generator uses four program structures; dev structures/wording
   change considerably. Broaden the train curriculum, but retain independently
   sealed structure and representation holdouts.
2. Equal gradient steps do not equalize supervised tokens, backward compute
   or output-token budgets. Predeclare controls and report each budget.
3. The model can emit `STATE`-like syntax without executing the correct
   transition. Evaluate intermediate trajectories under a separately frozen
   explicit trace request.
4. Test multiple training seeds and a competence-calibrated task ladder
   before any final set; Dev2 has already served model selection.
5. A more capable pretrained model and a verified accelerated runtime on
   Lappy may be needed. Lappy's observed CUDA PyTorch was unavailable at
   this check; no installation or active GPU-service change was made.
6. Checkpoint persistence/reload is proven for these concrete host artifacts,
   not proof of deployment, continual learning, general intelligence, or
   reasoning improvement.
