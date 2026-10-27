# Minotaur correctness-critic — real pretrained CPU pilot E-018 (2026-10-27)

**Evidence:** Live Windows workstation process output, independently read back
from two locally persisted JSON reports; source/unit tests and separate-process
reload. This is NOT an independent reasoning examination or a model-improvement
qualification. Raw model weights and run artifacts remain outside Git.

## Exact subject and controlled environment

- Labyrinth published Draft PR #20 head: `c749f9497fe2f182a6e2085767d3a95f2d7bfc45`.
- Git tree: `e9e33031bebcd020c74e41f92ac33429440e1fa1`.
- Source-state SHA-256 reported by the actual run:
  `cc18050d1ee9da8166447ec39527750ff555eb6e04a35e91455913b697985238`; clean Git checkout.
- Pretrained local donor: `Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B`,
  Qwen2 causal LM. Local `model.safetensors`: 988,097,824 bytes,
  SHA-256 `f3fabf5933c996348e19cc65f2eaf4ceb4a036cb9981d2da1142c2e8b4020938`.
  Local provenance is described in earlier E-017; the directory and hash
  alone do not independently prove the original model's training history.
- Device: Lappy Windows 11 CPU, no Torch CUDA. Python `3.14.5`, Torch
  `2.14.1+cpu`, Transformers `5.14.1`, PEFT `0.19.1`, deterministic
  Torch algorithms enabled. No dependency or checkpoint download.
- The tokenizer mapped ` Yes` and ` No` to distinct single tokens `7414`
  and `2308`, meeting the critic preflight.
- Fresh private experiment directory under `D:\VERA\experiments`; not a
  production model location. Vera identity/inference route unchanged.

## Exact bounded protocol

The train process imported `CriticConfig, run_critic` from the published
`labyrinth.minotaur_critic_pilot` and called:

```python
config = CriticConfig(
    model_path="D:/VERA/models/donors/huggingface/Open-Reasoner-Zero-0.5B",
    output_dir="D:/VERA/experiments/minotaur-critic-20261027-run1",
    steps=2,
    eval_count_per_tier=2,
)
report = run_critic(config)
```

All other config fields remained at module defaults: seed 777;
train/dev/transfer seeds 1357/2468/3579; 24 training tasks per tier;
384-token cap; learning rate 0.0003; LoRA rank 4/alpha 8; CPU.
Each training arm received exactly two forward/backward/AdamW steps from
identical LoRA initialization, with candidate supervision, a zeroed-gradient
sham, or inverted labels. Baseline was frozen. The same development and
reserved tier-4 transfer problems were scored in all four arms. The
training-run process exited 0 after approximately 53 seconds.

A **different Python process** invoked
`reload_critic(run_dir, new_reload_report_path)`, exited 0 after approximately
21 seconds, and verified distinct process IDs, unchanged adapter tensor values,
and exact reproduction of evaluation metrics/predictions.

## Observed metrics — counts matter

| Arm | Development (8 judgments) | Transfer (4 judgments) | Development false approvals | Transfer false approvals |
|---|---:|---:|---:|---:|
| Frozen pretrained | 5/8 = 62.5% | 2/4 = 50% | 3/4 | 2/2 |
| Correct-label candidate | 4/8 = 50% | 2/4 = 50% | 4/4 | 2/2 |
| Zero-update sham | 5/8 = 62.5% | 2/4 = 50% | 3/4 | 2/2 |
| Corrupted-label | 4/8 = 50% | 2/4 = 50% | 0/4 | 0/2 |

The candidate predicted **Yes on every held-out verdict**; the
corrupted-label control predicted **No on every verdict**. This is a warning
of decision collapse under tiny-budget, one-label-per-task training, not
evidence of improved verification or transfer. The corrupted arm trades
away all false approvals by falsely rejecting every correct answer.

### Adapter and report receipts

- Trainable adapter SHA-256 before updates:
  `f4a000673d03e8abc09f31058feb5a8305b983c962836b734d673c3152a2245d`.
- Candidate after two genuine AdamW updates:
  `f65a4a13060f044cae9d964fc3beea356bc51580a3215dc870abee1b558b1e9e`.
- Zero-update after two matched optimizer calls: **identical to initial**.
- Corrupted-label after two updates:
  `79b0e93d017d7fac1caff6ae6112ffe2af1142f50525df984e88ce0481db2736`.
- Persisted candidate `adapter_model.safetensors` file SHA-256:
  `836375415c600f2267139107994c330d79c3fa384dc4c080516bbac862dd8077`.
- Fresh-process reloaded tensor digest: **identical to candidate**.
- Training JSON report SHA-256:
  `27283f50c4bc10515d658d1382acd4738e084b3b3b0d4346cb20582fa2ba390f`.
- Fresh-process reload JSON SHA-256:
  `183c2915c6e84899a08a784db22d03b5304cd251a12bbced5508c13561acf2a2`.

The private raw records contain exact predictions, IDs, manifests and losses.
These hashes demonstrate internal byte consistency after readback, **not**
independent authorship or examiner attestation.

## Verdict and falsifier

- **TRAINING MECHANISM VERIFIED:** genuine pretrained base loaded, candidate
  and corrupted LoRA tensor values changed after optimizer steps, matched
  zero-update tensor did not, saved candidate loaded in a new Python process.
- **BEHAVIORAL CHANGE OBSERVED:** identical development cases show different
  candidate predictions and accuracy from frozen/zero-update controls; the
  change was adverse, not a reasoning improvement.
- **REASONING IMPROVEMENT SUPPORTED:** NO. Candidate development accuracy and
  false approvals worsened; transfer did not improve.
- **GENERALIZATION SUPPORTED:** NO. Tiny public templates and n=4 transfer
  are inadequate even if a gain existed.
- **OPERATIONAL GRADUATION:** NOT ESTABLISHED. External private examiner,
  independent structural families and authority custody remain required.

> A critic that approves every wrong answer is not fit to stand at the exit.
> More optimization steps could amplify this error. The next controlled
> experiment should train *both* correct and wrong answers of the **same**
> task in each update, while preserving equal-budget controls, then examine
> false approvals and larger independent slices before any gate authority.
