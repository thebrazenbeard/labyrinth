# Chat-template CPU SFT: developmental negative result (2026-10-27)

Status: **real pretrained neural updates and reloads PASS; held-out reasoning gain NOT demonstrated.**

This study extends Draft PR #7 with `--prompt-mode chat` training. It does not use
the independently sealed Dev3 evaluation keys. The six one-/two-instruction
calibration cases were first used to choose chat rendering, so their subsequent
results are **developmental**, not a final unbiased benchmark.

## Environment and exact evidence

- VN1 Windows CPU; offline HF SmolLM2-360M-Instruct, base `model.safetensors` SHA256
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- 361,821,120 pretrained parameters; 9,832,320 final-transformer-block
  trainable parameters; all other model parameters frozen.
- Train-only seed 1807 corpus, 128 distinct examples, SHA256
  `a18d0a9dcf9e7c3a8f4803e3582315aa30c31964edd67b1b9559a7d0981c6ad0`.
  Full generator membership attestation passed before model loading. No Dev3
  gold, prompt, or key was used by the training process.
- Each arm: optimizer AdamW, learning rate 0.0001, seed 1855, 32 train-case
  visits and backward passes, `answer` target, chat tokenizer rendering, and
  zero weight decay. Sham gradients explicitly zeroed after backward, before
  optimizer step. Correct/sham saw 283 supervised tokens; corrupt saw 282.
- All three checkpoints written to
  `C:\Users\Vera\Documents\LabyrinthTrainingRuns\20261027-chat32-seed1807`
  with copies of training source and data, per-arm receipts, exact predictions,
  and evaluation code. All three fresh-process reloads passed.

| Arm | Backward steps | Real gradient updates | Checkpoint SHA256 (prefix) | Layer changed? |
| --- | ---: | ---: | --- | --- |
| Correct chat SFT | 32 | 32 | `a6a64b9a5924653a89` | yes |
| Zero-update sham | 32 | 0 | `d387d4323e5314a3d8` | no |
| Corrupted-label chat SFT | 32 | 32 | `13a3e20dc34c89d6988` | yes |

Physical pretrained tokenizer prefix test: the assistant-training transcript
started with exactly the frozen-inference generation prefix. Training source
was reviewed independently and 123 unit tests passed before host execution.

## Paired six-case developmental outcome

Identical public prompts and generation budgets for all arms; greedy decode,
max 24 new tokens, explicit attention masks. A separate deterministic
arithmetic recalculation and JSON consistency check confirmed stored scores;
this is internal verification, not an independent model review.

| Checkpoint | Raw exact | Chat exact | Chat strict-format |
| --- | ---: | ---: | ---: |
| Frozen pretrained | 0/6 | **4/6** | 6/6 |
| Chat-trained 32-step | 0/6 | **2/6** | 6/6 |
| Compute-matched sham | 0/6 | **4/6** | 6/6 |
| Corrupted-label 32-step | 0/6 | **3/6** | 6/6 |

The frozen raw-v-chat developmental calibration was replicated with an explicit
attention mask, with **12/12 response strings unchanged**. Frozen masked
prediction SHA256: `cf6f69162be65f7d77242ea5f1d993e1901049d58a679130562bf5b6125aa882`.
Post-training prediction SHA256s: correct `166cdecbf2602433538ee4ec0a236b41062e0b3cccdf27633038d92c2e19ef18`,
sham `340d2efbcba8dac52e0f75600c385623bdc57d55c397772d55f47cca6f2ab03e`,
corrupt `094cf00308a219505063d2a709ed2e19934e3c6f60e11b21348ecd359495b637`.

**Result:** Chat formatting materially improved this frozen six-case baseline;
the 32-step chat SFT did **not** improve it and lost two correct cases versus
frozen/sham. That observation does not establish universal degradation, nor
does it measure structural generalization: six small development cases are
inadequate, the two-step and modular cases remained unsolved, and the
corrupted arm differs by one supervised token.

## Next gate

Do not consume Dev3 merely to tune against these outcomes. Construct a new
train-only graduated one-step-to-long-program curriculum, predeclare its
matched controls, and calibrate frozen one-/two-/structural-step competence
on a *different* development set. Freeze the protocol before asking the
independent Dev3 custodian for a final evaluation. No RL, continual learning,
reasoning transfer, or production integration was established here.
