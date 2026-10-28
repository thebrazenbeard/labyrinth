# Minotaur first-error real CPU pilot — E-020 (2026-10-27)

Evidence status: LIVE PRETRAINED TRAIN/RELOAD; NO REASONING QUALIFICATION.

## Exact subject

- Labyrinth source at execution: `ec77e38c9c3f7c0928b5b561fd52a6beece2580d`, source SHA-256 `12361141514bcb8a1b20915d19ad4edddd55860c1992e167d9d390513de39057`.
- Pre-existing local `Open-Reasoner-Zero-0.5B` checkpoint; base safetensors SHA-256 `f3fabf5933c996348e19cc65f2eaf4ceb4a036cb9981d2da1142c2e8b4020938`.
- Windows 11 CPU, Python 3.14.5, Torch 2.14.1+cpu, Transformers 5.14.1, PEFT 0.19.1.
- `FirstErrorConfig(steps=2, eval_count=2)`; 2 train-program pairs, correct/wrong prefix at each optimizer step; matched zero-update and inverted-label controls, frozen base.
- Both candidate and corrupted-label controls had 2 genuine AdamW steps / 4 forward-backward passes; zero-update took equal steps with zero gradients and retained initial adapter digest. Candidate and corrupted LoRA value digests differ from the initial digest.

## Measured outcomes

| Arm | Development 4 traces | Composition transfer 4 traces | Length transfer 4 traces | Development false approvals | Development false rejections |
|---|---:|---:|---:|---:|---:|
| Frozen base | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) | 2/2 | 0/2 |
| Zero-update | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) | 2/2 | 0/2 |
| Correct-label candidate | 1/4 (25%) | 1/4 (25%) | 0/4 (0%) | 0/2 | 2/2 |
| Inverted-label control | 1/4 (25%) | 1/4 (25%) | 0/4 (0%) | 0/2 | 2/2 |

Both adapted arms made identical tiny-suite predictions, unlike the frozen and zero-update controls. Candidate behavior regressed; it rejected correct traces and did not distinguish trained-label from corrupted-label effects. No useful first-error discrimination, training benefit, generalized skill, or graduation is established.

## Persistence / integrity receipts

- Training report SHA-256: `20244e1dba49afb8f680b19435c7da15c0e7b8641200a1ac09a36b8721d94915`.
- Adapter safetensors SHA-256: `3b2b168ef7d22b47bf9c83f15b7571f776d3845a90be26cb07dc966b25b538e0`.
- Candidate LoRA tensor-value SHA-256: `f6ada039727c7151c7585c7e4e18a386a6692c9d4db53875c37e2fe308e05e81`.
- Fresh-process reload report SHA-256: `718d492328353afc478feeb8226068bb06b87e02c9fded498badec07384b9cb4`.
- Training process ID 31512, reload process ID 35356; `different_process_id=true` and `same_predictions_and_metrics=true`, with identical adapter tensor values.

Reports and binary adapter remain in an isolated local research directory, not Git or production. They are self-reported training records validated by same-host readback, not independently authenticated model inference. The synthetic generator's small, public template set and only two examples per class per evaluation slice prohibit statistical or generalization claims. Minotaur and all trainees remain unqualified for release.

## Independent review finding and correction

The independent Systems Architect identified that the synthetic examiner once produced the same receipt schema and signature domain as production attestation and could trigger `GRADUATED` under a test key. This was reproduced as a RED test; the synthetic examiner now uses a separate schema and HMAC domain and attests `fresh_process_reload=false`, `independent_private_oracle=false`. The normal graduation gate must always reject that synthetic fixture; no actual private examiner or production release authority is implemented.
