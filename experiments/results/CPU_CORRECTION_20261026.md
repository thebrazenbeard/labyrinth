# Corrective CPU SFT — 32-step development experiment

Status: executed on VN1 after two-step pilot failed; no held-out reasoning accuracy improvement.

## Frozen intervention

- Train-only source fixture: `experiments/data/train_cpu_pilot_seed1702_32.json`, generator seed 1702, 32 distinct cases; SHA256 `1ddde0f4831617e53e6280ef78578ea018f732cfa5fb15ad8d34290cec841c84`.
- Curriculum fixture commit: `a6fb612a544c06defae441d27dfac4278dc042ce`.
- Same local SmolLM2-135M-Instruct pretrained weights and tokenizer as the two-step pilot; all parameters frozen except 3,540,096 final-transformer-block parameters.
- 32 train case visits, 32 AdamW calls, LR `1e-4`, seed 1702; 32 gradient-bearing updates for trained/corrupted, zero gradient updates for sham.
- Correct-label loss first/last: `2.8380089` / `0.8258467`. This is a training-set loss only.
- Actual local VN1 checkpoint digests and full receipt records are adjacent:

| Arm | Final block SHA256 | Model checkpoint SHA256 |
| --- | --- | --- |
| Trained | `1b527ae5c4cb8624bd5c29d107f69f5a17d37b3c692c138d68f5d6233e3eab77` | `fc400ef922a8618c00e68bbd28db0061eb6040601e4395ce21c7fdc9c4b789b1` |
| Sham | `05e2bfb7ec08dd65d97c82bc1ffaceef5b4dbb4be33aeb497f40f37863746b58` | `8bc78203ed842936183342dd1224509b34a00059dde9f0b33aca0f1e093a488e` |
| Corrupted | `6deb7048c121759eafb5609dcf1a4fe5b05b3ce4853f247b82900bce0f45cb68` | `fe3bd2d485dfbddbfe42705dc2b7a79fe57bd404a6ed1c15ebb067fbc063b2bf` |

Each of these three checkpoints passed a new-process loader+digest check on VN1. The raw artifact bytes remain at `%TEMP%\labyrinth-one-cpu-20261026-1600\{trained32,sham32,corrupted32}`.

## Development outcomes, not untouched final outcomes

- 16 identical prompts from previously unused matched-length structural pairs (dev seed 9704), canonical/paraphrased, no demonstrations.
- Frozen base, trained, sham, and corrupted labels used identical order and greedy 12-token decoding.
- Gold remained evaluator-local and never entered remote inference. Independent paired-analysis output is `dev9704_32_report.json`; adjacent JSON files preserve the public predictions.

| Arm | Exact correct | Strict response format |
| --- | ---: | ---: |
| Baseline | 0/16 | 0/16 |
| Correct-label SFT | 0/16 | 11/16 |
| Zero-update sham | 0/16 | 0/16 |
| Corrupted-label SFT | 0/16 | 7/16 |

**Inference:** the trained model learned some answer formatting. This is not evidence of program execution or answer generalization. Corrupted labels also improved formatting; longer training alone does not establish reasoning learning.

## Next scientific decision

Do not select hyperparameters from final seed 9703 or reuse dev seed 9704 as a fresh final. A next attempt should introduce explicit intermediate-state supervision or an easier competence-calibrated task family, use a development split, freeze configuration, and only then generate a new sealed final family. Report transfer separately from output-format compliance and compare a matching-budget corrupted-label control.

The source and unit tests validate protocol mechanics; the tiny public set and single seed are not a generalization qualification. No main merge, deployment, or in-place model replacement occurred.
