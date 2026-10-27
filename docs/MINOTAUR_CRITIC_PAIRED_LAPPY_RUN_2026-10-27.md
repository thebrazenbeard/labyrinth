# Minotaur correctness critic E-019 — paired-label CPU trials (2026-10-27)

**Evidence class:** Two live pretrained Windows CPU training runs, local JSON
readback and two genuine fresh-process LoRA reloads. This is a development
experiment with a tiny publicly generated evaluation set, **not** an
independent qualification or a proof of improved reasoning.

## Frozen subject and change from E-018

- Exact Labyrinth Draft PR #20 source commit:
  `84f1736e8f293252737a5f17878608331174baf8`;
  tree `ac8936f0b9c2308daafc98e65ef93e2477d34226`;
  run-reported source SHA-256
  `7102322998c8cba9b308807aaafb728716699828bfb29c564b84559e08c7507f`.
- Local pretrained Open-Reasoner-Zero-0.5B checkpoint and tokenizers identical
  to E-018. Safetensors SHA-256:
  `f3fabf5933c996348e19cc65f2eaf4ceb4a036cb9981d2da1142c2e8b4020938`.
- Python `3.14.5`, Torch `2.14.1+cpu`, Transformers `5.14.1`, PEFT
  `0.19.1`, deterministic CPU execution on Lappy. No packages downloaded,
  model files changed, production route touched, or external API inference.
- Critic schema **v2**: each optimizer update accumulates equal half-loss
  gradients for a *correct and incorrect answer to the same task*. Arms
  receive identical task pairs and budgets; the zero-update gradients are
  zeroed before AdamW. This differs from E-018 v1's one answer per step.

## Runs and readback

Both runs called `run_critic(CriticConfig(...))` with identical model path,
seeds, learning rate 0.0003, LoRA rank 4/alpha 8, development count 2 per
tier, transfer count 2 per tier, and new distinct experiment output paths.
Only `steps` changed: two in the first run, eight in the second. Each used
24 available training tasks per tier and the same complete evaluation cases.

| Subject | v2 paired 2 updates | v2 paired 8 updates |
|---|---:|---:|
| Candidate optimizer updates | 2 | 8 |
| Candidate forward/backward passes | 4 | 16 |
| Same-budget sham passes | 4 | 16 |
| Same-budget corrupted-label passes | 4 | 16 |
| Frozen development | 5/8 (62.5%) | 5/8 (62.5%) |
| Candidate development | 4/8 (50%) | 4/8 (50%) |
| Zero-update development | 5/8 (62.5%) | 5/8 (62.5%) |
| Corrupted-label development | 4/8 (50%) | 4/8 (50%) |
| All four arms' transfer | 2/4 (50%) | 2/4 (50%) |
| Candidate false approvals in development | 4/4 (100%) | 4/4 (100%) |
| Candidate false approvals on transfer | 2/2 (100%) | 2/2 (100%) |

Candidate and corrupted-label arms both answered **Yes on every trial** in
both runs. V2 paired exposure did **not** correct the class-prior collapse on
these cases. The corrupted-label arm shows that substantial tensor adaptation
does not imply a useful correctness discriminant. Eight updates cannot prove
the optimal budget, and the sample counts preclude a broad statistical claim.

### Exact artifact and process identity receipts

The identical initial adapter tensor digest in both runs was
`f4a000673d03e8abc09f31058feb5a8305b983c962836b734d673c3152a2245d`.
Both zero-update arms retained this exact digest.

| Receipt | v2 two-update trial | v2 eight-update trial |
|---|---|---|
| Candidate LoRA tensor SHA-256 | `5589e1f09592d0b9b182659b56f36ad423b98f541bcdc1346708d5ce7cf7b631` | `184e8b3e1d25246c8eb7b69d00e6193eb0110e5d5f9dd9f779157f50ed3a6445` |
| Corrupted tensor SHA-256 | `d0996f14d9d6ba456b22042c51126a792d644b34ae55ad6cabfc17bbf953a81e` | `0aaac9b8891556cfd4dd017720ea277c41f4128eedde4e34bb76dbe028615aa0` |
| Adapter safetensors SHA-256 | `f7ed2c78ec1a15c872433ca66a1bcd58a8c3abb5d39dab9ad225ef174949b34c` | `8db5af7b845d261e85457ddf3f07d9a073cc13891d31448f3285b4e5f7b2c591` |
| Training JSON SHA-256 | `e628568ebcc29d9f288a0535bcb62d53b34b9435c8b6ce68ed064e3eafe7c422` | `1865561295b4f02b7d2c6ce849496efe86161044bb184c66f358a284e070ac08` |
| Reload JSON SHA-256 | `fddaac8910cf76a604383a3c878180e26c356b3058b8b34fa142c5d3fec5f210` | `ee43819a7d2c53df1f2169ac1bf02409d128e843c86ceb963b9fd6f8efef88ca` |
| Training / reload process IDs | 21608 / 38188 | 36856 / 13432 |

Both fresh reload processes exited 0, authenticated internally unchanged
base/adapter files, found distinct process IDs, recovered the **same candidate
LoRA tensor digest**, and reproduced all development/transfer predictions.
The training processes finished after approximately 59 and 99 seconds,
respectively; reloads after approximately 21 seconds each. These self-checked
records are reproducibility evidence, not an independent examiner's signature.

## Judgment and next falsifier

1. **Training mechanism and persistence verified** for both real pretrained
   pilot runs, including equal-budget sham controls.
2. **Adverse behavioral change observed:** candidate changed a baseline
   negative verdict to Yes, increasing false approvals. No useful improvement.
3. **Reasoning improvement and structural generalization NOT ESTABLISHED.**
   A repeat of 100% false approvals is a guardian failure, irrespective of
   optimizer success.
4. **Paired training alone is insufficient at two or eight updates** on this
   checkpoint/evaluation. No broad conclusion about other models or budgets.
5. More responsible next experiments: inspect actual logit margins and token
   priors, use a development-only calibrated abstention threshold with risk and
   coverage readout, consider direct process-error supervision, and test
   larger genuinely independent structures. Retain frozen, zero-update and
   corrupted-label controls; do not keep blindly increasing optimizer steps.

> The Minotaur that says “Yes” to every trainee is not guarding an exit.
> The external graduation authority must continue to **block**.
