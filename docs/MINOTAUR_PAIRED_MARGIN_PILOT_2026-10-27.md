# Minotaur E-020 ? same-task paired-margin ranking trial (2026-10-27)

**Evidence class:** actual pretrained neural CPU LoRA training on Lappy; public development data only. **Qualification: NOT_ESTABLISHED.**

## Motivating defect

Two earlier balanced two-step losses led to degenerate global verdict priors: causal-token CE approved every ordinary development answer; direct binary-verdict CE rejected every ordinary development answer. A seemingly low false-approval rate therefore did not demonstrate discrimination.

## Exact experiment

- Source under test: `thebrazenbeard/labyrinth@4e163eaed161802cc4dba97d9aace9f88485db43`, clean tree at invocation; source tree SHA-256 `ed3a88a4f2de03a2e6bc54e75989ff7383bb0256f19c93a2e2fce2e2c0128cc6`.
- Donor: already-local pretrained Open-Reasoner-Zero-0.5B; no download and no production Vera changes.
- Loss: `pairwise_margin_rank`; for paired correct and incorrect proposed answers to *the same question*, minimize softplus(-(correct Yes?No margin ? incorrect Yes?No margin)).
- One update uses the same balanced problem pair across all candidate, zero-update and corrupted-label arms. To keep CPU RAM bounded, detached margins are measured first, then the exact analytic pairwise gradient is applied in two separate autograd passes. This requires two extra no-gradient forwards per optimizer step. The sham executes the same computations but zeros its adapter gradients before AdamW.
- Two AdamW updates; frozen, zero-update and corrupted-label arms; development eight answers/four pairs, register-program transfer four answers/two pairs. Public generated examples are not independent final-exam evidence.
- Local report SHA-256: `8ef476cdae217971b57eccd40422b38635c34ad9a08dd6b3169d84db128f15e6`.
- Adapter safetensors SHA-256: `878ed4c501b654725d482be927579545a93c6ac390add2a27990246343efcdf1`.
- LoRA parameter digest: initial/zero-update `f4a000673d03e8abc09f31058feb5a8305b983c962836b734d673c3152a2245d`, candidate `3d3fafa6d9293f86886e24678616b236fca72c8c64dcd1ecc0f5763c24b04859`, corrupted `6b26d1b7b8359141d3290cb8b58b1255b989484ddcc272276a59ebcdfc51b905`.

## Observed paired and absolute scores

| Arm | Development accuracy | Development pair rank | Transfer accuracy | Transfer pair rank | Dev false approvals | Dev false rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen pretrained | 5/8 | 3/4 | 2/4 | 0/2 | 3/4 | 0/4 |
| Zero-update | 5/8 | 3/4 | 2/4 | 0/2 | 3/4 | 0/4 |
| True paired-ranking LoRA | 4/8 | **1/4** | 2/4 | 0/2 | 0/4 | **4/4** |
| Corrupted-label LoRA | 4/8 | 2/4 | 2/4 | 0/2 | 4/4 | 0/4 |

At these small budgets, pairwise training **worsened** development ranking and did not fix structural transfer. Lower false approvals arise alongside rejected correct claims, not credible guardian competence.

## Independent mechanical verification

- Training process PID `36440`; fresh separate Python reload PID `38392`; reload JSON SHA-256 `2bde4e91e442d34fd8d8af87ee585d0c1569bd3f487a8124b4f29a61273a31a9`.
- Reload confirmed the exact saved LoRA tensor digest and stable discrete predictions, false approval/rejection counts, pairwise rank, and selective decisions. Raw floating-point margin summaries were not byte-identical on CPU; the contract's narrowly bounded 1e-4 descriptive-float tolerance accepted them. Therefore raw `same_predictions_and_metrics` is **false**, even though the strict reload contract succeeded. Do not claim full report equivalence.
- Independent scalar PyTorch comparison reproduced exact pairwise-logistic gradients for four positive/negative margin pairs.
- The separate experimental feature branch subsequently fixed its selective-risk denominator: zero coverage now reports an unavailable false-approval rate rather than a misleading zero. The current Minotaur draft uses a different logit-diagnostics interface and has no selective-abstention risk score to inherit that repair.
- The source integration onto the newer draft passed 203/203 local Python tests. The pretrained experiment remains bound to its original exact source commit; test success on the integrated code does not qualify model reasoning.

## Safety and next falsifier

Do not graduate trainees or grant Minotaur deployment authority from this experiment. The shared public arithmetic/templates are small and correlated. The private independent examiner, scoring-custody isolation, enough disjoint structural units, calibrated class-conditional abstention, and genuine process first-error localization remain unproven.

Next test should diagnose pairwise gradient signal-to-noise and stronger structural program diversity *before* increasing parameter count or training budget. A ranking improvement is meaningful only if it also improves independently calibrated absolute decisions with useful coverage, and beats the frozen, zero-update and corrupted-label controls.
