# 2026-10-27 pretrained numeric-answer SFT — negative result

Disposition: NO MINOTAUR GRADUATION / NO VERA DEPLOYMENT. A real
pretrained-model LoRA run trained numeric-answer-only targets, with a
same-exposure zero-learning-rate control and deterministic tier interleaving.
The measured semantic accuracy advantage over corrupted labels was ZERO.

## Exact experimental subject

- Training source commit: ac23852818f5f92de1f310dbc18b9d96f0631c1a; source-tree digest:
  08a6cc8a027d0207ef7500235bda4b5b7aa859c2ca5244f12db9c0f60b544631.
- Current source repair in main: Labyrinth PR #23, merge
  0b2c4f1f7509484ffde83614f5ac164b2c70ec02.
- Model: HuggingFaceTB/SmolLM2-135M-Instruct, HF revision
  12fd25f77366fa6b3b4b768ec3050bf629380bac, local safetensors SHA-256
  5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c.
- Frozen training plan SHA-256: f79040cc4a123af80e5cd88d8b7fbdae313861f174f3d5bb4f5c08379ab923fc.
- Seed 31711, two task families (tiers 1 and 2),
  8 cases per family, 32 optimizer steps
  for each trained arm and 32 zero-LR sham optimizer steps.
- Device CPU, learning rate 0.0002, LoRA rank
  4, alpha 8, max tokens
  384.
- Reserved public final seed 92513,
  8 cases per tier.
  The final evaluator script was frozen before final scoring at SHA-256
  07a7abba3c94b95034bd114c0ae668994288552dd7f1fb59ce741f5c0ea9e99e.
  The public deterministic task generator is not a private blind oracle.

## Neural update and negative controls

The v3 objective supplies the fixed ANSWER: prefix as part of the input
prompt; only numerical output tokens and EOS receive supervision. Therefore
numeric suffix validity does NOT establish independent learned answer
formatting. Candidate LoRA parameter value digest changed from
896ea9653db778268908a093eb6a5c199b05e072942598ccdc3b4306807ac799 to
67e000f186e3a94a1bf2aa1ebbfebed20511907a4f04c724c735573c39e7132a. Only the separate adapter was
saved; the donor weights were not changed.

The zero-LR sham saw the same examples and optimizer-step count and
preserved its trainable LoRA parameter digest. Reviewer-discovered
LIMITATION: a zero-LR forward may mutate model buffers even when
trainable parameters are unchanged. Labyrinth PR #23 later added a
fail-closed persistent-buffer digest check, AFTER this run; this exact
experimental run did not pass that later safeguard.

## Development scores (16 cases)

| Arm | Semantic correct | Numeric suffix valid |
|---|---:|---:|
| baseline | 0/16 | 0/16 |
| sham | 0/16 | 0/16 |
| corrupted | 0/16 | 16/16 |
| candidate | 0/16 | 16/16 |

## Reserved public final scores (16 cases)

| Arm | Semantic correct | Numeric suffix valid |
|---|---:|---:|
| baseline | 0/16 | 0/16 |
| sham | 0/16 | 0/16 |
| corrupted | 1/16 | 16/16 |
| candidate | 1/16 | 16/16 |

The candidate and corrupted-label arm got the same single problem correct,
case ID reserved_v3-t2-2-da4b57203e409cd7, with the same answer. Candidate-only wins
over corrupted labels: 0. Accuracy delta: 0.0. Hence no evidence of
reasoning improvement from correct supervision.

Fresh-process reload verified saved adapter tensor values and reproduced
all development candidate trials: True.
The reported distinct training/reload process IDs were
39592 and 19412; local process
difference alone is not an independently attested execution chain.

## Evidence checks and custody

- HF training report SHA-256: 73472c15ea29d114a46eae3bd9c03be13aa02bd71b9291b0a2193e619404cdfe
- Fresh-process reload report SHA-256: e02f3e71a76ca71472ff31b048df2d042d89315619a441176a8a83a132b1559f
- Reserved public final report SHA-256: 5c27ec04f7b58ffbf867f79a0943a4e5b113c5ea4f0ff46dc7fab9b144612912
- Internal development audit: PASS, 64 records.
- Independent read-only final-record consistency audit: PASS,
  64 records, hash checked, case order checked and
  scored against regenerated task answers.
- Local artifacts remain on Lappy under:
  C:\Tempt2-semantic-phase-20261027
  (frozen plan, source snapshot, adapters, raw reports, evaluator and audits).
- No checkpoint was activated, promoted, installed into Vera, or signed by
  an independent Minotaur authority.

Internal PASS audits do not authenticate model execution, report custody,
private unseen-task independence, or neural competence. The public
finite-generator questions are structurally related and the dataset is
small. This result should remain explicitly NEGATIVE and the Minotaur
graduation gate must stay closed.
