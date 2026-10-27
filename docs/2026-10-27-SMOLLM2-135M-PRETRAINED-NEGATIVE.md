# 2026-10-27: pretrained 135M LoRA reasoning pilot — negative result

**Disposition: NO GRADUATION.** This records one genuine local pretrained-language-model
LoRA update and reproduced negative reasoning evidence. It is **not** a Vera
production checkpoint, independent model certification, general reasoning
gain, or installed autonomous learning. No adapter was activated.

## Source, checkpoint, and frozen plan

- Labyrinth exact source main: `d51d3928e84b47ac5695eb36c991112c8db04516`.
- Hugging Face donor: `HuggingFaceTB/SmolLM2-135M-Instruct`, immutable upstream revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0 model-card declaration.
- Original-format safetensors checkpoint SHA-256:
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`. The tokenizer and config are additionally
  bound to per-file SHA-256 values in the private local pilot report.
- Frozen initial plan: `2016d1f7f78c939d650f3347edf0f5e0b9f8e63a5267a0d7657cb559b9d178b7`;
  CUDA was refused before training because `torch.cuda.is_available()`
  was false in the selected Python interpreter. No silent CPU fallback occurred.
- Frozen CPU variant plan: `803981cd491291e8171d1b560858c4530f015567f04b04b229536c527b323a40`.
  Only the execution device and isolated run output were changed.
- Local Python/package versions reported: {"peft": "0.19.1", "python": "3.14.5", "torch": "2.14.1", "transformers": "5.14.1"}.
- Training used exactly 16 public generated problems from tiers
  1–2, `seed=28041`, 16 optimizer steps for
  candidate and 16 corrupted-label steps.
  A base and a zero-update initialized-adapter control were also scored.
  The no-update adapter is NOT compute-matched for optimizer time/visits.
- Development holdout: 16 disjoint exact prompts with
  seed `seed XOR 0x5C18A7`. Reserved final set: seed
  `93017`, four cases per tier 1–4, reserved before
  viewing development scores. Public templated tasks are not a blind oracle.

## Observable neural update and fresh-process persistence

- LoRA trainable-tensor initial SHA-256:
  `fc3c771712e8ab10c655f32f581c245878a7183a906d4a3c24cc8a777a4bf8cf`.
- Candidate trainable-tensor SHA-256 after supervised steps:
  `0feda4d240a9a60aae97b424ea3437cc023ddfeb2304810942f3ca5d8cef9669`.
- These digests differ; the saved candidate adapter's file hashes and reloaded
  LoRA tensor-value digest were checked against training receipts.
- Independent process invocation `labyrinth eval-hf` reports
  `process_id_differs=true` and
  `matches_original_candidate_trials=true`.
  This supports persistence, not independent execution authentication.
- Original base checkpoint and metadata were rechecked across training and reload.
  Only the separate adapter was saved.

## Development comparison (16 problems)

| Arm | Semantic correct | Strict-format valid |
|---|---:|---:|
| baseline | 0/16 | 0/16 |
| sham | 0/16 | 0/16 |
| corrupted | 0/16 | 16/16 |
| candidate | 0/16 | 16/16 |

**Reasoning improvement: NONE DETECTED.** Both correct-label and corrupted-label
training achieved full required output syntax while scoring zero semantic answers.
The format change therefore does not establish learned reasoning.

## Reserved public-generator final comparison (16 different problems)

| Arm | Semantic correct | Strict-format valid |
|---|---:|---:|
| baseline | 0/16 | 0/16 |
| sham | 0/16 | 0/16 |
| corrupted | 0/16 | 16/16 |
| candidate | 0/16 | 16/16 |

Candidate-only semantic wins over frozen base: 0.
Over zero-update adapter: 0.
Over corrupted-label control: 0.
Accuracy delta against each comparator: **0.0**.
All four arms were compared on the same 16 cases in fixed order.

The evaluator was committed locally before inspecting final outcomes with SHA-256
`5165abba3c94b95034bd114c0ae668994288552dd7f1fb59ce741f5c0ea9e99e`.
It replayed corrupted-label training from the same initial adapter state using
training cases only, checked the resulting tensor digest against the original
training report, and reloaded the candidate from safetensors in another process.
This is a same-owner technical replay, not a separate trusted custodian.

## Audits, limitations and status

- Labyrinth HF development-report audit: **PASS**; source/digest and
  64 development trial records internally consistent.
- Reserved evaluation separate offline trial audit: **PASS**;
  64/64 final trial previews fully rehashed and independently
  rescored against regenerated case answers, with no truncated cases.
- Original pilot report SHA-256: `17cd3f76773062772fd96bdd9ee7e37a8e546f8c4349bb85078bfc427c14db89`.
- Reserved final output SHA-256: `d5c6828a55eda97e054ec0dc7ff89be2267a3fe425e04afef224cd261c66b446`.
- Local CPU-variant plan SHA-256: `803981cd491291e8171d1b560858c4530f015567f04b04b229536c527b323a40`.
- Candidate remains **UNINSTALLED / NOT GRADUATED**. No authorized model promotion
  or native independent Minotaur evaluator signature was created.

**Critical limitations:** File/model identity receipts and self-generated
reports are not authenticated external execution evidence. The public seeded
challenge generator permits report fabrication and substantial task-family
similarity. The sham did zero optimizer steps, so training compute is unmatched.
Only 16 final cases were scored, far below a statistically informative
multi-task learning claim. The reserved evaluator reviewer identified missing
extra source/plan binding and paired-record cross-checks; a separate post hoc
read-only trial audit filled the latter consistency gap without changing scores.
The two audits provide internal consistency, not private blind evaluation.
The measured improvement over all comparator arms was zero, so there is no
basis for Minotaur to admit this candidate.

## Local artifact custody

All weight files, per-trial outputs, reports, source/checkpoint manifests,
final evaluator code, and original experiment plans remain in the operator's
isolated Lappy experiment directory. They are not committed to this source
repository. The project can reproduce the same bounded pilot by pinning the
above revision and Labyrinth source, recreating the private local source tree,
running `train-hf`, then separately invoking `eval-hf` and the exact
source-hashed reserved-set evaluator. The documented final evaluator is an
experiment-specific local script rather than a general supported Labyrinth
command; it must not be presented as a production independent Minotaur service.
