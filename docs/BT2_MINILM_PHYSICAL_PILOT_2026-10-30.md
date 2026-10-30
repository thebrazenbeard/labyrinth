# BT2 Labyrinth — physical pretrained MiniLM development pilot (2026-10-30)

This is a **bounded physical neural-learning observation**, not independent final
qualification, a reasoning-generalization result, or a graduation receipt.

## Exact subject and execution

- Source parent: Labyrinth Draft PR #62 published head
  `534e6441ef41808517ed0a206203c57bd6411eff`; the pilot source below is a
  subsequent, separately reviewed candidate change.
- Pilot script: `scripts/qualify_minilm_cpu.py`, SHA-256
  `c4d42ce02c29cfa256cb5fa6ef77e239c56b00c7c28777b16ee6d1a2d9654b99`.
- Host: `LAPPY`, Windows, Intel i5-11400H, 31.7 GiB installed RAM; Python
  3.14.5; PyTorch `2.14.1+cpu` (CUDA unavailable to this Python); PyTorch CPU
  threads limited to two. Lappy's NVIDIA RTX 3050 was **not** used.
- Offline local pretrained model identifier:
  `sentence-transformers/all-MiniLM-L6-v2` (`BertModel`, 22,713,216 base
  parameters observed on load). No online download was performed. Model byte
  provenance was not independently attested.
- PEFT LoRA: rank 4, alpha 8, query/value modules, plus a trainable 2-class
  projection head. Two cycles, eight AdamW updates per cycle for the correct
  and inverted-label arms; equal recorded forward/backward visits but **zero
  optimizer updates** for the sham arm.
- Isolated host artifacts: `C:\Temp\bt2-minilm-q3-one-534e644-20261030\outputs\pilot01`.
  The preflight confirmed the run directory did not exist; unrelated host
  processes and earlier training directories were left untouched.
- Process completion: exit code **0**, runtime approximately 21.81 seconds
  after model-library startup; report `report.json` SHA-256
  `4c27ce5ce5553456b5d8d52b24bc4ba3e8b07ed4391423b0ae1db1251e66c6fb`.

## Measured results

All arms began with the same trainable-value digest
`12d34afe7fe8d453a4e41924663d5cc1f56440ca6d1cb6ec4e3d4bed7a88fee6`
and were checked to have identical training-prompt exposure sequences.
Train fixtures have 12 examples per cycle; each development set contains ten
public, predefined examples. These are not private final examinations.

| Arm | Held-out accuracy, cycle 2 | Alternative-template transfer, cycle 2 | Optimizer updates per cycle |
| --- | ---: | ---: | ---: |
| Correct labels | 10/10 (100%) | 8/10 (80%) | 8 |
| Zero update | 5/10 (50%) | 6/10 (60%) | 0 |
| Inverted labels | 0/10 (0%) | 2/10 (20%) | 8 |

The correct-label arm beat both controls in this single small public fixture.
It is **not** a statistically independently qualified effect and does not
establish broader reasoning generalization. The correct arm's held-out score
was 10/10 after cycle 1 and remained 10/10 after cycle 2; that is narrowly
consistent with retention on this fixed public fixture, not a robust forgetting
assessment.

### Additional initialization seeds

Two further *sequential*, isolated Lappy runs used the same script, fixed
development fixtures, two cycles and eight updates, but different initial
seeds. Both completed with exit code 0. These are sensitivity checks against
initialization, **not independent new examinations**.

| Seed | Held-out: correct / zero / inverted | Alternative template: correct / zero / inverted | Report SHA-256 |
| --- | --- | --- | --- |
| `20261030` | 10/10 / 5/10 / 0/10 | 8/10 / 6/10 / 2/10 | `4c27ce5ce5553456b5d8d52b24bc4ba3e8b07ed4391423b0ae1db1251e66c6fb` |
| `20261031` | 9/10 / 6/10 / 0/10 | 8/10 / 5/10 / 2/10 | `d92d43a3fde3d83f5f33589cc976219c6a16df2219984032de3e14fcc0816eca` |
| `20261032` | 10/10 / 5/10 / 1/10 | 8/10 / 5/10 / 2/10 | `efaa97b9d8d103771766328f249735abaff522a799da72680a782935e8ab8514` |

The correct-label arm outperformed both comparison arms on both development
sets in each of the three runs. These scores remain dependent on a single,
exposed set of hand-authored examples. No blinded evaluator, uncertainty
interval, or LoRA-only causal ablation was established.

### Parameter and persistence readback

- Initial LoRA tensor-value digest:
  `b0ffa71d51da42c55c5913ecff07ed6f8dedc60d44155182ab7cf2806b690e5b`.
- Correct-label final LoRA tensor-value digest:
  `a8b019e7586ad7467ef49cdc880d8c32cb6b55799949a1fa925a6cf2725edad2`.
- Zero-update final LoRA tensor-value digest: unchanged from initial.
- Inverted-label final LoRA tensor-value digest:
  `4a374a8c4dc88d26cfc44c745e64d68be45412098ece02e4f92fc1b9e2d2f418`.
- The experiment required the loaded cycle-2 trainable parameter digest to
  equal the saved cycle-1 digest, for every arm. Independent PowerShell report
  checks confirmed `reload_continuity=true`, `same_initial=true`, and
  `matched_visits=true`.
- Independent PowerShell SHA-256 readback of all six LoRA adapter files and
  six head files matched the respective reported file digests
  (`artifact_hashes_match=true`). As examples, correct-label cycle-1 adapter
  file SHA-256 was
  `eb7f304697ec1ac7d87e65ced784aafdc4e56f811fa4b3b74d33e670e7da2ef5`;
  correct-label cycle-2 adapter file SHA-256 was
  `55692012c56b7d46affd8f13d43e1319d4169f26e152c010906eb8f9f8583dbd`.

The changed LoRA tensors and cross-cycle reload are physical local evidence
for this narrow pretrained encoder/adapter run. The self-authored trainer and
development fixture remain under the same operator's custody; the values are
not independent scientific replication.

## Q8 checkpoint substrate — separately exercised

`labyrinth/neural_resume_checkpoint.py` and its test module were copied into
an isolated Lappy test root. SHA-256 readback matched the local source:

- Module: `28343bb1d088610d177486df4aa5b2cf475b77019b245105accd359cdad0ea57`.
- Tests: `a4d8c2d4cf83f738bd6e66a2e81b18ce04cbdc1d3038593c989e6fbc04c70332`.

`py -m unittest discover -s <isolated-q8-tests> -p test_neural_resume_checkpoint.py -v`
on Lappy returned **6 tests, 0 skipped, PASS**, exit 0. These included
full AdamW optimizer/RNG/tensor restore and an interrupted-vs-uninterrupted
second-step equivalence check using real CPU PyTorch. This proves the small
checkpoint module behavior exercised by those tests, **not** its integration
with the signed development transition or a real interrupted HF training run.

In the local source checkout (where Torch is unavailable), the full suite after
adding the pilot and checkpoint modules returned **523 tests, 42 skipped,
PASS**; four new PyTorch-dependent tests were skipped locally and are covered
by the separate Lappy result above.

## Qualification boundaries

- Q3: physical pretrained encoder/LoRA weight update observed for this narrow
  text-classification experiment; generalized reasoning learning still UNKNOWN.
- Q4: local LoRA save/reload continuity across two cycles observed; full
  Labyrinth/Minotaur neural lifecycle composition still UNKNOWN.
- Q5: one matched-input, three-arm public development comparison observed;
  independent examiner custody and repeatability across seeds NOT ESTABLISHED.
- Q6: alternative-template responses improved in one tiny task; broad
  transfer/retention and independent blind testing UNKNOWN.
- Q7: private final examiner/custody NOT RUN. No graduation authority.
- Q8: standalone full-state checkpoint tests PASS on real PyTorch; integration
  with recovery journal, training cursor and real interruption/restart is NOT
  ESTABLISHED. Full Q8 gate remains FAIL.
- Q9: Draft/HOLD; no merge or deployment is authorized by this experiment.

No additional model, private dataset, credential, or unrelated host service
was installed, overwritten, restarted, or deployed for this experiment.
