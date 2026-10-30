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

### Physical four-arm parameter-family ablation

An additional isolated CPU run on the same Lappy host used
`scripts/qualify_minilm_ablation_cpu.py` (SHA-256
`32e3cf11b9350a7f92bda415c0c5c703f6cb7fb1f207feb7525e95c7927c3eb9`)
under `C:\Temp\bt2-minilm-q3-one-534e644-20261030\outputs\ablation01`.
It finished with exit code 0 in approximately 18.25 seconds. Its report
SHA-256 was
`d86b6160e0c80195d53f50ff4db8c1c251cafad2440a25151d6467e6e81fd849`.

| Arm | Held-out | Alternative-template transfer | Trainable family |
| --- | ---: | ---: | --- |
| Full LoRA + head | 10/10 | 8/10 | LoRA and head |
| Head only | 9/10 | 8/10 | Head; LoRA frozen |
| LoRA only | 10/10 | 9/10 | LoRA; head frozen |
| Zero update | 5/10 | 6/10 | Neither updated |

All four arms had identical initial LoRA/head tensor digests and recorded
training exposures. Expected frozen and changed parameter families were
observed, saved/reloaded cycle continuity was checked, and eight LoRA/head
artifact file digests independently matched PowerShell readback. This
demonstrates that the LoRA-only arm improved on these public fixtures with the
classifier head frozen, limiting the alternative explanation that all gains
came from the head. However, it is **one seed on the same small exposed
fixtures**, not independently verified causal generalization. The full arm
did not outperform the head-only arm on transfer and scored below the LoRA-only
arm there; no broader superiority claim is supported.

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

### Fresh-process pretrained MiniLM checkpoint continuation

An isolated Lappy run using `scripts/qualify_minilm_resume_cpu.py` (SHA-256
`02825411693f79681e84516a49c85f3e0d8ef06fb151cbcf8ab774471e0e1d29`)
completed under
`C:\Temp\bt2-minilm-q3-one-534e644-20261030\resume_lab\outputs\resume02`.
The first tunnel invocation timed out, leaving `resume01` absent; a read-only
directory and active-job check preceded a unique `resume02` launch. Its
three subprocesses wrote `reference.json`, `checkpoint_receipt.json`, and
`resumed.json`; the final report states `exact_resume_match=true` and
`fresh_processes=3`.

- Report SHA-256 independently read on Lappy:
  `7c05882ec5bdcc6e599e12f332be358d3049f4945d3d8bfc6224f23e9a5c5856`.
- Checkpoint identity digest:
  `ebfeaa2f6275991c15b479f5879b318aee0700da7f9274e01485396a98bddbfb`.
- Safetensors file SHA-256:
  `90bdffe64cb642b83688b6fb315e3747e3c226df964a2462497fb438502e9e29`.
- LoRA value digest changed from
  `e5cce615ff993cc3977f25a0a868ff8fe271abdb3585c495c86db57cfb4da859`
  to `f3cc7b0cb128d36cfd009d35cd39266eb53405ad6e3175a1dcb007123b4335f8`.
- Public development predictions: 9/10 held-out and 8/10
  alternative-template transfer; `graduation_authorized=false`,
  `signed_event_journal_integrated=false`.

The orchestrator checks exact final trainable and LoRA tensor values,
predictions, exposure records, and distinct process IDs against an
uninterrupted reference. The checkpoint carries AdamW moments and Python,
NumPy, and Torch RNG states. This demonstrates **physical clean-boundary
restart continuity of a pretrained model**, not abrupt worker loss, signed
journal recovery, or independent exam custody. Full Q8 remains FAIL/HOLD.

### Forced-process-kill pretrained recovery observation

`scripts/qualify_minilm_fault_cpu.py` (SHA-256
`fe41df96b3f1e46ca6a68e35623e55508e9e66b34ea20376000865f109c1404e`)
was copied into the isolated `resume_lab` script root on Lappy and its byte
hash verified before execution. A distinct opt-in `outputs\fault01` run
completed, with the following PowerShell readback of its report and
post-update marker:

- Report SHA-256:
  `31010a8b189dc89db5db14b1670ae0819eb3c93771b6db5b9e01c58655334c12`.
- Forced worker PID `30264`, return code `1`; `forced_process_kill=true`,
  `fresh_processes=4`; the marker confirms a changed AdamW parameter-value
  digest after the checkpoint reload and before process termination.
- Checkpoint identity digest:
  `dc13271c668da616d9efe420300623c19cec415e8c40134c16f79bf13aab16e6`.
- Checkpoint combined-byte SHA-256:
  `4330e5ecb1fbe4872d40077a08a4f7c623769fd6a8bb0cb5a528a7347572af7a`;
  the experiment asserts these bytes were unchanged after termination and
  subsequent resume.
- LoRA value digest moved from
  `b6c14cb13e3f3ae26fa1deeedd57652e0ab7807edd7acb7160f54e0fb6d7aeed`
  to `bd86a7abe5f35b158181c855683cdde8cb8fbd4b06a4b970b83341e3ce4e6599`.
- The fresh post-kill continuation matched the uninterrupted reference's
  exact tensors, predictions and full training exposures; public development
  scores were 7/10 held-out, 7/10 alternative-template transfer.

The termination was performed by the experiment's owning parent process,
which killed its own child only after a fsynced post-update marker. This is
**real controlled process-kill recovery with pretrained PEFT and optimizer
state**. It does not simulate power failure, prove storage durability,
authenticate a signed development event or enforce external journal fences.
The report explicitly has `signed_event_journal_integrated=false`,
`durable_power_loss_tested=false`, and `graduation_authorized=false`.

### Physical frozen-base binding tests

The opt-in `labyrinth/frozen_base_binding.py` module and its tests were
copied into Lappy's isolated `resume_lab` with exact SHA-256 readback:

- Module SHA-256:
  `f602bd69e8c4c11fc629f1d1505acdc63d45756c2f1af5b9c21c17d13a4f818d`.
- Corrected test SHA-256:
  `c72f7ce61a85e7ca57f54833298736a7c82fde71b350fcfb4bfd4557d4615a3d`.

The **first** real CPU run revealed a regression in the test itself:
`add_(1)` followed by `sub_(1)` did not reproduce the original frozen
floating-point tensor bytes. The hostile failure correctly rejected the
altered base, but the test incorrectly expected a later checkpoint-digest
error. The test was corrected to restore an exact cloned tensor, and its
non-CPU probe now uses the PyTorch `meta` device rather than depending on CUDA.
The updated Lappy test run returned **9 tests, 0 skipped, PASS**. This proves
the opt-in binding checks as exercised on real PyTorch, including changed
frozen tensors/buffers and checkpoint restoration. It does **not** establish
that existing production training entry points exclusively use this binding.

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
