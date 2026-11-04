# HF/LoRA pilot v3-exposure.v1: required in-loop TRAIN ledger (source-only)

The HF pilot now emits `labyrinth.hf-lora-pilot.v3-exposure.v1` (the unrelated
MeZO experimental v4 identifier is reserved). Its report records
`optimizer_exposure_recording` as
`in-loop-completed-optimizer-step.v1:source-only` and a complete ordered
`visited_exposures` list for **each** sham, candidate and corrupted-label arm.
Each exposure binds the source case ID and SHA-256 of the actual TRAIN prompt.
The source process checks the pinned schedule before each optimizer step and
appends the observed case after that step.

`audit_hf_report(report)` and the CLI `audit-hf-report --input ...` are strict
by default: they require v3-exposure.v1 and reject a missing, altered or incomplete
exposure marker/ledger. Merely removing the marker, all ledgers, or changing the
reported schema to v2/v3 cannot turn a current submission into a passing strict
audit. The expected schema must still be independently pinned by any future
qualification workflow; a report cannot attest its own historical provenance.

Historical v2/v3 reports have a separate, explicit compatibility path:

```shell
python -m labyrinth audit-hf-report --input older_pilot_report.json --allow-legacy
```

The compatibility result is labeled
`evidence_contract=legacy-internal-consistency-only`. It does **not** assert
the new in-loop exposure coverage and must not be admitted as a current witness.

Both strict and compatibility results remain **internal-consistency-only**,
trainer-reported development artifacts. Neither proves independent optimizer
observation, physical pretrained-model improvement, checkpoint custody,
cross-process replay resistance, examination, graduation, or host authority.
