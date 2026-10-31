# R53 — Source-only measured MiniLM CLI opt-in

Status: **SOURCE-ONLY / NOT HOST-ATTESTED / NO GRADUATION AUTHORITY**.

This source change adds a strictly optional way to feed a previously proposed,
signed-development-derived cycle-one TRAIN schedule into the existing physical
five-arm MiniLM runner. The other three-arm and operator-declared dry-run paths
remain unchanged. It does not run a model, obtain a lease, fetch credentials,
authenticate a host-owned fence, provide a sealed examination or certify
learning by itself.

## Producer and separate pins

First, a source producer obtains a `BoundMeasuredNeuralBatches` by calling
`propose_signed_neural_batch_schedule(...)` with the existing four-key signed
development evidence and authentic cycle-zero TRAIN inventory. It may export
the strict canonical JSON manifest with:

```python
from pathlib import Path
from scripts.qualify_maze_conditioned_minilm_cpu import export_measured_cli_schedule_json

# `schedule` must already have been constructed from verified development data.
Path("measured-source-schedule.json").write_bytes(
    export_measured_cli_schedule_json(schedule)
)
```

The exporter does **not** include the signatures and the importing CLI does
**not** reverify their original signing keys. The actual schedule checksum,
candidate run, candidate artifact, predecessor checkpoint and fence must be
obtained independently of the manifest and pinned by a separate authority.
Passing these flags does not prove the caller has such an authority; this is
only a check that self-contained source bytes match the supplied values.

## CLI preflight (no Torch, no files written by the runner)

```sh
python -m scripts.qualify_maze_conditioned_minilm_cpu \
  --output-parent ./runs --run-name source-probe \
  --preregistered-controls \
  --full-optimizer-cycle-continuity \
  --steps-per-cycle 16 \
  --optimizer-run-id "$PINNED_RUN" \
  --optimizer-prior-sha256 "$PINNED_PREDECESSOR" \
  --optimizer-fence "$PINNED_FENCE" \
  --measured-schedule-file ./measured-source-schedule.json \
  --measured-schedule-sha256 "$PINNED_SCHEDULE_SHA256" \
  --measured-run-id "$PINNED_RUN" \
  --measured-candidate-sha256 "$PINNED_CANDIDATE" \
  --measured-predecessor-sha256 "$PINNED_PREDECESSOR" \
  --measured-fence "$PINNED_FENCE"
```

An optional `--execute` advances past preflight into the existing offline
neural trainer only after source, schedule, control-arm and optimizer pins have
matched. **Do not run `--execute` on a shared host until the independent
operator has released the exclusive host lease and authenticated Q8 custody.**

## Fail-closed properties

- All six measured parameters are required together: source manifest path and
  five separately supplied schedule/candidate/lineage pins.
- `--weak-cycle-one` is forbidden in measured mode; the exact weakness is
  reconstructed from the committed source schedule.
- Five-arm `--preregistered-controls` and full optimizer continuity are
  mandatory; optimizer run/predecessor/fence and optional v2 candidate must
  agree with the measured source pins.
- The manifest is bounded to 64 KiB, canonical JSON with no duplicate,
  missing, extra, or privileged fields. Case IDs are resolved solely against
  regenerated authentic TRAIN rows; ordered visits must cover all rows.
- The existing typed `verify_precommitted_neural_batch_schedule` checks the
  exact regenerated Maze, schedule digest, source event/exam IDs, arm list,
  candidate, predecessor, scope and fence before any neural import.
- CLI preflight reports
  `EXTERNALLY_PINNED_SOURCE_SCHEDULE_HOST_UNATTESTED` and explicitly marks
  source signatures as **not reverified at the CLI**.

## Independent gates still required

This remains a caller-supplied source manifest. Independent host-owned
nonrollback authority, event consumption, credential custody, physical
pretrained optimizer updates, matched control arms, multi-seed reasoning
improvement, retained prior skills, structural transfer, and independent Q7
private examination are all outside this change. A matching digest, even with
an exact fence, cannot authorize training, graduation, merge or deployment.

Test with `python -m unittest tests.test_measured_cli_binding_r53 -v`. The
test suite deliberately substitutes the neural backend rather than updating
weights, so it qualifies only the source preflight and actual argument bridge.
