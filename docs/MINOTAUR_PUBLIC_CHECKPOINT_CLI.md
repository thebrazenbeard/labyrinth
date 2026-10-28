# Minotaur source-bound public checkpoint diagnostic

Status: **public research diagnostic only**. This command does not install a
model, issue a graduation decision, sign evidence, or modify custodian state.

Run against an existing output directory from
`python -m labyrinth.minotaur_algebraic_experiment train`:

```bash
python -m labyrinth.minotaur_public_checkpoint \
  --artifact-dir /path/to/training-output \
  --challenge-json /path/to/public-challenge.json \
  --report /path/to/new-public-report.json
```

The report path must not exist. Its parent directory must exist. The command
will refuse symlinked input/output path components, oversized or nonregular
input files, malformed challenge JSON, invalid public claims, and failed
source/checkpoint replay. It publishes a complete report without replacing
an existing file. No behavior against hostile concurrent filesystem mutation
is claimed.

Example challenge JSON (modulo 97, publicly disclosed cases):

```json
{
  "schema": "labyrinth.minotaur-public-challenge.v1",
  "atomic_cases": [
    {"case_id": "add-wrap", "op": "ADD", "x": 95, "y": 5,
     "claimed_output": 3, "group_id": "public-wrap"}
  ],
  "program_cases": [
    {"case_id": "clean", "initial_state": [95, 5],
     "instructions": [["ADD", "A", "B"], ["MUL", "B", "A"]],
     "observed_states": [[3, 5], [3, 15]],
     "claimed_first_error": null, "group_id": "public-program"}
  ]
}
```

Each batch requires 1–4096 atomic cases and permits 0–4096 programs. The
existing independent V5 evaluator recomputes public arithmetic and program
claims before the neural bridge scores them. Exact training-atom overlap is
rejected. To evaluate the complete 28,227-case domain, use independently
audited batches and an exact-coverage manifest; this CLI does not invent one.

The command verifies the training report checksum, checkpoint digest, source
commit and training-source manifest, fresh reconstruction of the training
sample digest, and the known 12-coefficient float64 model state. Its derived
report records source-file hashes observed at import and evaluation for the
checkpoint CLI, public bridge, and V5 evaluator. The local guard rejects
ordinary edits after module import. These file hashes are **not proof of the
bytecode actually loaded**, particularly if a dependency was already imported
before the caller imported this module. Execute from a fresh Python process
and stable, reviewed checkout for meaningful source provenance. Optional
`--expected-source-commit` and
`--expected-checkpoint-sha256` cross-check caller-supplied expected values;
the CLI cannot authenticate where those values came from.

The existing experiment reload contract requires the artifact's Git commit
to equal the **current** checkout's HEAD. Consequently, an older checkpoint
from before this CLI was added is refused on a newer commit even when its
training-file digests are identical. Re-run the bounded training experiment
under the exact CLI-bearing source commit; no implicit cross-commit adoption
is performed.

A successful result is `PUBLIC_DIAGNOSTIC`, never qualification. Both success
and refusal retain `qualification=NOT_ESTABLISHED`,
`graduation_authority=none`, `minotaur_graduated=false`, and `deployed=false`.
The numeric training/reload PID comparison is reported only as a diagnostic;
PID reuse across process namespaces is possible, and this command does not
claim independently trusted process or private-evaluator attestation.

## Windows file-read and main CI qualification boundary (2026-10-28)

The bounded report, checkpoint, challenge, and source reads now use a
native non-following file handle on Windows, refuse directory junctions
and final symlinks, and compare the opened object's identity with its
pre-open metadata. The evaluator checks the report training-sample digest
before spending resources on model replay and rereads report, checksum and
checkpoint bytes after replay to reject observed generation changes.
A transient adversarial replacement of an ancestor directory can still
escape pathname checks, and a hostile actor able to swap files away and
back between checks is outside this local custody model. Neither control
is a cryptographic attestation or an independent private examiner.

The pinned Minotaur neural CPU workflow runs on relevant pull requests
**and pushes to main**, separately from the general Python workflow.
On Windows the exact POSIX FIFO security test cannot execute; its one
known platform-specific skip is recorded explicitly, while all other
required-test omissions and skips fail qualification. Linux must run
the FIFO test. The neural workflow is the definitive CPU-dependency
qualification: a green general CI job without PyTorch is not equivalent.
