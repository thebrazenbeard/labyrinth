# Single-case Switchback process trial — private receipt export

**Status:** source-only CLI privacy correction stacked directly on the Wave16
P0 single-trial runner. This does not establish a trained model, private heldout
custody, authenticated child execution, OS sandboxing or graduation.

`scripts.switchback_model_trial` now emits a small, machine-parseable
`labyrinth.switchback-real-model-trial-public-summary.v1` JSON object on stdout
by default. Its public values are source-reported verdict and exit status,
explicitly **not** independently attested model claims. It does not expose
case seeds, model responses, prompts, trace, model paths or child receipts.
The summary is **not replayable** as a complete evaluation receipt.

For an actual private offline receipt, specify an absolute file path:

```sh
python3 -m scripts.switchback_model_trial --seed 0 --mode fixed \
  --max-calls 1 --timeout 1 --total-timeout 3 \
  --private-report-file /absolute/private/single-trial.json
```

The complete JSON is staged in the target directory in a mode `0600` file,
flushed, then published through an atomic exclusive hard link. An existing
target or final-path symlink is never overwritten. Failed saves return nonzero
and print only a redacted failure summary. The parent directory and same-user
local processes remain outside the source-level privacy boundary; file
permissions are not an independent custody attestation.

**Windows qualification:** The private file exporter requires `os.fchmod`.
On Windows Python 3.12 and earlier that API is unavailable; attempting
`--private-report-file` now fails closed with a redacted
`private_report_save_failed` summary and exit code 2, without publishing
private output to stdout. Python 3.13 introduced Windows `fchmod`, but its
permission-bit behavior does **not** prove POSIX-style mode `0600` access
control or a verified Windows ACL. Do not claim private file export is
qualified on Lappy until its exact Python version, filesystem ACLs and
readback have been checked. The OS-missing-API adversarial regression is a
simulated source test, not that physical Windows qualification.

For older offline parsers that explicitly require the full receipt on stdout,
`--emit-private-json-stdout` restores that behavior. It is **unsafe** for CI
logs, bug-report uploads, shared terminals and ordinary transcript capture.
The private-file and unsafe-stdout switches are mutually exclusive. Always
protect the complete file or stdout stream: it can contain host-only case
seeds, raw model text, full source receipts and local model/artifact paths.
Never place credentials in command arguments.

The Python `run_trial` API and source verification/scoring contract are
unchanged. Neither a public summary nor a self-hashed private report proves
neural-weight updates, model identity, generalization or physical model
attribution. Original reports already written to logs are not retroactively
redacted by this change.

```sh
python3 -m unittest -v tests.test_switchback_single_cli_privacy
python3 -O -m unittest -v tests.test_switchback_single_cli_privacy
```
