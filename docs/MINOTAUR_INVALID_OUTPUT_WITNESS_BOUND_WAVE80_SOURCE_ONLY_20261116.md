# Wave 80 — Audit recorded invalid-output witness length

## Pinned source and threat
- Source parent: draft PR #261, commit
  `d5764a3de7f6d24a6cfd75e8f5a193e5a5874070`.
- In `run_candidate`, malformed text is recorded as
  `response[:MAX_RESPONSE_CHARACTERS + 1]` (at most 4097 characters).
- `audit_receipt` previously accepted a new self-hash with longer
  `MODEL_OUTPUT_INVALID` text, even though no runner call could store it.

## RED / GREEN host evidence
- Isolated Lappy Wave80 branch; baseline bridge/process: 59 PASS.
- Red-first test created real invalid-output receipt (9000-char
  adapter response, 4097-char stored witness, audit=True).
- Rehashed two impossible stored witnesses (4098 and 9000 chars).
  Pre-fix replay returned True for both: two assertion FAILs,
  `RED_OUTPUT_WITNESS_EXIT=1`.
- Fix: reject `MODEL_OUTPUT_INVALID` if the stored witness exceeds
  `MAX_RESPONSE_CHARACTERS + 1`, before parsing it.
- Identical focused test plus legitimate 4097-overflow test: 2 PASS,
  `GREEN_OUTPUT_WITNESS_EXIT=0`.

## Regression results and limits
- Windows focused bridge/subprocess/private-read/graduation/learning:
  108 PASS.
- WSL normal relevant suites: 102 PASS.
- WSL optimized bridge/graduation: 75 PASS.
- Total: 285 targeted post-fix executions. `compileall` and
  `git diff --check`: exit 0.
- Receipt digest is an unkeyed self-hash, not model attestation.
  This establishes only a runner/auditor source consistency invariant.
- Examiner same-user private-file exposure (#146) and independently
  custodied semantic mechanisms/model execution (#147) remain open.
- No real model training, independent evaluation, graduation or merge.
