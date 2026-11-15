# Wave 72: model-call exception receipts are diagnostic, not replay verified

## Exact parent and attack
- Parent: `db491344d23a9439c2d9551e96d007779cf75cc2` (Labyrinth draft PR #253).
- Original `audit_receipt` accepted a digest-resealed `MODEL_CALL_ERROR` receipt
  with empty response even when the actual runner generated a valid ACTION.
- A receipt self-hash is not a model invocation or exception attestation.
- RED regression: `test_model_call_error_cannot_be_audited_without_replaying_failure`
  failed because `audit_receipt(forged, ...)` returned True; `RED_EXIT=1`.

## Narrow change and expected behavior
- Auditor now rejects unprovable `MODEL_CALL_ERROR` as well as `EVALUATOR_ERROR`.
- Production runner still records an actual exception as `MODEL_CALL_ERROR`;
  a synthetic throwing adapter regression verifies recording and failed audit.
- Existing subprocess timeout test now checks recorded error and unavailable
  independent audit, rather than claiming a timeout was replay verified.
- Parseable action/claim and syntactically invalid-output replay remains intact.
- GREEN focused regression: 1/1 PASS, `GREEN_EXIT=0`.

## Local verification (Lappy)
- Windows: bridge, subprocess and private-read suites: 57/57 PASS.
- Windows: Python compileall source: exit 0.
- WSL: bridge and subprocess: 51/51 PASS.
- WSL optimized Python (-O): bridge 37/37 PASS.
- `git diff --check` exit 0; 145 targeted test executions pass.

## Hostile limits
- Neither source checks nor self-hashes authenticate actual model responses.
- Independent signer/custody, same-principal private file denial (issue #146),
  and heldout mechanism independence (issue #147) remain unqualified.
- No model learning, model graduation, runtime deploy, or protected-host effect.
- DRAFT/HOLD only; hosted CI is a separate gate.
