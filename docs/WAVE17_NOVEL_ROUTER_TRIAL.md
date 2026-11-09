# Novel Router local subprocess trial — Wave 17

**Status:** runnable source-public examination instrument, with real child
process fixtures and explicit external-command opt-in. Neither an independently
heldout challenge nor evidence of neural model learning.

The examined problem is the delayed two-lane packet router in
`labyrinth/switchback_novel_transfer.py` (PR159 source). It is mechanically
different from the five published three-bit Switchback families, but all code
and generator seeds are source-visible to a source-aware candidate.

## Run a real zero-update child fixture

```bash
python3 -m scripts.novel_router_model_trial --seed 0 --max-calls 1 \
  --total-timeout 2 --fixture-json '{"kind":"ABSTAIN"}'
python3 -m unittest discover -s tests -p test_novel_router_model_trial.py -q
python3 -O -m unittest discover -s tests -p test_novel_router_model_trial.py -q
```

The default runner does **not** start a neural model. A fixture test launches a
separate Python child process and transmits one bounded JSON public view. It
receives public levels, pending packet state, action choices, prior charged
observations and a **public generation seed**, but **never receives the private
case seed, private rules, case ID or complete oracle result**. The public
deterministic source can still be read by a same-user process; the boundary is
limited to the request bytes. Delivery of the generation seed does not prove
the child actually used it for sampling.

To run an explicitly selected local command, supply its **absolute executable
path** and arguments after `--command`, together with `--authorize-command`.
No shell interpretation, provider credentials, network service or paid compute
is requested by the runner. The child uses a minimal environment without
inherited proxy or token variables, but this is not a strong OS sandbox.

**Default CLI output is a redacted machine-readable summary.** It excludes the
private case seed, child stdout, candidate narrative, command arguments and
full trial receipt. To deliberately print complete host-private JSON, add
`--emit-private-json-stdout` before `--command` (if using that mode). Never
publish complete trial JSON without reviewing its sensitive contents.

## What a receipt records

Frozen `TrialConfig` fixes case/generation seed, calls, per-call and total wall
timeout, and output bound. The adapter records an exact immutable fixture
response sequence or explicitly authorized argv. For each child call, the
runner stores:

- SHA-256 of the exact public JSON request and captured child stdout;
- bounded base64 child stdout bytes, exact length, process status and PID;
- candidate decision recovered from the actual bytes and its charged probe;
- the final source `ProbeSession` receipt, raw proposal and replayable trace;
- source and executable file digests, plus optional explicitly provided local
  model-artifact digest.

The verifier reconstructs each public prompt, checks stdout hashes and byte
limits, re-parses the raw child JSON through a different decoder path,
reconstructs the stateful candidate probe sequence, and independently checks
the source session and underlying router plan result. A replay-valid score for
the *wrong* seed, a substituted PLAN after actual ABSTAIN bytes, or a rehashed
child-response hash cannot count as success.

Source receipts retain verdicts such as `VERIFIED_WIN`, `VERIFIED_FAILURE`,
`PROVEN_UNSAT_UNDER_DECLARED_RULES`, `NOVEL_RULESET_CHALLENGE` and `UNKNOWN` as
appropriate. A declared finite UNSAT proof is scoped strictly to the declared
source rules. `NOVEL_RULE` is an unverified candidate hypothesis, never an
automatic discovery or graduation victory. An unsuccessful process, illegal
model output or exhausted call budget does not earn a verified win.

## Restrictions and unresolved evidence

- `private_custody`, `model_weight_update_verified`, `model_identity_qualified`,
  `neural_learning_verified`, `generalization_qualified` and
  `graduation_eligible` are all explicitly **false**.
- The full report contains a **host-private source seed and raw model output**;
  do not automatically publish its full JSON in public PR comments. The
  default CLI omits private fields; the opt-in version does not.
- A local executable hash is a measured file on the current host, not a
  verified model-weights chain of custody. A same-user process can read source,
  files and deterministic generation rules; process PID/time and byte hashes
  are not independent physical attestations.
- All hashes are unkeyed integrity checks. An attacker who controls the
  runner, both parser implementations or Python itself remains inside the
  trusted computing base. No assertion of resistance to arbitrary same-process
  mutation, OS compromise or fabricated signed custody is made.
- Source-public synthetic cases are not a secret evaluation split. Multiple
  seeds are not independent mechanical families. No weight updates or actual
  neural reasoning transfer were measured in the fixture tests.

Evidence classifications remain **SOURCE**, **TEST**, and **UNKNOWN** for
unperformed host/model experimentation; a passing source test cannot be
promoted to a host-qualified model-learning result.
