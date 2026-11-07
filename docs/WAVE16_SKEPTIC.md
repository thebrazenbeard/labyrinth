# Wave16 Skeptic — model-trial evidence and common-mode trust

**Baseline:** isolated PR157 integration source
`16f84839df869d2a89243d55baa212cf92527944`.

**Artifacts:** `tests/test_wave16_skeptic.py` and this document only. The
candidate exercised here is an unchanged local fixed child subprocess; no
neural model is installed, modified, or scored as learned. No remote PR or
host service is touched.

Run: `python -B -m unittest discover -s tests -p test_wave16_skeptic.py -v`
and `python -O -B -m unittest discover -s tests -p test_wave16_skeptic.py -v`.

## S16-01 — common-mode JSON decoder falsely credits an abstaining model

**FAILS (reproduced).** The model-response scorer `_parse_response` and
supposed independent source-wire witness `_independent_wire_decision` both
call the identical `trial._read_json` function. One host-side mutation of
that function changes the parsed result of exact child `_ABSTAIN` bytes to
an already published correct seed-0 PLAN. The fixed subprocess never
proposed a plan, yet the runner returns:

- actual response hash = SHA-256 of the child **ABSTAIN** bytes;
- source receipt replay = true;
- trial binding = true;
- `verified_success = true`.

The two validators shared a single source implementation. Their agreement
was not independent physical or semantic evidence. A fix must independently
parse bounded raw output through a separate strict decoder path, preserving
failure-closed detection if one parser is mutated. A second call to the same
parser is not an independent witness.

This deliberately mutates trusted host Python, not the isolated child, so
it demonstrates the evaluator's *common-mode software-fault* boundary,
not a separately proven process escape.

### Follow-up S16-01b — shared stdlib decoder beneath separated wrappers

Root's first in-progress repair used a second `json.loads` invocation rather
than the scorer's `_read_json` wrapper. That stops the original single-helper
mutation but **still delegates both paths to the same mutable standard-library
`json.loads` function**. A second adversarial test patches only `json.loads`
for the actual fixed-child `ABSTAIN` bytes. The repaired WIP then also
reported a false `verified_success=true` despite a valid ABSTAIN response
hash. This is an intentionally strong *same-interpreter TCB* challenge,
not evidence that an external candidate could patch the host stdlib through
the child protocol. The genuine custody boundary remains unqualified.

## S16-02 — outer trial report SHA-256 is not an authenticator

**SURVIVES as explicitly unqualified evidence, not a secure receipt.** A
recipient can rewrite `model_neural_identity_verified`,
`model_weight_update_verified`, `generalization_qualified`, the trial
verdict, and success flag, then recompute an internally consistent unkeyed
`receipt_sha256` without any evaluator authority. The embedded source
receipt still disagrees with the fake outer WIN. This is a controlled
demonstration of why a hash alone cannot prove provenance; the source
correctly labels model identity unverified. Do not interpret a matching
checksum as independent signed custody.

## S16-03 — candidate-private versus host report disclosure

**SURVIVES at the configured JSON child boundary.** The child-facing public
view omits `seed` and `case_id`; the host's completed trial report includes
the exact case seed twice (`case_seed_host_only` and configuration). The
unqualified report is not an evaluation prompt and must not be distributed
to candidate models during an active withheld campaign. Model/source
contamination remains UNKNOWN until genuine hidden seed custody exists.

## S16-04 — batch receipt aggregation and admission semantics

During an unfrozen Wave16 batch prototype, the batch replay validator
accepted a rehashed receipt whose skipped second row was falsely relabeled
`COMPLETED`, and accepted independently altered `calls_reserved=999` and
`skipped_or_invalid=0` despite a one-call/two-case manifest. An independent
regression now mutates each of those three fields and requires fail-closed
rejection. The author corrected these invariants in the live WIP; the
focused regression became GREEN. Because the original file was untracked
and uncommitted when first tested, the RED is documented as a bounded
observed WIP test, **not** a reproducible historical Git commit. Freeze and
reattack the final source SHA before reporting a durable fix.

## S16-05 — typed receipt schema laundered through Python bool/int equality

**FAILS at immutable Experimenter commit**
`0ffcaba17f326a8b5f9c7ff094775a2bd7d4e8f9`. A genuine fixed-subprocess
one-case batch with `case_seed=1`, `calls_reserved=1`, and
`precommitted_case_count=1` independently verifies. Mutating **any one** of
those three nested integer fields to boolean `True`, then recomputing the
unkeyed `batch_receipt_sha256`, still returns
`verify_batch_receipt(forged) is True` because Python `True == 1`.

`test_batch_reviewer_rejects_bool_spoofed_integer_fields` is a separate
executable red regression with **three failing subtests** against that exact
source. To reproduce, run from the Experimenter checkout:

```bash
PYTHONPATH=. python3 -B -m unittest discover \
  -s /workspace/scratch/633ecda5b2d8/labyrinth-wave16-skeptic/tests \
  -p test_wave16_skeptic.py -v
```

The command runs the full file; the two earlier decoder mutations remain
RED on Experimenter's base because the root's separate decoder hardening
is absent from that source. Do not misclassify these branch differences
as new regressions. Require exact built-in integer typing throughout
receipt-schema admission, not ordinary equality against an integer.

### S16-05b — manifest checksum does not guarantee nested type safety

**FAILS on follow-up WIP atop `0ffcaba` (three independent ERROR subtests).**
Changing `manifest.config` from a valid dictionary to `null`, an array,
or a string, then recomputing **both** the manifest and outer receipt
hashes, causes `verify_batch_receipt` to raise uncaught `AttributeError`
at `manifest["config"].items()`. A malformed receipt must return `False`
without crashing the verifier. Regression:
`test_batch_verifier_fails_closed_on_double_rehashed_wrong_manifest_type`.
Unlike S16-05, this was first run against the author's *uncommitted*
bool-typing follow-up, so exact immutable follow-up SHA is pending.

## S16-06 — child-wire nested receipt and genuine control recheck

**SURVIVES bounded attacks** on clean integrated source
`a0994536e117ed612ccc649c94dc6b4f559a7f89`: the ordinary fixed
child's seed-1 ABSTAIN is now classified `COMPLETED` with
`SOURCE_PROCESS_TRIAL_REPLAY_VALID`; a forged `[None]` process receipt,
forged `raw_response="{}"`, or forged nested child response hash is rejected
even when both trial and batch outer self-hashes are recomputed. This is
an executable 3-mutation regression, *not* independent process identity
attestation or a model win.

The complete independent Skeptic file now passes **8/8 normal and 8/8
optimized (`python -O`)** against exact clean `a099453` source. This
includes both earlier failure classes S16-05 and S16-05b; their historical
RED source/WIP provenance is preserved separately above. No neural
training or network/host model installation was run.

## Evidence limits

These are source and executable TEST results on exact commits, not trained
parameter updates, authenticated model artifacts, independent external
reviewer custody, or hidden-generation qualification. Public deterministic
Switchback source permits catalog/seed memorization. A classifier claiming
neural generalization or graduation from these tests would exceed the
evidence, even after every regression goes green.
