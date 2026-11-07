# Wave 17 — child-wire / decision binding repair

**Boundary:** source-public process-trial receipts, on the same OS-user trust
surface. This is source replay validation, not a signature, private model
attestation, training result, or independent custody.

The independent Wave 17 Skeptic reproduced a real forged-win path on the
published Wave 16 batch verifier: a receipt retaining the child's literal
`{"kind":"ABSTAIN"}` response could relabel the recorded parsed decision as
an otherwise genuine source-oracle winning `PLAN`, recompute unkeyed hashes,
and obtain a positive batch verification and win numerator. Valid child-wire
hashes alone did not establish that the child actually authored the plan.

The repair makes the batch verifier independently decode the bounded raw
child JSON and demand equality with its recorded decision (including the
declared legacy `CLAIM`-as-ABSTAIN fixture). It rejects duplicate fields,
nonfinite numbers, invalid actions/plans, and malformed decisions, even when
the attacker recomputes every receipt SHA. The verifier also enforces exact
integer types for nested process-call accounting and the stated structure of
outer and nested Git source metadata. These metadata remain **self-reported**;
no Git content or host trust is attested by their correct shape.

Independent test-only counterexamples were first RED against Wave 16 source:
five test methods, six reproduced falsifiers (one source/wire mismatch, one
duplicate-key malformed wire, two provenance-schema mutations, two Boolean
budget mutations). After repair all seven Wave 17 Skeptic methods, including
two negative controls, and the earlier 19 batch tests passed under normal
and optimized Python. The adjacent 17 single-case process-trial tests also
passed in both modes. None of this demonstrates neural weight updates or
out-of-sample transfer.

**Remaining:** Full host-private case seeds and candidate stdout still live
inside the complete batch result. CLI redaction is a separate issue. Do not
publish those full receipts or embed credentials in `command_argv`.
