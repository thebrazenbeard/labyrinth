# Wave15 Skeptic — real-model evidence and evaluator trust

Evidence: local source/test, not a neural model trial or authenticated runtime.

- Target: isolated combined Labyrinth `2345c9c69ad0882cb32c8a584a3d0677af986123`.
- Live GitHub draft PR #155 when read: `915d29c72220cec585d49bb35a01ccd412c9b1ee`, source-only Switchback stacked on #154.
- Isolated test-only branch: `wave15-skeptic`; no production or remote write.
- Reproduction: `python -B -m unittest discover -s tests -p test_wave15_skeptic.py -v` and `python -O -B -m unittest discover -s tests -p test_wave15_skeptic.py -v`.

## Defect S15-01 — class-wide equality spoof breaks canonical admission

Source: `labyrinth.switchback_oracle._valid_case` delegated complete canonical
trust to `case == oracle_case(case.seed)`. An otherwise canonical `Case` with
the wrong target or mechanism signature passes after **one** patched Python
`Case.__eq__` returning True. `ProbeSession(forged_case)` admits that same
noncanonical case even though later receipt verification may reject claims.

Two independent RED tests preserve direct validation and session-constructor
admission failures. This is a same-interpreter adversarial mutation, not a
candidate-isolated subprocess exploit. A minimal fix must compare exact
primitive fields and nested Rule types rather than invoking mutable equality.

## Negative control S15-02 — a script obtains a perfect diagnostic score

`test_script_only_can_score_perfect_public_diagnostic_without_training`
supplies a model-free Python object implementing `.predict/.eval/.train`, plus
an equally synthetic `torch` shim. It scores **3/3 exact arithmetic cases**
via the `evaluate_compact_public` adapter, with no weights or neural updates.
The diagnostic correctly retains `qualification=NOT_ESTABLISHED` and
`graduation_authority=none`. Therefore this is a **PASS-as-negative-control**,
not a flaw in that evaluator. A later model runner must not automatically
promote a scored public response to a verified neural model trial.

## Source-only conclusions and explicit non-claims

`minotaur_trusted_launcher.py` limits candidate call surfaces to primitive
configuration and subprocess transport, while declaring the same-OS-user
boundary **not a sandbox**. `candidate_public_summary` uses an unkeyed
self-hash and labels all results `UNATTESTED_HOST_SELF_HASH`; these receipts
do not authenticate neural weights, hidden data custody or training updates.

The existing `hf_report_audit.py` explicitly classifies even a passing pilot
report as internal consistency only, and documents that a fabricated but
consistent report may PASS. No later conversion from these source-only
diagnostics to neural learning or graduation evidence is warranted.

The source-visible deterministic exam generation and public case metadata
leave seed identity and training/evaluation contamination unresolved until a
separate private evaluator demonstrably freezes split and seed custody.

**Verdict:** FAILS under S15-01 canonical admission mutation on frozen base;
model identity/weight learning/generalization remains UNKNOWN. See new-model
runner-specific trials when actual frozen source becomes available.

### Post-fix readback (separate working tree)

Combined integration advanced to exact clean `d89b1affc65939fd42cb23358800b4699230821c`.
Replaying all three tests from this isolated branch against that tree gave
**3/3 PASS** normally and **3/3 PASS** with `python -O`. The two former RED
admissions now fail closed. This verifies the corrected source at that SHA;
it does not retroactively erase the original counterexamples or establish
execution custody of a future model runner.
