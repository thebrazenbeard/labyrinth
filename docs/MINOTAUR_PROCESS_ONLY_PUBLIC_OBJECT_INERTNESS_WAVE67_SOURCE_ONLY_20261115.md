# Wave67 — Process-only public-view object inertness and replay guard

Date: 2026-11-15. Parent source: Labyrinth Draft PR #248
`0e7da28602275698f6b6b5943ee0812df4071a99`.
Branch: `fix/minotaur-process-public-view-inertness-20261115`.
Evidence class: SOURCE_INSPECTION + TEST only. This is **not**
independent examiner custody or a qualified sandbox.

## Reproduced host callback boundary gaps

The public-view snapshot routine `_public` accepted arbitrary
`collections.abc.Mapping` objects and called `items()` in the
examiner Python process. `run_candidate` serialized
caller-supplied public objects **before** checking adapter mode or
authorization. Thus a user-defined dictionary subclass could
execute Python methods on the host side, despite subsequently asking
for a subprocess-only candidate.

**Actual RED on the parent source:** the process-only regression
`test_process_only_does_not_invoke_python_mapping_callbacks_in_public_input`
failed `AssertionError: ValueError not raised` (one failure, exit1).
Source inspection showed the programmable `items()` path was reached.
This is a synthetic local Python object, not proof of access to
a real private examiner key or evidence that subprocess stdout carries
arbitrary Python objects.

**Second independent RED on source before replay correction:** an
actual synthetically invoked process-only run produced a replayable
self-hashed receipt. Passing a dictionary subclass with programmable
`items()` to `audit_receipt` returned **True**, not refusal:
`AssertionError: True is not false` (one failure, exit1).
Replay had not applied process-mode plain-container rules.

## Repair and behavioral scope

- Validate adapter type, process-mode requirement and explicit
  authorization **before** traversing any caller public view.
- `_public(..., require_plain_containers=True)` permits only exact
  inert builtin `dict`, `list`, `tuple` and JSON scalar types.
  Inspect `type()` before any mapping or sequence methods.
  Recursively enforce on input, examiner initial view, public-rule
  binding and action/claim feedback for **process-only** execution.
- Process-only receipt replay applies matching validation when
  `adapter_provenance == "MODEL_SUBPROCESS_UNATTESTED"`.
- The existing *explicitly unqualified* in-process development
  path retains legacy custom-`Mapping` support, including the
  preexisting deceptive-mapping and split-feedback tests.
- No receipt schema, examiner signature, candidate command, private
  host ACL, model identity or training machinery changes.

## Execution receipts

- Same first hostile test GREEN (1/1, 0 failure).
- New process-only cases for nested list subclass, programmable
  examiner feedback and unauthorized adapter ordering passed.
- Same receipt-replay hostile test GREEN (1/1, 0 failure), and
  an ordinary process-only receipt still replays correctly.
- Lappy Windows Python: `test_minotaur_candidate_bridge.py`
  **28/28 PASS**; `test_minotaur_subprocess_adapter.py` **14/14
  PASS**; `test_round3_bridge_public_query.py` **2/2 PASS**;
  `test_round3_bridge_oracle_integration.py` **2/2 PASS**.
- WSL Ubuntu26.04 Python: bridge **28/28 normal**,
  **28/28 optimized**; subprocess **14/14 PASS**.
- `compileall` and `git diff --check` PASS locally.
- Hosted CI on exact published head must be separately established;
  do not infer full-suite PASS from the focused evidence above.

## Material hostile limitation

> A plain JSON container does not restrict the trusted examiner's own
> Python methods, and a same-user child can still read known private
> case files. An untrusted operator can fabricate or relabel the
> self-hashed receipt's `adapter_provenance`. The guard rejects
> programmable public data in the stated process/replay path; it is
> **not** independent examiner identity, credential custody,
> filesystem isolation or attested neural execution.

Issues #146 and #147 remain open. No neural training, model
graduation, protected merge, deployment or host ACL/credential
changes. Do not elevate source/test PASS to private-exam qualification.
