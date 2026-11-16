# Wave68 — Receipt object inertness at audit and persistence seams

Date: 2026-11-15.
Parent: Labyrinth Draft PR #249 at `0603cfbfdddf17ae766472378ab18512738a3983`.
Branch: `fix/minotaur-receipt-object-inertness-20261115`.
Evidence: SOURCE_INSPECTION + TEST; NOT authenticated neural execution
or independent private-case examiner custody.

## Hostile source-state counterexamples (actual RED)

An untrusted `RunReceipt` Python subclass passed `isinstance`
and `audit_receipt` accepted it after reading fields that could invoke
subclass `__getattribute__`. The exact hostile test
`test_receipt_subclass_never_runs_host_callback_during_audit_or_save`
failed `AssertionError: True is not false`; exit 1.

An exact `RunReceipt` with a subclassed nested `TurnReceipt`
reached `dataclasses.asdict` through write-once serialization.
`test_nested_programmable_turn_cannot_execute_during_audit_or_save`
failed `AssertionError: TypeError not raised`; exit 1.
The malformed nested subclass reached persistence.

An exact `RunReceipt` carrying a programmable scalar with an
attacker-authored `__deepcopy__` likewise reached serialization.
`test_receipt_scalar_object_cannot_run_deepcopy_callback`
failed `AssertionError: TypeError not raised`; exit 1.
These are programmable *Python objects* supplied to local replay
and persistence, not an exploit through JSON model stdout or
a demonstration of reading private case material.

## Minimal repair

`_inert_receipt_shape` checks `type(receipt) is RunReceipt`,
exact built-in types of all receipt fields, an exact tuple of no
more than `MAX_ACTIONS + 1` turns, and exact `TurnReceipt`
instances with primitive text or nullable text fields.
Before any `asdict`, `audit_receipt` returns False for malformed
objects, while `save_receipt_new` raises TypeError.
Legacy valid `RunReceipt` instances and published schema remain
unchanged; no model logic or signed-trial/evaluation gate changed.

The same hostile tests are GREEN after the repair, without calling
malicious receipt/turn/scalar methods or creating output files.

## Qualification and follow-up

The candidate still runs without an authenticated separate security
principal on Windows and same-user subprocesses can open known
private files. Inert Python receipt objects block a callback seam;
they do not make self-hashed evidence authentic or establish signer
independence. Issues #146 and #147 remain OPEN.

No host ACL or credentials changed. No real neural training,
model graduation, protected merge, deployment, or runtime
qualification. Hosted CI and separate operating-system read-denial
must be verified independently against the published exact head.
