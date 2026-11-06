# R68 Q7 source-only public-summary mutation probe

Status: **SOURCE FIX VERIFIED; CHRONOLOGY AND GRADUATION HOLD**.

This probe tests the Python value returned by
`verify_pretraining_precommit`, not a host-owned signer, evaluator process, or
external event anchor. The public `FinalExamPrecommitSummary` was a frozen
dataclass. Its ordinary assignment defense did not prevent this real local
mutation:

```python
s = FinalExamPrecommitSummary("x", "y", "z", "a" * 64, "b" * 64)
object.__setattr__(s, "graduation_authorized", True)
assert s.graduation_authorized is True
```

The RED test reproduced that mutation, plus a constructor path that permitted
the positive flag. This was **mutable local receipt metadata**, not a proof
that the sealed final could be passed or that the host would graduate a model.

The production summary now uses a tuple-backed return value with a constant
`graduation_authorized=False` property. `object.__setattr__` cannot replace
its exam identity or add synthetic `external_anchor_verified` or private-bank
claims, and callers cannot construct or `dataclasses.replace` it with a
positive graduation claim. The signed input and the existing independent
source/inventory pin checks remain in place.

## Additional R70 hostile finding: signed-byte verification/parse TOCTOU

The signed input is itself a frozen dataclass and still supports low-level
`object.__setattr__`. In the old verifier, Ed25519 authenticated
`precommit.public_bytes`, and then `json.loads` re-read the wrapper. A
deterministic `sys.settrace` hook replaced the wrapper bytes at precisely
that line with an **unsigned** but canonical document whose source SHA-256
matched the caller's expectation; the verifier returned a successful source
receipt despite checking the signature over the *earlier* document.

The regression first failed (the verifier returned without raising). The
verifier now snapshots the public bytes, signature and source-only authority
flag once after the exact wrapper type check. The **same immutable bytes**
are used for Ed25519 verification, JSON parsing and canonical re-encoding.
The trace hook still executes, but the attack returns a pins failure. That
fix is a Python source-level atomicity guarantee against mutation of the
wrapper, not protection from a hostile interpreter capable of altering local
frames, patching the verifier, obtaining signer private keys, or controlling
the independent source expectations.

Regression:

```bash
python -m unittest tests.test_q7_precommit_summary_immutability_r68 tests.test_final_exam_precommit tests.test_sealed_semantic_exam_protocol -q
```

Still **not established** by this fix: genuine pre-training chronology,
independently anchored and read-back source inventory, host-controlled signer
custody, rollback-resistant consumed-event persistence, private-bank isolation,
independent pinned-model execution, sealed scoring, or graduation. This is a
source-only **instance-field** immutability guard, not an authorization object
that can be trusted across untrusted in-process Python execution. Code able to
monkeypatch the `FinalExamPrecommitSummary` class (including its descriptor),
or to substitute a subclass, can still forge displayed values; that boundary
must be enforced by the host and separately witnessed. The Q7/Q8/Q9 physical
gates remain HOLD.
