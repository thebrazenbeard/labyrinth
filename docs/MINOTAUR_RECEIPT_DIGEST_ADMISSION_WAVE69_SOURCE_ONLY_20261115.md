# Minotaur Wave 69 — Receipt self-digest admission

Status: SOURCE_ONLY / DRAFT / HOLD — 2026-11-15

## Exact subject
Parent: Labyrinth Draft PR #250 commit `1b6828b7df2adcf7a285d4e75959d0a2ea200d1c`.
Changed: `save_receipt_new` in `labyrinth/minotaur_candidate_bridge.py`.
The incoming `RunReceipt` is already checked for exact inert Python types. The writer previously did not compare its claimed self-digest with its serialized fields.

## Hostile red, real production seam
A locally altered `final_verdict` retained the original receipt digest.
`audit_receipt` returned False, but `save_receipt_new` created the file.
New `test_write_once_rejects_stale_self_digest_before_file_creation` failed on the unmodified parent: `AssertionError: ValueError not raised`; focused exit 1.
The test exercises actual serialization and disk persistence, not a mock of the writer.

## Minimal change
The writer now calculates SHA-256 over the same canonical serialized fields used by `_seal` (excluding `receipt_sha256`) and rejects a mismatch with `ValueError` before opening the target file.
Exact-type inertness, exclusive file creation, existing valid receipt schema, and conditional replay behavior are preserved.

## Observed verification on Lappy
- Windows Python 3.14.5: focused new test GREEN, full candidate bridge 32 PASS; subprocess adapter 14 PASS.
- WSL Python: candidate bridge 32 PASS normal, 32 PASS optimized (`-O`).
- `python -m compileall -q labyrinth` PASS; `git diff --check` PASS.
- Total reported focused Windows cases 46; WSL bridge cases 64. These are not full-suite or GPU benchmarks.

## Strict inference limits
A consistent self-digest is NOT signer authentication, independent private-case custody, model identity, sandbox containment, or proof of training/generalization.
An adversary able to rewrite both fields and digest can still produce a self-consistent receipt; `audit_receipt` itself only confirms conditional replay.
Labyrinth issues #146 and #147 remain open. Do not merge stacked drafts or graduate any model on this evidence alone.
No neural weights, GPU workload, installed runtime, protected branch, host ACL, credentials, or external examiner authority were changed.
