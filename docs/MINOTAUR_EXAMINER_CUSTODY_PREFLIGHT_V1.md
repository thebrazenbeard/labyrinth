# Wave42–43 — examiner private-file ACL and local read-open negative gates (Windows)

**Status: executable read-only hostile preflight, NOT independent examiner custody, verified candidate exclusion, or permission repair.**

This source exists because [the blind structural examiner](MINOTAUR_BLIND_STRUCTURAL_EVALUATOR_V1.md) prepares a private answer-key file but source-level separation into different *paths* is not OS access isolation. A committed hash or valid signer key does not prevent a candidate or a common operator from reading/altering answers before testing.

## What the tool verifies—and what it refuses to claim

`labyrinth.examiner_custody_acl` makes a bounded read-only Windows `Get-Acl` pass over a specified **existing private-case file** and every ancestor up to the volume root. Optionally it repeats that pass for the **examiner's actual private signing-key file**, if one exists. It resolves ACL identities to well-known Windows SIDs rather than locale-specific account names.

The following *Allow* ACEs are treated as potential failures:

- High-bit Windows `GENERIC_READ` (`0x80000000`), `GENERIC_WRITE` (`0x40000000`) or `GENERIC_ALL` (`0x10000000`) are conservatively treated as data read/write even when a `FileSystemRights` signed 32-bit mask lacks corresponding low-order file bits. Wave43 demonstrated the previous silent miss with four red test cases.
- File data READ allowed to Everyone (`S-1-1-0`), Authenticated Users (`S-1-5-11`), Builtin Users (`S-1-5-32-545`), Interactive, Network, Builtin Guests or Power Users
- File/ancestor write, append, delete, permissions-change or ownership-change allowed to any of those broad principals

The resulting exclusive-create report is an **unkeyed local receipt** with conservative findings and a self-hash. The tool does **not** change accounts, directory permissions, ownership, ACLs, model weights, running services, authentication policies, or credentials.

Run before a candidate process gains access to public cases:

```powershell
python -m labyrinth.examiner_custody_acl --private D:/EXAM_PRIVATE/private-cases.json --out D:/EXAM_AUDITS/private-acl-before-candidate.json
```

If an examiner-owned private signing key exists, add `--signer-private-key D:/EXAM_PRIVATE/examiner-ed25519.key`. Never pass a public key as if it were secret. Never commit the actual private case file, private signing key, Windows account SID, or local audit artifact to public Git.

The CLI returns **exit 2** when broad-allow exposure is observed and **exit 3** when no broad allowance was detected or inspection is incomplete; there is deliberately **no exit 0 qualifying independent custody**. It prints a concise finding status and original local receipt SHA.

Status meanings:

- `INSECURE_BROAD_ALLOW`: at least one broad-group *Allow* ACE potentially exposes private file contents or modification rights. More restrictive *Deny* ACEs could alter effective access, so this is a conservative potential-exposure finding
- `ACL_INSPECTION_INCOMPLETE`: read-only ACL data were malformed, unavailable or unresolved; never count this as protected
- `NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED`: these particular broad *Allow* ACEs were not seen, but separate-principal access, group membership, nested rights, administrative privileges, host ownership and independence remain unverified

The report **always** sets `independent_custody_verified=false`, `candidate_principal_read_denial_verified=false` and `acl_isolation_verified=false`. A locally produced report, even for apparent least-privilege permissions, cannot authenticate independent examiner control.

## Concrete Lappy readback — 2026-11-13

On Lappy, the current Windows process identity was `LAPPY\patri`. The actual existing examiner answer-key file at `D:\VERA\experiments\minotaur-wave38-authored-20261113\private-cases.json` inherited **Authenticated Users: Modify** and **Builtin Users: ReadAndExecute**. Its ancestors also granted inherited broad modify rights. The source-host audit:

```text
status                         INSECURE_BROAD_ALLOW
findings                       BROAD_READ_PRIVATE_CASES
                               BROAD_WRITE_PRIVATE_CASES
                               BROAD_WRITE_ANCESTOR_OF_PRIVATE_CASES
acl_inspection_complete        true
independent_custody_verified   false
candidate_principal_read_denial_verified false
exit code                      2 (expected denial)
receipt SHA-256                 c792a0763fc240b1a272ea9b268d65d640d1dc6481616abd8d268b0bb1c618de
```

Only Lappy was reported connected by Executor at this check. No second examiner host or separately authenticated user process was available to verify actual **denied** access. Nothing was elevated, re-permissioned, or moved; the existing private exam has **not** been upgraded to an independent holdout.

## Wave43 real local process read-open probe — negative evidence only

The separate `--probe-local-read` mode attempts to open the **existing original private answer file** using the actual running Windows process token, then closes the file handle **without reading any bytes**. It does not impersonate a separate account. The optional examiner SID is supplied by the caller and only checks whether the current token SID matches that declaration. That declaration is **not** an authenticated examiner identity or a trust anchor.

```powershell
# Run this from the exact Windows account which is intended to be the candidate.
# 'S-1-...' is the examiner SID supplied externally, not a credential.
python -m labyrinth.examiner_custody_acl --probe-local-read --private D:/EXAM_PRIVATE/private-cases.json --examiner-sid S-1-... --out D:/EXAM_AUDITS/local-read-open-receipt.json
```

The executable returns **exit 2** on `PRIVATE_READ_HANDLE_OPEN_SUCCEEDED` and **exit 3** on `PRIVATE_READ_OPEN_DENIED_LOCAL_ONLY`; neither is an admission or custody PASS. Other errors fail without a qualification artifact. The receipt contains the status, original path **hash** and running SID **hash**, the claimed-examiner-SID hash if supplied, an exact same-claimed-principal boolean, `private_file_bytes_read=0` and explicit **false** for both `candidate_principal_read_denial_verified` and `independent_custody_verified`. No private answers, credential material, raw SID or private file path are written into the receipt.

On **actual Lappy**, the Wave43 run used the same account (`LAPPY\\patri`) as the existing private-case preparation and produced:

```text
status                           PRIVATE_READ_HANDLE_OPEN_SUCCEEDED
read_open_succeeded              true
same_declared_examiner_principal true
private_file_bytes_read          0
candidate_principal_read_denial_verified false
independent_custody_verified     false
CLI exit                         2
receipt_sha256                   9f4739104e5c382c48fe483672a611934b0c86691e4dc29c33be1b80da43c5a8
```

The actual file handle was successfully opened and immediately closed without reading contents. The locally held report resides outside Git at `D:\\VERA\\experiments\\minotaur-wave43-local-access-20261114\\same-token-access-probe.json`. This is direct proof of read access under **that same current token**, not an attempt as a separately authenticated candidate user.

A *denied* open under the same identity, or a fabricated local report, cannot establish independent examiner custody. An actual protected held-out run still requires a **different authenticated candidate principal** to attempt a denied read under a separately controlled examiner. This source will **never** set independent custody true.

## Remaining qualification gates

1. Establish an examiner-controlled separate Windows principal or independently administered host with protected credentials and recoverability, subject to current authorization.
2. Confirm private cases and private signing keys cannot be read/modified by the **actual candidate execution identity** using a real OS denied-access probe under that identity. Merely reviewing ACL text is insufficient.
3. Preserve original case commitment and trusted examiner public-key anchor outside model/candidate/operator control before model execution; bind any signed receipt to that exact source exam.
4. Verify model-serving identity, installed model/adapter weight bytes, training receipts, matched controls and held-out semantics separately. Signed source commits and green CI are not evidence that a neural model learned.

These are distinct **permission audit**, **real candidate access probe**, **independent key custody**, **model-serving verification**, and **neural-learning** evidence classes. None can substitute for another.

To reproduce the source and host tests:

```powershell
python -m unittest discover -s tests -p 'test_examiner_custody_acl.py' -q
python -m unittest discover -s tests -p 'test_examiner_private_read_probe.py' -q
python -O -m unittest discover -s tests -p 'test_examiner_custody_acl.py' -q
python -O -m unittest discover -s tests -p 'test_examiner_private_read_probe.py' -q
python -m unittest discover -s tests -q
```

The Windows integration test reads ACLs of a throwaway source-only test file. Other operating systems skip that integration while still running pure policy-unit tests.
