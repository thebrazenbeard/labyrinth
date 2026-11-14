# Source-only staged examiner bundle V1

**Status:** executable local-development feature; **not** a protected
independent examiner, a trained model or crash-durable transactional store.
This is an optional companion to the original three-file
`labyrinth.blind_structural_exam prepare` protocol, which remains unchanged.

## Use on an authorized local examiner machine

The containing directory must already exist and be under the **examiner's**
control. No real candidate process should be given access to the bundle:
`private.json` contains the answer key. A staging directory also contains
this private key, and inherited Windows ACLs are **not** modified or verified.

```powershell
python -m labyrinth.exam_bundle prepare --out-bundle D:\VERA\experiments\example-exam-ready --count 4
```

The command returns a JSON object with a
`manifest_sha256` and a source-only status. Retain that original receipt
SHA independently of the published directory; an operator controlling both
the directory and supposed original pin can fabricate local consistency.

```powershell
python -m labyrinth.exam_bundle verify --bundle D:\VERA\experiments\example-exam-ready --expected-manifest-sha256 YOUR_ORIGINAL_MANIFEST_SHA256
```

Verification checks four exactly named files (no extras), strict canonical
manifest schema and digest, intended ready-directory name, all three original
file hashes and original precommit receipt, and the existing finite-mechanics
exam's private witnesses against public tables. Verification **does read the
private answer key**; it must be performed only by the examiner. It does
not check candidate credentials or infer model performance.

The underlying `private.json`, `public.json` and `precommit.json` retain
their previous compatible format, and can still be used with the original
scorer. Do not hand the bundle directory to a candidate; supply only the
separately scoped `public.json` content over an authorized interface.

## What publication does

1. A sibling directory on the same parent/filesystem is freshly created
   with a staging name such as `.example-exam-ready.stage-...`.
2. The existing original three-file preparation is executed there. It writes
   public questions and source precommit before the private key; the
   original source hashes are read back and validated.
3. A strict source-only manifest is written and the entire staged bundle is
   verified, including original finite mechanics witnesses and exact bytes.
4. Only after these checks, an `os.rename(stage, ready)` exposes the final
   directory name. The verifier rejects staging names and NTFS junction
   aliases as ready bundles.
5. Ordinary staging/write/rename exceptions remove the **newly owned**
   temporary directory and do not create the ready destination. Existing
   destination files or directories are not intentionally replaced.

**Never infer actual examiner custody from these properties.** The SHA is a
caller-held consistency pin. A user with access to the staging parent can
read stage files if ACLs permit. This is not a security sandbox.

## Verified failure tests, and unresolved cases

Tests cover valid preparation/verification, a missing parent, existing
destination, dangling output symlink, symlinked parent, missing bundle,
manifest-write refusal, rename refusal, wrong external manifest SHA,
modified private/public/manifest files, extra/missing members, rejected
staging-directory name, and rejected actual Windows NTFS junction alias.

This is **not** a guaranteed crash-atomic or power-loss-durable directory:
a crash before the final rename may leave an unready staging directory;
filesystem durability and parent-directory fsync have not been qualified.
Even if final rename is ordinarily one namespace operation, its exact
security, filesystem and power-loss behavior must be proved for the
target volume and environment. An attacker racing directory mappings can
still bypass static pathname preflight. Python documents that `os.rename`
has platform-dependent behavior on an existing destination; on Unix, an
existing **empty directory** can be replaced. Therefore the
check-then-rename sequence is not a race-free no-clobber guarantee against
another actor. See:

- Python 3.12 `os.rename`: https://docs.python.org/3.12/library/os.html#os.rename
- Microsoft `MoveFileEx`: https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexa
- Microsoft [Moving and Replacing Files](https://learn.microsoft.com/en-us/windows/win32/fileio/moving-and-replacing-files)

To qualify this for adversarial or protected exams, an independently
administered examiner environment, verified file DACL/handle isolation,
a genuinely distinct candidate identity with an actual denied-read probe,
a separately held signer/receipt, and filesystem crash/fault tests are
required. None are supplied by this new local-development API.

**Source evidence ceiling:** this bundle protects normal source publication
from accidental incomplete-directory visibility and checks integrity.
It does not establish trained neural generalization, independently served
model weights, private examiner custody or a production-ready deployment.
