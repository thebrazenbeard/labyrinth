# Minotaur: examiner isolation and served-model identity — unresolved qualification gates

Status: **RESEARCH / SOURCE-ONLY / NOT IMPLEMENTED** (2026-11-14). This
note records supported engineering alternatives without pretending any
examiner identity, protected case key, served weight pages, or independent
custody have been qualified. The minimal main-based public mechanics runtime
remains usable without these gates, but is not a neural graduation test.

## Problem evidenced by previous local waves

Local examiner case answers on Lappy were previously observed as accessible
under the same operating identity that runs the model-development tools.
A Windows ACL report and a file-open attempt under that *same* identity do
not demonstrate denial of the file to a different candidate process.

Existing public exam code reports independently controlled custody as false.
Local Ollama `/api/tags` exposes an advertised model digest, which can
identify the server's claimed artifact. It does **not** establish that the
model weights, adapters, or in-memory execution pages observed by a candidate
were independently authenticated or immutable.

These are separate trust problems. Do not treat a passing source test or
a public-rule deterministic solver as proof of either.

## Windows examiner/candidate execution boundary

**Supported external facts:**

- Microsoft documents that a process security context is determined by its
  access token. The file's DACL and the enabled SIDs, group memberships and
  access rights of that token determine whether the requested file read is
  granted.
  https://learn.microsoft.com/en-us/windows/win32/secauthz/how-dacls-control-access-to-an-object
  https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights
- `CreateProcessWithLogonW` can launch a process using separately supplied
  credentials. Its `LOGON_NETCREDENTIALS_ONLY` flag **retains the local
  caller's security token**, so it is **NOT** a valid technique for proving
  a different local candidate identity.
  https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createprocesswithlogonw
- `CreateRestrictedToken` can remove privileges, set deny-only SIDs, and
  restrict a process launched by an ordinary user. That is useful for
  optional sandbox hardening and bounded candidate tests, but a token
  derived and administered by the same operator is not independent
  examiner custody.
  https://learn.microsoft.com/en-us/windows/win32/secauthz/restricted-tokens
- `AuthzAccessCheck` evaluates an access request against supplied security
  descriptors and a client context. An `AuthzAccessCheck` result can be a
  *preflight*, but is not the same evidence class as an OS file-open attempt
  made by the actual candidate process.
  https://learn.microsoft.com/en-us/windows/win32/api/authz/nf-authz-authzaccesscheck
- NTFS supports hard links, directory junctions and reparse points.
  Pathname-only ACL walks require scrutiny of alternate paths and changes
  to directory entries during a check.
  https://learn.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions
  https://learn.microsoft.com/en-us/windows/win32/fileio/reparse-points

**Smallest authorized future qualification trial:**

1. Discover and verify a **genuinely authorized distinct** examiner/candidate
   execution identity and its SID. Do not create an account, request
   plaintext passwords, modify a system ACL, or change sharing in order to
   fabricate an identity. Do not use `/netonly` to simulate this.
2. On a separately approved controlled exam directory, have the examiner
   owner create a new answer file and protect the file **and ancestor access
   paths** using deliberate DACLs. The examiner key should be maintained
   outside candidate control.
3. Verify the actual candidate process token identity and perform an
   `O_RDONLY` / `CreateFile` read-open *without consuming answer bytes*.
   Capture an OS-denied result under that **specific** token. Also perform
   an examiner-owned positive control that opens the file. Log IDs, test
   path digests and receipt provenance, not secret plaintext.
4. Recheck ACLs after the attempt, detect unexpected inheritance, reparse
   points and hard-link bypasses, and preserve the exact original receipt
   under a custodian that the candidate cannot rewrite.
5. Only then consider the statement
   `candidate_principal_read_denial_verified=true` for that exact process,
   file and observation. Keep the broader
   `independent_examiner_custody_verified` gate **UNKNOWN** until
   distinct key management, clean precommit timing and audit control are
   independently demonstrated.

There is **no verified executable second-principal runner** in the current
Lappy-Dedicated tool session. Previous attempts were same-account probes.
The steps above are a researched trial, **not a performed test**. Permission
and account changes require separate authorization and a reversible plan.

## Ollama and model weight identity

**Supported external fact:** Ollama's official API defines `GET /api/tags`
with model names, digests and metadata, and `GET /api/ps` with running-model
metadata. Those are responses from the Ollama service itself, not external
attestation of the pages used to compute a given response.
https://github.com/ollama/ollama/blob/main/docs/api.md
https://github.com/ollama/ollama/blob/main/docs/openapi.yaml

**Candidate engineering route, not yet qualified:**

- In a controlled local deployment, pin the model and adapter files by
  original SHA-256, quantization, runner build, relevant configuration and
  process identity *before* a trial. Capture model-file provenance and
  reproducible command arguments.
- Make the runner load *only* the pinned files and isolate modification
  rights. Independently observe process creation and file-open/mapping
  events; compare with the runner's actual loaded-model state.
- Preserve a verifier-controlled observation and candidate result digest
  rather than relying only on the Ollama endpoint's self-report. Confirm
  the same model stays active across before/trial/after observations.
- Test a deliberately replaced tag or file under a controlled experiment.
  An attestation method must fail on a swapped file, modified adapter or
  differently served model while the displayed model tag remains stable.

Even this is a staged approach. **Hashing a GGUF on disk does not itself
prove the same weight bytes were used during inference**. For true
independent loaded-weight attestation, the trusted runtime boundary and
its instrumentation must be established first. Mark that result UNKNOWN,
not PASS, until a real probe supplies affirmative evidence.

## Current applicable results

- Source-defined candidate actions, explicit deterministic corrections
  and V2 provenance replay are locally testable now. These are different
  from examiner independence and neural-learning tests.
- The actual main-based PR #229 passed source tests on Windows and hosted
  Linux. No trained Minotaur checkpoint or semantic generalization
  qualification was demonstrated by those passes.
- The separate precommit-destination preflight fix on the current branch
  prevents one reproducible class of partial exam creation (missing
  output parent). It does **not** promise atomic multi-file publishing
  or protection from crash, disk-full, permission-denied or a hostile
  operator changing files after the precommit.

**Disposition:** implement and test narrow source repairs now; hold the
independent-identity, DACL mutation, key custody and loaded-weight
attestation effects until a verified, authorized execution environment
exists. Source readiness is not runtime qualification.
