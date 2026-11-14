# Minotaur examiner source publication and path-safety repair

**Evidence classification:** SOURCE INSPECTION, TEST, LIVE WINDOWS PROBES, and RESEARCH are kept separate. No independent examiner custody, true transactional filesystem, authenticated model weights, or model training is claimed here.

## Actual defects and repairs — 2026-11-14 on Lappy

These repairs build on the three-file source-defined examiner in main-based Draft PR #229 and destination preflight Draft PR #230.

**Failure 1: private answers published before public source or precommit.** The old routine wrote the private answer file first, then public cases, then hashed them and wrote the precommit. In a test-controlled failure of the public writer or precommit writer, the private answer file **remained** even though source preparation failed. Two tests confirmed this as RED exit1. A third test established that private output was first, contrary to the intended commit-before-secret-publication boundary. Source repair computes SHA-256 from the exact canonical serialized private/public bytes **in memory**, writes and read-back checks the public file, then writes and read-back checks the receipt committing to both exact byte strings, and **only then** writes and read-back checks the private file. Those same three tests passed. Normal original scorer cases validate that receipt hashes still bind both exact files.

**Failure 2: output-parent symlink or NTFS junction redirects private answers.** Actual isolated Lappy reproductions created a directory symlink and a Windows junction to another temporary directory. Both passed the earlier checks for a valid parent and generated private answers under the target rather than the caller's literal path. Test-only assertions failed RED exit1 on both; source now checks each ancestor of every output path for final-component symlinks or Windows reparse-point file attributes. Under the actual Lappy filesystem, both tests pass and the redirected private target remains absent.

**Failure 3: readback differs from intended source bytes.** After writes, the current source compares the complete read-back public/receipt/private bytes against its canonical in-memory source before reporting success. Injected mismatching readbacks for public or precommit fail before private publication. A mismatching private readback fails the preparation operation rather than returning a success receipt. This is local consistency, not an independent attestation.

Existing defenses remain: the three output names must be distinct and freshly absent, final-component dangling symlinks are explicitly rejected, and output parent directories must exist. None of these changes modify the evaluator's original scoring rubric or assert neural improvement.

## What this does NOT solve

The legacy API accepts **three independently addressed files**, possibly on different volumes. The new order is not a transactional commit across three distinct paths:

- A crash or disk-write failure while writing the **private** file can still leave a partially created private file. Public and precommit might already exist.
- A crash after public and precommit publication but before private publication can leave a public question set and commitment with no private case file. `score_exam` must reject absent or corrupted files, not assume `precommit.json` means the bundle is complete.
- Static ancestor checks do not prevent a privileged concurrent process from swapping an ancestor directory or final path **after** preflight. They do not establish DACL isolation or absence of hard-link aliases.
- `_write_once` writes with exclusive creation and fsync per file; it is not a Windows no-reparse handle API or a multi-file transaction. It can fail after creating a partial file.
- An examiner-controlled precommit hash cannot authenticate an independent signer or prove that an answer key was withheld from the candidate.

## Researched stronger publication design — not implemented

1. Offer a new **bundle-directory API** alongside the historical three-path interface. Stage all files in an examiner-owned private directory *on the same NTFS volume* with restricted ACLs, verified SHA-256 contents, and a manifest recording the case count, source version and file hashes. The private answer must remain outside any candidate-readable directory.
2. Publish a *single examiner-controlled ready directory* (or an examiner-held completion pointer) **only after** staged files are complete and verified. One same-volume directory rename provides a more coherent visibility boundary than three unrelated writes; it does not by itself guarantee data durability after abrupt power loss or solve permission control.
3. Have consumers accept only a verified **manifest/pointer** and validate the exact path identities and bytes before scoring. Never promote legacy files to a completed bundle merely because their hash receipt exists. Test crash injection at every write/rename boundary, disk-full and deliberate competing symlink/junction changes.
4. For Windows path-race resistance, use handle-relative access where supported or a narrowly reviewed `CreateFileW` wrapper with appropriate reparse-point behavior, captured file/volume identities, and policy for directory reparse points; verify on real Windows. Do not claim `Path.is_symlink()` preflight alone is a race-free sandbox.
5. Keep independently administered examiner and candidate OS identities, signer custody, actual candidate denied-read and model weight attestation as **separate** gating requirements.

Microsoft documents [same-volume directory moves](https://learn.microsoft.com/en-us/windows/win32/fileio/moving-directories) and [MoveFileEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexa). Python documents `os.replace` rename semantics, including possible cross-filesystem failure:
https://docs.python.org/3.11/library/os.html#os.replace

**Do not select Transactional NTFS (TxF) as a new dependency.** Microsoft strongly recommends alternatives because TxF may disappear in future Windows releases:
https://learn.microsoft.com/en-us/windows/win32/fileio/transactional-ntfs-portal

This is a researched design direction rather than executed cross-crash qualification. A source-only commit is not an OS isolation test, and a deterministic public-rule solver is not neural generalization.
