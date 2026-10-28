# Reload evaluation exit status and qualification boundary

The pretrained-Lora eval command always writes a separate diagnostic report
including process IDs and whether candidate trials matched the training report.
The CLI now exits nonzero for a false trial match or a non-distinct process ID
instead of returning success with a failed reproduction flag. A successful
result reports its limited scope: independent_executor_attestation=false.

This is an executable process-exit guard, not an authenticated independent
Minotaur graduation, proof of genuine learning, or a cryptographic signer.
Callers that use the Python reload_evaluate function directly must inspect its
diagnostic status; the CLI is the fail-closed operational boundary.

Reload report creation now uses descriptor-relative, exclusive file creation
through a verified non-symlink parent directory to reject ordinary path
replacement and symlink-parent races. This requires POSIX `dir_fd`,
`O_DIRECTORY`, and `O_NOFOLLOW` support. On Windows, where standard Python does
not provide equivalent guarantees, reload-report creation refuses operation;
the rest of the research pipeline is not thereby qualified on Windows.

This guard does not protect against an adversary who already has permission to
move the opened output directory into a model/checkpoint directory during
the final write. Keep the output and pinned checkpoint trees under separate,
trusted filesystem custody. Passing the source-level tests does not establish
an atomic checkpoint-integrity guarantee or an independent evaluator.
