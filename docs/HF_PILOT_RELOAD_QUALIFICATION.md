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
