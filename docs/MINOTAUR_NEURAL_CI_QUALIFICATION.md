# Minotaur saved-checkpoint CI qualification

`.github/workflows/minotaur-neural-qualification.yml` is a separate, CPU-only
qualification job for the committed `models/minotaur_scratch_v1` artifact. It
exists because the ordinary dependency-free test workflow intentionally does
not install PyTorch or Safetensors and therefore cannot execute this neural
checkpoint.

The job is pinned to Ubuntu 22.04, x64 CPython 3.12, `torch==2.5.1+cpu` from
the official PyTorch CPU index, `safetensors==0.4.5`, and `numpy==1.26.4`.
NumPy is an explicit digest/runtime dependency: the model parameter
fingerprint uses torch.Tensor.numpy(), which fails when NumPy is absent. Its GitHub actions
are full commit SHAs, its token has read-only contents permission, credentials
are not persisted, and package installation accepts wheels only. Pip resolves
PyTorch's runtime dependencies from its standard package indexes; these
transitive versions are not individually pinned, so this job is a compatible
runtime smoke rather than a fully reproducible dependency lock. The core project retains `dependencies = []`; these
packages are not installed by its normal CI job or library users.

It has a five-minute job limit and a 90-second command limit. The execution
test performs exactly one `reload_scratch` validation of the committed
Safetensors/report pair and one guarded inference on a fixed three-step trace.
Reload checks the stored file hash, strict state-dict keys, complete parameter
digest, and frozen dev/transfer prediction digests. Guarded inference reloads
the same checkpoint again before producing an advisory verdict. Any runtime,
artifact, digest, state-dict, inference-result, or timeout mismatch fails the
job.

This is a software-integrity and compatibility check only. It does not train,
download a model, load pickle or remote code, execute generated model text, or
qualify the model for graduation. The asserted outcome remains
`qualification: NOT_ESTABLISHED` and `graduation_authority: none`.

`tests/test_minotaur_neural_ci.py` also statically guards the workflow's
separation and version pins during ordinary dependency-free tests. Its actual
checkpoint test is skipped when optional packages are absent; the dedicated
workflow preflight requires those packages and versions before invoking it, so
that job cannot pass by skipping neural execution.
