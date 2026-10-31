# R80 Minotaur-to-HF source integration (non-authorizing)

This source-only integration connects four independently reviewed changes:

- The Minotaur journey operator stages feedback-conditioned **TRAIN-only** cases
  and produces the exact ordered `(case_id, prompt_sha256)` exposure roster.
- The HF `_train_steps` source loop checks the pinned roster before each
  optimizer step and records completed exposures after `optimizer.step()`.
  It snapshots caller-provided TRAIN and label lists before tokenizer callbacks,
  preventing a restored mutable list from hiding an encoded-case substitution.
- New HF reports use the strict `labyrinth.hf-lora-pilot.v3-exposure.v1` schema.
  A removed marker or exposure ledger is not accepted as legacy by default.
  Historic v2/v3 reports require explicit `allow_legacy` and cannot claim the
  new exposure contract; the incompatible MeZO v4 schema remains rejected.
  The source-only checkpoint classifier recognizes the new AdamW schema only
  to route it to the **actual strict preflight**, never as validated evidence;
  MeZO v4 or renamed MeZO metadata still cannot use that route. Its optimizer
  restart, byte-authentication, and graduation indicators are hardcoded
  read-only `False` properties, not caller-supplied dataclass fields; its
  `may_call_v3_preflight` routing hint is also derived read-only from the
  non-authorizing status, never accepted as a caller-supplied boolean.
- Signed Q5 source-screen success, local toy parameter changes, and a
  development exam result are tested not to compose into physical/graduation
  authorization.

The pinned neural CPU workflow requires all of the new source-level cases to
run without skips on both pull-request and `main` path triggers, while retaining
the existing Q5 witness, Minotaur, and Rezon source gates.

## Boundaries

The current roster is staged **before** HF optimization and is not an online
neural-feedback curriculum. The standard `run_pilot` entrypoint is not yet
wired to consume that Minotaur catalog; `_train_steps` integration is exercised
with a bounded fake optimizer test. Source loop and report data share the
trainer's trust domain. They do **not** independently attest host identity,
pre-outcome commitment timing, actual physical LoRA/pretrained parameter
updates, a saved model artifact, independent exam, or graduation authority.
Resume/reload paths that reject `FAIL` can still accept an `INCOMPLETE`
internal audit for unrelated checks; such a result is non-authorizing resume
bookkeeping only and must never be promoted to examination or graduation.

`ROUTING != AUTHORITY` and `RESOURCE_QOS != EPISTEMIC_CONFIDENCE` remain
binding. The R80 branch is eligible only for source-level Draft review,
without merge, live host use, training graduation, or deployment.
