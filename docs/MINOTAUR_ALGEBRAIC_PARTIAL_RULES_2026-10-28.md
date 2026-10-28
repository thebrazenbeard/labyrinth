# Minotaur V4 algebraic neural successor — partial supervision

The historical V4 learned_table contains 3 * 97 * 97 * 97 =
2,738,019 independent trainable logits. Complete enumeration produces
perfect finite-domain recall but its before-exposure score on withheld
atomic combinations was only about 1%. Retain this historical path as
the explicit MEMORIZATION NEGATIVE CONTROL, not a learned rule.

## Compact replacement

labyrinth.minotaur_algebraic_rules.build_algebraic_head contains 12
neural parameters, four coefficients for each of ADD/SUB/MUL over
features [1, x/8, y/8, xy/64]. The operation selector and register
routing are supplied by V4's parser. A fixed differentiable circular
modulo-97 logit projection handles wraparound. These are substantial,
explicit ARCHITECTURAL PRIORS, including multiplication in the basis:
the model learns operation-conditioned coefficients, not
multiplication itself or the grammar of operation words.

TransitionConfig(architecture="algebraic_rule") plugs the same head
into V4's six-step free-running state predictor, preserving its
untouched-register copy prior. It never indexes a table of atomic
operand pairs. Training labels are outputs for 12 tuples per
operation, sampled from operands 0–8 with no modular overflow during
supervision. The remaining 28,191 atomic tuples are withheld.

## Executable local research trial

With optional Torch and Safetensors, use:

    python -m labyrinth.minotaur_algebraic_experiment train --output-dir NEW_PATH
    python -m labyrinth.minotaur_algebraic_experiment reload --output-dir NEW_PATH

The training command refuses overwrite and saves a separate 12-coefficient
checkpoint plus a source-bound report. The reload command verifies
checkpoint file and tensor digests, a fresh process, all-atom scores
and 128 previously seeded free-running programs. A green result is
local reproducibility, NOT an independent trusted examiner.

The candidate is compared against frozen initialization, a same-step
zero-learning-rate control, a wrong-label model trained with
operation-preserving derangement, and a literal lookup baseline that
abstains on unknown tuples. A 128-program evaluation uses unseen
residues, register/constant operands, both targets, wraparound and
lengths 1–6. True states are computed by the separate offline evaluator,
not supplied to model.forward. The inference encoder accepts
include_supervision=False and eval-mode teacher forcing is prohibited.

No checkpoint is activated in Vera and no independent private
Minotaur graduation certificate is issued. Success of an algebra-
biased 12-parameter model is narrower than unconstrained arithmetic
discovery, untrained operation semantics, different moduli or
arbitrarily long programs. The public exhaustive audit is not a
private blind test.

## Evidence integrity and hostile-review limits

The experiment report includes hashes of relevant source files observed at
module import and report creation (not independent attestation of the exact
bytecode executed), the serialized coefficient tensors, saved checkpoint and ordered atomic
and program predictions. A sibling report.sha256 detects unaccompanied
report modifications. Reload bounds local report/checkpoint sizes and rejects
changed source, missing/truncated state vectors and mismatched trial digests.
The module refuses ordinary disk changes after import, but an already-stale
dependency imported before the experiment module cannot be proven fresh by
disk hashes alone. For source-provenance claims, use a fresh Python process
in a stable, reviewed checkout; hostile concurrent mutation is not covered.

These are local **consistency checks**, not signatures or trustworthy
third-party custody: an operator able to rewrite both a report and its
checksum may still fabricate evidence. PID inequality similarly cannot
authenticate process isolation. The operator never receives graduation
authority from a local score, even at 100% on this synthetic family.
