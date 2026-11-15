# Wave66 — Cross-attempt training-family leakage in Minotaur graduation gate

Date: 2026-11-15. Source base: Labyrinth Draft PR #247,
`79a58048076ec5582d2b1eef938d8dbd54f298d7`.
Branch: `fix/minotaur-training-eval-family-disjoint-20261115`.
Scope: source-only policy-validation repair to issue #147, not an
independent semantic-custody solution or actual neural model evaluation.

## Literal hostile finding: signed two-attempt false graduation

`GateState` stores consumed **evaluation** semantic-family IDs but not
prior **training** families. The existing evaluator blocked same-attempt
training/final overlap and previous evaluation/future training overlap.
It did NOT recognize an earlier training family reused as a later
held-out evaluation family. The policy-frozen independent semantic
registry was present, so this condition was statically detectable
without adding a state schema or touching the evaluator's HMAC key.

In a real code-path synthetic fixture, attempt 1 received a valid
test-key-signed attestation with training family `train-1`, failed the
graduation objective and advanced to `STAY`. In attempt 2 the signed
receipt supplied distinct case IDs, prompt commitments, and a
correctly matching frozen registry, but relabeled one held-out case as
semantic family `train-1`. The other current training families
were distinct, so the existing current-attempt check did not catch it.

**RED observation, on unmodified PR #247 source:** the focused hostile
test invoked `decision(second, first.state, policy=contaminated_policy)`
and got `GRADUATED`; its assertion failed:
`AssertionError: 'GRADUATED' == 'GRADUATED' : previously trained
mechanics were laundered into a later holdout`.
Command: `python -m unittest discover -s tests
-p test_graduation_gate.py
-k prior_training_mechanism_cannot_graduate_as_later_holdout -v`.
Result: 1 test, 1 failure, exit 1.

The fixture is **synthetic signed evidence**, with a known test-only
HMAC key and declared scores; it is not a forged real evaluator
attestation or evidence that a genuine trained model generalizes.
Its relevance is the demonstrated decision logic despite all
cryptographic and per-attempt structural checks passing.

## Smallest source correction

In `GraduationPolicy.__post_init__`, derive the union of all
precommitted `train` semantic-family IDs and the union of all
precommitted evaluation-slice semantic-family IDs across every
attempt. Reject any intersection with explicit
`ValueError("training/evaluation semantic-family overlap across attempts")`.
This checks the *existing frozen registry*, not prompt similarity;
case IDs, prompt hashes, and HMACs alone cannot prove distinct mechanics.

Training-only semantic-family repetition across different attempts
remains allowed. No policy/receipt/state schema was changed.
The existing signed-source, immutable registry, paired controls,
strict objective, and previous evaluation replay checks remain.

## Actual GREEN and regression receipts

The identical hostile test passed after the source change:
1 test, 0 failures, exit 0. An edge-case test confirms the existing
policy's two repeated training families are still accepted, while
same-attempt training/evaluation family collision is rejected.

Lappy Windows Python: `test_graduation_gate.py`: **30 PASS**,
`test_graduation_state.py`: **6 PASS**,
`test_round4_learning_controls.py`: **6 PASS**.
`python -m compileall` and `git diff --check` passed.
Further WSL and hosted immutable-head test results should be attached
as separate receipts; do not infer those from focused local checks.

## Remaining hostile challenge — NOT QUALIFIED

> An independently precommitted registry can still lie about the
> semantic mechanics: a malicious or mistaken custodian can assign
> different family hashes to effectively identical tasks. Signed trial
> origin and model execution are likewise not proved by a valid HMAC.
> This gate now rejects *declared* train/eval overlap across attempts,
> not undisclosed equivalence between distinct family identifiers.

This fixes a software leak in source-level admission but does not
resolve the underlying semantic-diversity or examiner-separation
requirements of issues #147 and #146. No model training, deploy,
protected merge, examiner ACL, credential, or shared runtime changed.

## Hosted-CI red follow-up and compatibility correction (2026-11-15)

First immutable commit `9a92578244162eb81f863d5c03300b3fe3c298a4`
was **NOT** full-suite green: hosted Python 3.12 job
`132744036210` in workflow `43691870990` ran **1,283 tests**,
**one ERROR** in `test_round6_semantic_gate.
test_precommitted_family_seen_in_training_blocks_transfer`.
The old test assumed a contaminated registry could be instantiated and
then receive a signed `BLOCKED` GateDecision; the new necessary preflight
invariant instead raised `ValueError` during policy construction.
Reproduced locally on Windows: same error, focused exit1.

**Correction:** update only that preexisting hostile test to assert an
explicit `ValueError` containing
`training/evaluation semantic-family overlap` at registry freeze.
The rejected condition is unchanged; rejection moves earlier.
Windows post-correction: 16 Round6 + 30 graduation + 6 state
+ 6 Round4 control tests PASS. WSL Ubuntu26.04 post-correction:
16 Round6 normal + 16 Round6 optimized + 30 graduation + 6 state
tests PASS. Compiler and Git whitespace checks PASS.
Full hosted CI on the amended immutable head is a separate verdict.
