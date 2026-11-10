# Exact-Cover Rescue Tribunal — Wave21 hostile family

**Evidence class:** source-public executable finite mechanics. This is a *new*
combinatorial family, not neural learning, private evaluator custody, or a
deployment qualification.

## The dilemma

All endangered sites must receive exactly one rescue crew. Every declared crew
covers three particular sites. No site can receive two crews; no site may be
left behind. The candidate may select no more than the declared crew budget.
The underlying task is **exact cover**, a distinct combinatorial constraint
problem rather than navigation, probing, or causal-actuator intervention.

The same fundamental dilemma can be described from a rescue, fleet, supply, or
allocation perspective. Wording variants keep the same rules fingerprint;
they do **not** count as independent semantic held-outs.

Four source-public cases exhibit (1) a cyclic impossible coverage constraint,
(2) an actual added declared crew changing UNSAT to WIN, (3) two different
valid perfect allocations, and (4) an insufficient crew budget. The single
crew addition is a change in mechanics, not a lexical paraphrase.

## Contract and verdicts

- `VERIFIED_WIN`: an exact tuple of valid declared crew indices covers every
  site once, with no overlap, within the action budget. The direct-set
  verifier does **not** call or trust the search engine.
- `VERIFIED_FAILURE`: candidate allocation malformed, overlapping, incomplete,
  or over budget; it does not imply global impossibility.
- `NO_KNOWN_WIN`: the search budget ends before finite exploration completes.
- `PROVEN_UNSAT_UNDER_DECLARED_RULES`: exhaustive bounded search finds no cover
  and a **separately coded subset-enumeration oracle** also finds none. This is
  conditional on precisely the declared crews, sites, and budget.
- `UNKNOWN`: invalid search settings, conflicting search/replay/oracle, or
  malformed novel proposal. It must not be counted as a proof.
- `NOVEL_RULESET_CHALLENGE`: another intelligence proposes an undeclared crew.
  The rule addition is preserved as a proposal with verification explicitly
  `NOT_PERFORMED`, never silently declared a win or rejected as nonsense.

`adjudicate_unexpected` independently checks a novel allocation made solely
of declared crews **even if a previous solver said UNSAT**. A valid witness
overturns that prior verdict. This is the deliberate proof-of-error seam: the
solver's answer key does not veto a valid counterexample.

## Test the tester

The red-first regression `test_malicious_python_mapping_key_fails_closed`
provoked an uncaught `RuntimeError` in `novel_rule_challenge` via a deceptive
`str` subclass's `__eq__`. Exact-type checking now precedes set comparisons.

Additional controls:

- 540 randomized, mechanically distinct graphs, independently enumerated with
  a **test-only multiset-counting oracle**, compared with the solver and
  module's separate subset oracle.
- A deliberately faulty slot-counting solver and replay both falsely accept
  two overlapping crews; the independent direct-set verifier rejects them.
- A valid alternative solution can overturn a previously declared UNSAT.
- Invalid `bool`, `str`, duplicate, over-budget, and oversized inputs never
  produce a win; limited search never produces an UNSAT certificate.
- A zero-update answer catalogue gets all four known public fixture labels
  exactly but answers `UNKNOWN` on a genuinely different graph fingerprint.
- A zero-update fixed-script control chooses the same two crew indices in
  every fixture and wins **0/4**, compared with **2/4** symbolic source-public
  wins. Neither control updates neural parameters.

Run focused checks:

```sh
python -m unittest tests.test_exact_cover_tribunal -q
python -O -m unittest tests.test_exact_cover_tribunal -q
```

## Qualification ceiling

Even a perfect symbolic score here proves only performance on declared finite
rules. The future neural gate must compare a real changed checkpoint with the
base, zero-update, corrupted-label, and public-catalogue controls on genuinely
unseen mechanisms held by an independent evaluator. A complete source-public
fixture catalogue can be memorized without learning. No such neural trial is
claimed by this family.

`verify_allocation` charges indices in one submitted plan, but it is **not** an
authenticated, persistent attempt ledger across repeated requests. A genuine
trial runner must separately enforce total attempts and adversarial candidate
isolation. The `fingerprint` identifies exact source bytes, not semantic
isomorphism under arbitrary site relabeling, so fresh fingerprints alone are
not evidence of new mechanics.
