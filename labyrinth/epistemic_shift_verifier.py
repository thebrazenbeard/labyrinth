"""Independent causal replay and counterfactual falsification of hypotheses.

Separate source path, not independent ownership, private evaluator custody,
or a proof of model learning. It never calls the exam's transition().
"""

from __future__ import annotations

from itertools import product

from .epistemic_shift_exam import (ALL_KINDS, CHANGE_AFTER, INPUTS,
                                   MAX_ATTEMPTS, Case, State)


def _apply(state: State, pair: object, active_kernel: str) -> tuple[State, int | None]:
    if (type(pair) is not tuple or len(pair) != 2 or
            type(pair[0]) is not int or type(pair[1]) is not int or pair not in INPUTS):
        return state, None
    a, b = pair
    prior_a, prior_b = state.previous_x, state.previous_y
    values = {
        "copy": a,
        "xor": int(a != b),
        "xnor": int(a == b),
        "and": int(a + b == 2),
        "or": int(a + b > 0),
        "edge": int(b == 1 and a != prior_a),
        "lag": int(a == 1 and prior_b == 1),
    }
    if active_kernel not in values:
        raise ValueError("unsupported declared verifier rule")
    return State(state.interventions + 1, a, b), values[active_kernel]


def _actual_kernel(case: Case, state: State) -> str:
    return "copy" if case.sealed or state.interventions < CHANGE_AFTER else case.kind


def independent_replay(case: Case, attempts: object) -> tuple[State, tuple[int | None, ...]] | None:
    if not isinstance(attempts, (list, tuple)) or len(attempts) > MAX_ATTEMPTS:
        return None
    state = State()
    observations: list[int | None] = []
    for pair in attempts:
        state, effect = _apply(state, pair, _actual_kernel(case, state))
        observations.append(effect)
    return state, tuple(observations)


def assess_hypothesis(case: Case, current: State, candidate_kernel: object) -> dict[str, object]:
    """Falsify candidate post-change law on every 3-intervention continuation.

    A candidate can only qualify a *post-change* hypothesis once the change
    boundary has been crossed, to prevent a correct prelaw 'copy' from being
    spuriously graded as the persistent law before contradictory evidence.
    """
    if current.interventions < CHANGE_AFTER + 1:
        return {"verdict": "UNKNOWN", "hypothesis_verified": False,
                "reason": "insufficient post-change intervention history"}
    if type(candidate_kernel) is not str or candidate_kernel not in ALL_KINDS:
        return {"verdict": "NOVEL_RULESET_CHALLENGE", "hypothesis_verified": False,
                "proposal": candidate_kernel,
                "reason": "unrecognized mechanism requires new-rules review"}
    checked = 0
    for sequence in product(INPUTS, repeat=3):
        actual, proposed = current, current
        for depth, pair in enumerate(sequence):
            actual, truth = _apply(actual, pair, _actual_kernel(case, actual))
            proposed, forecast = _apply(proposed, pair, candidate_kernel)
            checked += 1
            if truth != forecast:
                return {"verdict": "VERIFIED_FAILURE", "hypothesis_verified": False,
                        "counterexample": {"sequence": [list(p) for p in sequence],
                                           "failed_at": depth,
                                           "observed": truth, "predicted": forecast},
                        "checks": checked,
                        "reason": "candidate law falsified by a future intervention"}
    return {"verdict": "VERIFIED_WIN", "hypothesis_verified": True,
            "checks": checked,
            "reason": "all three-step declared-world counterfactuals agree"}


def adjudicate_claim(case: Case, claim: object, attempts: object,
                     observations: object, belief_log: object) -> dict[str, object]:
    replay = independent_replay(case, attempts)
    if replay is None or replay[1] != tuple(observations):
        return {"verdict": "EVALUATOR_DISAGREEMENT", "verified_success": False}
    current = replay[0]
    if type(claim) is not dict or type(claim.get("kind")) is not str:
        return {"verdict": "UNKNOWN", "verified_success": False}
    kind = claim["kind"]
    if kind == "PREDICT":
        sequence, prediction = claim.get("sequence"), claim.get("outputs")
        if (type(sequence) not in (list, tuple) or type(prediction) not in (list, tuple)
                or len(sequence) not in (2, 3) or len(prediction) != len(sequence)):
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "invalid counterfactual prediction contract"}
        if any(type(p) not in (list, tuple) or len(p) != 2 or
               type(p[0]) is not int or type(p[1]) is not int or tuple(p) not in INPUTS
               for p in sequence):
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "undeclared intervention in prediction"}
        state = current
        actual = []
        for pair in sequence:
            state, output = _apply(state, tuple(pair), _actual_kernel(case, state))
            actual.append(output)
        correct = all(type(x) is int and x == truth
                      for x, truth in zip(prediction, actual))
        return {"verdict": "VERIFIED_WIN" if correct else "VERIFIED_FAILURE",
                "verified_success": False, "prediction_verified": correct,
                "actual_outputs": actual,
                "reason": "one short forecast never proves model learning"}
    if kind == "HYPOTHESIS":
        reviewed = assess_hypothesis(case, current, claim.get("kernel"))
        return {**reviewed, "verified_success": False,
                "revision_verified": False}
    if kind == "REVISION":
        revised = claim.get("revised_kernel")
        reviewed = assess_hypothesis(case, current, revised)
        if reviewed["verdict"] != "VERIFIED_WIN":
            return {**reviewed, "verified_success": False,
                    "revision_verified": False}
        # Evidence of real contradiction must precede the recorded revision.
        contradiction = None
        valid_count = 0
        for pair, effect in zip(attempts, observations):
            if effect is None:
                continue
            if valid_count >= CHANGE_AFTER and effect != pair[0]:
                contradiction = valid_count + 1
                break
            valid_count += 1
        correct_prior = any(t <= CHANGE_AFTER and law == "copy"
                            for t, law in belief_log)
        revised_after = (contradiction is not None and any(
            t >= contradiction and law == revised for t, law in belief_log))
        accepted = correct_prior and revised_after
        return {**reviewed, "verdict": "VERIFIED_WIN" if accepted else "UNKNOWN",
                "verified_success": False, "revision_verified": accepted,
                "reason": ("recorded revision follows observed prelaw falsifier"
                           if accepted else "no recorded contradiction-driven revision")}
    if kind == "ABSTAIN":
        assessment = claim.get("assessment")
        return {"verdict": assessment if type(assessment) is str and
                assessment in ("UNKNOWN", "NO_KNOWN_WIN") else "UNKNOWN",
                "verified_success": False}
    if kind == "NOVEL_RULE":
        description = claim.get("rule_change")
        if type(description) is not str or not description.strip():
            return {"verdict": "UNKNOWN", "verified_success": False,
                    "reason": "exact nonempty rule proposal required"}
        predictions = claim.get("immediate_outputs")
        consistent = None
        counterexample = None
        if type(predictions) in (list, tuple) and len(predictions) == len(INPUTS):
            expected = [_apply(current, pair, _actual_kernel(case, current))[1]
                        for pair in INPUTS]
            consistent = all(type(value) is int and value == truth
                             for value, truth in zip(predictions, expected))
            if not consistent:
                counterexample = {"inputs": [list(x) for x in INPUTS],
                                  "actual_outputs": expected}
        return {"verdict": "NOVEL_RULESET_CHALLENGE", "verified_success": False,
                "proposal": description, "local_consistency": consistent,
                "counterexample": counterexample,
                "reason": "local fit is not validation of a new global causal law"}
    return {"verdict": "UNKNOWN", "verified_success": False,
            "reason": "unsupported claim"}
