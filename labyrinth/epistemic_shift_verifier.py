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
    # Subclasses may lie independently about __len__ and __iter__. Take one
    # immutable snapshot of an exact builtin witness and enforce the charge
    # budget against that same snapshot before replay.
    if type(attempts) not in (list, tuple):
        return None
    trace = tuple(attempts)
    if len(trace) > MAX_ATTEMPTS:
        return None
    state = State()
    observations: list[int | None] = []
    for pair in trace:
        state, effect = _apply(state, pair, _actual_kernel(case, state))
        observations.append(effect)
    return state, tuple(observations)


def exact_observations(expected: tuple[int | None, ...], observed: object) -> bool:
    """Reject custom equality objects masquerading as truthful binary effects."""
    if type(observed) not in (list, tuple) or len(expected) != len(observed):
        return False
    for truth, candidate in zip(expected, observed):
        if truth is None:
            if candidate is not None:
                return False
        elif (type(truth) is not int or truth not in (0, 1)
              or type(candidate) is not int or candidate != truth):
            return False
    return True


def exact_replay_match(expected: tuple[State, tuple[int | None, ...]] | None,
                       current: object, observations: object) -> bool:
    """Compare typed state and effects; equality callbacks are not evidence."""
    if expected is None or type(current) is not State:
        return False
    if (type(current.interventions) is not int or current.interventions < 0
            or type(current.previous_x) is not int or current.previous_x not in (0, 1)
            or type(current.previous_y) is not int or current.previous_y not in (0, 1)):
        return False
    state = expected[0]
    return (state.interventions == current.interventions
            and state.previous_x == current.previous_x
            and state.previous_y == current.previous_y
            and exact_observations(expected[1], observations))


def assess_hypothesis(case: Case, current: State, candidate_kernel: object) -> dict[str, object]:
    """Falsify candidate post-change law on every 3-intervention continuation.

    A candidate can only qualify a *post-change* hypothesis once the change
    boundary has been crossed, to prevent a correct prelaw 'copy' from being
    spuriously graded as the persistent law before contradictory evidence.
    """
    if current.interventions < CHANGE_AFTER + 1:
        return {"verdict": "UNKNOWN", "hypothesis_verified": False,
                "reason": "insufficient post-change intervention history"}
    if type(candidate_kernel) is not str or not candidate_kernel.strip():
        return {"verdict": "UNKNOWN", "hypothesis_verified": False,
                "reason": "missing or malformed causal mechanism"}
    if candidate_kernel not in ALL_KINDS:
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


def observed_trace_identifies(attempts: tuple[object, ...],
                              observations: tuple[int | None, ...],
                              proposed_kernel: str) -> bool:
    """Test identification from observed evidence, not hidden oracle truth.

    Unique only relative to the seven declared causal kernels; it does not
    exclude previously unknown causal mechanisms.
    """
    compatible = []
    for candidate in ALL_KINDS:
        state = State()
        predicted = []
        for pair in attempts:
            active = "copy" if state.interventions < CHANGE_AFTER else candidate
            state, effect = _apply(state, pair, active)
            predicted.append(effect)
        if exact_observations(tuple(predicted), observations):
            compatible.append(candidate)
    return compatible == [proposed_kernel]


def adjudicate_claim(case: Case, claim: object, attempts: object,
                     observations: object, belief_log: object) -> dict[str, object]:
    replay = independent_replay(case, attempts)
    if replay is None or not exact_observations(replay[1], observations):
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
        identified = observed_trace_identifies(tuple(attempts), tuple(observations),
                                               revised)
        if not identified:
            return {**reviewed, "verdict": "UNKNOWN", "verified_success": False,
                    "revision_verified": False, "observationally_identified": False,
                    "reason": "observed trace does not distinguish declared laws"}
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
                "observationally_identified": True,
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
