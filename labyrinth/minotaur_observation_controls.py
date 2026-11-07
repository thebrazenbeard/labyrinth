"""Opt-in, source-only controls for causal observation versus prior knowledge.

This audits evidence, NOT neural learning. The public rebound rule can be
submitted correctly with zero probes; an exhaustive synthetic counterfactual
match is therefore not evidence that it was inferred from experience.

Observation coverage can itself be scripted or memorized, and the posterior
uses only the declared 242-conjunction hypothesis language. Neither a narrow
posterior nor this module authorizes Minotaur graduation or model promotion.

Bounded independent dynamic programming over every legal action prefix of
length <=10 finds the strongest rebound-world posterior still contains four
conjunctions. Thus the `OBSERVATIONALLY_IDENTIFIED_WITHIN_DSL` guard is
unreachable for this particular fixed world and budget. That is a formal
finite-DSL, finite-budget statement, not proof that an overlooked real-world
solution is impossible. The executable certificate is in the separate
`test_minotaur_identifiability_bound` regression.
Independent hypothetical thirteen-action analysis can narrow this DSL to one
rule, but thirteen actions are forbidden by the actual ten-action exam.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from .novel_discovery_reviewer import FIELDS, normalize_proposal, replay_world
from .novel_discovery_world import (CHANGE_AFTER, INPUTS, MAX_ATTEMPTS,
                                    POST_PROBES, PRE_PROBES, CandidateSession,
                                    World)


Hypothesis = tuple[tuple[str, int], ...]
TOTAL_HYPOTHESES = 3 ** len(FIELDS) - 1
MIN_DISTINCT_POST_CONTEXTS = 4
FIXED_PUBLIC_RULE = {"form": "and_literals", "literals": [
    {"field": "prev2_x", "equals": 1},
    {"field": "prev_x", "equals": 0},
    {"field": "x", "equals": 1},
    {"field": "y", "equals": 1},
]}


@dataclass(frozen=True)
class ObservationAssessment:
    status: str
    attempts_charged: int
    post_observations: int
    distinct_post_contexts: int
    positive_post_observations: int
    negative_post_observations: int
    consistent_hypotheses: int
    information_gain_bits: float | None
    proposal_consistent: bool | None
    counterfactual: dict[str, object] | None
    neural_learning_established: bool = False
    graduation_authority: str = "none"


@dataclass(frozen=True)
class MatchedControlReceipt:
    zero_probe: ObservationAssessment
    memorized_fixed_script: ObservationAssessment
    symbolic_same_observations: ObservationAssessment
    symbolic_abstained: bool
    optimizer_updates: int = 0
    model_checkpoint: str = "NONE"
    qualification: str = "NOT_ESTABLISHED"


def _all_conjunctions() -> tuple[Hypothesis, ...]:
    return tuple(
        tuple(sorted((field, value) for field, value in zip(FIELDS, states)
                     if value is not None))
        for states in product((None, 0, 1), repeat=len(FIELDS))
        if any(value is not None for value in states)
    )


_DECLARED_HYPOTHESES = _all_conjunctions()
if len(_DECLARED_HYPOTHESES) != TOTAL_HYPOTHESES:
    raise RuntimeError("declared conjunction catalogue has incorrect cardinality")


def _effect(hypothesis: Hypothesis, context: dict[str, int]) -> int:
    return int(all(context[field] == value for field, value in hypothesis))


def _normalized_hypothesis(proposal: object) -> Hypothesis | None:
    normalized, status = normalize_proposal(proposal)
    if status != "EXECUTABLE" or normalized is None:
        return None
    return tuple(sorted((item["field"], item["equals"])
                        for item in normalized["literals"]))


def _post_contexts(attempts: tuple[object, ...], effects: tuple[object, ...]
                   ) -> list[tuple[dict[str, int], int]]:
    """Derive contexts from observed interventions, not from the world law."""
    prev2_x = prev_x = prev_y = valid_count = 0
    rows: list[tuple[dict[str, int], int]] = []
    for pair, observed in zip(attempts, effects):
        if (type(pair) is not tuple or len(pair) != 2 or
                type(pair[0]) is not int or type(pair[1]) is not int or pair not in INPUTS):
            continue  # Still charged, but not a physical intervention.
        x, y = pair
        context = {"x": x, "y": y, "prev_x": prev_x,
                   "prev2_x": prev2_x, "prev_y": prev_y}
        if valid_count >= CHANGE_AFTER:
            rows.append((context, observed))
        valid_count += 1
        prev2_x, prev_x, prev_y = prev_x, x, y
    return rows


def _posterior(rows: list[tuple[dict[str, int], int]]) -> tuple[Hypothesis, ...]:
    return tuple(h for h in _DECLARED_HYPOTHESES
                 if all(_effect(h, context) == observed for context, observed in rows))


def _counterfactual(proposal: Hypothesis | None, posterior: tuple[Hypothesis, ...],
                    rows: list[tuple[dict[str, int], int]]) -> dict[str, object] | None:
    if proposal is None:
        return None
    for context, observed in rows:
        predicted = _effect(proposal, context)
        if predicted != observed:
            return {"kind": "OBSERVED_REFUTATION", "context": context,
                    "proposal_prediction": predicted, "observed": observed}
    competitor = next((hypothesis for hypothesis in posterior if hypothesis != proposal), None)
    if competitor is None:
        return None
    for bits in product((0, 1), repeat=len(FIELDS)):
        context = dict(zip(FIELDS, bits))
        proposed, alternative = _effect(proposal, context), _effect(competitor, context)
        if proposed != alternative:
            return {"kind": "UNOBSERVED_DISCRIMINATOR", "context": context,
                    "proposal_prediction": proposed, "alternative_prediction": alternative,
                    "alternative_hypothesis": competitor,
                    "next_action_reachability": "NOT_ESTABLISHED"}
    return None


def assess_observational_identification(world: World, attempts: object,
                                        effects: object, proposal: object
                                        ) -> ObservationAssessment:
    """Require authentic world replay, post-change coverage and unique DSL fit.

    Does not inspect candidate weights, update an optimizer, or know whether a
    fixed script chose the observations. `OBSERVATIONALLY_IDENTIFIED_WITHIN_DSL`
    is a conditional data-adequacy label, never a learning/graduation award.
    """
    invalid = ObservationAssessment("INVALID_EVIDENCE", 0, 0, 0, 0, 0,
                                    TOTAL_HYPOTHESES, 0.0, None, None)
    if (type(world) is not World or type(attempts) not in (tuple, list)
            or type(effects) not in (tuple, list) or len(attempts) != len(effects)
            or len(attempts) > MAX_ATTEMPTS):
        return invalid
    snapshot = replay_world(world, attempts)
    if snapshot is None:
        return invalid
    expected = snapshot[1]
    for truth, observed in zip(expected, effects):
        if (truth is None and observed is not None) or (truth is not None and
               (type(truth) is not int or type(observed) is not int or truth != observed)):
            return invalid
    rows = _post_contexts(tuple(attempts), tuple(effects))
    posterior = _posterior(rows)
    candidate = _normalized_hypothesis(proposal)
    consistent = (None if candidate is None else candidate in posterior)
    count = len(posterior)
    # An empty posterior has zero likelihood under this DSL, not infinite
    # identification information. Information gain is then undefined.
    gain = math.log2(TOTAL_HYPOTHESES / count) if count else None
    positives = sum(effect == 1 for _, effect in rows)
    negatives = len(rows) - positives
    contexts = len({tuple(context[field] for field in FIELDS) for context, _ in rows})
    if rows and not posterior:
        status = "OUTSIDE_DECLARED_HYPOTHESIS_CLASS"
    elif candidate is None:
        status = "UNTESTED_PROPOSAL"
    elif not rows:
        status = "NO_POST_CHANGE_EVIDENCE"
    elif not consistent:
        status = "FALSIFIED_BY_OBSERVATIONS"
    elif (count == 1 and contexts >= MIN_DISTINCT_POST_CONTEXTS
          and positives >= 1 and negatives >= 1):
        status = "OBSERVATIONALLY_IDENTIFIED_WITHIN_DSL"
    else:
        status = "UNDERDETERMINED"
    return ObservationAssessment(status, len(attempts), len(rows), contexts,
                                 positives, negatives, count, gain, consistent,
                                 _counterfactual(candidate, posterior, rows))


def matched_nonlearning_controls(world: World, *,
                                 fixed_probes: tuple[tuple[int, int], ...] =
                                 PRE_PROBES + POST_PROBES) -> MatchedControlReceipt:
    """Compare fixed answers and symbolic fitting on the SAME observed trace.

    The memorized policy is handed the source-known rule. The symbolic arm
    uses only the observed trace and abstains unless one DSL hypothesis remains.
    Neither arm trains a model; both have zero optimizer updates.
    """
    zero = assess_observational_identification(world, (), (), FIXED_PUBLIC_RULE)
    session = CandidateSession(world)
    for pair in fixed_probes:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("fixed probe must be a pair")
        session.intervene(*pair)
    attempts = session.attempts()
    effects = tuple(row["effect"] for row in session.view()["observations"])
    scripted = assess_observational_identification(world, attempts, effects,
                                                   FIXED_PUBLIC_RULE)
    candidates = _posterior(_post_contexts(attempts, effects))
    if len(candidates) == 1:
        inferred = {"form": "and_literals", "literals": [
            {"field": field, "equals": equals} for field, equals in candidates[0]]}
        abstained = False
    else:
        inferred = None
        abstained = True
    symbolic = assess_observational_identification(world, attempts, effects, inferred)
    return MatchedControlReceipt(zero, scripted, symbolic, abstained)
