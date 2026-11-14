"""Executable cross-family Minotaur evaluation with honest no-update controls.

Four source-only exam families are intentionally kept separate: constant,
diverse (three mechanics), timegate, and Oracle Chamber (policy/certificate,
not physical escape). A deterministic baseline for each consumes only public
rules and observations. None trains or launches a model or proves transfer.

An authorized caller may inject an existing ModelAdapter through the same
public JSON bridge. That optional callback is UNATTESTED for locality, exact
weights and provider; in-process Python remains unsuitable for blind custody.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import secrets
from typing import Any, Callable, Mapping

from .adapters import ModelAdapter
from .minotaur_candidate_bridge import RunReceipt, run_candidate


FAMILIES = ("constant", "diverse", "timegate", "oracle")
_CONSTANT_SCRIPT = ("north", "east", "take", "west", "north", "offer", "leave")
_ORACLE_SCHEDULE = ("alpha", "alpha", "beta", "beta", "parity")
_ROOM_ROUTES = re.compile(r"(north|east|south|west) to ([a-z_]+)")


@dataclass(frozen=True)
class ExamSpec:
    family: str
    seed: int
    variant: int = 0
    sealed: bool = False
    split: str = "dev"
    weak_cell: bool = False


@dataclass(frozen=True)
class CrossCaseReceipt:
    family: str
    seed: int
    baseline_kind: str
    baseline: RunReceipt
    candidate: RunReceipt | None
    baseline_escape_verified: bool
    baseline_policy_verified: bool
    baseline_certificate_verified: bool


class _ConstantFacade:
    """Translate #139's start/act protocol into bridge readback semantics."""

    def __init__(self, exam: Any):
        from .constant_exam import CandidateSession, MAX_ACTIONS
        self._session = CandidateSession(exam)
        public = self._session.start()
        self._latest = {"observation": public["initial_observation"],
                        "remaining_actions": MAX_ACTIONS, "finished": False,
                        "verdict": "UNKNOWN", "escape_verified": False}

    def view(self) -> dict[str, Any]:
        return dict(self._latest)

    def act(self, action: str) -> dict[str, Any]:
        result = self._session.act(action)
        terminal = result["status"] in ("VERIFIED_WIN", "UNSOLVED", "INVALID_TRACE")
        self._latest = {"observation": result["observation"],
                        "remaining_actions": result["actions_remaining"],
                        "finished": terminal, "verdict": result["status"],
                        "escape_verified": result["status"] == "VERIFIED_WIN",
                        "action_status": result["status"]}
        return self.view()


class _DiverseFacade:
    """Expose independently certified escape evidence from #140's session."""

    def __init__(self, exam: Any):
        from .diverse_exam import CandidateSession
        self._session = CandidateSession(exam)

    def view(self) -> dict[str, Any]:
        feedback = dict(self._session.view())
        feedback["escape_verified"] = feedback.get("verdict") == "VERIFIED_WIN"
        return feedback

    def act(self, action: str) -> dict[str, Any]:
        self._session.act(action)
        return self.view()


def _validated(spec: ExamSpec) -> None:
    if not isinstance(spec, ExamSpec) or type(spec.family) is not str or spec.family not in FAMILIES:
        raise ValueError("unknown Minotaur examiner family")
    if type(spec.seed) is not int or not 0 <= spec.seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if type(spec.variant) is not int or not 0 <= spec.variant < 10000:
        raise ValueError("variant must be a nonnegative integer below 10000")
    if type(spec.sealed) is not bool or type(spec.weak_cell) is not bool:
        raise ValueError("sealed/weak_cell must be booleans")
    if type(spec.split) is not str or spec.split not in ("train", "dev", "test"):
        raise ValueError("invalid split")
    if spec.weak_cell and spec.family != "timegate":
        raise ValueError("weak-cell rule belongs only to timegate")


def make_public_case(spec: ExamSpec) -> tuple[dict[str, Any], Callable[[], Any]]:
    """Return a candidate view and private host-only fresh-session factory.

    The returned factory must stay on the trusted evaluator side. Passing it,
    the underlying Case, or a Python inspector into a model prompt is forbidden.
    """
    _validated(spec)
    if spec.family == "constant":
        from .constant_exam import make_exam, public_view
        exam = make_exam(spec.seed, split=spec.split, variant=spec.variant,
                         kind="sealed_control" if spec.sealed else "apparently_sealed")
        return public_view(exam), lambda: _ConstantFacade(exam)
    if spec.family == "diverse":
        from .diverse_exam import make_exam, public_view
        exam = make_exam(spec.seed, split=spec.split, variant=spec.variant,
                         sealed=spec.sealed)
        return public_view(exam), lambda: _DiverseFacade(exam)
    if spec.family == "timegate":
        from .timegate_exam import CandidateSession, candidate_view, make_case
        case = make_case(spec.seed, variant=spec.variant, sealed=spec.sealed,
                         weak_cell=spec.weak_cell)
        return candidate_view(case), lambda: CandidateSession(case)
    from .oracle_chamber_exam import CandidateSession, candidate_view, make_case
    case = make_case(spec.seed, variant=spec.variant, sealed=spec.sealed)
    # Pass the host's exact public rules without adding local runner fields.
    # The bridge itself reserves one final model response after five queries;
    # altering this view would defeat the full host/public binding check.
    return candidate_view(case), lambda: CandidateSession(case)


def _routes(observation: str) -> dict[str, str]:
    return {destination: direction for direction, destination in
            _ROOM_ROUTES.findall(observation)}


class PublicZeroUpdateAdapter:
    """Frozen public-rule interpreter, not a fitted or trained language model.

    Its family tag names a *public task protocol*, not a private Case identity.
    The diverse branch intentionally mirrors the strong public landmark parser
    used as a negative control in test_psychologist_zero_update.py.
    """

    def __init__(self, family: str):
        if family not in FAMILIES:
            raise ValueError("unknown public baseline family")
        self.family = family
        self.model_id = f"zero-update:public-{family}-v1"
        self._position = 0
        self._script: tuple[str, ...] | None = None

    def _timegate(self, public: dict[str, Any]) -> tuple[str, ...]:
        prompt = public["prompt"]
        charge, period = map(int, re.search(
            r"Charging requires clock phase (\d+) in a cycle of (\d+)", prompt).groups())
        gate = int(re.search(r"gate accepts escape only at phase (\d+)", prompt).group(1))
        phase = int(re.search(r"Clock: (\d+)/(\d+)", public["observation"]).group(1))
        actions = ["forward", "take", "forward"]
        phase = (phase + len(actions)) % period
        while phase != charge:
            actions.append("wait")
            phase = (phase + 1) % period
        actions.extend(("arm", "forward"))
        phase = (phase + 2) % period
        while phase != gate:
            actions.append("wait")
            phase = (phase + 1) % period
        actions.append("escape")
        return tuple(actions)

    def _diverse(self, public: dict[str, Any], history: list[dict[str, Any]]) -> str:
        observation = (history[-1]["feedback"]["observation"]
                       if history else public["observation"])
        if not history:
            routes = _routes(observation)
            if len(routes) != 1:
                raise ValueError("expected one public starting route")
            return next(iter(routes.values()))
        if self._script is None:
            routes = _routes(observation)
            if "archive" in routes:
                self._script = ("archive", "take", "junction", "gate", "offer", "leave")
            elif "lever" in routes:
                self._script = ("lever", "activate", "junction", "bridge", "leave")
            elif "west_shrine" in routes:
                self._script = ("west_shrine", "take", "hub", "east_shrine", "take",
                                "hub", "portal", "combine", "leave")
            else:
                raise ValueError("unknown visible landmark ontology")
        if self._position >= len(self._script):
            return "wait"  # invalid/nonwinning no-update fallback, never new logic
        token = self._script[self._position]
        self._position += 1
        return _routes(observation).get(token, token)

    def _oracle(self, public: dict[str, Any]) -> str:
        match = re.search(r"at most (\d+) answers? may be inverted adversarially",
                          public["prompt"])
        if match is None:
            claim = {"kind": "ABSTAIN", "assessment": "UNKNOWN"}
        elif int(match.group(1)) == 1:
            claim = {"kind": "POLICY", "policy": {
                "questions": list(_ORACLE_SCHEDULE), "decoder": "nearest_codeword"}}
        else:
            claim = {"kind": "UNSAT_CERTIFICATE", "declared_rules": {
                "secrets": 4, "binary_answers": 2, "max_queries": 5,
                "lie_budget": int(match.group(1))}}
        return json.dumps({"kind": "CLAIM", "claim": claim}, separators=(",", ":"))

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        request = json.loads(prompt)
        public, history = request["public_start"], request["history"]
        if self.family == "oracle":
            return self._oracle(public)
        if self.family == "diverse":
            action = self._diverse(public, history)
        else:
            if self._script is None:
                self._script = (_CONSTANT_SCRIPT if self.family == "constant"
                                else self._timegate(public))
            action = (self._script[self._position] if self._position < len(self._script)
                      else "wait")
            self._position += 1
        return json.dumps({"kind": "ACTION", "action": action}, separators=(",", ":"))


def _feedback_flag(receipt: RunReceipt, name: str) -> bool:
    for turn in reversed(receipt.turns):
        if turn.public_feedback_json is not None:
            value = json.loads(turn.public_feedback_json).get(name)
            return value is True
    return False


def run_cross_exam(specs: tuple[ExamSpec, ...], *,
                   candidate_factory: Callable[[Mapping[str, Any]], ModelAdapter] | None = None,
                   authorize_local_adapter: bool = False) -> tuple[CrossCaseReceipt, ...]:
    """Run separate public-only baseline and optional candidate, no host launch.

    Each arm receives the same case's public view, deterministic generation
    seed and a newly constructed examiner session. This is source-level input
    matching; no GPU-time, token-budget, model-weight or private-set custody is
    attested. The baseline is pure code and explicitly zero-update.
    """
    if type(specs) is not tuple or not specs or len(specs) > 64:
        raise ValueError("specs must be a bounded nonempty tuple")
    if len(set(specs)) != len(specs):
        raise ValueError("duplicate exact exam configuration")
    for spec in specs:
        _validated(spec)
    results = []
    for spec in specs:
        public, session_factory = make_public_case(spec)
        # Hidden examiner seed and model-sampling seed are separate. Exposing
        # the former lets a candidate reconstruct deterministic private cases
        # from public source, even across an OS process boundary.
        # Use one independent draw for both matched arms, receipt-bound.
        model_rng_seed = secrets.randbelow(2**31)
        # A provider/factory may be untrusted; never give it ExamSpec (whose
        # sealed control and seed are host-side), Case, or the session factory.
        # The copied candidate view is exactly the public JSON projection.
        candidate = (candidate_factory(json.loads(json.dumps(public, allow_nan=False)))
                     if candidate_factory is not None else None)
        if candidate is not None and not authorize_local_adapter:
            from .minotaur_candidate_bridge import FixedActionAdapter
            from .adapters import ReplayAdapter
            if type(candidate) not in (FixedActionAdapter, ReplayAdapter):
                raise PermissionError("unattested candidate requires separate local adapter authorization")
        candidate_receipt = (run_candidate(public, session_factory, candidate,
                                          seed=model_rng_seed,
                                          authorize_local_adapter=authorize_local_adapter)
                             if candidate is not None else None)
        baseline = run_candidate(public, session_factory, PublicZeroUpdateAdapter(spec.family),
                                 seed=model_rng_seed, authorize_local_adapter=True)
        results.append(CrossCaseReceipt(
            family=spec.family, seed=spec.seed, baseline_kind="ZERO_UPDATE_PUBLIC_ONLY",
            baseline=baseline, candidate=candidate_receipt,
            baseline_escape_verified=baseline.verified_escape,
            baseline_policy_verified=_feedback_flag(baseline, "policy_verified"),
            baseline_certificate_verified=_feedback_flag(baseline, "certificate_verified"),
        ))
    return tuple(results)


def main() -> int:
    """Stdlib-only no-model source demonstration; no network or host writes."""
    cases = run_cross_exam(tuple(ExamSpec(name, 0) for name in FAMILIES))
    print(json.dumps({
        "schema": "labyrinth.minotaur-cross-exam.v1",
        "model_execution": "DETERMINISTIC_CONTROLS_ONLY",
        "neural_training_performed": False,
        "private_examiner_custody": "NOT_ESTABLISHED",
        "cases": [{"family": case.family,
                   "baseline_verdict": case.baseline.final_verdict,
                   "baseline_escape_verified": case.baseline_escape_verified,
                   "baseline_policy_verified": case.baseline_policy_verified,
                   "baseline_certificate_verified": case.baseline_certificate_verified,
                   "receipt_sha256": case.baseline.receipt_sha256}
                  for case in cases],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
