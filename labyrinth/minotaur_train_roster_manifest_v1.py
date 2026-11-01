"""Source-only Minotaur TRAIN roster exchange; NOT model-training admission.

`minotaur.train-roster.v1` deliberately differs from TRAININGDAY's existing
`vera.trainingday.corpus.v1` integer-chain corpus. A consumer must implement
and qualify a separate admission path, with its own arithmetic oracle and
external custody. Raw UTF-8 and TRAININGDAY R18-normalized prompt commitments
are separate; a unique-case catalog is not a repeated optimizer visit ledger.
The public JSON contains TRAIN prompts, but never answer keys or answer hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_policy_runner import TrainingCorridorRunner
from .maze import Challenge
from .minotaur_hf_roster_bridge import MinotaurHfTrainRoster
from .minotaur_policy_learning import MinotaurTrainingPolicy, TrainCaseApproval


SCHEMA = "minotaur.train-roster.v1"
NORMALIZATION = "nfkc-whitespace-casefold.v1"
MAX_PUBLIC_JSON_BYTES = 262_144
_HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TRAIN_ID = re.compile(r"train-[A-Za-z0-9_.:-]{1,120}\Z", re.ASCII)
_ANY_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_INSTRUCTION = "Respond with exactly one line: ANSWER: <integer>."
_PRACTICE = re.compile(r"Calculate ([0-9]+) ([+-]) ([0-9]+)\.\n", re.ASCII)
_CONTRAST = re.compile(
    r"Expression A is ([0-9]+) \+ ([0-9]+)\. Expression B is ([0-9]+) - ([0-9]+)\. "
    r"Calculate A minus B\.\n", re.ASCII,
)
_TRANSFER_PLUS = re.compile(
    r"A workshop starts with ([0-9]+) components and receives ([0-9]+) more\. "
    r"How many components does it now have\?\n", re.ASCII,
)
_TRANSFER_MINUS = re.compile(
    r"A workshop starts with ([0-9]+) components and uses ([0-9]+)\. "
    r"How many components remain, allowing a negative balance\?\n", re.ASCII,
)


@dataclass(frozen=True)
class TrainRosterManifestExport:
    """Public prompt catalog and separate operator-private gold commitment."""

    public_json: bytes
    public_sha256: str
    operator_private_answers_sha256: str

    @property
    def physically_authenticated(self) -> bool:
        return False

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


@dataclass(frozen=True)
class VerifiedTrainRosterManifest:
    """Content/schema validation; all authority must live outside this type."""

    public_sha256: str
    scope_sha256: str
    case_count: int
    optimizer_visit_count: int

    @property
    def physically_authenticated(self) -> bool:
        return False

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any, label: str, *, git: bool = False) -> str:
    pattern = _HEX40 if git else _HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label} digest")
    return value


def _id(value: Any, label: str) -> str:
    if type(value) is not str or _TRAIN_ID.fullmatch(value) is None:
        raise ValueError(f"invalid TRAIN {label}")
    return value


def _prompt(value: Any) -> str:
    if type(value) is not str or not value or "\x00" in value or any(
        ord(ch) < 32 and ch not in "\n\t" for ch in value
    ):
        raise ValueError("invalid public TRAIN prompt")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("TRAIN prompt is not encodable UTF-8") from exc
    if not 1 <= len(encoded) <= 8192:
        raise ValueError("TRAIN prompt exceeds bounded UTF-8 length")
    return value


def normalized_prompt_sha256(prompt: str) -> str:
    """Exactly R18: NFKC, split/join whitespace, then casefold, then UTF-8."""
    _prompt(prompt)
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()
    if not normalized:
        raise ValueError("TRAIN prompt is empty after normalization")
    return _sha(normalized.encode("utf-8"))


def _unique_forbidden(values: tuple[str, ...], label: str, *, ids: bool = False) -> set[str]:
    if type(values) is not tuple or len(values) > 4096:
        raise ValueError(f"{label} must be a bounded tuple")
    for value in values:
        if ids:
            if type(value) is not str or _ANY_CASE_ID.fullmatch(value) is None:
                raise ValueError(f"invalid {label}")
        else:
            _digest(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} entry")
    return set(values)


def _answer_from_source_prompt(prompt: str, tactic: str) -> str:
    """Independent, bounded oracle for current three TRAIN arithmetic templates."""
    if not prompt.endswith(_INSTRUCTION) or type(tactic) is not str:
        raise ValueError("unsupported arithmetic TRAIN source prompt")
    intro = prompt[: -len(_INSTRUCTION)]
    if tactic == "practice":
        match = _PRACTICE.fullmatch(intro)
        if match is not None:
            a, op, b = match.groups()
            answer = int(a) + int(b) if op == "+" else int(a) - int(b)
    elif tactic == "contrast":
        match = _CONTRAST.fullmatch(intro)
        if match is not None:
            a, b, a2, b2 = map(int, match.groups())
            if (a, b) != (a2, b2):
                raise ValueError("contrast operands are inconsistent")
            answer = (a + b) - (a - b)
    elif tactic == "transfer":
        plus = _TRANSFER_PLUS.fullmatch(intro)
        minus = _TRANSFER_MINUS.fullmatch(intro)
        match = plus or minus
        if match is not None:
            a, b = map(int, match.groups())
            answer = a + b if plus is not None else a - b
    else:
        match = None
    if match is None or any(len(group) > 8 for group in match.groups() if group.isdigit()):
        raise ValueError("TRAIN arithmetic source is not a bounded recognized tactic")
    return str(answer)


def _public_case(case: Challenge, approval: TrainCaseApproval) -> dict[str, Any]:
    if (type(case) is not Challenge or type(case.split) is not str
            or case.split != "train" or case.tier != 1
            or type(case.tier) is not int or case.kind not in ("arithmetic", "arithmetic_contrast")
            or type(case.kind) is not str):
        raise ValueError("only exact tier-one arithmetic TRAIN cases may be exported")
    _id(case.id, "case ID")
    prompt = _prompt(case.prompt)
    if (type(approval) is not TrainCaseApproval or approval.case_id != case.id
            or approval.split != "train" or approval.prompt_sha256 != _sha(prompt.encode())
            or (case.kind == "arithmetic_contrast") != (approval.tactic == "contrast")):
        raise ValueError("TRAIN case does not match its trusted registry approval")
    if type(case.answer) is not str or case.answer != _answer_from_source_prompt(prompt, approval.tactic):
        raise ValueError("TRAIN private answer differs from source-controlled arithmetic oracle")
    return {
        "case_id": case.id,
        "kind": case.kind,
        "tier": case.tier,
        "split": "train",
        "prompt": prompt,
        "raw_prompt_sha256": _sha(prompt.encode("utf-8")),
        "normalized_prompt_sha256": normalized_prompt_sha256(prompt),
        "approval": {
            "room_id": approval.room_id,
            "skill": approval.skill,
            "tactic": approval.tactic,
            "difficulty": approval.difficulty,
        },
    }


def export_minotaur_train_roster_v1(
    roster: MinotaurHfTrainRoster,
    runner: TrainingCorridorRunner,
    config: PilotConfig, *,
    producer_commit_sha: str,
    producer_tree_sha: str,
    forbidden_case_ids: tuple[str, ...] = (),
    forbidden_raw_prompt_sha256: tuple[str, ...] = (),
    forbidden_normalized_prompt_sha256: tuple[str, ...] = (),
) -> TrainRosterManifestExport:
    """Export trusted operator TRAIN staging without exposing private answers.

    The supplied Git pins and forbidden sets are caller claims, not host-owned
    attestations. No file, network, model-training intake or GPU is touched.
    """
    _digest(producer_commit_sha, "producer commit", git=True)
    _digest(producer_tree_sha, "producer tree", git=True)
    if (type(roster) is not MinotaurHfTrainRoster
            or type(runner) is not TrainingCorridorRunner
            or type(config) is not PilotConfig or not runner.is_complete
            or runner.state.pending is not None or runner._in_flight is not None
            or roster.cycle_index != 0 or roster.cycle_index != runner.state.plan.cycle_index
            or runner.state.scope.fingerprint != roster.scope_sha256
            or config.train_count != len(roster.train_cases)
            or config.steps != len(roster.ordered_optimizer_exposures)
            or not 2 <= len(roster.train_cases) <= 64 or not 1 <= config.steps <= 128):
        raise ValueError("complete matching cycle-zero TRAIN roster required")
    admissions = runner.training_registry
    observed = runner.observations
    transitions = runner.registry_transitions
    cases = roster.train_cases
    if (type(cases) is not tuple or len(admissions) != len(cases)
            or len(observed) != len(cases) or len(transitions) != len(cases) - 1
            or type(runner.policy) is not MinotaurTrainingPolicy):
        raise ValueError("TRAIN registry and observation lineage incomplete")
    public_cases: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        entry = _public_case(case, admissions[idx])
        observation = observed[idx]
        if (observation.case_id != case.id
                or observation.prompt_sha256 != entry["raw_prompt_sha256"]
                or observation.room_id != admissions[idx].room_id):
            raise ValueError("TRAIN observations disagree with admitted source")
        public_cases.append(entry)

    source_visits = _training_visit_exposures(list(cases), seed=config.seed,
                                              steps=config.steps)
    if (type(roster.ordered_optimizer_exposures) is not tuple
            or roster.ordered_optimizer_exposures != source_visits):
        raise ValueError("optimizer visit order does not match staged TRAIN cases")
    lookup = {entry["case_id"]: entry for entry in public_cases}
    visits = [{"case_id": cid, "raw_prompt_sha256": sha,
               "normalized_prompt_sha256": lookup[cid]["normalized_prompt_sha256"]}
              for cid, sha in source_visits]
    for idx, transition in enumerate(transitions, start=1):
        prev_observation, next_observation = observed[idx - 1], observed[idx]
        if (transition.added_case_ids != (cases[idx].id,)
                or transition.before_policy_sha256 != prev_observation.policy_after_sha256
                or transition.after_policy_sha256 != next_observation.policy_before_sha256):
            raise ValueError("TRAIN registry transition policy chronology is inconsistent")
        if idx > 1 and transition.before_registry_sha256 != transitions[idx - 2].after_registry_sha256:
            raise ValueError("TRAIN registry hash chain is broken")
    initial_registry = MinotaurTrainingPolicy(
        scope=runner.state.scope, approved_training_cases=(admissions[0],)
    )._registry_digest()
    if transitions[0].before_registry_sha256 != initial_registry or (
            transitions[-1].after_registry_sha256 != runner.policy._registry_digest()):
        raise ValueError("TRAIN registry first/final digest mismatch")
    transition_json = [{
        "before_policy_sha256": t.before_policy_sha256,
        "after_policy_sha256": t.after_policy_sha256,
        "before_registry_sha256": t.before_registry_sha256,
        "after_registry_sha256": t.after_registry_sha256,
        "added_case_ids": list(t.added_case_ids),
    } for t in transitions]
    data = {
        "schema": SCHEMA,
        "normalization": NORMALIZATION,
        "producer": {"commit_sha": producer_commit_sha, "tree_sha": producer_tree_sha},
        "scope_sha256": roster.scope_sha256,
        "cycle_index": roster.cycle_index,
        "seed": config.seed,
        "train_cases": public_cases,
        "optimizer_visits": visits,
        "admission_transitions": transition_json,
        "registry_after_sha256": runner.policy._registry_digest(),
        "policy_after_sha256": runner.policy.model_hash,
    }
    public_json = _canonical(data)
    if len(public_json) > MAX_PUBLIC_JSON_BYTES:
        raise ValueError("TRAIN roster manifest exceeds maximum size")
    public_digest = _sha(public_json)
    # This digest never leaves the trusted operator export object. Low-entropy
    # arithmetic labels are brute-forceable from an unkeyed public hash.
    private_digest = _sha(_canonical({
        "public_sha256": public_digest,
        "private_train_answers": [
            {"case_id": case.id, "answer": case.answer} for case in cases
        ],
    }))
    verify_minotaur_train_roster_v1(
        public_json, expected_manifest_sha256=public_digest,
        expected_source_commit_sha=producer_commit_sha,
        expected_source_tree_sha=producer_tree_sha,
        expected_scope_sha256=roster.scope_sha256,
        expected_cycle_index=roster.cycle_index,
        forbidden_case_ids=forbidden_case_ids,
        forbidden_raw_prompt_sha256=forbidden_raw_prompt_sha256,
        forbidden_normalized_prompt_sha256=forbidden_normalized_prompt_sha256,
    )
    return TrainRosterManifestExport(public_json, public_digest, private_digest)


def _strict_json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_PUBLIC_JSON_BYTES:
        raise ValueError("bounded canonical UTF-8 public TRAIN JSON required")

    def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for key, value in items:
            if key in d:
                raise ValueError("duplicate TRAIN manifest JSON key")
            d[key] = value
        return d

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON number {value}")

    try:
        d = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                       parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 TRAIN JSON") from exc
    try:
        canonical = _canonical(d)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("unsupported TRAIN JSON structure") from exc
    if type(d) is not dict or canonical != raw:
        raise ValueError("TRAIN JSON must be exactly canonical")
    return d


def verify_minotaur_train_roster_v1(
    raw: bytes, *, expected_manifest_sha256: str,
    expected_source_commit_sha: str, expected_source_tree_sha: str,
    expected_scope_sha256: str, expected_cycle_index: int,
    forbidden_case_ids: tuple[str, ...] = (),
    forbidden_raw_prompt_sha256: tuple[str, ...] = (),
    forbidden_normalized_prompt_sha256: tuple[str, ...] = (),
) -> VerifiedTrainRosterManifest:
    """Strict self-consistency against caller-pinned source/manifest commitments.

    Neither the commitments nor the caller are authenticated here. Never feed
    this receipt to host permission checks, Q5–Q9 graduation, or legacy
    TRAININGDAY corpus admission; a separate verified adapter is required.
    """
    _digest(expected_manifest_sha256, "expected public manifest")
    _digest(expected_source_commit_sha, "expected source commit", git=True)
    _digest(expected_source_tree_sha, "expected source tree", git=True)
    _digest(expected_scope_sha256, "expected scope")
    if type(expected_cycle_index) is not int or expected_cycle_index != 0:
        raise ValueError("source-only roster verifier supports cycle zero only")
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError("TRAIN manifest does not match caller-pinned digest")
    d = _strict_json(raw)
    if set(d) != {"schema", "normalization", "producer", "scope_sha256",
                  "cycle_index", "seed", "train_cases", "optimizer_visits",
                  "admission_transitions", "registry_after_sha256", "policy_after_sha256"}:
        raise ValueError("unexpected TRAIN manifest fields")
    if d["schema"] != SCHEMA or d["normalization"] != NORMALIZATION:
        raise ValueError("unsupported TRAIN manifest contract")
    producer = d["producer"]
    if (type(producer) is not dict or set(producer) != {"commit_sha", "tree_sha"}
            or _digest(producer["commit_sha"], "source commit", git=True)
            != expected_source_commit_sha or _digest(producer["tree_sha"], "source tree", git=True)
            != expected_source_tree_sha):
        raise ValueError("TRAIN producer source pins mismatch")
    if (_digest(d["scope_sha256"], "scope") != expected_scope_sha256
            or type(d["cycle_index"]) is not int or d["cycle_index"] != expected_cycle_index
            or type(d["seed"]) is not int or not 0 <= d["seed"] < 2**31):
        raise ValueError("TRAIN scope/cycle/seed mismatch")
    forbidden_ids = _unique_forbidden(forbidden_case_ids, "forbidden case", ids=True)
    forbidden_raw = _unique_forbidden(forbidden_raw_prompt_sha256, "forbidden raw prompt")
    forbidden_norm = _unique_forbidden(forbidden_normalized_prompt_sha256,
                                        "forbidden normalized prompt")
    cases = d["train_cases"]
    visits = d["optimizer_visits"]
    transitions = d["admission_transitions"]
    if (type(cases) is not list or not 2 <= len(cases) <= 64
            or type(visits) is not list or not 1 <= len(visits) <= 128
            or type(transitions) is not list or len(transitions) != len(cases) - 1):
        raise ValueError("TRAIN catalog/visit/transition cardinality invalid")
    ids: set[str] = set()
    raw_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    schedule_cases: list[Challenge] = []
    lookup: dict[str, dict[str, Any]] = {}
    for case in cases:
        if (type(case) is not dict or set(case) != {
            "case_id", "kind", "tier", "split", "prompt",
            "raw_prompt_sha256", "normalized_prompt_sha256", "approval",
        }):
            raise ValueError("unexpected public TRAIN case fields")
        cid = _id(case["case_id"], "case ID")
        prompt = _prompt(case["prompt"])
        raw_digest = _sha(prompt.encode("utf-8"))
        normalized_digest = normalized_prompt_sha256(prompt)
        if (case["raw_prompt_sha256"] != raw_digest
                or case["normalized_prompt_sha256"] != normalized_digest
                or cid in ids or raw_digest in raw_hashes
                or normalized_digest in normalized_hashes
                or cid in forbidden_ids or raw_digest in forbidden_raw
                or normalized_digest in forbidden_norm):
            raise ValueError("TRAIN case identity/prompt/normalization collision or replay")
        if (case["split"] != "train" or type(case["split"]) is not str
                or case["tier"] != 1 or type(case["tier"]) is not int
                or case["kind"] not in ("arithmetic", "arithmetic_contrast")
                or type(case["kind"]) is not str):
            raise ValueError("public catalog contains unsupported/non-TRAIN cases")
        approval = case["approval"]
        if type(approval) is not dict or set(approval) != {
                "room_id", "skill", "tactic", "difficulty"}:
            raise ValueError("public TRAIN case missing approval metadata")
        if (type(approval["room_id"]) is not str
                or type(approval["skill"]) is not str
                or type(approval["tactic"]) is not str
                or type(approval["difficulty"]) is not int):
            raise ValueError("invalid public TRAIN approval metadata")
        # Reuse the complete schema checks for an exact operator-side approval.
        _ = TrainCaseApproval(
            room_id=approval["room_id"], case_id=cid, skill=approval["skill"],
            tactic=approval["tactic"], difficulty=approval["difficulty"],
            prompt_sha256=raw_digest,
        )
        if (case["kind"] == "arithmetic_contrast") != (approval["tactic"] == "contrast"):
            raise ValueError("TRAIN tactic and challenge kind mismatch")
        _answer_from_source_prompt(prompt, approval["tactic"])
        ids.add(cid)
        raw_hashes.add(raw_digest)
        normalized_hashes.add(normalized_digest)
        schedule_cases.append(Challenge(cid, case["kind"], 1, "train", prompt, "0"))
        lookup[cid] = case

    expected_visits = _training_visit_exposures(
        schedule_cases, seed=d["seed"], steps=len(visits),
    )
    for index, visit in enumerate(visits):
        if (type(visit) is not dict or set(visit) != {
            "case_id", "raw_prompt_sha256", "normalized_prompt_sha256"}):
            raise ValueError("invalid optimizer visit event")
        cid, sha = expected_visits[index]
        if (visit["case_id"] != cid or visit["raw_prompt_sha256"] != sha
                or visit["normalized_prompt_sha256"] != lookup[cid]["normalized_prompt_sha256"]):
            raise ValueError("TRAIN optimizer visits differ from pinned order")

    for i, transition in enumerate(transitions, start=1):
        if type(transition) is not dict or set(transition) != {
            "before_policy_sha256", "after_policy_sha256",
            "before_registry_sha256", "after_registry_sha256", "added_case_ids"}:
            raise ValueError("invalid public registry admission transition")
        for key in ("before_policy_sha256", "after_policy_sha256",
                    "before_registry_sha256", "after_registry_sha256"):
            _digest(transition[key], key)
        if (transition["added_case_ids"] != [cases[i]["case_id"]]
                or (i > 1 and transition["before_registry_sha256"]
                    != transitions[i - 2]["after_registry_sha256"])):
            raise ValueError("TRAIN admission transition replay/order mismatch")
        # The registry digest is independently derivable from the public
        # admitted case catalog; no private learner labels are required.
        def registry_digest(prefix: list[dict[str, Any]]) -> str:
            entries = [{
                "room_id": c["approval"]["room_id"],
                "case_id": c["case_id"],
                "skill": c["approval"]["skill"],
                "tactic": c["approval"]["tactic"],
                "difficulty": c["approval"]["difficulty"],
                "prompt_sha256": c["raw_prompt_sha256"],
                "split": "train",
            } for c in prefix]
            return _sha(_canonical({"approved_training_cases": entries}))
        if (transition["before_registry_sha256"] != registry_digest(cases[:i])
                or transition["after_registry_sha256"] != registry_digest(cases[:i + 1])):
            raise ValueError("public TRAIN registry commitments are not reproducible")
    _digest(d["registry_after_sha256"], "final registry")
    _digest(d["policy_after_sha256"], "final policy")
    if transitions[-1]["after_registry_sha256"] != d["registry_after_sha256"]:
        raise ValueError("final registry commitment does not match transitions")
    return VerifiedTrainRosterManifest(
        _sha(raw), expected_scope_sha256, len(cases), len(visits),
    )
