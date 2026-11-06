"""Open-ended Kobayashi proposal triage, *not* a mechanism adjudicator.

Ordinary wording for a recognized physical manipulation stays a recognized
operation. A detailed proposal outside that grammar may be recorded as a
NOVEL_CANDIDATE, but neither novelty nor success is established by prose.
Only independently checked world/canon evidence may adjudicate its physics.
Unrecognized language must never be assigned an invented catastrophic effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
import unicodedata
from typing import Literal

from .open_action_v1 import OpenAction, parse_open_action


ProposalStatus = Literal["KNOWN_MECHANISM", "NOVEL_CANDIDATE", "UNKNOWN"]

_LABEL = re.compile(
    r"^\s*(action|intervention|proposal|mechanism|causal\s+mechanism|"
    r"assumption|assumptions|premise|premises|proof|evidence|support|"
    r"prediction|predicted\s+observation|expected\s+observation|"
    r"falsifier|falsification|disconfirmation|refutation)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_MAX_PROPOSAL_CHARS = 4000  # matches KobayashiWorld.record_unverified_proposal
_INJECTION = re.compile(
    r"\b(?:ignore\s+(?:all\s+|previous\s+|prior\s+|system\s+)*instructions|"
    r"(?:system|developer)\s+(?:prompt|message|instruction)|"
    r"you\s+are\s+now|\brole\s*:\s*(?:system|developer)|"
    r"override\s+(?:the\s+)?(?:evaluator|grade|verdict|instructions)|"
    r"print\s+(?:the\s+)?(?:hidden|private|secret)\b)",
    re.IGNORECASE,
)
_ASSUMPTION = re.compile(r"\b(?:assum(?:e|ed|ing|ption)|presum(?:e|ed)|provided\s+that)\b", re.I)
_PROOF = re.compile(r"\b(?:proof|evidence|observ(?:e|ed|ation)|visible|measur(?:e|ed)|indicat(?:e|es))\b", re.I)
_PREDICTION = re.compile(r"\b(?:predict|expect|would\s+(?:cause|produce|result)|will\s+(?:cause|produce|result)|should\s+(?:cause|produce|result))\b", re.I)
_FALSIFIER = re.compile(r"\b(?:falsif\w*|disprov\w*|refut\w*|contradict\w*|fails?\s+if|wrong\s+if|would\s+fail\s+if)\b", re.I)
_MECHANISM = re.compile(r"\b(?:because|thereby|through|mechanism|caus(?:e|es|ing)|transfer(?:ring)?|coupl(?:e|es|ing))\b", re.I)
_INTENT = re.compile(
    r"^(?:i\s+(?:(?:will|shall|intend\s+to|choose\s+to|decide\s+to|"
    r"am\s+going\s+to|propose\s+to)\s+)?|i['’]ll\s+|"
    r"(?:use|using)\s+(?:the\s+)?(?:robotic\s+)?arm\s+to\s+|"
    r"my\s+(?:action|intervention|proposal)\s+is\s+to\s+|"
    r"(?:execute|perform)\s+)",
    re.IGNORECASE,
)
_NON_ACTION_FIRST_VERBS = frozenset({
    "think", "believe", "wonder", "suspect", "guess", "hope", "feel",
    "say", "claim", "explain", "consider", "recommend", "ask", "want",
    "know", "assume", "predict", "expect", "propose", "would", "might",
    "could", "should", "do", "don't", "not", "never", "refuse",
})
_EXPLANATORY_FIRST_VERBS = _NON_ACTION_FIRST_VERBS | frozenset({
    "observe", "observed", "infer", "estimate", "reason", "note", "explain",
    "understand", "conclude", "calculate", "acknowledge", "describe",
    "see", "am", "shall", "can", "cannot", "may",
})
_NO_COMMITMENT = re.compile(r"\b(?:if|maybe|perhaps|might|could|should|would|not|never|don't|won't)\b", re.I)
_PHYSICAL_QUALIFIERS = (
    # The canon models "turn", not direction, partial travel, torque, dwell
    # time, or a compound operation. Collapsing these into "turn" would make
    # an unmodelled mechanism falsely inherit a known catastrophic outcome.
    re.compile(r"\b(?:clockwise|counterclockwise|anti[- ]clockwise|upwards?|"
               r"downwards?|leftwards?|rightwards?)\b", re.I),
    re.compile(r"\b(?:half|quarter|eighth|third|partial(?:ly)?|fractional|"
               r"one[- ](?:half|quarter|eighth|third)|"
               r"(?:one|two|three|five|ten|\d+(?:\.\d+)?)\s+"
               r"(?:degrees?|turns?|seconds?|millimeters?|inches?))\b|"
               r"\b\d+\s*/\s*\d+\b", re.I),
    re.compile(r"\b(?:gently|forcefully|slowly|rapidly|gradually|lightly|"
               r"hard|softly|abruptly|carefully|only\s+until|"
               r"precise\s+torque|controlled\s+force|maximum\s+torque)\b", re.I),
    re.compile(r"\b(?:by\s+[a-z][a-z-]*|via\s+|through\s+|while\s+|"
               r"simultaneously|at\s+the\s+same\s+time|during\s+)\b", re.I),
    # The scenario's arm is an ordinary control interface; invoking it alone
    # does not establish a novel physical mechanism. Other instruments might.
    re.compile(r"\b(?:with|using)\s+(?!"
               r"(?:(?:the|a)\s+)?(?:robotic\s+)?arm\b)[a-z]", re.I),
)

_LABEL_GROUP = {
    "action": "action", "intervention": "action", "proposal": "action",
    "mechanism": "mechanism", "causal mechanism": "mechanism",
    "assumption": "assumptions", "assumptions": "assumptions",
    "premise": "assumptions", "premises": "assumptions",
    "proof": "proof", "evidence": "proof", "support": "proof",
    "prediction": "prediction", "predicted observation": "prediction",
    "expected observation": "prediction",
    "falsifier": "falsifier", "falsification": "falsifier",
    "disconfirmation": "falsifier", "refutation": "falsifier",
}


@dataclass(frozen=True)
class ProposalAssessment:
    """Reviewable receipt, not a physics verdict or grade."""

    status: ProposalStatus
    raw_text: str
    raw_sha256: str
    action: OpenAction | None
    proposed_action_text: str | None
    claimed_mechanism: tuple[str, ...]
    claimed_assumptions: tuple[str, ...]
    claimed_proof: tuple[str, ...]
    claimed_prediction: tuple[str, ...]
    claimed_falsifier: tuple[str, ...]
    parser_reason: str
    reason: str
    novelty_verified: bool = False

    def __post_init__(self) -> None:
        if (self.status not in ("KNOWN_MECHANISM", "NOVEL_CANDIDATE", "UNKNOWN")
                or type(self.raw_text) is not str
                or self.raw_sha256 != sha256(self.raw_text.encode("utf-8")).hexdigest()
                or not isinstance(self.claimed_mechanism, tuple)
                or not isinstance(self.claimed_assumptions, tuple)
                or not isinstance(self.claimed_proof, tuple)
                or not isinstance(self.claimed_prediction, tuple)
                or not isinstance(self.claimed_falsifier, tuple)
                or not self.parser_reason or not self.reason
                or self.novelty_verified is not False
                or (self.status == "KNOWN_MECHANISM") != (type(self.action) is OpenAction)
                or (self.action is not None
                    and self.action.raw_text != self.proposed_action_text)):
            raise ValueError("invalid Kobayashi proposal receipt")


def _words(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return tuple(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalized))


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip())


def _proposal_parts(raw_text: str) -> dict[str, tuple[str, ...]]:
    extracted: dict[str, list[str]] = {
        "action": [], "mechanism": [], "assumptions": [], "proof": [],
        "prediction": [], "falsifier": [],
    }
    narrative: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LABEL.fullmatch(line)
        if match is not None:
            group = _LABEL_GROUP[" ".join(match.group(1).casefold().split())]
            extracted[group].append(match.group(2).strip())
        else:
            narrative.extend(_sentences(line))
    if not extracted["action"] and narrative:
        extracted["action"].append(narrative[0])
    # Narrative statements are retained verbatim as *claims*. Finding words
    # is not validation of their substance, observational truth, or novelty.
    for sentence in narrative:
        if _MECHANISM.search(sentence):
            extracted["mechanism"].append(sentence)
        if _ASSUMPTION.search(sentence):
            extracted["assumptions"].append(sentence)
        if _PROOF.search(sentence):
            extracted["proof"].append(sentence)
        if _PREDICTION.search(sentence):
            extracted["prediction"].append(sentence)
        if _FALSIFIER.search(sentence):
            extracted["falsifier"].append(sentence)
    return {key: tuple(values) for key, values in extracted.items()}


def _contains_second_explicit_action(raw_text: str) -> bool:
    """Do not let a multi-line answer silently execute its first command."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    labelled_actions = sum(
        re.match(r"^(?:action|intervention|proposal)\s*:", line, re.I) is not None
        for line in lines
    )
    if labelled_actions > 1:
        return True
    # An unlabeled paragraph may describe its rationale over many sentences.
    # Only obvious additional executable clauses are rejected here; unresolved
    # prose remains inert even if this conservative detector misses one.
    sentences = [
        sentence
        for line in lines if _LABEL.fullmatch(line) is None
        for sentence in _sentences(line)
    ]
    if labelled_actions == 0:
        # The opening narrative sentence is the proposed primary action.
        sentences = sentences[1:]
    else:
        # With an Action: field, *every* additional imperative narrative
        # sentence would be an unsanctioned second manipulation.
        pass
    for sentence in sentences:
        words = _words(sentence)
        if words and words[0] == "then":
            words = words[1:]
        if not words:
            continue
        if words[0] == "i":
            words = words[1:]
            if words and words[0] in ("will", "shall"):
                words = words[1:]
            # A second sentence that asserts a new physical operation, even
            # with a previously unknown verb, is not a single manipulation.
            if words and words[0] not in _EXPLANATORY_FIRST_VERBS:
                return True
        elif words[0] in (
            "turn", "pull", "cut", "push", "open", "close", "rotate",
            "twist", "sever", "press", "wedge", "shim", "jam", "bridge",
            "activate", "energize", "rewire", "insulate", "strike", "attach",
        ):
            return True
    return False


def _is_committed_unrecognized_action(action_text: str) -> bool:
    """Require an asserted operation, never treat commentary as an action."""
    if not action_text or len(action_text) > 1024:
        return False
    text = " ".join(_words(action_text))
    # A trailing rationale may mention uncertainty without making the action
    # conditional. The operator's clause still has to be a commitment.
    clause = re.split(r"\b(?:because|although|since)\b", text, maxsplit=1)[0].strip()
    if _NO_COMMITMENT.search(clause) or not _INTENT.match(action_text.strip()):
        return False
    words = _words(clause)
    if not words or len(words) < 3:
        return False
    # This rejects "I think ..." or "I wonder ..." without enumerating
    # allowed physical operations or silently guessing their effects.
    if words[0] == "i":
        after = words[1:]
        if after and after[0] in ("will", "shall"):
            after = after[1:]
        if after[:2] == ("propose", "to"):
            after = after[2:]
        if after and after[0] in _NON_ACTION_FIRST_VERBS:
            return False
    if re.search(r"\b(?:then|or|and)\b", clause):
        return False
    return True


def _has_unmodelled_physical_qualifier(action_text: str) -> bool:
    """Detect physically distinct parameters absent from the toy action API."""
    action_clause = re.split(r"\b(?:because|although|since)\b", action_text,
                             maxsplit=1, flags=re.I)[0]
    return any(pattern.search(action_clause) for pattern in _PHYSICAL_QUALIFIERS)


def assess_kobayashi_proposal(
    raw_text: str,
    visible_objects: tuple[str, ...],
) -> ProposalAssessment:
    """Classify a *claim's reviewability*, never its correctness or outcome.

    `KNOWN_MECHANISM` means the operation is normalized by the public physical
    interface; it does not mean the claimed causal explanation is correct.
    `NOVEL_CANDIDATE` means an out-of-grammar claim warrants review, *not*
    that it uses a new mechanism, satisfies scenario constraints, or succeeds.
    `UNKNOWN` means insufficiently identifiable/reviewable; no physical event.

    Hidden graph, feasible actions, outcomes, and score are never inputs.
    """
    if type(raw_text) is not str:
        raise ValueError("text proposal required")
    # Reuse visible-object contract, with an inert sentinel that cannot act.
    # Invalid scene configuration is an operator error, never a model failure.
    parse_open_action("", visible_objects)
    digest = sha256(raw_text.encode("utf-8")).hexdigest()
    parts = _proposal_parts(raw_text) if 0 < len(raw_text) <= _MAX_PROPOSAL_CHARS else {
        key: () for key in ("action", "mechanism", "assumptions", "proof", "prediction", "falsifier")
    }
    action_text = parts["action"][0] if parts["action"] else None

    def outcome(status: ProposalStatus, reason: str, parser_reason: str,
                action: OpenAction | None = None) -> ProposalAssessment:
        return ProposalAssessment(
            status=status, raw_text=raw_text, raw_sha256=digest, action=action,
            proposed_action_text=action_text,
            claimed_mechanism=parts["mechanism"],
            claimed_assumptions=parts["assumptions"],
            claimed_proof=parts["proof"],
            claimed_prediction=parts["prediction"],
            claimed_falsifier=parts["falsifier"],
            parser_reason=parser_reason, reason=reason,
        )

    if not 1 <= len(raw_text) <= _MAX_PROPOSAL_CHARS:
        return outcome("UNKNOWN", "UNBOUNDED_PROPOSAL", "NOT_PARSED")
    if (any(ord(c) < 32 and c not in "\r\n\t" for c in raw_text)
            or _INJECTION.search(unicodedata.normalize("NFKC", raw_text))
            or "```" in raw_text or "<script" in raw_text.casefold()):
        return outcome("UNKNOWN", "UNTRUSTED_INSTRUCTION_OR_CONTROL_TEXT", "NOT_PARSED")
    if action_text is None:
        return outcome("UNKNOWN", "NO_PROPOSED_ACTION", "NOT_PARSED")
    if _contains_second_explicit_action(raw_text):
        return outcome("UNKNOWN", "MULTIPLE_PROPOSED_ACTIONS", "NOT_PARSED")

    resolved = parse_open_action(action_text, visible_objects)
    if resolved.status == "RESOLVED" and not _has_unmodelled_physical_qualifier(action_text):
        assert resolved.action is not None
        # OpenAction's provenance is the exact *action clause* (bounded to
        # 1024); the ProposalAssessment preserves the entire unabridged text,
        # including lengthy hypotheses and proof claims, separately.
        canonical = OpenAction(resolved.action.verb, resolved.action.target, action_text)
        return outcome("KNOWN_MECHANISM", "RECOGNIZED_PHYSICAL_OPERATION", resolved.reason, canonical)
    if resolved.status == "RESOLVED":
        # The underlying input verb/target is known, but this particular
        # qualified manipulation is not in the committed mechanism graph.
        qualifier_reason = "UNMODELLED_PHYSICAL_QUALIFIER"
    else:
        qualifier_reason = None
    if (resolved.status != "RESOLVED"
            and resolved.reason not in ("NO_PHYSICAL_VERB", "NO_EXACT_VISIBLE_TARGET")):
        return outcome("UNKNOWN", "AMBIGUOUS_OR_NONCOMMITTAL_OPERATION", resolved.reason)
    if not _is_committed_unrecognized_action(action_text):
        return outcome("UNKNOWN", "NO_COMMITTED_PHYSICAL_OPERATION",
                       qualifier_reason or resolved.reason)
    if not (parts["mechanism"] and parts["assumptions"]
            and parts["prediction"] and parts["falsifier"]):
        return outcome("UNKNOWN", qualifier_reason or "INSUFFICIENT_FALSIFIABLE_SUPPORT",
                       qualifier_reason or resolved.reason)
    # A paper claim is not an executable physical intervention. Proof, even
    # when provided, is only claimed until checked against independent canon.
    return outcome("NOVEL_CANDIDATE", "UNVERIFIED_REVIEWABLE_CAUSAL_HYPOTHESIS",
                   qualifier_reason or resolved.reason)


__all__ = ["ProposalAssessment", "ProposalStatus", "assess_kobayashi_proposal"]
