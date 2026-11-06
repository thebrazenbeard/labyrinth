"""Bounded free-text physical-action interpreter for experimental Labyrinths.

The learner proposes its own instruction in ordinary language. This module
never publishes action/answer options, scores a proposal, selects an action for
the learner, or reads hidden causal mechanisms. It only resolves an unambiguous
one-operation description against the physical objects currently visible.

Unknown language is *not* a failed physical intervention. The caller must
preserve the unresolved original for review without mutating the world.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal


Status = Literal["RESOLVED", "UNKNOWN"]

_VERBS = {
    "turn": "turn", "rotate": "turn", "twist": "turn", "crank": "turn",
    "pull": "pull", "yank": "pull", "tug": "pull",
    "cut": "cut", "sever": "cut", "snip": "cut", "slice": "cut",
    "push": "push", "press": "push", "depress": "push", "shove": "push",
    "open": "open", "unseal": "open",
    "close": "close", "shut": "close", "seal": "close",
}
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NEGATION = re.compile(r"\b(?:not|never|without|don't|won't|cannot|can't)\b")
_MULTI_STEP = re.compile(r"\b(?:then|afterward|afterwards|next|otherwise|instead|unless|either|or|and)\b")
_CONDITIONAL = re.compile(r"\b(?:if|perhaps|maybe|might|could|would rather|should)\b")
_INJECTION = re.compile(
    r"\b(?:ignore\s+(?:previous|prior|all|system)\s+instructions|"
    r"system\s+prompt|you\s+are\s+now|developer\s+message|"
    r"assistant\s+role)\b"
)
# These introduce *explanations*, not additional physical operations. A model
# may correctly acknowledge uncertainty in its rationale while committing to
# an unambiguous manipulation. The rationale is saved in raw_text.
_RATIONALE = frozenset({"because", "although", "since"})
_COMMITMENT_PREFIXES = (
    (), ("will",), ("shall",), ("intend", "to"), ("choose", "to"),
    ("decide", "to"), ("am", "going", "to"), ("propose", "to"),
)
_ARM_PREFIXES = (
    (), ("use", "arm", "to"), ("use", "the", "arm", "to"),
    ("use", "robotic", "arm", "to"),
    ("use", "the", "robotic", "arm", "to"),
)


@dataclass(frozen=True)
class OpenAction:
    """An interpretation, not permission or a claim about consequences."""

    verb: str
    target: str
    raw_text: str

    def __post_init__(self) -> None:
        if (type(self.verb) is not str or self.verb not in set(_VERBS.values())
                or type(self.target) is not str or not 1 <= len(self.target) <= 96
                or type(self.raw_text) is not str or not 1 <= len(self.raw_text) <= 1024):
            raise ValueError("invalid bounded interpreted action")


@dataclass(frozen=True)
class ActionResolution:
    """Contestable parser receipt, with the exact model utterance retained."""

    status: Status
    raw_text: str
    action: OpenAction | None
    reason: str

    def __post_init__(self) -> None:
        if (self.status not in ("RESOLVED", "UNKNOWN")
                or type(self.raw_text) is not str
                or type(self.reason) is not str or not self.reason.isupper()
                or (self.status == "RESOLVED") != (type(self.action) is OpenAction)
                or (self.action is not None and self.action.raw_text != self.raw_text)):
            raise ValueError("invalid open-action resolution receipt")


def _normalized_words(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    return tuple(_WORD.findall(text.replace("-", " ").replace("_", " ")))


def _unknown(raw: str, reason: str) -> ActionResolution:
    return ActionResolution("UNKNOWN", raw, None, reason)


def _clear_committed_action(prefix: tuple[str, ...]) -> bool:
    """Accept an imperative or a small declarative commitment syntax only.

    Descriptions *of* an action ("the manual says pull...") do not authorize
    execution. New syntax may be added without curating allowed action pairs.
    """
    if not prefix:
        return True
    if prefix in (
        ("i'll",), ("i", "ll"),
        ("my", "action", "is", "to"), ("my", "choice", "is", "to"),
        ("my", "intervention", "is", "to"),
        ("execute",), ("perform",), ("do",),
    ):
        return True
    if prefix[0] == "i":
        rest = prefix[1:]
        return any(rest == lead + arm for lead in _COMMITMENT_PREFIXES for arm in _ARM_PREFIXES)
    if prefix in _ARM_PREFIXES[1:]:
        return True
    return False


def parse_open_action(raw_text: str, visible_objects: tuple[str, ...]) -> ActionResolution:
    """Resolve one freely composed physical proposal without hidden semantics.

    A closed *verb grammar* is a transport/physics interface, not a list of
    complete choices. The learner invents its own verb/target combination.
    Ambiguity is explicitly UNKNOWN rather than being assigned a catastrophe.
    Source configuration errors are exceptional, not attributed to the model.
    """
    if type(visible_objects) is not tuple or not 1 <= len(visible_objects) <= 32:
        raise ValueError("bounded tuple of visible physical objects required")
    normalized_targets: dict[tuple[str, ...], str] = {}
    for obj in visible_objects:
        if type(obj) is not str or not 1 <= len(obj) <= 96 or any(ord(c) < 32 for c in obj):
            raise ValueError("invalid public physical object label")
        key = _normalized_words(obj)
        if not key or key in normalized_targets:
            raise ValueError("duplicate or empty normalized visible object label")
        normalized_targets[key] = obj

    if type(raw_text) is not str:
        raise ValueError("model proposal must be text")
    if not 1 <= len(raw_text) <= 1024:
        # Preserve the exact unparsed proposal for a contestable receipt.
        return _unknown(raw_text, "UNBOUNDED_PROPOSAL")
    if (any(ord(c) < 32 for c in raw_text)
            or raw_text.count(".") > 1 or ";" in raw_text
            or "?" in raw_text or '"' in raw_text
            or re.search(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", raw_text)
            or "“" in raw_text or "”" in raw_text
            or "{" in raw_text or "}" in raw_text or "```" in raw_text):
        return _unknown(raw_text, "MULTIPLE_OR_UNPARSEABLE_STATEMENTS")

    words = _normalized_words(raw_text)
    text = " ".join(words)
    if _INJECTION.search(text):
        return _unknown(raw_text, "UNTRUSTED_INSTRUCTION_ATTEMPT")
    # Evaluate only the executable clause. A justification is evidence, not a
    # second manipulation. Splitting before identifying the verb also rejects
    # questions such as "because the valve could break, should I open it?".
    action_words = words[:next((i for i, w in enumerate(words) if w in _RATIONALE), len(words))]
    action_text = " ".join(action_words)
    if _NEGATION.search(action_text) or _CONDITIONAL.search(action_text):
        return _unknown(raw_text, "NONCOMMITTAL_PROPOSAL")
    if _MULTI_STEP.search(action_text):
        return _unknown(raw_text, "MULTIPLE_OR_ALTERNATIVE_ACTIONS")

    physical_verbs = [(i, _VERBS[word]) for i, word in enumerate(action_words) if word in _VERBS]
    if len(physical_verbs) == 0:
        return _unknown(raw_text, "NO_PHYSICAL_VERB")
    if len(physical_verbs) != 1:
        return _unknown(raw_text, "AMBIGUOUS_PHYSICAL_VERB")
    verb_index, verb = physical_verbs[0]
    if not _clear_committed_action(action_words[:verb_index]):
        return _unknown(raw_text, "NO_CLEAR_ACTION_COMMITMENT")

    matches: list[tuple[int, str]] = []
    for tokens, original in normalized_targets.items():
        for at in range(verb_index + 1, len(action_words) - len(tokens) + 1):
            if action_words[at:at + len(tokens)] == tokens:
                matches.append((len(tokens), original))
                break
    if not matches:
        return _unknown(raw_text, "NO_EXACT_VISIBLE_TARGET")
    # A public label may be nested in another ("valve", "red valve"). Use
    # the most specific *exact contiguous name*. Never select between two
    # equally-specific visible targets.
    longest = max(length for length, _ in matches)
    matches = [(length, label) for length, label in matches if length == longest]
    if len(matches) > 1:
        return _unknown(raw_text, "AMBIGUOUS_VISIBLE_TARGET")

    action = OpenAction(verb, matches[0][1], raw_text)
    return ActionResolution("RESOLVED", raw_text, action, "PARSED_PHYSICAL_ACTION")


__all__ = ["OpenAction", "ActionResolution", "parse_open_action"]
