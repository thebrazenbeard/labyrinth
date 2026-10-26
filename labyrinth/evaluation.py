"""Strict, side-effect-free scoring of untrusted model responses."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .maze import Challenge


_ANSWER_RE = re.compile(r"\s*ANSWER:\s*([+-]?[0-9]{1,12})\s*", flags=re.IGNORECASE | re.ASCII)


@dataclass(frozen=True)
class Score:
    correct: bool
    points: int
    reason: str
    parsed_answer: str | None


def score_response(challenge: Challenge, response: str) -> Score:
    """Award exactly 0 or 1 point using the generator's evaluator-only key.

    Only an entire `ANSWER: <integer>` line is accepted. Explanation text,
    multiple answers, pasted prompt injections, and oversized outputs cannot
    change the scoring rule or cause the evaluator to execute model content.
    """

    if not isinstance(response, str) or len(response) > 8192:
        return Score(False, 0, "invalid_response", None)
    parsed = _ANSWER_RE.fullmatch(response)
    if parsed is None:
        return Score(False, 0, "invalid_format", None)
    # Bounded numeric tokens avoid Python's integer-conversion digit limit and
    # ensure all numeric renderings are canonicalized before comparison.
    value = str(int(parsed.group(1)))
    correct = value == challenge.answer
    return Score(correct, int(correct), "correct" if correct else "incorrect", value)
