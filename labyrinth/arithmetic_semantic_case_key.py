"""Narrow, fail-closed arithmetic case-equivalence guard.

This is deliberately NOT a natural-language semantic judge.  It recognizes
only a small, auditable grammar of nonnegative, single-addition expressions,
to prevent a known same-instance TRAIN/retention leakage route.  Unsupported
wording must be adjudicated by a separately governed evaluator, not guessed.

The guard is opt-in source code and is not currently enforced by the existing
retention entry points or an independently hosted examination authority.
"""

from __future__ import annotations

import re


class UnsupportedArithmeticCase(ValueError):
    """The restricted grammar cannot establish an arithmetic case identity."""


class ArithmeticCaseOverlap(ValueError):
    """The same arithmetic instance occurs across declared partitions."""


_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
)
_NUMBER_WORDS = {word: value for value, word in enumerate(_WORDS)}
_ATOM = r"(?P<left>[a-z]+|[0-9]{1,2})"
_OTHER = r"(?P<right>[a-z]+|[0-9]{1,2})"
_GRAMMAR = (
    re.compile(rf"what is {_ATOM} plus {_OTHER}"),
    re.compile(rf"what is {_ATOM} \+ {_OTHER}"),
    re.compile(rf"compute {_ATOM} plus {_OTHER}"),
    re.compile(rf"compute {_ATOM} \+ {_OTHER}"),
    re.compile(rf"add {_ATOM} to {_OTHER}"),
    re.compile(rf"{_ATOM} plus {_OTHER}"),
    re.compile(rf"{_ATOM} \+ {_OTHER}"),
)


def _number(atom: str) -> int:
    if atom.isascii() and atom.isdecimal():
        if len(atom) > 1 and atom.startswith("0"):
            raise UnsupportedArithmeticCase("leading-zero numeral is unsupported")
        value = int(atom)
        if value <= 99:
            return value
    elif atom in _NUMBER_WORDS:
        return _NUMBER_WORDS[atom]
    raise UnsupportedArithmeticCase("unsupported arithmetic operand")


def arithmetic_semantic_case_key(prompt: str) -> str:
    """Return ``add:min:max`` for exactly one supported addition operation.

    Integer literals are ASCII 0..99; English numerals are zero..twenty.
    Expressions with subtraction, multiplication, extra operators, clauses,
    variables, hidden Unicode characters, or unknown words are rejected.
    Operand order is normalized because addition is commutative.
    """
    if type(prompt) is not str or not prompt or len(prompt) > 160:
        raise UnsupportedArithmeticCase("invalid or oversized prompt")
    if not re.fullmatch(r"[A-Za-z0-9+?. ]+", prompt):
        raise UnsupportedArithmeticCase("non-ASCII, control or unsupported punctuation")
    normalized = " ".join(prompt.strip().lower().split(" "))
    if normalized.endswith(("?", ".")):
        normalized = normalized[:-1]
    for rule in _GRAMMAR:
        match = rule.fullmatch(normalized)
        if match is not None:
            left, right = _number(match.group("left")), _number(match.group("right"))
            first, second = sorted((left, right))
            return f"add:{first}:{second}"
    raise UnsupportedArithmeticCase("expression is not a single approved addition")


def require_disjoint_arithmetic_cases(
    train_prompts: tuple[str, ...], exam_prompts: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reject same arithmetic instance across TRAIN and exam, or unknown text.

    Returns the two canonical key rosters only when every expression is in
    the restricted grammar and no same-instance reuse exists. This helper
    does not attest source custody, semantic diversity, or evaluator isolation.
    """
    if (type(train_prompts) is not tuple or type(exam_prompts) is not tuple
            or not train_prompts or not exam_prompts):
        raise UnsupportedArithmeticCase("nonempty tuple partitions required")
    train = tuple(arithmetic_semantic_case_key(row) for row in train_prompts)
    exam = tuple(arithmetic_semantic_case_key(row) for row in exam_prompts)
    if set(train).intersection(exam):
        raise ArithmeticCaseOverlap("same arithmetic instance crosses partitions")
    return train, exam
