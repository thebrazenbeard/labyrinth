"""Bounded propositional-logic curriculum with a separate exact examiner.

This is deliberately unrelated to the modulo-97 register trace domain.  Its
grammar is ASCII, has at most four atoms and nesting depth three, and its
oracle is exhaustive truth-table entailment.  Generated prompts contain a
question (or an unlabelled proposed proof), never an answer key or a
``positive``/``negative`` marker.  As with every public generator, reserved
seeds and a custodian outside the trainee boundary are still needed for a
genuinely blind exam.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import random
import re

from .evaluation import Score
from .maze import Challenge


_SPLITS = {"train", "holdout"}
_ANSWER = re.compile(r"ANSWER: (YES|NO)\Z", re.ASCII)
_FIRST_ERROR = re.compile(r"FIRST_ERROR: (NONE|[1-8])\Z", re.ASCII)
_SHAPES = {
    "train": (
        ("modus_ponens", ("A", "(A>B)"), "B"),
        ("and_introduction", ("A", "B"), "(A&B)"),
        ("implication_chain", ("A", "(A>B)", "(B>C)"), "C"),
    ),
    "holdout": (
        ("disjunction_cases", ("(A|B)", "(A>C)", "(B>C)"), "C"),
        ("and_elimination", ("(A&B)",), "A"),
        ("negated_disjunction", ("!A", "(A|B)"), "B"),
    ),
}


def _validate(seed: int, count: int, split: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
        raise ValueError("count must be an integer between 1 and 1000")
    if split not in _SPLITS:
        raise ValueError("split must be train or holdout")


def _rng(seed: int, split: str, index: int, purpose: str) -> tuple[random.Random, str]:
    # Domain separation ensures that an identical numeric seed cannot select a
    # training instance in the holdout population.
    digest = sha256(f"labyrinth-proplog-v1\0{purpose}\0{seed}\0{split}\0{index}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big")), digest.hex()[:16]


def _rename(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in text)


def _shape(seed: int, split: str, index: int, purpose: str) -> tuple[str, tuple[str, ...], str, str]:
    rng, digest = _rng(seed, split, index, purpose)
    name, premises, conclusion = rng.choice(_SHAPES[split])
    atoms = list("ABCD")
    rng.shuffle(atoms)
    mapping = dict(zip("ABC", atoms))
    return name, tuple(_rename(item, mapping) for item in premises), _rename(conclusion, mapping), digest


class _FormulaParser:
    """Parser for atom | !formula | (formula [&|>] formula), depth <= 3."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not 1 <= len(text) <= 64 or not text.isascii():
            raise ValueError("formula outside bounded grammar")
        self.text, self.pos = text, 0

    def parse(self) -> tuple:
        tree = self._formula(0)
        if self.pos != len(self.text):
            raise ValueError("trailing formula text")
        return tree

    def _formula(self, depth: int) -> tuple:
        if depth > 3 or self.pos >= len(self.text):
            raise ValueError("formula depth or syntax")
        char = self.text[self.pos]
        if char in "ABCD":
            self.pos += 1
            return ("atom", char)
        if char == "!":
            self.pos += 1
            return ("not", self._formula(depth + 1))
        if char == "(":
            self.pos += 1
            left = self._formula(depth + 1)
            if self.pos >= len(self.text) or self.text[self.pos] not in "&|>":
                raise ValueError("missing binary operator")
            op = self.text[self.pos]
            self.pos += 1
            right = self._formula(depth + 1)
            if self.pos >= len(self.text) or self.text[self.pos] != ")":
                raise ValueError("missing closing parenthesis")
            self.pos += 1
            return (op, left, right)
        raise ValueError("formula syntax")


def _parse(text: str) -> tuple:
    return _FormulaParser(text).parse()


def _atoms(tree: tuple) -> set[str]:
    if tree[0] == "atom":
        return {tree[1]}
    if tree[0] == "not":
        return _atoms(tree[1])
    return _atoms(tree[1]) | _atoms(tree[2])


def _value(tree: tuple, assignment: dict[str, bool]) -> bool:
    if tree[0] == "atom":
        return assignment[tree[1]]
    if tree[0] == "not":
        return not _value(tree[1], assignment)
    left, right = _value(tree[1], assignment), _value(tree[2], assignment)
    return left and right if tree[0] == "&" else left or right if tree[0] == "|" else (not left or right)


def entails(premises: tuple[str, ...], conclusion: str) -> bool:
    """Return semantic entailment by enumerating the bounded valuation space."""
    parsed_premises = tuple(_parse(item) for item in premises)
    parsed_conclusion = _parse(conclusion)
    names = sorted(set().union(*(_atoms(item) for item in parsed_premises), _atoms(parsed_conclusion)))
    return all(
        not all(_value(item, assignment) for item in parsed_premises) or _value(parsed_conclusion, assignment)
        for bits in itertools.product((False, True), repeat=len(names))
        for assignment in (dict(zip(names, bits)),)
    )


def generate_logic_challenges(seed: int, count: int, split: str) -> list[Challenge]:
    """Create balanced YES/NO entailment questions on split-exclusive shapes."""
    _validate(seed, count, split)
    cases: list[Challenge] = []
    for index in range(count):
        structure, premises, entailed, digest = _shape(seed, split, index, "question")
        is_yes = index % 2 == 0
        conclusion = entailed if is_yes else f"!{entailed}"
        answer = "YES" if entails(premises, conclusion) else "NO"
        # The assertion catches accidental changes to a template's semantics.
        if (answer == "YES") != is_yes:
            raise RuntimeError("invalid internal logic template")
        prompt = (
            "In bounded propositional logic, decide whether the conclusion follows from all premises. "
            "Operators are ! (not), & (and), | (or), and > (if...then). "
            "Respond with exactly one line: ANSWER: <truth value>.\nPremises:\n"
            + "\n".join(f"- {item}" for item in premises)
            + f"\nConclusion: {conclusion}"
        )
        cases.append(Challenge(
            id=f"proplog-{split}-{index}-{digest}", kind="propositional_entailment", tier=5,
            split=split, prompt=prompt, answer=answer,
            metadata=(("generator", "labyrinth-proplog-v1"), ("structure", structure),
                      ("structure_fingerprint", sha256(structure.encode()).hexdigest())),
        ))
    return cases


def score_logic_response(challenge: Challenge, response: str) -> Score:
    """Strictly score a response without accepting prose, injections, or aliases."""
    if challenge.kind != "propositional_entailment" or not isinstance(response, str) or len(response) > 128:
        return Score(False, 0, "invalid_response", None)
    match = _ANSWER.fullmatch(response)
    if match is None:
        return Score(False, 0, "invalid_format", None)
    correct = match.group(1) == challenge.answer
    return Score(correct, int(correct), "correct" if correct else "incorrect", match.group(1))


@dataclass(frozen=True)
class FirstErrorExample:
    """Proof-audit task; expected_first_error remains evaluator-only data."""

    prompt: str
    premises: tuple[str, ...]
    claims: tuple[str, ...]
    expected_first_error: int | None
    structure: str


def _proof_prompt(premises: tuple[str, ...], claims: tuple[str, ...]) -> str:
    return (
        "Audit the proposed bounded propositional-logic proof. A CLAIM is valid only if it is entailed "
        "by the initial premises together with earlier valid CLAIM lines. Operators are !, &, |, and >. "
        "Return exactly FIRST_ERROR: NONE or FIRST_ERROR: <positive integer>.\nPremises:\n"
        + "\n".join(f"- {item}" for item in premises)
        + "\nProposed proof:\n" + "\n".join(f"{i}. CLAIM {claim}" for i, claim in enumerate(claims, 1))
    )


def generate_first_error_examples(seed: int, count: int, split: str) -> list[FirstErrorExample]:
    """Generate alternating valid and deliberately first-invalid proof audits."""
    _validate(seed, count, split)
    examples = []
    for index in range(count):
        structure, premises, conclusion, _ = _shape(seed, split, index, "proof")
        if index % 2 == 0:
            claims, expected = (conclusion, premises[0]), None
        else:
            valid_prefix = (conclusion,) if index % 4 == 1 else ()
            claims = valid_prefix + (f"!{conclusion}", premises[0])
            expected = len(valid_prefix) + 1
        examples.append(FirstErrorExample(_proof_prompt(premises, claims), premises, claims, expected, structure))
    return examples


def evaluate_first_error(example: FirstErrorExample) -> int | None:
    """Exact proof examiner. Malformed claims fail closed at their first line."""
    if not isinstance(example, FirstErrorExample) or not example.claims or len(example.claims) > 8:
        return 1
    valid = list(example.premises)
    try:
        # Parse all initial premises before considering a trainee-provided trace.
        for premise in valid:
            _parse(premise)
    except ValueError:
        return 1
    for number, claim in enumerate(example.claims, 1):
        try:
            if not entails(tuple(valid), claim):
                return number
        except (TypeError, ValueError):
            return number
        valid.append(claim)
    return None


def score_first_error_response(example: FirstErrorExample, response: str) -> Score:
    """Strictly score an audit verdict against the independently recomputed error."""
    if not isinstance(response, str) or len(response) > 128:
        return Score(False, 0, "invalid_response", None)
    match = _FIRST_ERROR.fullmatch(response)
    if match is None:
        return Score(False, 0, "invalid_format", None)
    parsed = None if match.group(1) == "NONE" else int(match.group(1))
    correct = parsed == evaluate_first_error(example)
    return Score(correct, int(correct), "correct" if correct else "incorrect", match.group(1))
