"""Development-only bridge to an exactly pinned Continuum Calc algebra source.

This creates public, deterministic *training* tasks.  It does not issue data
rights, create a private held-out evaluator, train weights, or establish that
the installed Calc package has an independently authenticated publisher.
Labyrinth's sealed DEV/final paths must never call this public generator.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import random

from .maze import Challenge


CALC_REPOSITORY = "thebrazenbeard/calc"
CALC_COMMIT = "17e12d0f659b42732fb98e5730732aba9ff1b488"
CALC_ALGEBRA_SHA256 = "6a1b269d14ddb34fbe4e8e5a9f56ac98c401fa98636cd3b13ab423cba1bead08"
SCHEMA = "labyrinth.calc-public-training.v1"
_INSTRUCTION = "Respond with exactly one line: ANSWER: <integer>."


def _load_exact_calc_algebra():
    """Require the audited Calc algebra source bytes, not merely its version.

    This is a *source-file equality* check for local development, not a trusted
    host attestation. A process that replaces imported Python objects can still
    forge results, so this module is not a protected evaluator.
    """
    try:
        algebra = importlib.import_module("calc.algebra")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("optional Continuum Calc algebra is not installed") from exc
    module_path = getattr(algebra, "__file__", None)
    if type(module_path) is not str:
        raise RuntimeError("Calc algebra has no inspectable source file")
    try:
        data = Path(module_path).read_bytes()
    except OSError as exc:
        raise RuntimeError("Calc algebra source cannot be read") from exc
    if hashlib.sha256(data).hexdigest() != CALC_ALGEBRA_SHA256:
        raise RuntimeError("Calc algebra source differs from reviewed source pin")
    if not callable(getattr(algebra, "Polynomial", None)) or not callable(
        getattr(algebra, "determinant", None)
    ):
        raise RuntimeError("pinned Calc algebra API unavailable")
    return algebra


def _case_rng(seed: int, index: int) -> tuple[random.Random, str]:
    tag = f"{SCHEMA}\x00{seed}\x00train\x00{index}".encode("ascii")
    digest = hashlib.sha256(tag).digest()
    return random.Random(int.from_bytes(digest, "big")), digest.hex()[:16]


def generate_calc_training_challenges(
    seed: int, count: int, *, split: str = "train"
) -> list[Challenge]:
    """Generate exact polynomial-derivative and determinant training cases.

    These are public development challenges, not private DEV/final holdouts or
    calibrated difficulty tiers.  Tier 5 is deliberately outside the existing
    `generate_maze` CLI tier contract, so no existing workflow silently changes.
    Answers use exact rational Calc algebra; the tests also check each answer
    with an independent power-rule or 2x2 determinant calculation.
    """
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be a nonnegative signed-31-bit integer")
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("count must be an integer between 1 and 128")
    if type(split) is not str or split != "train":
        raise ValueError("public Calc challenge bridge supports TRAIN only")
    algebra = _load_exact_calc_algebra()
    rows: list[Challenge] = []
    for index in range(count):
        rng, digest = _case_rng(seed, index)
        if index % 2 == 0:
            coeffs = [rng.randint(-8, 8) for _ in range(3)]
            # Disjoint index bands prevent duplicate public prompts even when
            # two seeded RNG streams happen to draw identical small operands.
            coeffs[0] = index * 41 + coeffs[0]
            coeffs.append(rng.choice(tuple(n for n in range(-8, 9) if n)))
            x = rng.randint(-4, 4)
            result = algebra.Polynomial(coeffs).differentiate()(x)
            if result.denominator != 1:
                raise ValueError("Calc polynomial derivative is unexpectedly fractional")
            independent = coeffs[1] + 2 * coeffs[2] * x + 3 * coeffs[3] * x * x
            if result != independent:
                raise RuntimeError("Calc polynomial derivative disagrees with independent power rule")
            prompt = (
                f"For p(x) = {coeffs[0]} + ({coeffs[1]})x + ({coeffs[2]})x^2 "
                f"+ ({coeffs[3]})x^3, compute p'({x}).\n{_INSTRUCTION}"
            )
            kind = "calc_polynomial_derivative"
            answer = str(result.numerator)
        else:
            a, b, c, d = (rng.randint(-12, 12) for _ in range(4))
            a = index * 41 + a
            result = algebra.determinant(((a, b), (c, d)))
            if result.denominator != 1:
                raise ValueError("Calc integer determinant is unexpectedly fractional")
            if result != a * d - b * c:
                raise RuntimeError("Calc determinant disagrees with independent 2x2 formula")
            prompt = (
                f"Compute the determinant of the integer matrix "
                f"[[{a}, {b}], [{c}, {d}]].\n{_INSTRUCTION}"
            )
            kind = "calc_matrix_determinant"
            answer = str(result.numerator)
        rows.append(Challenge(
            id=f"calc-train-t5-{index}-{digest}",
            kind=kind,
            tier=5,
            split="train",
            prompt=prompt,
            answer=answer,
            metadata=(
                ("generator", SCHEMA),
                ("calc_repository", CALC_REPOSITORY),
                ("calc_source_commit", CALC_COMMIT),
                ("calc_algebra_sha256", CALC_ALGEBRA_SHA256),
                ("evidence_scope", "public-development-only"),
            ),
        ))
    if len({row.id for row in rows}) != count or len({row.prompt for row in rows}) != count:
        raise ValueError("Calc challenge id or prompt collision")
    return rows
