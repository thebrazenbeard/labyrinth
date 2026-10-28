"""Experimental learned multiplication interaction without an x*y input feature.

This is a small NumPy MLP study, not a Minotaur admission candidate. Only the
offline label generator and evaluator multiply operand values. The network's
forward pass receives two normalized scalars, never a supplied product, a
lookup table, operation-specific arithmetic, or teacher states.

Even operand identities are supervised; odd operand identities are fully
withheld. This is *interpolation* within a bounded numeric interval, not
evidence of discovering multiplication across new mathematical domains.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

MOD = 97
SCALE = 20.0
OUTPUT_SCALE = SCALE * SCALE
EXPERIMENTAL_ONLY = True


def _np() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("NumPy is required for the optional research experiment") from exc
    return numpy


def split_pairs() -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Operand identities are disjoint across supervised and primary holdout."""
    even = range(0, 21, 2)
    odd = range(1, 20, 2)
    outside = range(21, 33)
    return (
        [(x, y) for x in even for y in even],
        [(x, y) for x in odd for y in odd],
        [(x, y) for x in outside for y in outside],
    )


class NeuralInteraction:
    """A trainable two-input nonlinear function, with no product feature."""

    def __init__(self, *, seed: int = 41, width: int = 32) -> None:
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("invalid seed")
        if type(width) is not int or not 4 <= width <= 64:
            raise ValueError("width outside bounded research range")
        np = _np()
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(0, 0.34, size=(2, width)),
            "b1": np.zeros(width),
            "w2": rng.normal(0, 0.21, size=(width, width)),
            "b2": np.zeros(width),
            "w3": rng.normal(0, 0.14, size=(width, 1)),
            "b3": np.zeros(1),
        }

    def forward(self, inputs: Any) -> tuple[Any, tuple[Any, ...]]:
        """Only unary tanh and trainable affine layers consume raw operands."""
        np = _np()
        p = self.parameters
        first = np.tanh(inputs @ p["w1"] + p["b1"])
        second = np.tanh(first @ p["w2"] + p["b2"])
        output = second @ p["w3"] + p["b3"]
        return output, (inputs, first, second)

    def predict(self, pairs: list[tuple[int, int]]) -> Any:
        np = _np()
        values = np.asarray(pairs, dtype=np.float64) / SCALE
        output, _ = self.forward(values)
        return np.rint(output.ravel() * OUTPUT_SCALE).astype(np.int64) % MOD

    def weights(self) -> Any:
        np = _np()
        return np.concatenate([v.ravel() for v in self.parameters.values()]).copy()


def train(
    model: NeuralInteraction, pairs: list[tuple[int, int]], labels: Any,
    *, steps: int = 10000, learning_rate: float = 0.01,
) -> None:
    """Full-batch Adam, including genuine zero-LR matched-compute control."""
    np = _np()
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("steps outside bounded research range")
    if type(learning_rate) is not float or not 0.0 <= learning_rate <= 0.05:
        raise ValueError("invalid learning rate")
    if not pairs or len(pairs) > 500:
        raise ValueError("invalid training size")
    inputs = np.asarray(pairs, dtype=np.float64) / SCALE
    targets = np.asarray(labels, dtype=np.float64).reshape(-1, 1) / OUTPUT_SCALE
    if targets.shape[0] != len(pairs) or not np.isfinite(targets).all():
        raise ValueError("invalid labels")
    first_moment = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    for iteration in range(1, steps + 1):
        prediction, (inp, first, second) = model.forward(inputs)
        grad_out = (2.0 / len(pairs)) * (prediction - targets)
        p = model.parameters
        gradients = {
            "w3": second.T @ grad_out,
            "b3": grad_out.sum(axis=0),
        }
        grad_second = (grad_out @ p["w3"].T) * (1.0 - second * second)
        gradients["w2"] = first.T @ grad_second
        gradients["b2"] = grad_second.sum(axis=0)
        grad_first = (grad_second @ p["w2"].T) * (1.0 - first * first)
        gradients["w1"] = inp.T @ grad_first
        gradients["b1"] = grad_first.sum(axis=0)
        magnitude = np.sqrt(sum(float(np.square(g).sum()) for g in gradients.values()))
        if not np.isfinite(magnitude):
            raise RuntimeError("nonfinite neural gradient")
        scale = min(1.0, 1.0 / (magnitude + 1e-12))
        for name, grad in gradients.items():
            grad = grad * scale
            first_moment[name] = 0.9 * first_moment[name] + 0.1 * grad
            second_moment[name] = 0.999 * second_moment[name] + 0.001 * grad * grad
            corrected_m = first_moment[name] / (1.0 - 0.9 ** iteration)
            corrected_v = second_moment[name] / (1.0 - 0.999 ** iteration)
            p[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)


def _score(model: NeuralInteraction, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    np = _np()
    prediction = model.predict(pairs)
    expected = np.asarray([(x * y) % MOD for x, y in pairs], dtype=np.int64)
    return {"correct": int((prediction == expected).sum()), "total": len(pairs)}


def study(*, seed: int = 41, steps: int = 10000) -> dict[str, Any]:
    """Train controlled arms from the same initialization; do not certify."""
    np = _np()
    training_pairs, withheld_pairs, outside_pairs = split_pairs()
    labels = np.asarray([x * y for x, y in training_pairs], dtype=np.float64)
    rng = np.random.default_rng(seed + 100)
    corrupted_labels = labels[rng.permutation(len(labels))]
    train_operands = {value for pair in training_pairs for value in pair}
    withheld_operands = {value for pair in withheld_pairs for value in pair}
    frozen = NeuralInteraction(seed=seed)
    candidate = NeuralInteraction(seed=seed)
    sham = NeuralInteraction(seed=seed)
    corrupted = NeuralInteraction(seed=seed)
    before = frozen.weights()
    train(candidate, training_pairs, labels, steps=steps, learning_rate=0.01)
    train(sham, training_pairs, labels, steps=steps, learning_rate=0.0)
    train(corrupted, training_pairs, corrupted_labels, steps=steps, learning_rate=0.01)
    if not np.array_equal(sham.weights(), before):
        raise RuntimeError("zero-LR weights changed")
    if np.array_equal(candidate.weights(), before):
        raise RuntimeError("candidate did not update")
    arms = {"frozen": frozen, "zero_lr_sham": sham,
            "corrupted_labels": corrupted, "trained": candidate}
    return {
        "schema": "labyrinth.neural-mul-interpolation-research.v1",
        "parameter_count": len(candidate.weights()),
        "seed": seed, "steps": steps,
        "training_pairs": len(training_pairs),
        "primary_holdout_operand_identity_overlap": len(train_operands & withheld_operands),
        "corrupted_label_matches": int((labels == corrupted_labels).sum()),
        "arms": {name: {"training": _score(model, training_pairs),
                        "withheld_odd_operands": _score(model, withheld_pairs),
                        "outside_21_32": _score(model, outside_pairs)}
                 for name, model in arms.items()},
        "architecture": "two scalar inputs, two trainable tanh layers, linear output; no x*y input feature",
        "limitations": [
            "Odd-operand evaluation is interpolation inside seen numeric bounds, not novel domains",
            "Modulo-97 reduction is supplied after neural prediction",
            "Training uses unwrapped integer products, stronger than modulo-only supervision",
            "Offline arithmetic oracle supplies labels and evaluation truth",
            "This is an explored public split, not a preregistered sealed examination",
            "No independently sealed benchmark, pretrained backbone, or private examiner",
        ],
        "minotaur_graduated": False, "deployed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research only: learned multiplication without product feature")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(study(seed=args.seed, steps=args.steps), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
