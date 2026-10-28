"""Public neural multiplication transfer experiment; no model admission authority.

This compares a generic tanh MLP with a quadratic-activation neural interaction.
Neither forward pass receives an x*y feature or a fixed multiplication routine.
The square activation is a strong polynomial inductive bias: even success outside
the training interval does not establish general-purpose neural reasoning.
"""

from __future__ import annotations

import argparse
import json

from .minotaur_neural_interaction import (
    MOD, OUTPUT_SCALE, SCALE, _np, _score, split_pairs,
)


class QuadraticInteraction:
    """Trainable affine -> square -> affine network using only raw x,y inputs."""

    def __init__(self, *, seed: int, width: int = 8) -> None:
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("invalid seed")
        if type(width) is not int or not 2 <= width <= 32:
            raise ValueError("invalid width")
        np = _np()
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w": rng.normal(0, 0.2, (2, width)),
            "b": np.zeros(width),
            "v": rng.normal(0, 0.2, (width, 1)),
            "c": np.zeros(1),
        }

    def forward(self, inputs):
        np = _np()
        p = self.parameters
        pre = inputs @ p["w"] + p["b"]
        hidden = np.square(pre)
        output = hidden @ p["v"] + p["c"]
        return output, (inputs, pre, hidden)

    def predict(self, pairs):
        np = _np()
        inputs = np.asarray(pairs, dtype=np.float64) / SCALE
        output, _ = self.forward(inputs)
        return np.rint(output.ravel() * OUTPUT_SCALE).astype(np.int64) % MOD

    def weights(self):
        np = _np()
        return np.concatenate([v.ravel() for v in self.parameters.values()]).copy()

    def polynomial_coefficients(self):
        """Posthoc exact decomposition; not used for training or inference."""
        np = _np()
        p = self.parameters
        xweights, yweights = p["w"]
        biases = p["b"]
        output_weights = p["v"].ravel()
        return {
            "x_squared": float(np.dot(output_weights, xweights * xweights)),
            "xy": float(2.0 * np.dot(output_weights, xweights * yweights)),
            "y_squared": float(np.dot(output_weights, yweights * yweights)),
            "x": float(2.0 * np.dot(output_weights, xweights * biases)),
            "y": float(2.0 * np.dot(output_weights, yweights * biases)),
            "constant": float(np.dot(output_weights, biases * biases) + p["c"][0]),
        }


def train_quadratic(model, pairs, labels, *, steps: int = 3000, learning_rate: float = 0.01):
    """Train the entire network; zero-LR still runs identical optimizer work."""
    np = _np()
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("invalid steps")
    if type(learning_rate) is not float or not 0 <= learning_rate <= 0.05:
        raise ValueError("invalid learning rate")
    if not pairs or len(pairs) > 500:
        raise ValueError("invalid pairs")
    inputs = np.asarray(pairs, dtype=np.float64) / SCALE
    target = np.asarray(labels, dtype=np.float64).reshape(-1, 1) / OUTPUT_SCALE
    if target.shape[0] != len(pairs) or not np.isfinite(target).all():
        raise ValueError("invalid labels")
    moments = {k: np.zeros_like(v) for k, v in model.parameters.items()}
    variances = {k: np.zeros_like(v) for k, v in model.parameters.items()}
    for t in range(1, steps + 1):
        pred, (inp, pre, hidden) = model.forward(inputs)
        p = model.parameters
        grad_out = 2.0 * (pred - target) / len(pairs)
        grad_pre = (grad_out @ p["v"].T) * (2.0 * pre)
        grads = {
            "w": inp.T @ grad_pre,
            "b": grad_pre.sum(axis=0),
            "v": hidden.T @ grad_out,
            "c": grad_out.sum(axis=0),
        }
        magnitude = np.sqrt(sum(float(np.square(g).sum()) for g in grads.values()))
        if not np.isfinite(magnitude):
            raise RuntimeError("nonfinite gradients")
        factor = min(1.0, 1.0 / (magnitude + 1e-12))
        for name, gradient in grads.items():
            gradient = gradient * factor
            moments[name] = 0.9 * moments[name] + 0.1 * gradient
            variances[name] = 0.999 * variances[name] + 0.001 * gradient * gradient
            mhat = moments[name] / (1 - 0.9 ** t)
            vhat = variances[name] / (1 - 0.999 ** t)
            p[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)


def _score_shifted(model, pairs, *, coordinate_scale: float = 1.0,
                   modulus: int = MOD) -> dict[str, int]:
    """Declared diagnostic shifts; not evidence of learned input parsing/modulus."""
    np = _np()
    if not 0.1 <= coordinate_scale <= 4 or modulus <= 1:
        raise ValueError("invalid shift")
    encoded = np.asarray(pairs, dtype=np.float64) * coordinate_scale / SCALE
    raw, _ = model.forward(encoded)
    prediction = np.rint(raw.ravel() * OUTPUT_SCALE).astype(np.int64) % modulus
    expected = np.asarray([(x * y) % modulus for x, y in pairs], dtype=np.int64)
    return {"correct": int((prediction == expected).sum()), "total": len(pairs)}


def experiment(*, seed: int, steps: int = 3000) -> dict:
    """Public disjoint-family and signed-range probes with matched controls."""
    np = _np()
    training, odd, outside = split_pairs()
    signed_shift = [(x, y) for x in range(-12, 0) for y in range(-12, 0)]
    labels = np.asarray([x * y for x, y in training], dtype=np.float64)
    rng = np.random.default_rng(seed + 100)
    corrupt_labels = labels[rng.permutation(len(labels))]
    arms = {name: QuadraticInteraction(seed=seed) for name in
            ("trained", "frozen", "zero_lr_sham", "corrupted_labels")}
    initial = arms["frozen"].weights()
    train_quadratic(arms["trained"], training, labels, steps=steps, learning_rate=0.01)
    train_quadratic(arms["zero_lr_sham"], training, labels, steps=steps, learning_rate=0.0)
    train_quadratic(arms["corrupted_labels"], training, corrupt_labels,
                    steps=steps, learning_rate=0.01)
    if not np.array_equal(initial, arms["zero_lr_sham"].weights()):
        raise RuntimeError("zero-learning-rate control changed")
    if np.array_equal(initial, arms["trained"].weights()):
        raise RuntimeError("candidate failed to train")
    scores = {name: {
        "train": _score(model, training),
        "heldout_odd": _score(model, odd),
        "outside_21_32": _score(model, outside),
        "negative_minus12_minus1": _score(model, signed_shift),
        "outside_modulus_101_external_projection": _score_shifted(
            model, outside, modulus=101),
        "odd_coordinate_units_doubled": _score_shifted(
            model, odd, coordinate_scale=2.0),
    } for name, model in arms.items()}
    return {
        "schema": "labyrinth.neural-mul-polynomial-bias-research.v1",
        "seed": seed, "steps_per_train_arm": steps,
        "parameter_count": len(arms["trained"].weights()),
        "training_examples": len(training),
        "heldout_operand_overlap": len(set(range(0, 21, 2)) & set(range(1, 20, 2))),
        "architecture": "trainable two-input quadratic-activation hidden layer; no x*y input",
        "trained_posthoc_polynomial_coefficients": arms["trained"].polynomial_coefficients(),
        "scores": scores,
        "limits": [
            "Quadratic activation strongly biases model toward polynomial functions",
            "Training labels contain unwrapped integer products rather than modulo residues",
            "Modulo-97 output projection is engineered, not learned",
            "Modulus-101 diagnostic also uses an externally supplied projection",
            "Negative inputs test a signed numerical range, not textual representation learning",
            "Doubled numeric coordinate units are an adversarial encoding shift without an adaptation key",
            "Public fixed split and research design are not independent sealed evaluation",
            "No pretrained language model, private examiner, or graduation authority",
        ],
        "minotaur_graduated": False,
        "deployed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpromotable public neural transfer research")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(experiment(seed=args.seed, steps=args.steps), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
