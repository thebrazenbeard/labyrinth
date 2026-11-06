"""Exact, bounded falsifier for naïve neural weight averaging.

Two one-hidden-layer ReLU networks can compute IDENTICAL functions while their
weight arrays are different only because hidden neurons were permuted. A raw
coordinate-wise weight mean need not preserve that function. This module uses
exact fractions, no optimizers, no gradients, no pretrained weights and no
host model access. It is not a general model-merging or alignment algorithm.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from itertools import product

from labyrinth.boolean_neural_compiler import NeuralArtifact


MAX_INPUTS = 10
MAX_OUTPUTS = 64
MAX_HIDDEN = 1024
MAX_EXHAUSTIVE_RATIONAL_OPS = 250_000


def _bounded(model: NeuralArtifact) -> None:
    if type(model) is not NeuralArtifact:
        raise ValueError("expected an exact ReLU NeuralArtifact")
    if not (1 <= len(model.input_names) <= MAX_INPUTS
            and 1 <= len(model.output_names) <= MAX_OUTPUTS
            and 1 <= len(model.w1) <= MAX_HIDDEN):
        raise ValueError("model exceeds bounded weight-space ablation domain")


def _paired(left: NeuralArtifact, right: NeuralArtifact) -> None:
    _bounded(left)
    _bounded(right)
    if (left.input_names != right.input_names
            or left.output_names != right.output_names
            or len(left.w1) != len(right.w1)):
        raise ValueError("coordinate-mean requires identical tensor shapes and ABI")


def permute_hidden(model: NeuralArtifact, permutation: tuple[int, ...]) -> NeuralArtifact:
    """Reorder neurons and all corresponding outgoing weights exactly."""
    _bounded(model)
    hidden = len(model.w1)
    if (type(permutation) is not tuple or len(permutation) != hidden
            or any(type(index) is not int for index in permutation)
            or set(permutation) != set(range(hidden))):
        raise ValueError("hidden permutation must be an exact bijection")
    return NeuralArtifact(
        model.name + "_permuted", model.input_names, model.output_names,
        tuple(model.w1[index] for index in permutation),
        tuple(model.b1[index] for index in permutation),
        tuple(tuple(row[index] for index in permutation) for row in model.w2),
        model.b2,
    )


def _fingerprint(model: NeuralArtifact, index: int) -> tuple:
    """Exact full-neuron identity, including outgoing parameters."""
    return (model.w1[index], model.b1[index],
            tuple(row[index] for row in model.w2))


def align_identical_neurons(reference: NeuralArtifact,
                            candidate: NeuralArtifact) -> NeuralArtifact:
    """Undo ONLY exact hidden permutations; reject changed/missing neurons.

    Unlike learned matching or Git Re-Basin, this deliberately requires an
    exact neuron multiset match. It cannot merge heterogeneous checkpoints.
    """
    _paired(reference, candidate)
    if reference.b2 != candidate.b2:
        raise ValueError("output biases differ, exact alignment unavailable")
    available: dict[tuple, deque[int]] = defaultdict(deque)
    for index in range(len(candidate.w1)):
        available[_fingerprint(candidate, index)].append(index)
    order = []
    for index in range(len(reference.w1)):
        matching = available.get(_fingerprint(reference, index))
        if not matching:
            raise ValueError("no exact hidden-neuron permutation aligns models")
        order.append(matching.popleft())
    return permute_hidden(candidate, tuple(order))


def coordinate_mean_forward(left: NeuralArtifact, right: NeuralArtifact,
                            inputs: tuple[int, ...]) -> tuple[Fraction, ...]:
    """Evaluate naive 50:50 tensor average exactly, including layer 1 and 2.

    Fraction arithmetic makes a broken mean observable without rounding it to
    apparently valid Boolean answers or silently clipping nonbinary outputs.
    """
    _paired(left, right)
    if (type(inputs) is not tuple or len(inputs) != len(left.input_names)
            or any(type(bit) is not int or bit not in (0, 1) for bit in inputs)):
        raise ValueError("input outside exact Boolean ABI")
    hidden = []
    for row_a, row_b, bias_a, bias_b in zip(left.w1, right.w1, left.b1, right.b1):
        total = Fraction(bias_a + bias_b, 2)
        total += sum(Fraction(a + b, 2) * x
                     for a, b, x in zip(row_a, row_b, inputs))
        hidden.append(max(Fraction(0), total))
    return tuple(
        Fraction(bias_a + bias_b, 2)
        + sum(Fraction(a + b, 2) * value
              for a, b, value in zip(row_a, row_b, hidden))
        for row_a, row_b, bias_a, bias_b in zip(left.w2, right.w2, left.b2, right.b2)
    )


def ablate_permutation(model: NeuralArtifact, permutation: tuple[int, ...]) -> dict:
    """Exhaustive small-domain comparison with a matched-alignment control."""
    _bounded(model)
    estimated_rational_ops = (2 ** len(model.input_names) * len(model.w1)
                              * (len(model.input_names) + len(model.output_names)))
    if estimated_rational_ops > MAX_EXHAUSTIVE_RATIONAL_OPS:
        raise ValueError("exhaustive Fraction experiment exceeds CPU budget")
    permuted = permute_hidden(model, permutation)
    aligned = align_identical_neurons(model, permuted)
    changed_weights = (
        sum(a != b for row_a, row_b in zip(model.w1, permuted.w1)
            for a, b in zip(row_a, row_b))
        + sum(a != b for a, b in zip(model.b1, permuted.b1))
        + sum(a != b for row_a, row_b in zip(model.w2, permuted.w2)
              for a, b in zip(row_a, row_b))
    )
    cases = 0
    naive_failures = 0
    for inputs in product((0, 1), repeat=len(model.input_names)):
        reference = model.evaluate(inputs)
        if permuted.evaluate(inputs) != reference:
            raise ArithmeticError("permutation unexpectedly changed function")
        if aligned.evaluate(inputs) != reference:
            raise ArithmeticError("exact alignment failed functional check")
        naive = coordinate_mean_forward(model, permuted, inputs)
        corrected = coordinate_mean_forward(model, aligned, inputs)
        if corrected != reference:
            raise ArithmeticError("aligned average changed original function")
        naive_failures += naive != reference
        cases += 1
    return {
        "schema": "LABYRINTH_WEIGHT_SPACE_PERMUTATION_ABLATION_V1",
        "evidence_class": "EXACT_SYNTHETIC_NUMERICAL_EXHAUSTIVE_TEST",
        "original_artifact_sha256": model.sha256(),
        "permuted_artifact_sha256": permuted.sha256(),
        "changed_parameter_coordinates": changed_weights,
        "exhaustive_cases": cases,
        "permuted_function_mismatches": 0,
        "naive_mean_function_mismatches": naive_failures,
        "aligned_mean_function_mismatches": 0,
        "model_training_performed": False,
        "qwen_model_edited_or_executed": False,
    }
