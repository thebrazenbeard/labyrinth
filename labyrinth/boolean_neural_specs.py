"""Two genuinely separate deterministic functions and their shared typed ABI."""

from .boolean_neural_compiler import NeuralArtifact, compile_boolean, link_parallel


INPUT_ABI = ("a", "b", "carry_in")


def parity3(bits: tuple[int, ...]) -> tuple[int, ...]:
    return (sum(bits) % 2,)


def majority3(bits: tuple[int, ...]) -> tuple[int, ...]:
    return (int(sum(bits) >= 2),)


def build() -> tuple[NeuralArtifact, NeuralArtifact, NeuralArtifact]:
    parity = compile_boolean("parity3", INPUT_ABI, ("sum_bit",), parity3)
    majority = compile_boolean("majority3", INPUT_ABI, ("carry_out",), majority3)
    full_adder = link_parallel("full_adder", (parity, majority))
    return parity, majority, full_adder
