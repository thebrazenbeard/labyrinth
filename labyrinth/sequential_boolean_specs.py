"""Two independently compiled Boolean functions with explicit sequential ABI."""

from .sequential_boolean_compiler import compile_boolean, compose_sequential


def xor2(bits: tuple[int, ...]) -> tuple[int, ...]:
    return (bits[0] ^ bits[1],)


def invert1(bits: tuple[int, ...]) -> tuple[int, ...]:
    return (1 - bits[0],)


def build():
    xor_net = compile_boolean("xor2", ("a", "b"), ("xor_bit",), xor2)
    invert_net = compile_boolean("not1", ("xor_bit",), ("xnor_bit",), invert1)
    xnor_net = compose_sequential("xnor-via-neural-fusion", xor_net, invert_net)
    return xor_net, invert_net, xnor_net
