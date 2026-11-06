"""Source-only Gated DeltaNet RAM microtest and Qwenputer size lower bounds.

This is an independent plain-Python implementation of the published delta
overwrite equation for one-hot keys. It does not execute Qwenputer, Qwen3.5,
an actual transformer layer, CUDA, or any model file. Reported checkpoint
parameter counts below come from a pinned upstream *document*, not a local
measurement. No gradients, training, downloads or model effects occur.
"""

from __future__ import annotations

import random


QWENPUTER_SIZE_URL = "https://github.com/divagr18/qwenputer/blob/main/docs/q35-size.md"
QWENPUTER_SIZE_BLOB = "efdf8340aa3266e7b8f5d907fb951990c2f70520"
QWENPUTER_IR_BLOB = "0e380ce0fef075a8b5408e5a493ab4c723878e3c"
QWENPUTER_LICENSE_BLOB = "ccdbffd15bbbf20bc8127e113326cae649e8059f"
PARAMETERS_BILLION = 1_000_000_000
DECIMAL_GB = 1_000_000_000
GIB = 1024 ** 3
MAX_ROWS = 64
MAX_VALUE_BITS = 64


class OneHotDeltaRAM:
    """A tiny exact-bit memory instantiating S' = S + beta*k*(v - k^T*S).

    Not a Qwen layer: ignores convolutions, RMSNorm, SwiGLU, tokenization,
    routing, multi-head state, finite-precision rounding and generation.
    """

    def __init__(self, rows: int, width: int):
        if (type(rows) is not int or type(width) is not int
                or not 1 <= rows <= MAX_ROWS or not 1 <= width <= MAX_VALUE_BITS):
            raise ValueError("micro-RAM shape exceeds strict CPU fixture bounds")
        self.rows = rows
        self.width = width
        self._state = [[0] * width for _ in range(rows)]

    def _key(self, key: tuple[int, ...]) -> None:
        if (type(key) is not tuple or len(key) != self.rows
                or any(type(bit) is not int or bit not in (0, 1) for bit in key)
                or sum(key) != 1):
            raise ValueError("GDN micro-RAM needs a one-hot row key")

    def _value(self, value: tuple[int, ...]) -> None:
        if (type(value) is not tuple or len(value) != self.width
                or any(type(bit) is not int or bit not in (0, 1) for bit in value)):
            raise ValueError("GDN micro-RAM needs a bounded binary value")

    def key_for(self, address: int) -> tuple[int, ...]:
        if type(address) is not int or not 0 <= address < self.rows:
            raise ValueError("address outside micro-RAM")
        return tuple(int(index == address) for index in range(self.rows))

    def write(self, key: tuple[int, ...], value: tuple[int, ...], *, beta: int = 1) -> None:
        """Apply actual explicit dense delta equation, not a row assignment."""
        self._key(key)
        self._value(value)
        if type(beta) is not int or beta not in (0, 1):
            raise ValueError("only exact overwrite/hold beta values supported")
        previous = [sum(key[i] * self._state[i][j] for i in range(self.rows))
                    for j in range(self.width)]
        for i in range(self.rows):
            for j in range(self.width):
                self._state[i][j] += beta * key[i] * (value[j] - previous[j])

    def read(self, query: tuple[int, ...]) -> tuple[int, ...]:
        self._key(query)
        return tuple(sum(query[i] * self._state[i][j] for i in range(self.rows))
                     for j in range(self.width))

    def snapshot(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._state)


def micro_ram_control(seed: int = 20261106, *, rows: int = 16,
                      width: int = 8, transactions: int = 128) -> dict:
    """Independent dictionary-memory oracle and deterministic overwrite trial."""
    if (type(seed) is not int or not 0 <= seed < 2**64
            or type(transactions) is not int or not 1 <= transactions <= 4096):
        raise ValueError("bounded deterministic micro-RAM trial required")
    ram = OneHotDeltaRAM(rows, width)
    oracle = {address: (0,) * width for address in range(rows)}
    rng = random.Random(seed)
    comparisons = 0
    for _ in range(transactions):
        addr = rng.randrange(rows)
        value = tuple(rng.randrange(2) for _ in range(width))
        key = ram.key_for(addr)
        ram.write(key, value)
        oracle[addr] = value
        # Probe every row, not only the last writer, to detect cross-talk.
        for read_addr in range(rows):
            got = ram.read(ram.key_for(read_addr))
            if got != oracle[read_addr]:
                raise ArithmeticError("delta-rule RAM diverged from independent oracle")
            comparisons += 1
    return {
        "schema": "LABYRINTH_GDN_MICRO_RAM_CONTROL_V1",
        "evidence_class": "SYNTHETIC_INTEGER_DELTA_EQUATION_ONLY",
        "seed": seed,
        "rows": rows,
        "value_bits": width,
        "transactions": transactions,
        "independent_oracle_row_comparisons": comparisons,
        "all_matched": True,
        "qwenputer_ir_blob_audited": QWENPUTER_IR_BLOB,
        "actual_qwen_model_or_gpu_run": False,
    }


def qwenputer_stock_footprint_floor() -> dict:
    """Compute transparent *idealized* storage floors from author size counts.

    Upstream states ~6.421B parameters for its optimized text-only variant,
    with ~1.558B in added analytic chip layers. These are rounded metadata,
    not locally hashed safetensors. These floors exclude scales, metadata,
    runtime code, activations, KV/state cache and quantization accuracy loss.
    """
    text_params = 6_421_000_000
    chip_params = 1_558_000_000
    other_params = text_params - chip_params
    ideal_all_q4 = text_params * 4 // 8
    mixed_bf16_chip_q4_other = chip_params * 2 + other_params * 4 // 8
    return {
        "schema": "LABYRINTH_QWENPUTER_FOOTPRINT_LOWER_BOUND_V1",
        "evidence_class": "AUTHOR_REPORTED_ROUNDED_PARAMS_PLUS_LOCAL_ARITHMETIC",
        "author_source_url": QWENPUTER_SIZE_URL,
        "author_source_blob_sha1": QWENPUTER_SIZE_BLOB,
        "license_blob_sha1": QWENPUTER_LICENSE_BLOB,
        "author_base_bf16_decimal_gb": 9.34,
        "author_original_overlay_bf16_decimal_gb": 15.15,
        "author_optimized_overlay_bf16_decimal_gb": 13.53,
        "author_optimized_textonly_bf16_decimal_gb": 12.86,
        "author_optimized_textonly_parameter_count_approx": text_params,
        "author_chip_parameter_count_approx": chip_params,
        "ideal_all_q4_textonly_bytes_excluding_overhead": ideal_all_q4,
        "hypothetical_bf16_chip_q4_other_bytes_excluding_overhead": mixed_bf16_chip_q4_other,
        "ideal_all_q4_exceeds_3_decimal_gb": ideal_all_q4 > 3 * DECIMAL_GB,
        "ideal_all_q4_exceeds_3_gib_before_overhead": ideal_all_q4 > 3 * GIB,
        "mixed_bf16_chip_q4_other_exceeds_3_gib": mixed_bf16_chip_q4_other > 3 * GIB,
        "quantized_chip_exactness_qualified": False,
        "resident_memory_or_lappy_probe_performed": False,
    }
