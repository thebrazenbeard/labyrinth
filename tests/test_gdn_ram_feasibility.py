"""CPU-only negative controls for one-hot GDN memory and footprint bounds."""

import unittest

from labyrinth.gdn_ram_feasibility import (
    OneHotDeltaRAM,
    micro_ram_control,
    qwenputer_stock_footprint_floor,
)


class GatedDeltaFeasibilityTests(unittest.TestCase):
    def test_one_hot_overwrite_and_read_without_cross_talk(self):
        ram = OneHotDeltaRAM(4, 4)
        a, b = ram.key_for(0), ram.key_for(3)
        ram.write(a, (1, 0, 0, 1))
        ram.write(b, (1, 1, 0, 0))
        self.assertEqual(ram.read(a), (1, 0, 0, 1))
        self.assertEqual(ram.read(b), (1, 1, 0, 0))
        self.assertEqual(ram.read(ram.key_for(1)), (0, 0, 0, 0))
        ram.write(a, (0, 1, 1, 0))
        self.assertEqual(ram.read(a), (0, 1, 1, 0))
        self.assertEqual(ram.read(b), (1, 1, 0, 0))

    def test_beta_hold_and_zero_clear_are_mathematical_not_host_shortcuts(self):
        ram = OneHotDeltaRAM(2, 3)
        k = ram.key_for(1)
        ram.write(k, (1, 1, 0))
        old = ram.snapshot()
        ram.write(k, (0, 0, 1), beta=0)
        self.assertEqual(ram.snapshot(), old)
        ram.write(k, (0, 0, 0), beta=1)
        self.assertEqual(ram.read(k), (0, 0, 0))

    def test_multihot_and_invalid_numeric_inputs_fail_closed(self):
        ram = OneHotDeltaRAM(4, 2)
        for bad_key in ((0, 0, 0, 0), (1, 1, 0, 0), (True, 0, 0, 0),
                        (2, 0, 0, 0), (1, 0, 0)):
            with self.subTest(key=bad_key), self.assertRaises(ValueError):
                ram.write(bad_key, (1, 0))
        for bad_value in ((1, 0, 1), (False, 0), (1, 2), [1, 0]):
            with self.subTest(value=bad_value), self.assertRaises(ValueError):
                ram.write(ram.key_for(0), bad_value)
        for bad_beta in (True, False, 2, 0.5):
            with self.subTest(beta=bad_beta), self.assertRaises(ValueError):
                ram.write(ram.key_for(0), (1, 0), beta=bad_beta)
        for bad_shape in ((0, 8), (64, 0), (65, 8), (8, 65), (True, 8)):
            with self.subTest(shape=bad_shape), self.assertRaises(ValueError):
                OneHotDeltaRAM(*bad_shape)

    def test_seeded_oracle_probes_every_row_after_each_write(self):
        report = micro_ram_control(seed=20261106, rows=16, width=8, transactions=128)
        self.assertIs(report["all_matched"], True)
        self.assertEqual(report["independent_oracle_row_comparisons"], 2048)
        self.assertIs(report["actual_qwen_model_or_gpu_run"], False)
        self.assertEqual(report, micro_ram_control(seed=20261106, rows=16,
                                                   width=8, transactions=128))

    def test_author_parameter_counts_show_three_gb_is_not_qualified(self):
        report = qwenputer_stock_footprint_floor()
        # The absolute best-case ideal 4-bit *all* text-only weights is about
        # 3.21 decimal GB, with no quant metadata or runtime working memory.
        self.assertEqual(report["ideal_all_q4_textonly_bytes_excluding_overhead"],
                         3_210_500_000)
        self.assertIs(report["ideal_all_q4_exceeds_3_decimal_gb"], True)
        # 3.0 GiB is slightly larger; distinguish unit systems honestly.
        self.assertIs(report["ideal_all_q4_exceeds_3_gib_before_overhead"], False)
        # Keeping nonzero-sparse but dense-allocated chip layers at BF16 is
        # vastly larger: ~5.55 decimal GB *before* KV/state/scale overhead.
        self.assertEqual(report["hypothetical_bf16_chip_q4_other_bytes_excluding_overhead"],
                         5_547_500_000)
        self.assertIs(report["mixed_bf16_chip_q4_other_exceeds_3_gib"], True)
        self.assertIs(report["quantized_chip_exactness_qualified"], False)
        self.assertIs(report["resident_memory_or_lappy_probe_performed"], False)


if __name__ == "__main__":
    unittest.main()
