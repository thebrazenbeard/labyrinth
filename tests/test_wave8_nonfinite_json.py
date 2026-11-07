"""Hostile JSON numeric overflow must fail before a child is launched."""

import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import ProcessModelAdapter, _read_json


class ProcessAdapterNonfiniteJSONTests(unittest.TestCase):
    def test_deep_public_json_rejected_without_unhandled_recursion_or_launch(self):
        raw = '{"a":' + '[' * 10000 + '0' + ']' * 10000 + '}'
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen") as launched:
            with self.assertRaises(ValueError):
                ProcessModelAdapter(mode="probe").generate(raw)
            launched.assert_not_called()

    def test_exponent_overflow_is_rejected_without_child_launch(self):
        for raw in (b'{"turn":1e999}', b'{"nested":{"value":-1e999}}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    _read_json(raw)
                with patch("labyrinth.minotaur_process_adapter.subprocess.Popen") as launched:
                    with self.assertRaises(ValueError):
                        ProcessModelAdapter(mode="probe").generate(raw.decode("utf-8"))
                    launched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
