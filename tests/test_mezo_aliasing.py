"""Regression: a nominally frozen Parameter must never alias trained storage."""

import unittest

from labyrinth.mezo import mezo_step
from tests.test_mezo import FakeTorch, Model, Parameter


class _Storage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class _StorageParameter(Parameter):
    def __init__(self, values, *, trainable, pointer):
        super().__init__(values, trainable=trainable)
        self._storage = _Storage(pointer)

    def untyped_storage(self):
        return self._storage


class SharedStorageTests(unittest.TestCase):
    def test_rejects_trainable_tensor_sharing_frozen_parameter_storage(self):
        model = Model()
        model.parameter = _StorageParameter([2.0], trainable=True, pointer=12345)
        model.frozen = _StorageParameter([9.0], trainable=False, pointer=12345)
        called = False

        def objective():
            nonlocal called
            called = True
            return 3.0

        with self.assertRaisesRegex(ValueError, "overlapping.*storage"):
            mezo_step(model, FakeTorch, objective, epsilon=0.01,
                      learning_rate=0.05, seed=1)
        self.assertFalse(called, "alias check must precede model execution")
        self.assertEqual(model.parameter.values, [2.0])
        self.assertEqual(model.frozen.values, [9.0])

    def test_distinct_trainable_and_frozen_storage_is_permitted(self):
        model = Model()
        model.parameter = _StorageParameter([2.0], trainable=True, pointer=12345)
        model.frozen = _StorageParameter([9.0], trainable=False, pointer=12346)
        result = mezo_step(model, FakeTorch,
                           lambda: model.parameter.values[0] ** 2,
                           epsilon=0.01, learning_rate=0.05, seed=1)
        self.assertEqual(result["forward_evaluations"], 2)
        self.assertEqual(model.frozen.values, [9.0])


if __name__ == "__main__":
    unittest.main()
