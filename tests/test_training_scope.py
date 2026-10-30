"""Scope contracts must fail before any adaptive training or examination."""

from dataclasses import FrozenInstanceError
import json
import unittest

from labyrinth.training_scope import (
    MAX_ROOMS,
    SCHEMA,
    TrainingScopeSpec,
    parse_scope_json,
)


def scope(**changes):
    base = {
        "objective_id": "vera.empathy.v1",
        "domain": "empathy",
        "skills": ("cognitive_empathy", "compassionate_action", "boundary_recognition"),
        "specificity": 3,
        "scale": 2,
        "max_cycles": 4,
        "seed": 20261030,
    }
    return TrainingScopeSpec(**{**base, **changes})


class TrainingScopeTests(unittest.TestCase):
    def test_example_empathy_scope_and_bounded_chambers(self):
        planned = scope()
        self.assertEqual(planned.chamber_count_per_skill, 4)
        self.assertEqual(planned.total_chambers, 12)
        self.assertTrue(planned.total_chambers <= MAX_ROOMS)
        self.assertEqual(planned.domain, "empathy")

    def test_specificity_scale_both_change_extent(self):
        small = scope(scale=1, specificity=1)
        deeper = scope(scale=1, specificity=5)
        larger = scope(scale=5, specificity=5)
        self.assertLess(small.total_chambers, deeper.total_chambers)
        self.assertLess(deeper.total_chambers, larger.total_chambers)
        self.assertEqual(larger.chamber_count_per_skill, 15)

    def test_canonical_roundtrip_and_fingerprint(self):
        declared = scope()
        as_data = declared.to_dict()
        self.assertEqual(as_data["schema"], SCHEMA)
        self.assertEqual(TrainingScopeSpec.from_dict(as_data), declared)
        self.assertEqual(
            parse_scope_json(json.dumps(as_data, indent=2).encode("utf-8")), declared
        )
        self.assertEqual(
            parse_scope_json(json.dumps(dict(reversed(list(as_data.items())))).encode("utf-8")).fingerprint,
            declared.fingerprint,
        )
        self.assertRegex(declared.fingerprint, r"^[0-9a-f]{64}$")

    def test_material_fields_affect_fingerprint(self):
        base = scope()
        for name, changed in {
            "objective_id": "vera.empathy.v2",
            "domain": "reasoning",
            "skills": ("cognitive_empathy",),
            "specificity": 5,
            "scale": 5,
            "max_cycles": 2,
            "seed": 1,
        }.items():
            with self.subTest(name=name):
                self.assertNotEqual(base.fingerprint, scope(**{name: changed}).fingerprint)

    def test_invalid_identifiers_and_reserved_splits(self):
        for field, invalid in (
            ("objective_id", ""),
            ("objective_id", "bad name"),
            ("objective_id", "x" * 129),
            ("domain", "Empathy"),
            ("domain", "test/injection"),
            ("skills", ("good", "good")),
            ("skills", ("train", "eval")),
            ("skills", ("control_override",)),
            ("skills", ("holdout_set",)),
            ("skills", ("",)),
            ("skills", ("a b",)),
            ("skills", ()),
            ("skills", ("x",) * 13),
        ):
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                scope(**{field: invalid})

    def test_bad_numbers_rejected_even_bool_and_nonfinite(self):
        for name in ("specificity", "scale", "max_cycles", "seed"):
            for bad in (True, False, 0.5, float("nan"), "2"):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    scope(**{name: bad})
        for name, invalid in (
            ("specificity", 0), ("specificity", 6),
            ("scale", 0), ("scale", 6),
            ("max_cycles", 0), ("max_cycles", 9),
            ("seed", -1), ("seed", 2**31),
        ):
            with self.subTest(name=name, value=invalid), self.assertRaises(ValueError):
                scope(**{name: invalid})

    def test_large_scope_exceeds_room_cap(self):
        with self.assertRaises(ValueError):
            scope(skills=tuple(f"skill_{i}" for i in range(12)), specificity=5, scale=5)

    def test_immutable_and_mutation_revalidation(self):
        planned = scope()
        with self.assertRaises(FrozenInstanceError):
            planned.seed = 5
        object.__setattr__(planned, "scale", float("inf"))
        for operation in (lambda: planned.to_dict(), lambda: planned.fingerprint,
                          lambda: planned.chamber_count_per_skill):
            with self.assertRaises(ValueError):
                operation()

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        base = json.dumps(scope().to_dict())
        for payload in (
            base.replace('"scale": 2', '"scale": 2, "scale": 2'),
            base.replace('"seed": 20261030', '"seed": NaN'),
            base.replace('"seed": 20261030', '"seed": Infinity'),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_scope_json(payload.encode("utf-8"))

    def test_bounded_json_and_schema_validation(self):
        base = scope().to_dict()
        for changed in (
            {**base, "schema": "v2"},
            {**base, "unknown": 3},
            {key: value for key, value in base.items() if key != "seed"},
            {**base, "skills": "cognitive_empathy"},
        ):
            with self.assertRaises(ValueError):
                TrainingScopeSpec.from_dict(changed)
        for bad_raw in (b"", b"x" * 16385, b"{", b"\xff", "json", None):
            with self.subTest(raw=type(bad_raw).__name__), self.assertRaises(ValueError):
                parse_scope_json(bad_raw)


if __name__ == "__main__":
    unittest.main()
