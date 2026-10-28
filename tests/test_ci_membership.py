import unittest

from scripts.minotaur_ci import missing_required, test_ids


class MembershipGateTests(unittest.TestCase):
    def test_missing_case_is_reported_even_when_other_cases_remain(self):
        self.assertEqual(missing_required(["module.Case.first", "module.Case.second"],
                                          ["module.Case.first", "module.Case.extra"]),
                         ["module.Case.second"])

    def test_empty_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            missing_required([], ["module.Case.first"])

    def test_nested_suite_ids_are_discovered(self):
        suite = unittest.TestSuite([unittest.TestSuite([self])])
        self.assertEqual(list(test_ids(suite)), [self.id()])
