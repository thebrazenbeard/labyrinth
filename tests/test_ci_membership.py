import unittest

from scripts.minotaur_ci import missing_required, test_ids, unexpected_skips


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

class MinotaurPlatformSkipTests(unittest.TestCase):
    def test_only_named_fifo_probe_is_exempted_on_windows(self):
        class Fake:
            def __init__(self, ident): self.ident=ident
            def id(self): return self.ident
        case = Fake("test_minotaur_public_checkpoint.PublicCheckpointTests."
                    "test_fifo_challenge_refuses_without_blocking_open")
        skipped = [(case,"named pipes unsupported")]
        self.assertEqual(unexpected_skips(skipped,platform="nt"),[])
        self.assertEqual(unexpected_skips(skipped,platform="posix"),skipped)
        wrong_reason=[(case,"artificial skip")]
        self.assertEqual(unexpected_skips(wrong_reason,platform="nt"),wrong_reason)
        other=Fake("test_minotaur_public_checkpoint.PublicCheckpointTests."
                   "test_checkpoint_and_public_cases_score_without_graduation")
        self.assertEqual(unexpected_skips([(other,"named pipes unsupported")],
                                          platform="nt"),
                         [(other,"named pipes unsupported")])
