"""Run focused CPU tests, retaining the reviewed minimum test inventory.

New tests run automatically. Renaming or removing a baseline case requires an
explicit, reviewable update to tests/minotaur_required_tests.txt.
"""

from pathlib import Path
import sys
import unittest


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


def missing_required(required, discovered):
    if not required:
        raise ValueError("Required Minotaur test inventory is empty")
    return sorted(set(required) - set(discovered))


def main():
    inventory = Path("tests/minotaur_required_tests.txt")
    required = [line.strip() for line in inventory.read_text().splitlines() if line.strip()]
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_minotaur_*.py")
    discovered = list(test_ids(suite))
    missing = missing_required(required, discovered)
    if missing:
        sys.exit("Required Minotaur tests missing:\n" + "\n".join(missing))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        sys.exit(f"Skipped Minotaur tests are not allowed: {result.skipped!r}")
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
