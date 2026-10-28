"""Run focused CPU tests, retaining the reviewed minimum test inventory.

New tests run automatically. Renaming or removing a baseline case requires an
explicit, reviewable update to tests/minotaur_required_tests.txt.
"""

from pathlib import Path
import os
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


def unexpected_skips(skipped, *, platform: str):
    """Refuse every skipped neural case except an exact unsupported-OS probe.

    Windows has no POSIX named FIFO to construct; all other skips still
    invalidate qualification. This never exempts the same case on Linux.
    """
    allowed = {
        ("test_minotaur_public_checkpoint.PublicCheckpointTests."
         "test_fifo_challenge_refuses_without_blocking_open",
         "named pipes unsupported")
    } if platform == "nt" else set()
    return [(test, why) for test, why in skipped
            if (test.id(), why) not in allowed]


def main():
    inventory = Path("tests/minotaur_required_tests.txt")
    required = [line.strip() for line in inventory.read_text().splitlines() if line.strip()]
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_minotaur_*.py")
    discovered = list(test_ids(suite))
    missing = missing_required(required, discovered)
    if missing:
        sys.exit("Required Minotaur tests missing:\n" + "\n".join(missing))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unexpected = unexpected_skips(result.skipped, platform=os.name)
    if unexpected:
        sys.exit(f"Unexpected skipped Minotaur tests: {unexpected!r}")
    if result.skipped:
        print("Minotaur: POSIX FIFO probe unavailable on Windows; qualified "
              "platform-specific skip was recorded.", file=sys.stderr)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
