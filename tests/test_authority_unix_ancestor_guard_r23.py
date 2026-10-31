"""Source-level hostile checks for operator path ancestor custody.

The server's file and DB parent may be private yet replaceable through an
earlier directory. These checks do not establish independent OS administration.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.authority_service_core_r20 import SourceOnlyJourneyAuthorityCore
from labyrinth.authority_unix_service_r22 import (
    UnixAuthorityService, UnixAuthorityTransportError, _require_private_parent,
)
from tests.test_authority_unix_service_r22 import _binding, _pin


class OperatorAncestorGuardR23Tests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.risky = self.root / "writable-grandparent"
        self.risky.mkdir(mode=0o700)
        self.risky.chmod(0o777)
        self.private = self.risky / "private-child"
        self.private.mkdir(mode=0o700)
        self.safe = self.root / "safe"
        self.safe.mkdir(mode=0o700)

    def test_operator_config_key_and_database_reject_writable_grandparent(self):
        for label, target in (("operator config", "operator.json"),
                              ("operator signing key", "seed.bin"),
                              ("operator database", "authority.sqlite3")):
            with self.subTest(label=label):
                with self.assertRaises(UnixAuthorityTransportError):
                    _require_private_parent(self.private / target, label=label)

    def test_group_writable_grandparent_rejected_even_with_private_child(self):
        self.risky.chmod(0o770)
        with self.assertRaises(UnixAuthorityTransportError):
            _require_private_parent(self.private / "seed.bin", label="operator key")

    def test_intermediate_symlink_refused(self):
        alias = self.root / "alias"
        alias.symlink_to(self.risky, target_is_directory=True)
        with self.assertRaises(UnixAuthorityTransportError):
            _require_private_parent(alias / "private-child" / "seed.bin", label="operator key")

    def test_private_root_below_sticky_temporary_parent_is_permitted(self):
        # A root-owned sticky /tmp parent protects other users' private entries
        # against unlink/rename; this says nothing about same-UID or root attack.
        _require_private_parent(self.safe / "seed.bin", label="operator key")

    def test_socket_with_writable_grandparent_is_refused_before_bind(self):
        core = SourceOnlyJourneyAuthorityCore(
            database_path=self.safe / "authority.sqlite3",
            provider_id="provider-a", service_signing_key=os.urandom(32),
            provisioned_journeys=(_pin(_binding("run-a", "event-a", "exam-a")),),
        )
        with self.assertRaises(UnixAuthorityTransportError):
            UnixAuthorityService(
                socket_path=self.private / "authority.sock", core=core,
                principal_by_uid={os.getuid(): "trainer-a"},
            )


if __name__ == "__main__":
    unittest.main()
