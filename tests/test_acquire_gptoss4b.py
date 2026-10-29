"""Offline tests; never contact Hugging Face or download model weights."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import acquire_gptoss4b as candidate


class Response(io.BytesIO):
    def __init__(self, body: bytes, status=200, content_range=""):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}


class AcquisitionTests(unittest.TestCase):
    def test_candidate_is_pinned_under_four_gb(self):
        self.assertEqual(len(candidate.REVISION), 40)
        self.assertEqual(candidate.EXPECTED_BYTES, 3_233_019_552)
        self.assertLess(candidate.EXPECTED_BYTES, candidate.MAX_BYTES)
        self.assertEqual(len(candidate.EXPECTED_SHA256), 64)
        self.assertIn(candidate.REVISION, candidate.URL)

    def test_exact_hash_promotes_and_writes_receipt(self):
        body = b"GGUFoffline-test-payload"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = candidate.download(
                root, opener=lambda req, timeout: Response(body),
                expected_bytes=len(body), expected_sha256=hashlib.sha256(body).hexdigest(),
                max_bytes=100,
            )
            self.assertEqual(out["sha256_verified"], hashlib.sha256(body).hexdigest())
            self.assertEqual((root / candidate.FILENAME).read_bytes(), body)
            self.assertTrue((root / (candidate.FILENAME + ".receipt.json")).is_file())
            self.assertFalse((root / (candidate.FILENAME + ".part")).exists())

    def test_digest_mismatch_does_not_promote(self):
        body = b"GGUFbad-payload"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                candidate.download(root, opener=lambda req, timeout: Response(body),
                                   expected_bytes=len(body), expected_sha256="0" * 64,
                                   max_bytes=100)
            self.assertFalse((root / candidate.FILENAME).exists())
            self.assertTrue((root / (candidate.FILENAME + ".part")).exists())

    def test_refuses_range_ignored_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / (candidate.FILENAME + ".part")).write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "resumable range"):
                candidate.download(root, opener=lambda req, timeout: Response(b"replacement"),
                                   expected_bytes=9, expected_sha256="0" * 64,
                                   max_bytes=100)
            self.assertEqual((root / (candidate.FILENAME + ".part")).read_bytes(), b"GGUF")

    def test_complete_partial_is_verified_without_requesting_impossible_range(self):
        body = b"GGUFcomplete-but-not-promoted"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / (candidate.FILENAME + ".part")).write_bytes(body)
            out = candidate.download(
                root, opener=lambda req, timeout: self.fail("unexpected HTTP request"),
                expected_bytes=len(body), expected_sha256=hashlib.sha256(body).hexdigest(),
                max_bytes=100,
            )
            self.assertEqual(out["bytes"], len(body))
            self.assertEqual((root / candidate.FILENAME).read_bytes(), body)

    def test_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / candidate.FILENAME).write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                candidate.download(root, opener=lambda req, timeout: None)

    def test_refuses_insufficient_disk_space_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(candidate.shutil, "disk_usage", return_value=type("D", (), {"free": 1})()):
                with self.assertRaisesRegex(OSError, "insufficient free space"):
                    candidate.download(root, opener=lambda req, timeout: None)


if __name__ == "__main__":
    unittest.main()
