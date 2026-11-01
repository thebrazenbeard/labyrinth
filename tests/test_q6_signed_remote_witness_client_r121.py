"""Synthetic protocol tests, NOT a deployed independent witness service."""

import json
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import ssl
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.q6_durable_slot_authority_r120 import (
    Q6DurableHead, Q6DurableOutcomeUnknown,
)
from labyrinth.q6_signed_remote_witness_client_r121 import (
    Q6SignedMTLSRemoteWitness, Q6WitnessMTLSConfiguration, SCHEMA,
    _SIGNATURE_DOMAIN, verify_signed_reply,
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


class SignedRemoteWitnessSourceTests(unittest.TestCase):
    def setUp(self):
        self.signer = Ed25519PrivateKey.generate()
        self.pubkey = self.signer.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.before = {"commit_index": 12, "head_sha256": "1" * 64}
        self.after = {"commit_index": 13, "head_sha256": "2" * 64}
        self.read = {"schema": SCHEMA, "operation": "READ", "nonce": "a" * 32,
                     "expected": None, "new": None}
        self.cas = {"schema": SCHEMA, "operation": "CAS", "nonce": "b" * 32,
                    "expected": self.before, "new": self.after}

    def reply(self, request=None, *, ok=True, head=None, signer=None, **changed):
        request = self.read if request is None else request
        fields = {"schema": SCHEMA, "operation": request["operation"],
                  "nonce": request["nonce"],
                  "request_sha256": sha256(canonical(request)).hexdigest(),
                  "ok": ok,
                  "head": head or (self.after if request["operation"] == "CAS" else self.before)}
        fields.update(changed)
        signature = (self.signer if signer is None else signer).sign(
            _SIGNATURE_DOMAIN + canonical(fields))
        fields["signature_hex"] = signature.hex()
        return canonical(fields)

    def test_valid_read_signed_head(self):
        ok, head = verify_signed_reply(
            request=self.read, reply_bytes=self.reply(),
            pinned_signer_public_key=self.pubkey)
        self.assertTrue(ok)
        self.assertEqual(head, Q6DurableHead(12, "1" * 64))
        self.assertTrue(head.source_only_unqualified)

    def test_valid_cas_and_signed_refusal(self):
        ok, head = verify_signed_reply(
            request=self.cas, reply_bytes=self.reply(self.cas),
            pinned_signer_public_key=self.pubkey)
        self.assertTrue(ok)
        self.assertEqual(head, Q6DurableHead(13, "2" * 64))
        refused, prior = verify_signed_reply(
            request=self.cas,
            reply_bytes=self.reply(self.cas, ok=False, head=self.before),
            pinned_signer_public_key=self.pubkey)
        self.assertFalse(refused)
        self.assertEqual(prior, Q6DurableHead(12, "1" * 64))

    def test_untrusted_signer_cannot_acknowledge(self):
        with self.assertRaises(Q6DurableOutcomeUnknown):
            verify_signed_reply(request=self.cas,
                                reply_bytes=self.reply(self.cas, signer=Ed25519PrivateKey.generate()),
                                pinned_signer_public_key=self.pubkey)

    def test_replayed_nonce_and_wrong_operation_refused(self):
        other = dict(self.cas, nonce="c" * 32)
        for request in (other, self.read):
            with self.subTest(request=request["operation"]):
                with self.assertRaises(Q6DurableOutcomeUnknown):
                    verify_signed_reply(request=request, reply_bytes=self.reply(self.cas),
                                        pinned_signer_public_key=self.pubkey)

    def test_cas_expected_or_new_tamper_changes_signed_challenge(self):
        for field in ("expected", "new"):
            mutated = dict(self.cas)
            mutated[field] = {"commit_index": 99, "head_sha256": "3" * 64}
            with self.subTest(field=field), self.assertRaises(Q6DurableOutcomeUnknown):
                verify_signed_reply(request=mutated, reply_bytes=self.reply(self.cas),
                                    pinned_signer_public_key=self.pubkey)

    def test_ack_must_match_new_head_even_if_signed(self):
        with self.assertRaises(Q6DurableOutcomeUnknown):
            verify_signed_reply(request=self.cas,
                                reply_bytes=self.reply(self.cas, head=self.before),
                                pinned_signer_public_key=self.pubkey)

    def test_signed_malformed_bool_index_and_extra_field_refused(self):
        candidates = (
            {"commit_index": True, "head_sha256": "1" * 64},
            {"commit_index": 12, "head_sha256": "Z" * 64},
            {"commit_index": 12, "head_sha256": "1" * 64, "untrusted": True},
        )
        for value in candidates:
            with self.subTest(value=value), self.assertRaises(Q6DurableOutcomeUnknown):
                verify_signed_reply(request=self.read, reply_bytes=self.reply(head=value),
                                    pinned_signer_public_key=self.pubkey)

    def test_duplicate_key_and_noncanonical_json_refused(self):
        valid = self.reply()
        duplicate = valid.replace(b'"ok":true', b'"ok":false,"ok":true')
        for payload in (duplicate, valid.strip(), valid.replace(b':', b': ', 1)):
            with self.subTest(payload=payload[:60]), self.assertRaises(Q6DurableOutcomeUnknown):
                verify_signed_reply(request=self.read, reply_bytes=payload,
                                    pinned_signer_public_key=self.pubkey)

    def test_no_unpinned_transport_configuration(self):
        fake = Q6WitnessMTLSConfiguration(
            endpoint="http://witness.example/q6/witness/r121",
            ca_cert=Path("/nonexistent/ca.pem"),
            client_cert=Path("/nonexistent/client.pem"),
            client_key=Path("/nonexistent/key.pem"),
            pinned_server_certificate_sha256="1" * 64,
            pinned_signer_public_key=self.pubkey,
        )
        with self.assertRaises(Q6DurableOutcomeUnknown):
            Q6SignedMTLSRemoteWitness(fake)
        for endpoint in ("https://user:pass@witness.example/q6/witness/r121",
                         "https://witness.example/q6/witness/r121?unaudited=1",
                         "https://witness.example/other/path"):
            with self.subTest(endpoint=endpoint), self.assertRaises(Q6DurableOutcomeUnknown):
                Q6SignedMTLSRemoteWitness(Q6WitnessMTLSConfiguration(
                    endpoint=endpoint, ca_cert=fake.ca_cert, client_cert=fake.client_cert,
                    client_key=fake.client_key,
                    pinned_server_certificate_sha256="1" * 64,
                    pinned_signer_public_key=self.pubkey))

    def test_nonconsecutive_cas_denied_before_network(self):
        witness = Q6SignedMTLSRemoteWitness.__new__(Q6SignedMTLSRemoteWitness)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            witness.compare_and_advance(Q6DurableHead(1, "1" * 64),
                                        Q6DurableHead(3, "3" * 64))

    def test_signed_read_refusal_is_not_promoted(self):
        with self.assertRaises(Q6DurableOutcomeUnknown):
            verify_signed_reply(request=self.read,
                                reply_bytes=self.reply(ok=False),
                                pinned_signer_public_key=self.pubkey)

    def test_no_live_witness_server_claim_in_client(self):
        self.assertTrue(Q6SignedMTLSRemoteWitness.source_only_unqualified)
        self.assertFalse(Q6SignedMTLSRemoteWitness.independent_host_attested)

    def test_local_mutual_tls_signed_highwater_roundtrip_not_independent(self):
        """Real local mTLS handshake; server memory is not a physical witness."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Q6 test CA")])
            now = datetime.now(timezone.utc)
            ca_cert = (x509.CertificateBuilder().subject_name(ca_name)
                       .issuer_name(ca_name).public_key(ca_key.public_key())
                       .serial_number(x509.random_serial_number())
                       .not_valid_before(now - timedelta(minutes=1))
                       .not_valid_after(now + timedelta(days=1))
                       .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                       .sign(ca_key, algorithm=hashes.SHA256()))

            def identity(name, usage):
                key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
                builder = (x509.CertificateBuilder().subject_name(subject)
                           .issuer_name(ca_name).public_key(key.public_key())
                           .serial_number(x509.random_serial_number())
                           .not_valid_before(now - timedelta(minutes=1))
                           .not_valid_after(now + timedelta(days=1))
                           .add_extension(x509.ExtendedKeyUsage([usage]), critical=False))
                if name == "localhost":
                    builder = builder.add_extension(x509.SubjectAlternativeName([
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    ]), critical=False)
                cert = builder.sign(ca_key, algorithm=hashes.SHA256())
                (root / (name + ".pem")).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
                (root / (name + ".key")).write_bytes(key.private_bytes(
                    serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()))
                return cert

            server_cert = identity("localhost", ExtendedKeyUsageOID.SERVER_AUTH)
            identity("q6-client", ExtendedKeyUsageOID.CLIENT_AUTH)
            (root / "ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
            signer = self.signer
            state = {"head": self.before, "client_seen": False}

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *unused):
                    pass

                def do_POST(self):
                    state["client_seen"] = bool(self.connection.getpeercert())
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                    request = json.loads(body)
                    operation = request["operation"]
                    if operation == "CAS":
                        ok = request["expected"] == state["head"]
                        if ok:
                            state["head"] = request["new"]
                    else:
                        ok = True
                    payload = {"schema": SCHEMA, "operation": operation,
                               "nonce": request["nonce"],
                               "request_sha256": sha256(canonical(request)).hexdigest(),
                               "ok": ok, "head": state["head"]}
                    payload["signature_hex"] = signer.sign(
                        _SIGNATURE_DOMAIN + canonical(payload)).hex()
                    response = canonical(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)

            class QuietExpectedDisconnectServer(HTTPServer):
                def handle_error(self, request, client_address):
                    if not isinstance(sys.exc_info()[1], BrokenPipeError):
                        super().handle_error(request, client_address)

            server = QuietExpectedDisconnectServer(("127.0.0.1", 0), Handler)
            server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
            server_ctx.load_cert_chain(str(root / "localhost.pem"),
                                       str(root / "localhost.key"))
            server_ctx.verify_mode = ssl.CERT_REQUIRED
            server_ctx.load_verify_locations(cafile=str(root / "ca.pem"))
            server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                config = Q6WitnessMTLSConfiguration(
                    endpoint=f"https://localhost:{server.server_address[1]}/q6/witness/r121",
                    ca_cert=root / "ca.pem", client_cert=root / "q6-client.pem",
                    client_key=root / "q6-client.key",
                    pinned_server_certificate_sha256=sha256(server_cert.public_bytes(
                        serialization.Encoding.DER)).hexdigest(),
                    pinned_signer_public_key=self.pubkey)
                client = Q6SignedMTLSRemoteWitness(config)
                self.assertEqual(client.read_head(), Q6DurableHead(12, "1" * 64))
                self.assertTrue(client.compare_and_advance(Q6DurableHead(12, "1" * 64),
                                                           Q6DurableHead(13, "2" * 64)))
                self.assertFalse(client.compare_and_advance(Q6DurableHead(12, "1" * 64),
                                                            Q6DurableHead(13, "2" * 64)))
                self.assertEqual(client.read_head(), Q6DurableHead(13, "2" * 64))
                self.assertTrue(state["client_seen"])
                wrong_pin = Q6WitnessMTLSConfiguration(
                    endpoint=config.endpoint, ca_cert=config.ca_cert,
                    client_cert=config.client_cert, client_key=config.client_key,
                    pinned_server_certificate_sha256="f" * 64,
                    pinned_signer_public_key=config.pinned_signer_public_key)
                with self.assertRaises(Q6DurableOutcomeUnknown):
                    Q6SignedMTLSRemoteWitness(wrong_pin).read_head()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
