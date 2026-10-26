"""Hermetic adapter tests: no Ollama server or external network required."""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib import error, request

from labyrinth.adapters import (
    AdapterError,
    ModelAdapter,
    OllamaAdapter,
    ReplayAdapter,
    ScriptedAdapter,
    load_adapter,
)


class ReplayAdapterTests(unittest.TestCase):
    def test_exact_match_and_repeatability(self) -> None:
        adapter = ReplayAdapter({"maze?": "door B"}, model_id="fixture-v1")
        self.assertIsInstance(adapter, ModelAdapter)
        self.assertEqual(adapter.model_id, "fixture-v1")
        self.assertEqual(adapter.generate("maze?", seed=4), "door B")
        self.assertEqual(adapter.generate("maze?", seed=800), "door B")

    def test_unknown_prompt_fails_closed(self) -> None:
        with self.assertRaises(AdapterError):
            ReplayAdapter({}).generate("unknown")

    def test_explicit_default_and_copy(self) -> None:
        responses = {"prompt": "one"}
        adapter = ScriptedAdapter(responses, default="fallback")
        responses["prompt"] = "poisoned"
        self.assertEqual(adapter.generate("prompt"), "one")
        self.assertEqual(adapter.generate("missing"), "fallback")
        with self.assertRaises(TypeError):
            adapter.responses["prompt"] = "poisoned"  # type: ignore[index]

    def test_invalid_prompt_and_seed(self) -> None:
        adapter = ReplayAdapter({})
        for seed in (-1, 2**31, True, 2.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                adapter.generate("x", seed=seed)
        with self.assertRaises(TypeError):
            adapter.generate(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            adapter.generate("a" * 64_001)

    def test_factory(self) -> None:
        self.assertEqual(load_adapter("replay", responses={"x": "y"}).generate("x"), "y")
        self.assertIsInstance(load_adapter("scripted", responses={}), ReplayAdapter)
        with self.assertRaises(ValueError):
            load_adapter("import.anything")


class _FakeResponse(io.BytesIO):
    status = 200


class OllamaAdapterTests(unittest.TestCase):
    def test_local_url_validation(self) -> None:
        for host in ("127.0.0.1", "[::1]", "localhost"):
            with self.subTest(host=host):
                self.assertEqual(OllamaAdapter("model", base_url=f"http://{host}:11434").model_id, "ollama:model")
        for url in (
            "http://remote.example:11434",
            "http://192.168.1.71:11434",
            "http://169.254.169.254:11434",
            "https://127.0.0.1:11434",
            "http://user:pass@localhost:11434",
            "http://localhost:11434/proxy",
            "http://localhost:11434?token=x",
            "http://localhost:notaport",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaAdapter("model", base_url=url)
        self.assertEqual(OllamaAdapter("model", base_url="http://192.168.1.71:11434", allow_remote=True).model_id, "ollama:model")

    def test_request_payload_and_response(self) -> None:
        adapter = OllamaAdapter("local-model", timeout_seconds=1.5, max_output_tokens=17)
        opener = Mock()
        opener.open.return_value = _FakeResponse(json.dumps({"done": True, "response": "42"}).encode())
        object.__setattr__(adapter, "_opener", opener)
        self.assertEqual(adapter.generate("Question?", seed=123), "42")
        call = opener.open.call_args
        req = call.args[0]
        self.assertIsInstance(req, request.Request)
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(call.kwargs["timeout"], 1.5)
        self.assertEqual(req.get_method(), "POST")
        data = json.loads(req.data)
        self.assertEqual(data, {
            "model": "local-model",
            "prompt": "Question?",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 17, "seed": 123},
        })

    def test_rejects_oversize_bad_json_and_incomplete(self) -> None:
        adapter = OllamaAdapter("model", max_response_bytes=64)
        for payload in (b"a" * 65, b"not-json", b'{"response":"x","done":false}', b'{"done":true}', b"[]"):
            with self.subTest(payload=payload), self.assertRaises(AdapterError):
                opener = Mock()
                opener.open.return_value = _FakeResponse(payload)
                object.__setattr__(adapter, "_opener", opener)
                adapter.generate("test")

    def test_rejects_transport_errors_and_redirects(self) -> None:
        adapter = OllamaAdapter("model")
        for exception in (
            error.URLError("offline"),
            error.HTTPError("http://127.0.0.1:11434/api/generate", 302, "redirect", {}, None),
            TimeoutError("slow"),
        ):
            with self.subTest(exception=type(exception).__name__), self.assertRaises(AdapterError):
                opener = Mock()
                opener.open.side_effect = exception
                object.__setattr__(adapter, "_opener", opener)
                adapter.generate("test")

    def test_proxy_and_redirect_policy(self) -> None:
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.invalid:3128", "http_proxy": "http://proxy.invalid:3128"}):
            adapter = OllamaAdapter("model")
        handlers = adapter._opener.handlers
        # The explicit empty ProxyHandler removes the default environment
        # handler; urllib does not add a handler with no proxy routes.
        self.assertFalse(any(isinstance(h, request.ProxyHandler) for h in handlers))
        self.assertTrue(any(h.__class__.__name__ == "_NoRedirect" for h in handlers))

    def test_invalid_config(self) -> None:
        for kwargs in (
            {"timeout_seconds": 0},
            {"timeout_seconds": float("nan")},
            {"timeout_seconds": 1000},
            {"max_response_bytes": 0},
            {"max_output_tokens": 0},
            {"max_output_tokens": 99999},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaAdapter("model", **kwargs)


if __name__ == "__main__":
    unittest.main()
