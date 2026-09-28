"""
Integration tests for the FeelFish pass-through route.

The route is exercised through a real FastAPI app (TestClient) with:

- ``user_api_key_auth`` and ``get_feelfish_deps`` overridden, so no proxy DB or
  global router is needed, and
- an ``httpx.MockTransport`` upstream that records exactly what was forwarded.
"""

import time
from typing import Any, Dict, Iterable, List, Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.feelfish_passthrough.route import (
    FeelfishDeps,
    get_feelfish_deps,
    router as feelfish_router,
)
from litellm.proxy.proxy_server import openai_exception_handler

UPSTREAM_API_BASE = "https://upstream.example.com/api/v3/tts/unidirectional"

CHARACTER_RULE = {
    "from": "response_body",
    "path": "usage.text_words",
    "aggregate": "last",
    "unit": "character",
    "unit_price": 0.0005,
    "required": True,
}


class _FakeKeyAuth:
    api_key = "sk-test"
    user_id = "user-1"


class _RecordingSpendRecorder:
    """Callable spend recorder (same contract as ``record_feelfish_spend``)."""

    def __init__(self) -> None:
        self.records: List[Any] = []

    async def __call__(self, record: Any) -> None:
        self.records.append(record)


class _FakeRouter:
    def __init__(
        self, deployment: Optional[Dict[str, Any]], merged_model_info: Optional[dict] = None
    ) -> None:
        self._deployment = deployment
        self._merged = merged_model_info or {}

    async def async_get_available_deployment_for_pass_through(
        self, model: str, request_kwargs: Optional[dict] = None
    ) -> Dict[str, Any]:
        if self._deployment is None:
            raise ValueError(f"no deployment for {model}")
        return self._deployment

    def get_deployment_model_info(self, model_id: str, model_name: str) -> dict:
        return self._merged


def _deployment(
    billing: Optional[dict] = None,
    passthrough: Optional[dict] = None,
    merged_billing: Optional[dict] = None,
    **litellm_params_overrides: Any,
) -> Dict[str, Any]:
    litellm_params: Dict[str, Any] = {
        "model": "echo-upstream",
        "custom_llm_provider": "feelfish",
        "api_base": UPSTREAM_API_BASE,
        "api_key": "upstream-secret",
        "auth_style": "header:X-Api-Key",
        "extra_headers": {"X-Api-Resource-Id": "seed-tts-2.0"},
        "use_in_pass_through": True,
    }
    if passthrough is not None:
        litellm_params["passthrough"] = passthrough
    litellm_params.update(litellm_params_overrides)

    model_info: Dict[str, Any] = {"id": "deployment-id-1", "mode": "audio_speech"}
    pricing = billing if billing is not None else {"rules": [CHARACTER_RULE]}
    model_info["response_field_pricing"] = pricing

    return {
        "model_name": "doubao-tts",
        "litellm_params": litellm_params,
        "model_info": model_info,
        "merged_model_info": merged_billing,
    }


def _build_app(deployment: Optional[Dict[str, Any]], upstream_handler) -> tuple:
    recorder = _RecordingSpendRecorder()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
    merged = (deployment or {}).get("merged_model_info") or {}
    deps = FeelfishDeps(
        llm_router=_FakeRouter(deployment, merged),
        spend_recorder=recorder,
        http_client=http_client,
    )
    app = FastAPI()
    # Mirror the production error contract for ProxyException.
    app.add_exception_handler(ProxyException, openai_exception_handler)
    app.include_router(feelfish_router)
    app.dependency_overrides[user_api_key_auth] = lambda: _FakeKeyAuth()
    app.dependency_overrides[get_feelfish_deps] = lambda: deps
    return TestClient(app, raise_server_exceptions=False), recorder, http_client


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


async def _async_chunks(chunks: Iterable[bytes]):
    for chunk in chunks:
        yield chunk


# --------------------------------------------------------------------------- #
# forwarding
# --------------------------------------------------------------------------- #


def test_json_request_is_forwarded_verbatim_and_billed():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"code": 20000000, "usage": {"text_words": 14}})

    client, recorder, http_client = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            json={"req_params": {"text": "hello"}, "model": "upstream-owned-model"},
            headers={"X-Client-Trace": "abc"},
        )

        assert response.status_code == 200
        assert response.json()["usage"]["text_words"] == 14
        # non-streaming responses expose the exact cost for the client
        assert response.headers["x-litellm-response-cost"] == "0.007"

        assert len(captured) == 1
        forwarded = captured[0]
        assert str(forwarded.url) == UPSTREAM_API_BASE
        assert forwarded.headers["X-Api-Key"] == "upstream-secret"
        assert forwarded.headers["X-Api-Resource-Id"] == "seed-tts-2.0"
        # the body (including its own `model` field) belongs to the upstream protocol
        assert forwarded.content == response.request.content.replace(
            b"", b""
        ) or b'"model":"upstream-owned-model"' in forwarded.content
        # client credentials / litellm identity never leak upstream
        assert "authorization" not in {k.lower() for k in forwarded.headers}
        assert not any(
            k.lower().startswith("x-litellm-") for k in forwarded.headers
        )
        assert "x-client-trace" not in {k.lower() for k in forwarded.headers}

        assert _wait_for(lambda: len(recorder.records) == 1)
        record = recorder.records[0]
        assert record.model == "doubao-tts"
        assert record.cost == pytest.approx(0.007)
        assert record.stream is False
        assert record.billing is not None
        assert record.billing.to_breakdown()["rules"][0]["value"] == 14
    finally:
        client.close()


def test_subpath_and_query_are_appended_to_api_base():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts/extra/path?lang=zh")
        assert response.status_code == 200
        assert str(captured[0].url) == f"{UPSTREAM_API_BASE}/extra/path?lang=zh"
    finally:
        client.close()


def test_forward_headers_allow_list():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    deployment = _deployment(passthrough={"forward_headers": ["X-Trace-Id"]})
    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        client.post(
            "/feelfish/doubao-tts",
            json={},
            headers={"X-Trace-Id": "t-1", "X-Other": "nope"},
        )
        assert captured[0].headers["X-Trace-Id"] == "t-1"
        assert "x-other" not in {k.lower() for k in captured[0].headers}
    finally:
        client.close()


def test_strip_body_fields_removes_configured_keys():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    deployment = _deployment(passthrough={"strip_body_fields": ["model"]})
    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        client.post("/feelfish/doubao-tts", json={"model": "upstream-x", "text": "a"})
        assert b"upstream-x" not in captured[0].content
        assert b'"text":"a"' in captured[0].content
    finally:
        client.close()


def test_multipart_request_is_replayed():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 2}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            files={"file": ("a.wav", b"RIFFxxxx", "audio/wav")},
            data={"language": "zh"},
        )
        assert response.status_code == 200
        body = captured[0].content
        assert b'name="language"' in body and b"zh" in body
        assert b'name="file"' in body and b"RIFFxxxx" in body
        assert captured[0].headers["content-type"].startswith("multipart/form-data")
    finally:
        client.close()


def test_raw_binary_request_is_forwarded():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 3}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            content=b"\x00\x01\x02binary",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200
        assert captured[0].content == b"\x00\x01\x02binary"
        assert captured[0].headers["content-type"] == "application/octet-stream"
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# billing
# --------------------------------------------------------------------------- #


def test_streaming_response_is_billed_after_the_stream_ends():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_async_chunks(
                [
                    b'{"code":0,"data":"aa"}\n',
                    b'{"code":0,"data":"bb"}\n',
                    b'{"code":20000000,"usage":{"text_words":14}}\n',
                ]
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={"req_params": {"text": "hi"}})
        assert response.status_code == 200
        assert '"code":20000000' in response.text
        # chunked responses cannot carry the cost header
        assert "x-litellm-response-cost" not in response.headers

        assert _wait_for(lambda: len(recorder.records) == 1)
        record = recorder.records[0]
        assert record.stream is True
        assert record.cost == pytest.approx(0.007)
        assert record.extra_breakdown["buffer_skipped"] is False
    finally:
        client.close()


def test_binary_stream_is_not_buffered_for_billing():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_async_chunks([b"\xff\xfbaudio", b"\xff\xfbmore"]),
            headers={"content-type": "audio/mpeg", "x-audio-seconds": "2"},
        )

    billing = {
        "rules": [
            {
                "from": "response_headers",
                "path": "x-audio-seconds",
                "unit_price": 0.01,
                "required": True,
            }
        ]
    }
    client, recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={"stream": True})
        assert response.status_code == 200
        assert _wait_for(lambda: len(recorder.records) == 1)
        record = recorder.records[0]
        assert record.cost == pytest.approx(0.02)
        assert record.extra_breakdown["buffer_skipped"] is True
    finally:
        client.close()


def test_missing_required_field_uses_fallback_rule():
    billing = {
        "rules": [CHARACTER_RULE],
        "fallback": [
            {
                "from": "request_body",
                "path": "text",
                "transform": "characters_no_space",
                "unit_price": 0.0005,
            }
        ],
        "on_missing": "fallback",
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client, recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={"text": "你好 世界"})
        assert response.status_code == 200
        assert response.headers["x-litellm-response-cost"] == "0.002"
        assert recorder.records[0].billing.fallback_used is True
        assert recorder.records[0].billing.missing_required == ["usage.text_words"]
    finally:
        client.close()


def test_upstream_error_is_passed_through_without_charge():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 429
        assert response.json() == {"error": "rate limited"}
        assert recorder.records == []
    finally:
        client.close()


def test_bill_on_error_charges_the_request_fee():
    billing = {
        "rules": [CHARACTER_RULE],
        "cost_per_request": 0.03,
        "bill_on_error": True,
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client, recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 500
        assert _wait_for(lambda: len(recorder.records) == 1)
        assert recorder.records[0].cost == pytest.approx(0.03)
    finally:
        client.close()


def test_breakdown_header_is_opt_in():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"text_words": 14}})

    billing = {"rules": [CHARACTER_RULE], "expose_breakdown": True}
    client, _recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.headers["x-litellm-response-cost"] == "0.007"
        assert "x-litellm-cost-breakdown" in response.headers
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_path_traversal_is_rejected_without_calling_upstream():
    calls: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts/../../evil")
        assert response.status_code in (400, 404)
        assert calls == []
        assert recorder.records == []
    finally:
        client.close()


def test_unknown_model_is_rejected():
    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    client, recorder, _http = _build_app(None, upstream)
    try:
        response = client.post("/feelfish/unknown-model", json={})
        assert response.status_code == 400
        assert "not available for pass-through" in response.json()["error"]["message"]
        assert recorder.records == []
    finally:
        client.close()


def test_model_without_pricing_config_is_rejected_fail_closed():
    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    deployment = _deployment()
    deployment["model_info"].pop("response_field_pricing")
    deployment["litellm_params"].pop("response_field_pricing", None)

    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "feelfish_config_error"
    finally:
        client.close()


def test_bearer_and_query_auth_styles():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(auth_style="bearer"), upstream)
    try:
        client.post("/feelfish/doubao-tts", json={})
        assert captured[0].headers["authorization"] == "Bearer upstream-secret"
    finally:
        client.close()

    client, _recorder, _http = _build_app(
        _deployment(auth_style="query:access_token"), upstream
    )
    try:
        client.post("/feelfish/doubao-tts", json={})
        assert "access_token=upstream-secret" in str(captured[1].url)
    finally:
        client.close()
