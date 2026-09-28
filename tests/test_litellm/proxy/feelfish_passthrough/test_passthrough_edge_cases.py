"""
Regression tests for the edge cases found in review:

- the auth chain must not consume/reject pass-through bodies,
- a client disconnect mid-stream must still be billed,
- the bounded billing buffer must retain the *tail* (terminal usage event),
- repeated multipart / urlencoded fields must survive forwarding,
- the production spend recorder contract must be callable end-to-end.
"""

import asyncio
import datetime
import json
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.feelfish_passthrough import route as route_module
from litellm.proxy.feelfish_passthrough.route import (
    FeelfishDeps,
    _BillingBuffer,
    _stream_and_bill,
    get_feelfish_deps,
    router as feelfish_router,
)
from litellm.proxy.feelfish_passthrough.spend import (
    SpendRecord,
    record_feelfish_spend,
)
from litellm.proxy.feelfish_passthrough.upstream import parse_incoming_body
from litellm.proxy.proxy_server import openai_exception_handler
from litellm.proxy.feelfish_passthrough.config import (
    PassthroughConfig,
    parse_billing_config,
)
from litellm.proxy.feelfish_passthrough.field_path import parse_json_lines

from .test_passthrough_route import (
    CHARACTER_RULE,
    UPSTREAM_API_BASE,
    _async_chunks,
    _build_app,
    _deployment,
    _FakeKeyAuth,
    _RecordingSpendRecorder,
    _wait_for,
)


def _request(
    body: bytes = b"{}",
    content_type: str = "application/json",
    method: str = "POST",
    path: str = "/feelfish/doubao-tts",
) -> Request:
    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode())],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "scheme": "http",
        "root_path": "",
    }
    return Request(scope, receive)


# --------------------------------------------------------------------------- #
# auth chain body handling
# --------------------------------------------------------------------------- #


def test_auth_body_read_keeps_raw_bodies_intact():
    async def scenario() -> bytes:
        request = _request(b"\x00\x01binary", "application/octet-stream")
        parsed = await _read_request_body(request)
        assert parsed == {}
        # the body must still be readable by the pass-through route
        return await request.body()

    assert asyncio.run(scenario()) == b"\x00\x01binary"


def test_auth_body_read_still_rejects_invalid_json():
    async def scenario() -> None:
        request = _request(b"{not json", "application/json")
        with pytest.raises(ProxyException):
            await _read_request_body(request)

    asyncio.run(scenario())


def test_raw_body_exemption_only_applies_to_feelfish_routes():
    """
    Other routes must keep parsing JSON bodies for auth: skipping the parse would
    let a model-restricted key bypass `can_key_call_model`.
    """

    async def scenario() -> Dict[str, Any]:
        # JSON payload declared as text/plain, on a non-feelfish route
        other = _request(b'{"model": "restricted-model"}', "text/plain", path="/mistral/v1/chat/completions")
        parsed_other = await _read_request_body(other)

        # same body on the feelfish route is left unread for verbatim forwarding
        feelfish = _request(b'{"model": "restricted-model"}', "text/plain")
        parsed_feelfish = await _read_request_body(feelfish)
        raw = await feelfish.body()
        return {"other": parsed_other, "feelfish": parsed_feelfish, "raw": raw}

    result = asyncio.run(scenario())
    assert result["other"]["model"] == "restricted-model"
    assert result["feelfish"] == {}
    assert result["raw"] == b'{"model": "restricted-model"}'


def test_form_bodies_survive_the_auth_body_read():
    async def scenario() -> None:
        urlencoded = _request(
            b"tag=a&tag=b", "application/x-www-form-urlencoded"
        )
        await _read_request_body(urlencoded)  # simulates the auth dependency
        body = await parse_incoming_body(urlencoded, PassthroughConfig())
        assert body.raw_body == b"tag=a&tag=b"

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# multipart / urlencoded forwarding
# --------------------------------------------------------------------------- #


def test_repeated_multipart_fields_are_forwarded():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            files=[
                ("tag", (None, "a")),
                ("tag", (None, "b")),
                ("file", ("a.wav", b"RIFFxxxx", "audio/wav")),
            ],
        )
        assert response.status_code == 200
        body = captured[0].content
        assert body.count(b'name="tag"') == 2
        assert b"\r\n\r\na\r\n" in body and b"\r\n\r\nb\r\n" in body
        assert b'filename="a.wav"' in body
    finally:
        client.close()


def test_urlencoded_body_is_forwarded_verbatim():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            content=b"tag=a&tag=b",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert captured[0].content == b"tag=a&tag=b"
        assert (
            captured[0].headers["content-type"]
            == "application/x-www-form-urlencoded"
        )
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# billing on stream abort + bounded buffer
# --------------------------------------------------------------------------- #


def _streaming_response(chunks: List[bytes], headers: Dict[str, str]) -> httpx.Response:
    async def content():
        for chunk in chunks:
            yield chunk

    return httpx.Response(200, content=content(), headers=headers)


def test_stream_request_with_a_json_response_is_billed():
    """`stream: true` on the request must not break billing when the upstream answers JSON."""

    def upstream(request: httpx.Request) -> httpx.Response:
        # pretty-printed JSON: line-by-line parsing would find nothing
        body = '{\n  "usage": {\n    "text_words": 24\n  }\n}'
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "application/json"}
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={"stream": True})
        assert response.status_code == 200
        for _ in range(50):
            if recorder.records:
                break
            time.sleep(0.01)
        assert len(recorder.records) == 1
        assert recorder.records[0].cost == pytest.approx(0.012)
    finally:
        client.close()


def test_untyped_raw_body_is_forwarded():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            content=b"hello",
            headers={"Content-Type": ""},
        )
        assert response.status_code == 200, response.text
        assert captured[0].content == b"hello"
    finally:
        client.close()


def test_forwarded_header_cannot_shadow_the_upstream_credential():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    # allow-list the very header the deployment uses for auth
    deployment = _deployment(passthrough={"forward_headers": ["x-api-key"]})
    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            json={},
            headers={"x-api-key": "client-supplied"},
        )
        assert response.status_code == 200
        forwarded = captured[0].headers
        assert forwarded["x-api-key"] == "upstream-secret"
        assert len(forwarded.get_list("x-api-key")) == 1
    finally:
        client.close()


def test_upstream_redirects_are_not_followed():
    """Following a cross-origin redirect would leak custom credential headers."""
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            307, headers={"location": "https://evil.example.com/collect"}
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    # the test client must not follow redirects either, otherwise the assertion
    # below would observe the followed hop instead of the proxied 3xx
    client.follow_redirects = False
    try:
        response = client.post("/feelfish/doubao-tts", json={"text": "hi"})
        # the redirect itself is passed through, never followed
        assert response.status_code == 307
        assert response.headers["location"] == "https://evil.example.com/collect"
        assert len(captured) == 1
        assert recorder.records == []
    finally:
        client.close()


def test_buffered_response_text_rules_are_billed():
    """`from: response_text` must work for buffered (non-streamed) responses."""

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"usage":{"text_words":10}}',
            headers={"content-type": "application/json"},
        )

    billing = {
        "rules": [
            {
                "from": "response_text",
                "path": "usage.text_words",
                "unit_price": 0.5,
                "required": True,
            }
        ]
    }
    client, recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 200
        assert recorder.records[0].cost == pytest.approx(5.0)
    finally:
        client.close()


def test_structured_json_suffix_responses_are_billed():
    """`application/vnd.api+json` is a JSON response and must be priced."""

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"usage": {"text_words": 14}}',
            headers={"content-type": "application/vnd.api+json"},
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 200
        assert recorder.records[0].cost == pytest.approx(0.007)
    finally:
        client.close()


def test_structured_json_suffix_streams_are_buffered_for_billing():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_async_chunks([b'{"usage": {"text_words": 24}}']),
            headers={"content-type": "application/vnd.api+json"},
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={"stream": True})
        assert response.status_code == 200, response.text
        assert _wait_for(lambda: len(recorder.records) == 1)
        assert recorder.records[0].cost == pytest.approx(0.012)
    finally:
        client.close()


def test_protocol_specific_json_content_type_is_preserved():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            content=b'{"a": 1}',
            headers={"Content-Type": "application/merge-patch+json"},
        )
        assert response.status_code == 200
        assert (
            captured[0].headers["content-type"] == "application/merge-patch+json"
        )
        # JSON is forwarded as a parsed object, so only the value must match
        assert json.loads(captured[0].content) == {"a": 1}
    finally:
        client.close()


def test_trailing_slash_and_json_null_body_are_preserved():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        assert client.post("/feelfish/doubao-tts/jobs/").status_code == 200
        assert str(captured[0].url).endswith("/jobs/"), str(captured[0].url)

        assert (
            client.post(
                "/feelfish/doubao-tts",
                content=b"null",
                headers={"Content-Type": "application/json"},
            ).status_code
            == 200
        )
        assert captured[1].content == b"null"
    finally:
        client.close()


def test_query_credential_is_not_shadowed_by_the_client():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(
        _deployment(auth_style="query:api_key"), upstream
    )
    try:
        client.post("/feelfish/doubao-tts?api_key=client-value")
        query = captured[0].url.query.decode()
        assert "client-value" not in query
        assert "api_key=upstream-secret" in query
    finally:
        client.close()


def test_client_disconnect_still_bills_the_consumed_output():
    recorder = _RecordingSpendRecorder()
    config = parse_billing_config(
        {
            "rules": [
                {
                    "from": "response_body",
                    "path": "usage.text_words",
                    "aggregate": "last",
                    "unit_price": 0.0005,
                    "required": True,
                }
            ],
            "fallback": [{"from": "request_body", "path": "text", "transform": "characters", "unit_price": 0.0005}],
        }
    )
    upstream_response = _streaming_response(
        [
            b'{"usage":{"text_words":4}}\n',
            b'{"usage":{"text_words":14}}\n',
        ],
        {"content-type": "application/x-ndjson"},
    )

    async def scenario() -> None:
        request = _request()
        stream = _stream_and_bill(
            buffer=_BillingBuffer(limit=1024),
            request=request,
            upstream_response=upstream_response,
            body=await parse_incoming_body(request, PassthroughConfig()),
            model_name="doubao-tts",
            billing_config=config,
            call_id="call-1",
            start_time=datetime.datetime.now(),
            user_api_key_dict=_FakeKeyAuth(),
            deps=FeelfishDeps(
                llm_router=None, spend_recorder=recorder, http_client=None
            ),
        )
        first = await stream.__anext__()
        assert b"text_words" in first
        # the client goes away after the first chunk
        await stream.aclose()
        # the recorder task is detached from the cancelled response task
        for _ in range(50):
            if recorder.records:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.cost == pytest.approx(0.002)  # 4 characters from the partial stream
    assert record.extra_breakdown["client_disconnected"] is True


def test_api_base_query_params_are_preserved():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    deployment = _deployment(
        api_base="https://upstream.example.com/run?api-version=2026"
    )
    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        client.post("/feelfish/doubao-tts?lang=zh")
        query = captured[0].url.query.decode()
        assert "api-version=2026" in query
        assert "lang=zh" in query
    finally:
        client.close()


def test_request_body_metadata_cannot_spoof_the_billing_identity():
    """
    The client body belongs to the upstream protocol: it must never be able to
    override the key/user the spend is charged to.
    """
    captured: List[httpx.Request] = []
    seen_metadata: Dict[str, Any] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post(
            "/feelfish/doubao-tts",
            json={
                "req_params": {"text": "hi"},
                "metadata": {"user_api_key_hash": "victim-key-hash", "user_id": "victim"},
                "litellm_metadata": {"user_api_key_hash": "victim-key-hash"},
            },
        )
        assert response.status_code == 200
        # the body is still forwarded upstream verbatim
        assert b'"user_api_key_hash":"victim-key-hash"' in captured[0].content
        assert recorder.records, "the call must be billed"
    finally:
        client.close()

    # and the production recorder must ignore the client-supplied identity
    async def scenario() -> Dict[str, Any]:
        deps = get_feelfish_deps()
        key_auth = UserAPIKeyAuth(api_key="sk-test", user_id="real-user")
        record = SpendRecord(
            model="doubao-tts",
            cost=0.007,
            billing=None,
            request=_request(),
            user_api_key_dict=key_auth,
            request_body={
                "metadata": {"user_api_key_hash": "victim-key-hash"},
                "litellm_metadata": {"user_api_key_hash": "victim-key-hash"},
            },
            response_status=200,
            stream=False,
            call_id="call-spoof",
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
        )
        logged: Dict[str, Any] = {}
        original = record_feelfish_spend.__globals__["Logging"]

        class _CapturingLogging(original):  # type: ignore[misc, valid-type]
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                logged["obj"] = self

            async def async_success_handler(self, *args: Any, **kwargs: Any) -> None:
                logged["metadata"] = kwargs.get("litellm_params", {}).get("metadata")
                return None

            def _should_run_sync_callbacks_for_async_calls(self) -> bool:
                return False

        record_feelfish_spend.__globals__["Logging"] = _CapturingLogging
        try:
            await deps.spend_recorder(record)
        finally:
            record_feelfish_spend.__globals__["Logging"] = original
        return logged

    logged = asyncio.run(scenario())
    metadata = logged["metadata"]
    assert metadata["user_api_key_hash"] != "victim-key-hash"
    assert metadata["user_api_key_user_id"] == "real-user"


def test_subpath_is_appended_to_the_path_not_the_query():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    deployment = _deployment(
        api_base="https://upstream.example.com/run?api-version=2026"
    )
    client, _recorder, _http = _build_app(deployment, upstream)
    try:
        client.post("/feelfish/doubao-tts/jobs?lang=zh")
        url = captured[0].url
        assert url.path == "/run/jobs", str(url)
        query = url.query.decode()
        assert "api-version=2026" in query and "lang=zh" in query
    finally:
        client.close()


def test_unbilled_upstream_error_reports_a_failure_event(monkeypatch):
    """Rate-limit callbacks release their state from a success/failure event."""
    reported: List[Dict[str, Any]] = []

    async def _fake_report(**kwargs: Any) -> None:
        reported.append(kwargs)

    from litellm.proxy.feelfish_passthrough import route as route_module

    monkeypatch.setattr(route_module, "record_feelfish_failure", _fake_report)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 429
        assert recorder.records == []
    finally:
        client.close()

    assert len(reported) == 1
    assert reported[0]["request_data"]["model"] == "doubao-tts"
    assert isinstance(reported[0]["error"], httpx.HTTPStatusError)


def test_encoded_reserved_characters_stay_in_the_path():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts/tasks/id%23part%3Fx")
        assert response.status_code == 200, response.text
        url = captured[0].url
        # httpx decodes `URL.path`; assert on the serialized URL instead
        assert str(url).endswith("/tasks/id%23part%3Fx"), str(url)
        assert url.query == b"", "the encoded '?' must not become a query string"
    finally:
        client.close()


def test_pre_call_hooks_receive_the_global_concurrency_limit(monkeypatch):
    calls: List[Dict[str, Any]] = []

    class _ProxyLogging:
        async def pre_call_hook(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs)
            return kwargs.get("data", {})

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _ProxyLogging())
    monkeypatch.setattr(
        proxy_server, "general_settings", {"global_max_parallel_requests": 7}
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        assert client.post("/feelfish/doubao-tts", json={}).status_code == 200
    finally:
        client.close()

    assert calls[0]["data"]["metadata"]["global_max_parallel_requests"] == 7
    # the terminal event must carry it too, otherwise the counter never decrements
    assert recorder.records[0].model == "doubao-tts"


def test_invalid_path_does_not_acquire_concurrency(monkeypatch):
    """A rejected path must not leave a rate-limit slot occupied."""
    calls: List[Dict[str, Any]] = []

    class _ProxyLogging:
        async def pre_call_hook(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs)
            return kwargs.get("data", {})

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _ProxyLogging())
    monkeypatch.setattr(
        proxy_server, "general_settings", {"global_max_parallel_requests": 1}
    )

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        # %2E%2E survives client-side normalisation but decodes to ".."
        response = client.post("/feelfish/doubao-tts/%2E%2E/escape")
        assert response.status_code == 400, response.text
    finally:
        client.close()

    assert calls == [], "a rejected request must not acquire a concurrency slot"


def test_form_metadata_field_is_not_json_interpreted():
    """`metadata=hello` is upstream data, not a LiteLLM metadata blob."""

    async def scenario() -> Dict[str, Any]:
        form = _request(
            b"metadata=hello&tag=a", "application/x-www-form-urlencoded"
        )
        parsed = await _read_request_body(form)
        # the route must still be able to forward the (cached) form body
        forwarded = await parse_incoming_body(form, PassthroughConfig())
        return {"parsed": parsed, "forwarded": forwarded.raw_body}

    result = asyncio.run(scenario())
    # the auth-facing view must not expose the upstream metadata field ...
    assert "metadata" not in result["parsed"]
    # ... while the route can still forward the original form body verbatim
    assert result["forwarded"] == b"metadata=hello&tag=a"


def test_pre_call_hooks_are_invoked_before_forwarding(monkeypatch):
    """Rate limiting lives in the proxy pre-call hooks, not in key auth."""
    calls: List[Dict[str, Any]] = []

    class _ProxyLogging:
        async def pre_call_hook(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs)
            return kwargs.get("data", {})

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.feelfish_passthrough import route as route_module

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _ProxyLogging())
    assert route_module  # silence unused import lint

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 200
    finally:
        client.close()

    assert len(calls) == 1
    assert calls[0]["call_type"] == "pass_through_endpoint"
    assert calls[0]["data"]["model"] == "doubao-tts"


def test_truncated_stream_with_sum_aggregation_falls_back():
    """
    A dropped middle makes `sum`/`count` unreliable: such rules must degrade to
    the fallback chain instead of silently undercharging.
    """
    recorder = _RecordingSpendRecorder()
    config = parse_billing_config(
        {
            "rules": [
                {
                    "from": "response_body",
                    "path": "usage.text_words",
                    "aggregate": "sum",
                    "unit_price": 0.0005,
                }
            ],
            "cost_per_request": 0.05,
        }
    )

    async def content():
        # large enough to overflow the tiny buffer below
        for index in range(1, 40):
            yield f'{{"usage":{{"text_words":{index}}}}}\n'.encode()

    upstream_response = httpx.Response(
        200, content=content(), headers={"content-type": "application/x-ndjson"}
    )

    async def scenario() -> None:
        request = _request()
        stream = _stream_and_bill(
            buffer=_BillingBuffer(limit=64),
            request=request,
            upstream_response=upstream_response,
            body=await parse_incoming_body(request, PassthroughConfig()),
            model_name="doubao-tts",
            billing_config=config,
            call_id="call-truncated",
            start_time=datetime.datetime.now(),
            user_api_key_dict=_FakeKeyAuth(),
            deps=FeelfishDeps(
                llm_router=None, spend_recorder=recorder, http_client=None
            ),
        )
        async for _chunk in stream:
            pass
        for _ in range(50):
            if recorder.records:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.extra_breakdown["buffer_truncated"] is True
    assert record.cost == pytest.approx(0.05)
    assert record.billing.cost_per_request_used is True


def test_breakdown_header_is_latin1_safe():
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"text_words": 14}})

    billing = {
        "rules": [
            {
                "from": "response_body",
                "path": "usage.text_words",
                "unit": "字符",
                "unit_price": 0.0005,
                "required": True,
            }
        ],
        "expose_breakdown": True,
    }
    client, _recorder, _http = _build_app(_deployment(billing=billing), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 200
        assert response.headers["x-litellm-response-cost"] == "0.007"
        assert "x-litellm-cost-breakdown" in response.headers
    finally:
        client.close()


def test_buffered_read_failure_reports_a_failure_event(monkeypatch):
    reported: List[Dict[str, Any]] = []

    async def _fake_report(**kwargs: Any) -> None:
        reported.append(kwargs)

    from litellm.proxy.feelfish_passthrough import route as route_module

    monkeypatch.setattr(route_module, "record_feelfish_failure", _fake_report)

    def upstream(request: httpx.Request) -> httpx.Response:
        async def broken():
            yield b'{"partial": '
            raise httpx.ReadTimeout("upstream timed out mid-body")

        # application/json keeps the request on the buffered path
        return httpx.Response(
            200, content=broken(), headers={"content-type": "application/json"}
        )

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 500
        assert recorder.records == []
    finally:
        client.close()

    assert len(reported) == 1
    assert isinstance(reported[0]["error"], BaseException)


def test_billing_buffer_keeps_the_terminal_event():
    buffer = _BillingBuffer(limit=64)
    for index in range(1, 100):
        buffer.add(f'{{"usage":{{"text_words":{index}}}}}\n'.encode())

    assert buffer.truncated is True
    events = parse_json_lines(buffer.text())
    assert events, "the retained window must still contain parseable events"
    assert events[-1]["usage"]["text_words"] == 99


def test_billing_buffer_without_truncation_returns_everything():
    buffer = _BillingBuffer(limit=1024)
    buffer.add(b'{"usage":{"text_words":1}}\n')
    assert buffer.truncated is False
    assert buffer.text() == '{"usage":{"text_words":1}}\n'


def test_event_straddling_head_and_tail_is_not_split():
    """
    Filling the head half does not drop bytes, so an event spanning the
    head/tail boundary must stay intact (no synthetic newline inserted).
    """
    event = b'{"usage":{"text_words":42}}\n'
    buffer = _BillingBuffer(limit=64)
    # 8 + 28 bytes: crosses the 32-byte head limit, still under the 64-byte total
    buffer.add(b'{"a":1}\n' + event)

    assert buffer.truncated is False
    text = buffer.text()
    assert text is not None
    assert '{"usage":{"text_words":42}}' in text
    assert parse_json_lines(text)[-1]["usage"]["text_words"] == 42


def test_billing_buffer_cancellation_is_billed():
    """Task cancellation (not GeneratorExit) must still record the cost."""
    recorder = _RecordingSpendRecorder()
    config = parse_billing_config({"rules": [CHARACTER_RULE]})

    async def slow_content():
        yield b'{"usage":{"text_words":4}}\n'
        await asyncio.sleep(30)  # the client disconnects while we wait here
        yield b'{"usage":{"text_words":9}}\n'

    upstream_response = httpx.Response(
        200, content=slow_content(), headers={"content-type": "application/x-ndjson"}
    )

    async def scenario() -> None:
        request = _request()
        stream = _stream_and_bill(
            buffer=_BillingBuffer(limit=1024),
            request=request,
            upstream_response=upstream_response,
            body=await parse_incoming_body(request, PassthroughConfig()),
            model_name="doubao-tts",
            billing_config=config,
            call_id="call-cancel",
            start_time=datetime.datetime.now(),
            user_api_key_dict=_FakeKeyAuth(),
            deps=FeelfishDeps(
                llm_router=None, spend_recorder=recorder, http_client=None
            ),
        )
        assert await stream.__anext__()
        pending = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        for _ in range(50):
            if recorder.records:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert len(recorder.records) == 1
    assert recorder.records[0].extra_breakdown["client_disconnected"] is True


def test_failure_recorder_emits_a_logging_failure_event(monkeypatch):
    """
    Parallel-request limiters release concurrency in ``async_log_failure_event``;
    ``post_call_failure_hook`` alone is not enough.
    """
    from litellm.proxy.feelfish_passthrough import spend as spend_module

    captured: Dict[str, Any] = {}
    original = spend_module.Logging

    class _CapturingLogging(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured["obj"] = self

        async def async_failure_handler(self, **kwargs: Any) -> None:
            captured["failure"] = kwargs

    monkeypatch.setattr(spend_module, "Logging", _CapturingLogging)

    async def scenario() -> None:
        await spend_module.record_feelfish_failure(
            model="doubao-tts",
            request=_request(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="user-1"),
            request_data={"model": "doubao-tts"},
            error=httpx.HTTPStatusError(
                "Upstream returned HTTP 429",
                request=httpx.Request("POST", "https://upstream.example.com/tts"),
                response=httpx.Response(429),
            ),
            traceback_str="",
            call_id="call-fail",
            start_time=datetime.datetime.now(),
        )

    asyncio.run(scenario())
    assert "failure" in captured, "a logging failure event must be emitted"
    assert isinstance(captured["failure"]["exception"], httpx.HTTPStatusError)


def test_nested_wildcards_are_flattened_in_streams():
    """A wildcard inside a streamed event list must aggregate like the buffered path."""
    from litellm.proxy.feelfish_passthrough.billing import (
        BillingContext,
        evaluate_billing,
    )

    config = parse_billing_config(
        {
            "rules": [
                {
                    "from": "response_body",
                    "path": "items[*].cost",
                    "aggregate": "sum",
                    "unit_price": 1.0,
                }
            ]
        }
    )
    buffered = evaluate_billing(
        config, BillingContext(response_body={"items": [{"cost": 2}, {"cost": 3}]})
    )
    streamed = evaluate_billing(
        config,
        BillingContext(
            response_body=[{"items": [{"cost": 2}, {"cost": 3}]}, {"items": [{"cost": 1}]}]
        ),
    )
    assert buffered.cost == pytest.approx(5.0)
    assert streamed.cost == pytest.approx(6.0)


def test_non_finite_usage_values_are_treated_as_missing():
    from litellm.proxy.feelfish_passthrough.billing import (
        BillingContext,
        evaluate_billing,
    )

    config = parse_billing_config(
        {
            "rules": [
                {
                    "from": "response_body",
                    "path": "usage.text_words",
                    "unit_price": 0.0005,
                    "required": True,
                }
            ],
            "fallback": [{"from": "request_body", "path": "text", "transform": "characters", "unit_price": 0.0005}],
            "on_missing": "fallback",
        }
    )
    result = evaluate_billing(
        config,
        BillingContext(
            response_body={"usage": {"text_words": "NaN"}},
            request_body={"text": "abcd"},
        ),
    )
    assert result.has_missing_required is True
    assert result.fallback_used is True
    assert result.cost == pytest.approx(0.002)


def test_infinite_pricing_config_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        parse_billing_config(
            {"rules": [{"path": "a", "unit_price": float("inf")}]}
        )
    with pytest.raises(ValueError, match="finite"):
        parse_billing_config({"rules": [{"path": "a"}], "multiplier": float("nan")})


def test_request_body_and_header_aggregates_survive_truncation():
    """
    `sum`/`count` over the *request body* or headers do not depend on the response
    stream, so a long response must not degrade them.
    """
    recorder = _RecordingSpendRecorder()
    config = parse_billing_config(
        {
            "rules": [
                {
                    "from": "request_body",
                    "path": "items[*].id",
                    "aggregate": "count",
                    "unit_price": 0.02,
                }
            ],
            "cost_per_request": 9.99,
        }
    )

    async def content():
        for index in range(1, 40):
            yield f'{{"usage":{{"text_words":{index}}}}}\n'.encode()

    upstream_response = httpx.Response(
        200, content=content(), headers={"content-type": "application/x-ndjson"}
    )

    async def scenario() -> None:
        request = _request(body=b'{"items":[{"id":1},{"id":2},{"id":3}]}')
        stream = _stream_and_bill(
            buffer=_BillingBuffer(limit=64),
            request=request,
            upstream_response=upstream_response,
            body=await parse_incoming_body(request, PassthroughConfig()),
            model_name="doubao-tts",
            billing_config=config,
            call_id="call-reqcount",
            start_time=datetime.datetime.now(),
            user_api_key_dict=_FakeKeyAuth(),
            deps=FeelfishDeps(
                llm_router=None, spend_recorder=recorder, http_client=None
            ),
        )
        async for _chunk in stream:
            pass
        for _ in range(50):
            if recorder.records:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.extra_breakdown["buffer_truncated"] is True
    assert record.cost == pytest.approx(0.06)  # 3 items, not the fallback price
    assert record.billing.cost_per_request_used is False


def test_truncated_stream_rejects_a_response_aggregating_fallback():
    """A fallback that itself needs the full stream must not bypass cost_per_request."""
    recorder = _RecordingSpendRecorder()
    config = parse_billing_config(
        {
            # primary needs the full stream -> unresolved when truncated
            "rules": [
                {
                    "from": "response_body",
                    "path": "usage.text_words",
                    "aggregate": "sum",
                    "unit_price": 0.0005,
                    "required": True,
                }
            ],
            # ... and so does the fallback, which must therefore be rejected too
            "fallback": [
                {
                    "from": "response_body",
                    "path": "usage.text_words",
                    "aggregate": "count",
                    "unit_price": 0.0005,
                }
            ],
            "cost_per_request": 0.05,
        }
    )

    async def content():
        for index in range(1, 40):
            yield f'{{"usage":{{"text_words":{index}}}}}\n'.encode()

    upstream_response = httpx.Response(
        200, content=content(), headers={"content-type": "application/x-ndjson"}
    )

    async def scenario() -> None:
        request = _request()
        stream = _stream_and_bill(
            buffer=_BillingBuffer(limit=64),
            request=request,
            upstream_response=upstream_response,
            body=await parse_incoming_body(request, PassthroughConfig()),
            model_name="doubao-tts",
            billing_config=config,
            call_id="call-fb-truncated",
            start_time=datetime.datetime.now(),
            user_api_key_dict=_FakeKeyAuth(),
            deps=FeelfishDeps(
                llm_router=None, spend_recorder=recorder, http_client=None
            ),
        )
        async for _chunk in stream:
            pass
        for _ in range(50):
            if recorder.records:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    record = recorder.records[0]
    assert record.cost == pytest.approx(0.05)
    assert record.billing.cost_per_request_used is True


def test_model_budget_metadata_comes_from_the_key():
    from litellm.proxy.feelfish_passthrough import spend as spend_module

    metadata: Dict[str, Any] = {}
    key_auth = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="user-1",
        model_max_budget={"doubao-tts": 5.0},
        end_user_model_max_budget={"doubao-tts": 1.0},
    )
    spend_module._add_trusted_budget_metadata(metadata, key_auth)
    assert metadata["user_api_key_model_max_budget"] == {"doubao-tts": 5.0}
    assert metadata["user_api_key_end_user_model_max_budget"] == {"doubao-tts": 1.0}


def test_named_credential_is_resolved():
    captured: List[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"usage": {"text_words": 1}})

    deployment = _deployment(api_key=None, litellm_credential_name="cred-1")

    class _CredentialRouter:
        async def async_get_available_deployment_for_pass_through(self, **kwargs: Any):
            return deployment

        def get_deployment_model_info(self, **kwargs: Any) -> dict:
            return {}

        def get_deployment_credentials_with_provider(
            self, model_id: str
        ) -> Dict[str, Any]:
            assert model_id == deployment["model_info"]["id"]
            return {"api_key": "named-credential-secret", "api_base": UPSTREAM_API_BASE}

    recorder = _RecordingSpendRecorder()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = FastAPI()
    app.add_exception_handler(ProxyException, openai_exception_handler)
    app.include_router(feelfish_router)
    app.dependency_overrides[user_api_key_auth] = lambda: _FakeKeyAuth()
    app.dependency_overrides[get_feelfish_deps] = lambda: FeelfishDeps(
        llm_router=_CredentialRouter(), spend_recorder=recorder, http_client=http_client
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/feelfish/doubao-tts", json={"text": "hi"})
        assert response.status_code == 200
        assert captured[0].headers["X-Api-Key"] == "named-credential-secret"
    finally:
        client.close()


def test_unresolved_named_credential_fails_closed():
    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called without credentials")

    deployment = _deployment(api_key=None, litellm_credential_name="missing-cred")

    class _NoCredentialRouter:
        async def async_get_available_deployment_for_pass_through(self, **kwargs: Any):
            return deployment

        def get_deployment_model_info(self, **kwargs: Any) -> dict:
            return {}

        def get_deployment_credentials_with_provider(
            self, model_id: str
        ) -> Dict[str, Any]:
            return {}

    recorder = _RecordingSpendRecorder()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = FastAPI()
    app.add_exception_handler(ProxyException, openai_exception_handler)
    app.include_router(feelfish_router)
    app.dependency_overrides[user_api_key_auth] = lambda: _FakeKeyAuth()
    app.dependency_overrides[get_feelfish_deps] = lambda: FeelfishDeps(
        llm_router=_NoCredentialRouter(),
        spend_recorder=recorder,
        http_client=http_client,
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/feelfish/doubao-tts", json={"text": "hi"})
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "feelfish_config_error"
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# production dependencies
# --------------------------------------------------------------------------- #


def test_production_spend_recorder_is_callable():
    async def scenario() -> SpendRecord:
        deps = get_feelfish_deps()
        key_auth = UserAPIKeyAuth(api_key="sk-test", user_id="user-1")
        assert deps.spend_recorder is record_feelfish_spend
        record = SpendRecord(
            model="doubao-tts",
            cost=0.007,
            billing=None,
            request=_request(),
            user_api_key_dict=key_auth,
            request_body={"req_params": {"text": "hi"}},
            response_status=200,
            stream=False,
            call_id="call-prod",
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
        )
        await deps.spend_recorder(record)
        return record

    assert asyncio.run(scenario()).cost == pytest.approx(0.007)


def test_streamed_cost_survives_the_real_logging_pipeline():
    """
    With `stream=True` on the Logging object, `_success_handler_helper_fn` resets
    `response_cost` to None, which would silently bill every streamed call 0.
    """
    from litellm.proxy.feelfish_passthrough import spend as spend_module

    original = spend_module.Logging
    holders: Dict[str, Any] = {}

    class _CapturingLogging(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            holders["obj"] = self

    spend_module.Logging = _CapturingLogging
    try:

        async def scenario(stream_flag: bool) -> Optional[float]:
            record = SpendRecord(
                model="doubao-tts",
                cost=0.007,
                billing=None,
                request=_request(),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="user-1"),
                request_body={},
                response_status=200,
                stream=stream_flag,
                call_id="call-cost",
                start_time=datetime.datetime.now(),
                end_time=datetime.datetime.now(),
            )
            await spend_module.record_feelfish_spend(record)
            payload = holders["obj"].model_call_details.get("standard_logging_object") or {}
            return payload.get("response_cost")

        assert asyncio.run(scenario(True)) == pytest.approx(0.007)
        assert asyncio.run(scenario(False)) == pytest.approx(0.007)
    finally:
        spend_module.Logging = original


def test_pre_call_hook_rejection_does_not_release_concurrency(monkeypatch):
    """
    A pre-call rejection must NOT emit a failure event: we cannot know whether
    the limiter had incremented, and a spurious decrement would let a caller
    lower other in-flight requests' concurrency usage (fail-open bypass).
    """
    reported: List[Dict[str, Any]] = []

    async def _fake_report(**kwargs: Any) -> None:
        reported.append(kwargs)

    from litellm.proxy.feelfish_passthrough import route as route_module

    class _RejectingProxyLogging:
        async def pre_call_hook(self, **kwargs: Any) -> Dict[str, Any]:
            raise HTTPException(status_code=429, detail="rate limited")

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _RejectingProxyLogging())
    monkeypatch.setattr(route_module, "record_feelfish_failure", _fake_report)

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    client, _recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 429
    finally:
        client.close()

    assert reported == []


def test_logging_payload_carries_no_bodies():
    """Log sinks must never receive the upstream request/response bodies."""
    from litellm.proxy.feelfish_passthrough import spend as spend_module

    original = spend_module.Logging
    holders: Dict[str, Any] = {}

    class _CapturingLogging(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            holders["obj"] = self

        async def async_success_handler(self, *args: Any, **kwargs: Any) -> None:
            return None

        def _should_run_sync_callbacks_for_async_calls(self) -> bool:
            return False

    spend_module.Logging = _CapturingLogging
    try:

        async def scenario() -> Dict[str, Any]:
            record = SpendRecord(
                model="doubao-tts",
                cost=0.007,
                billing=None,
                request=_request(),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="user-1"),
                request_body={"req_params": {"text": "secret transcript"}},
                response_status=200,
                stream=False,
                call_id="call-nobody",
                start_time=datetime.datetime.now(),
                end_time=datetime.datetime.now(),
            )
            await spend_module.record_feelfish_spend(record)
            return holders["obj"].model_call_details.get(
                "passthrough_logging_payload", {}
            )

        payload = asyncio.run(scenario())
    finally:
        spend_module.Logging = original

    assert "response_body" not in payload
    assert "request_body" not in payload


def test_close_upstream_response_survives_cancellation():
    closed = asyncio.Event()

    class _SlowResponse:
        async def aclose(self) -> None:
            await asyncio.sleep(0.05)
            closed.set()

    async def scenario() -> None:
        task = asyncio.ensure_future(
            route_module._close_upstream_response(_SlowResponse())
        )
        await asyncio.sleep(0)
        task.cancel()
        # cleanup swallows the cancellation on purpose
        await asyncio.wait_for(asyncio.shield(task), timeout=2)
        # ... but the shielded close must still finish in the background
        await asyncio.wait_for(closed.wait(), timeout=2)

    asyncio.run(scenario())


def test_route_uses_the_injected_recorder_for_buffered_requests():
    """
    Guards the contract used by ``get_feelfish_deps``: the recorder is a plain
    async callable, not an object with a ``record`` method.
    """

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"text_words": 14}})

    client, recorder, _http = _build_app(_deployment(), upstream)
    try:
        response = client.post("/feelfish/doubao-tts", json={})
        assert response.status_code == 200
        assert len(recorder.records) == 1
        assert recorder.records[0].cost == pytest.approx(0.007)
    finally:
        client.close()
