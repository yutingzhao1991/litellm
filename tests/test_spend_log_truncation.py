import pytest

from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _truncate_spend_log_request_body,
)


@pytest.fixture
def settings(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {}, raising=False)
    return proxy_server


def _body():
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "r"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "x" * 300,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "description": "y" * 200}
                        },
                    },
                },
            }
        ],
    }


def test_disabled_by_default(settings):
    body = _body()
    out = _truncate_spend_log_request_body(body, metadata={"tags": ["agent"]})
    assert out == body


def test_truncates_from_last_user_and_compresses_tool_descriptions(settings):
    settings.general_settings = {
        "spend_log_truncation": {
            "enabled": True,
            "compress_tool_description_max_chars": 20,
        }
    }
    out = _truncate_spend_log_request_body(_body(), metadata={"tags": ["agent"]})
    assert [m["role"] for m in out["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert out["messages"][0]["content"] == "new"
    assert out["tools"][0]["function"]["description"] == "x" * 20 + "…"
    assert (
        out["tools"][0]["function"]["parameters"]["properties"]["a"]["description"]
        == "y" * 20 + "…"
    )


def test_summary_tag_skips_truncation(settings):
    settings.general_settings = {
        "spend_log_truncation": {
            "enabled": True,
            "compress_tool_description_max_chars": 20,
        }
    }
    body = _body()
    out = _truncate_spend_log_request_body(body, metadata={"tags": ["phase:summary"]})
    assert out == body
