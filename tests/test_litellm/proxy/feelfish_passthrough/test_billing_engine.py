"""
Unit tests for the FeelFish response-field billing engine.

Pure functions only: no proxy, no httpx, no database.
"""

import pytest

from litellm.proxy.feelfish_passthrough.billing import (
    BillingContext,
    evaluate_billing,
)
from litellm.proxy.feelfish_passthrough.config import parse_billing_config
from litellm.proxy.feelfish_passthrough.field_path import (
    lookup,
    lookup_header,
    parse_json_lines,
)


def _config(**overrides):
    payload = {
        "rules": [
            {
                "from": "response_body",
                "path": "usage.text_words",
                "aggregate": "last",
                "unit": "character",
                "unit_price": 0.0005,
                "required": True,
            }
        ]
    }
    payload.update(overrides)
    return parse_billing_config(payload)


# --------------------------------------------------------------------------- #
# field path dialect
# --------------------------------------------------------------------------- #


def test_lookup_supports_nested_keys_indexes_wildcards_and_lists():
    data = {
        "usage": {"text_words": 14},
        "items": [{"cost": 1}, {"cost": 2}],
        "nested": [{"a": {"b": 5}}],
    }
    assert lookup(data, "usage.text_words") == 14
    assert lookup(data, "items[1].cost") == 2
    assert lookup(data, "items[*].cost") == [1, 2]
    assert lookup(data, "nested[*].a.b") == [5]
    # implicit wildcard over a list of objects
    assert lookup(data, "items.cost") == [1, 2]
    assert lookup(data, "missing.path") is None
    assert lookup(data, "items[9].cost") is None


def test_lookup_header_is_case_insensitive():
    headers = {"X-Duration-Ms": "1500"}
    assert lookup_header(headers, "x-duration-ms") == "1500"
    assert lookup_header(headers, "missing") is None


def test_parse_json_lines_handles_ndjson_and_sse():
    text = '{"code":0,"data":"aa"}\n\n: keep-alive\ndata: {"code":20000000,"usage":{"text_words":7}}\ndata: [DONE]\n'
    events = parse_json_lines(text)
    assert [event["code"] for event in events] == [0, 20000000]
    assert parse_json_lines(None) == []
    assert parse_json_lines("not json") == []


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


def test_single_field_pricing_uses_last_event():
    config = _config(
        rules=[
            {
                "from": "response_text",
                "path": "usage.text_words",
                "aggregate": "last",
                "unit": "character",
                "unit_price": 0.0005,
                "required": True,
            }
        ]
    )
    result = evaluate_billing(
        config,
        BillingContext(
            response_text='{"usage":{"text_words":3}}\n{"usage":{"text_words":14}}'
        ),
    )
    assert result.cost == pytest.approx(0.007)
    assert result.rules[0].value == 14
    assert result.fallback_used is False
    assert result.has_missing_required is False


def test_multiple_fields_are_summed_with_scale_and_multiplier():
    config = _config(
        rules=[
            {"from": "response_body", "path": "usage.text_words", "unit_price": 0.0005},
            {"from": "response_body", "path": "duration", "unit_price": 0.01},
            {
                "from": "response_body",
                "path": "words",
                "unit_price": 0.002,
                "scale": 0.001,
            },
        ],
        multiplier=2.0,
    )
    result = evaluate_billing(
        config,
        BillingContext(response_body={"usage": {"text_words": 10}, "duration": 3, "words": 2500}),
    )
    # (10*0.0005 + 3*0.01 + 2500*0.002*0.001) * 2 = (0.005 + 0.03 + 0.005) * 2
    assert result.cost == pytest.approx(0.08)


@pytest.mark.parametrize(
    "aggregate,expected",
    [
        ("first", 1.0),
        ("last", 6.0),
        ("sum", 13.0),
        ("min", 1.0),
        ("max", 6.0),
        ("count", 4.0),
    ],
)
def test_aggregates(aggregate, expected):
    config = _config(
        rules=[
            {
                "from": "response_body",
                "path": "items[*].v",
                "aggregate": aggregate,
                "unit_price": 1.0,
            }
        ]
    )
    result = evaluate_billing(
        config, BillingContext(response_body={"items": [{"v": 1}, {"v": 2}, {"v": 4}, {"v": 6}]})
    )
    assert result.cost == pytest.approx(expected)
    assert result.rules[0].value == pytest.approx(expected)


def test_transforms():
    cases = [
        ("characters", "hello world", 11.0),
        ("characters_no_space", "hello world", 10.0),
        ("bytes", "héllo", 6.0),
        # count_items counts containers; arrays are counted with aggregate: count
        ("count_items", {"a": 1, "b": 2}, 2.0),
        ("truthy", True, 1.0),
    ]
    for transform, value, expected in cases:
        config = _config(
            rules=[
                {
                    "from": "response_body",
                    "path": "value",
                    "transform": transform,
                    "unit_price": 1.0,
                }
            ]
        )
        result = evaluate_billing(config, BillingContext(response_body={"value": value}))
        assert result.cost == pytest.approx(expected), transform


def test_array_length_is_counted_with_count_aggregate():
    config = _config(
        rules=[
            {
                "from": "response_body",
                "path": "data",
                "aggregate": "count",
                "unit_price": 0.02,
            }
        ]
    )
    result = evaluate_billing(
        config, BillingContext(response_body={"data": [{"url": "a"}, {"url": "b"}]})
    )
    assert result.cost == pytest.approx(0.04)


def test_ms_to_seconds_transform():
    config = _config(
        rules=[
            {
                "from": "response_headers",
                "path": "x-duration-ms",
                "transform": "ms_to_seconds",
                "unit_price": 0.01,
            }
        ]
    )
    result = evaluate_billing(
        config, BillingContext(response_headers={"X-Duration-Ms": "1500"})
    )
    assert result.cost == pytest.approx(0.015)


def test_empty_collection_counts_as_zero_not_missing():
    config = _config(
        rules=[
            {
                "from": "response_body",
                "path": "items",
                "aggregate": "count",
                "unit_price": 0.02,
                "required": True,
            }
        ],
        cost_per_request=5.0,
    )
    result = evaluate_billing(config, BillingContext(response_body={"items": []}))
    assert result.has_missing_required is False
    assert result.cost == pytest.approx(0.0)
    assert result.cost_per_request_used is False


def test_stream_count_without_the_field_is_missing_not_zero():
    config = _config(
        rules=[
            {
                "from": "response_text",
                "path": "usage.text_words",
                "aggregate": "count",
                "unit_price": 0.02,
                "required": True,
            }
        ],
        cost_per_request=5.0,
    )
    # events exist, but none carries the field -> missing, not "count 0"
    result = evaluate_billing(
        config, BillingContext(response_text='{"other":1}\n{"other":2}')
    )
    assert result.has_missing_required is True
    assert result.cost == pytest.approx(5.0)
    assert result.cost_per_request_used is True


def test_empty_wildcard_array_counts_as_zero():
    from litellm.proxy.feelfish_passthrough.field_path import lookup

    assert lookup({"items": []}, "items[*]") == []
    config = _config(
        rules=[
            {
                "from": "response_body",
                "path": "items[*]",
                "aggregate": "count",
                "unit_price": 0.02,
                "required": True,
            }
        ],
        cost_per_request=10.0,
    )
    result = evaluate_billing(config, BillingContext(response_body={"items": []}))
    assert result.has_missing_required is False
    assert result.cost == pytest.approx(0.0)
    assert result.cost_per_request_used is False


def test_missing_required_uses_fallback_rules():
    config = _config(
        fallback=[
            {
                "from": "request_body",
                "path": "text",
                "transform": "characters_no_space",
                "unit_price": 0.0005,
            }
        ]
    )
    result = evaluate_billing(config, BillingContext(request_body={"text": "你好 世界"}))
    assert result.cost == pytest.approx(0.002)
    assert result.fallback_used is True
    assert result.missing_required == ["usage.text_words"]


def test_fallback_with_missing_required_falls_through_to_cost_per_request():
    config = _config(
        fallback=[
            {
                "from": "request_body",
                "path": "text",
                "transform": "characters_no_space",
                "unit_price": 0.0005,
                "required": True,
            },
            {"from": "request_body", "path": "optional_field", "unit_price": 1.0},
        ],
        cost_per_request=0.02,
    )
    # one fallback rule matches, but a required fallback rule does not: the
    # partial fallback must not be accepted
    result = evaluate_billing(config, BillingContext(request_body={"optional_field": 5}))
    assert result.cost == pytest.approx(0.02)
    assert result.cost_per_request_used is True
    assert "text" in result.missing_required


def test_missing_required_falls_back_to_cost_per_request():
    config = _config(fallback=[], cost_per_request=0.02)
    result = evaluate_billing(config, BillingContext(request_body={}))
    assert result.cost == pytest.approx(0.02)
    assert result.cost_per_request_used is True
    assert result.has_missing_required is True


def test_missing_required_with_zero_policy_bills_only_matched_rules():
    config = _config(
        on_missing="zero",
        rules=[
            {"from": "response_body", "path": "usage.text_words", "unit_price": 0.0005, "required": True},
            {"from": "response_body", "path": "duration", "unit_price": 0.01},
        ],
    )
    result = evaluate_billing(config, BillingContext(response_body={"duration": 2}))
    assert result.cost == pytest.approx(0.02)
    assert result.has_missing_required is True


def test_min_cost_and_breakdown_payload():
    config = _config(min_cost=0.05)
    result = evaluate_billing(config, BillingContext(response_body={"usage": {"text_words": 1}}))
    assert result.cost == pytest.approx(0.05)
    breakdown = result.to_breakdown()
    assert breakdown["total"] == pytest.approx(0.05)
    assert breakdown["rules"][0]["path"] == "usage.text_words"
    assert breakdown["rules"][0]["matched"] is True


def test_bool_values_are_not_treated_as_numbers():
    config = _config(
        rules=[{"from": "response_body", "path": "ok", "unit_price": 1.0, "required": True}]
    )
    result = evaluate_billing(config, BillingContext(response_body={"ok": True}))
    assert result.has_missing_required is True
    assert result.cost == 0.0


def test_string_numbers_are_supported():
    config = _config(
        rules=[{"from": "response_body", "path": "usage.text_words", "unit_price": 0.0005}]
    )
    result = evaluate_billing(
        config, BillingContext(response_body={"usage": {"text_words": "42"}})
    )
    assert result.cost == pytest.approx(0.021)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="invalid response_field_pricing"):
        parse_billing_config({"rules": []})
    with pytest.raises(ValueError, match="invalid response_field_pricing"):
        parse_billing_config({"rules": [{"path": "a", "unit_price": -1}]})
    with pytest.raises(ValueError, match="invalid response_field_pricing"):
        parse_billing_config({"rules": [{"path": "  "}]})
    with pytest.raises(ValueError, match="invalid response_field_pricing"):
        parse_billing_config({"rules": [{"path": "a", "aggregate": "median"}]})
    with pytest.raises(ValueError, match="not configured"):
        parse_billing_config(None)
