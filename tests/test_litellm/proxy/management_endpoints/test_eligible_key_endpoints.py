from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.management_endpoints.key_management_endpoints import (
    _eligible_key_availability,
    _eligible_key_helper,
    router,
)


def test_eligible_key_endpoint_is_additive_post_route():
    route = next(route for route in router.routes if route.path == "/key/eligible")
    assert route.methods == {"POST"}


@pytest.mark.parametrize(
    "spend,max_budget,reset_delta,expires_delta,expected",
    [
        (10.0, None, None, None, "available"),
        (9.0, 10.0, None, None, "available"),
        (10.0, 10.0, 1, 2, "waiting_for_reset"),
        (10.0, 10.0, -1, 2, "reset_pending"),
        (10.0, 10.0, None, 2, None),
        (10.0, 10.0, 2, 1, None),
    ],
)
def test_eligible_key_availability(
    spend, max_budget, reset_delta, expires_delta, expected
):
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)
    reset_at = now + timedelta(days=reset_delta) if reset_delta is not None else None
    expires = now + timedelta(days=expires_delta) if expires_delta is not None else None

    assert (
        _eligible_key_availability(
            spend=spend,
            max_budget=max_budget,
            budget_reset_at=reset_at,
            expires=expires,
            now=now,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_eligible_key_helper_returns_all_useful_keys_with_summary_projection():
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)

    def key_row(
        name,
        *,
        spend,
        max_budget,
        reset_at=None,
        expires=None,
        updated_at=None,
    ):
        return {
            "key_name": name,
            "key_alias": None,
            "spend": spend,
            "max_budget": max_budget,
            "expires": expires,
            "budget_duration": "30d" if reset_at else None,
            "budget_reset_at": reset_at,
            "created_at": now - timedelta(days=10),
            "updated_at": updated_at or now,
            "blocked": False,
        }

    rows = [
        key_row("unlimited", spend=100.0, max_budget=None),
        key_row("available", spend=1.0, max_budget=2.0),
        key_row(
            "waiting",
            spend=2.0,
            max_budget=2.0,
            reset_at=now + timedelta(days=1),
            expires=now + timedelta(days=2),
            updated_at=now + timedelta(minutes=1),
        ),
        key_row(
            "pending",
            spend=2.0,
            max_budget=2.0,
            reset_at=now - timedelta(minutes=1),
            expires=now + timedelta(days=2),
        ),
        key_row("terminal", spend=2.0, max_budget=2.0),
        key_row(
            "expires-before-reset",
            spend=2.0,
            max_budget=2.0,
            reset_at=now + timedelta(days=2),
            expires=now + timedelta(days=1),
        ),
    ]
    mock_prisma_client = AsyncMock()
    mock_query_raw = AsyncMock(return_value=rows)
    mock_prisma_client.db.query_raw = mock_query_raw

    response = await _eligible_key_helper(
        prisma_client=mock_prisma_client,
        user_ids=["stable-id", "legacy@example.com", "stable-id", ""],
        now=now,
    )

    assert [key.key_name for key in response.keys] == [
        "waiting",
        "unlimited",
        "available",
        "pending",
    ]
    assert [key.availability for key in response.keys] == [
        "waiting_for_reset",
        "available",
        "available",
        "reset_pending",
    ]
    assert [key.usable_now for key in response.keys] == [False, True, True, False]

    mock_query_raw.assert_awaited_once()
    sql_query, *query_params = mock_query_raw.call_args.args
    assert 'FROM "LiteLLM_VerificationToken"' in sql_query
    assert "user_id IN ($3, $4)" in sql_query
    assert "team_id != $1" in sql_query
    assert "expires > $2::timestamp" in sql_query
    assert "COALESCE(spend, 0) < max_budget" in sql_query
    assert "budget_reset_at < expires" in sql_query
    assert "SELECT\n            key_name," in sql_query
    assert "token" not in sql_query
    assert "stable-id" not in sql_query
    assert "legacy@example.com" not in sql_query
    assert "LIMIT" not in sql_query
    assert "COUNT(" not in sql_query
    assert query_params[0] == "litellm-dashboard"
    assert query_params[1] == now.replace(tzinfo=None)
    assert query_params[2:] == ["stable-id", "legacy@example.com"]
