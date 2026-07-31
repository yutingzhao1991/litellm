from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

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
        return MagicMock(
            key_name=name,
            key_alias=None,
            spend=spend,
            max_budget=max_budget,
            expires=expires,
            budget_duration="30d" if reset_at else None,
            budget_reset_at=reset_at,
            created_at=now - timedelta(days=10),
            updated_at=updated_at or now,
            blocked=False,
        )

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
    mock_find_many = AsyncMock(return_value=rows)
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many

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

    mock_find_many.assert_awaited_once()
    query = mock_find_many.call_args.kwargs
    assert "take" not in query
    assert "order" not in query
    assert "include" not in query
    assert query["where"]["AND"][0] == {
        "user_id": {"in": ["stable-id", "legacy@example.com"]}
    }
    assert {"OR": [{"blocked": None}, {"blocked": False}]} in query["where"]["AND"]
    assert {"OR": [{"expires": None}, {"expires": {"gt": now}}]} in query["where"][
        "AND"
    ]
    assert query["select"] == {
        "key_name": True,
        "key_alias": True,
        "spend": True,
        "max_budget": True,
        "expires": True,
        "budget_duration": True,
        "budget_reset_at": True,
        "created_at": True,
        "updated_at": True,
        "blocked": True,
    }
