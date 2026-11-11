"""
Regression tests for CVE-2026-42208 — pre-auth SQL injection in the
virtual-key lookup (`PrismaClient.get_data(table_name="combined_view")`).

The lookup used to build its SQL with an f-string:

    WHERE v.token = '{token}'

and interpolated the *raw* bearer token. The auth-failure hook
(`_enrich_failure_metadata_with_key_info`) feeds this lookup the bearer of a
request that just failed authentication, i.e. a value that was never hashed, so
`Authorization: Bearer x' OR pg_sleep(4) IS NULL --` ended up as SQL and ran
before the 401 was returned.

The fix normalizes the token to its stored form and binds it as `$1`, so the
payload can only ever be data.
"""

import os
import sys
from typing import Any, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.utils import PrismaClient, _hash_token_if_needed


def _unwrap(func):
    """Drop the `backoff` / `log_db_metrics` decorators (they touch proxy globals)."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _make_client(query_first_result: Any = None) -> PrismaClient:
    """
    A `PrismaClient` without a real Prisma engine / DB connection: only the
    attributes that `get_data` touches on the combined_view path are set.
    """
    client = PrismaClient.__new__(PrismaClient)
    client.db = MagicMock()
    client.db.query_first = AsyncMock(return_value=query_first_result)
    client.proxy_logging_obj = MagicMock()
    # `get_data` logs failures through this hook and re-raises the original error;
    # it must be awaitable or the logging path masks the real exception.
    client.proxy_logging_obj.failure_handler = AsyncMock()
    return client


async def _lookup(client: PrismaClient, token: str) -> Tuple[str, Tuple[Any, ...]]:
    """Run the combined_view lookup and return (sql, bind_args)."""
    await _unwrap(PrismaClient.get_data)(
        client,
        token=token,
        table_name="combined_view",
        query_type="find_unique",
        check_deprecated=False,
    )
    args: List[Any] = list(client.db.query_first.call_args.args)
    assert client.db.query_first.call_args.kwargs == {}
    return args[0], tuple(args[1:])


@pytest.mark.asyncio
async def test_combined_view_binds_token_instead_of_interpolating():
    """The injected bearer must never appear in the SQL text."""
    payload = "x' OR pg_sleep(4) IS NULL -- cbaaf9261"
    client = _make_client()

    sql, bind_args = await _lookup(client, payload)

    assert "WHERE v.token = $1" in sql, sql
    assert payload not in sql
    assert "pg_sleep" not in sql
    assert "cbaaf9261" not in sql
    # Payload does not start with "sk-", so it is bound as-is (and will simply
    # never match a stored hash) — the important part is that it is data.
    assert bind_args == (payload,)


@pytest.mark.asyncio
async def test_combined_view_binds_hashed_form_of_plaintext_key():
    """Plaintext `sk-` keys are hashed before binding, as the DB stores hashes."""
    client = _make_client()

    sql, bind_args = await _lookup(client, "sk-plaintext-secret")

    assert "sk-plaintext-secret" not in sql
    assert bind_args == (_hash_token_if_needed("sk-plaintext-secret"),)
    assert len(bind_args[0]) == 64


@pytest.mark.asyncio
async def test_combined_view_hashed_token_is_passed_through_unchanged():
    """Callers that already hold a hash must keep hitting the same row."""
    stored_hash = _hash_token_if_needed("sk-something")
    client = _make_client()

    _, bind_args = await _lookup(client, stored_hash)

    assert bind_args == (stored_hash,)


@pytest.mark.asyncio
async def test_combined_view_rejects_non_string_token():
    client = _make_client()

    with pytest.raises(Exception) as exc_info:
        await _unwrap(PrismaClient.get_data)(
            client,
            token=["not", "a", "token"],
            table_name="combined_view",
            query_type="find_unique",
            check_deprecated=False,
        )

    assert getattr(exc_info.value, "status_code", None) == 400
    client.db.query_first.assert_not_called()


@pytest.mark.asyncio
async def test_cached_plan_fallback_keeps_bind_args():
    """The cached-plan retry must not drop the bound parameter."""
    client = _make_client()
    client.db.query_first = AsyncMock(
        side_effect=[
            Exception("cached plan must not change result type"),
            [{"token": "abc"}],
        ]
    )

    result = await client._query_first_with_cached_plan_fallback(
        "SELECT 1 FROM t WHERE token = $1", "x' OR 1=1 --"
    )

    assert result == [{"token": "abc"}]
    first_call, second_call = client.db.query_first.call_args_list
    assert first_call.args == ("SELECT 1 FROM t WHERE token = $1", "x' OR 1=1 --")
    assert second_call.args[1] == "x' OR 1=1 --"
    assert "x' OR 1=1 --" not in second_call.args[0]
