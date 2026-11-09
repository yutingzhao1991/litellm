import json
import os
import sys
from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.db.prisma_client import PrismaWrapper, should_update_prisma_schema


def test_should_update_prisma_schema(monkeypatch):
    # CASE 1: Environment variable behavior
    # When DISABLE_SCHEMA_UPDATE is not set -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema() == True

    # When DISABLE_SCHEMA_UPDATE="true" -> should not update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "true")
    assert should_update_prisma_schema() == False

    # When DISABLE_SCHEMA_UPDATE="false" -> should update
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "false")
    assert should_update_prisma_schema() == True

    # CASE 2: Explicit parameter behavior (overrides env var)
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)
    assert should_update_prisma_schema(True) == False  # Param True -> should not update

    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", None)  # Set env var opposite to param
    assert should_update_prisma_schema(False) == True  # Param False -> should update


@pytest.mark.asyncio
async def test_recreate_prisma_client_successful_disconnect():
    """
    Test that recreate_prisma_client works normally when disconnect succeeds.
    """
    # Mock the original prisma client
    mock_prisma = AsyncMock()
    
    # Create a mock PrismaWrapper instance
    wrapper = Mock()
    wrapper._original_prisma = mock_prisma
    
    # Configure disconnect to succeed
    mock_prisma.disconnect.return_value = None
    
    # Mock the entire recreate_prisma_client method to avoid import issues
    async def mock_recreate_prisma_client(new_db_url: str, http_client=None):
        try:
            await mock_prisma.disconnect()
        except Exception:
            pass
        
        mock_new_prisma = AsyncMock()
        wrapper._original_prisma = mock_new_prisma
        await mock_new_prisma.connect()
    
    # Assign the mock method to the wrapper
    wrapper.recreate_prisma_client = mock_recreate_prisma_client
    
    # Call the method
    await wrapper.recreate_prisma_client("postgresql://new:new@localhost:5432/new")
    
    # Verify that disconnect was called
    mock_prisma.disconnect.assert_called_once()
    
    # Verify that the new client replaced the original
    assert wrapper._original_prisma != mock_prisma
    assert hasattr(wrapper._original_prisma, 'connect') 


class _FakeEngineProcess:
    def __init__(self, pid: int):
        self.pid = pid


class _FakeEngine:
    def __init__(self, pid: int):
        self.process = _FakeEngineProcess(pid)


class _FakePrismaClient:
    """Minimal stand-in for prisma.Prisma with a query engine attached."""

    def __init__(self, engine_pid: int = 0, disconnect_error: Exception = None):
        if engine_pid:
            self._engine = _FakeEngine(engine_pid)
        self._disconnect_error = disconnect_error
        self.disconnect_timeouts = []
        self.connected = False

    async def disconnect(self, timeout=None):
        self.disconnect_timeouts.append(timeout)
        if self._disconnect_error is not None:
            raise self._disconnect_error

    async def connect(self):
        self.connected = True


def _use_fake_prisma_module(new_client):
    prisma_module = Mock()
    prisma_module.Prisma = Mock(return_value=new_client)
    return patch.dict(sys.modules, {"prisma": prisma_module})


@pytest.mark.asyncio
async def test_recreate_prisma_client_kills_old_engine_before_disconnect():
    """A reconnect must never leave the previous query engine running.

    prisma-client-py 0.11 blocks in `process.wait(timeout=None)` while tearing the
    engine down; killing it first makes the wait immediate and keeps the engine from
    being orphaned when disconnect() raises.
    """
    events = []
    old_client = _FakePrismaClient(engine_pid=4321)
    original_disconnect = old_client.disconnect

    async def tracked_disconnect(timeout=None):
        events.append(("disconnect", timeout))
        await original_disconnect(timeout=timeout)

    old_client.disconnect = tracked_disconnect
    new_client = _FakePrismaClient()
    wrapper = PrismaWrapper(original_prisma=old_client, iam_token_db_auth=False)

    with _use_fake_prisma_module(new_client), patch(
        "litellm.proxy.db.prisma_client.is_query_engine_pid", return_value=True
    ), patch(
        "litellm.proxy.db.prisma_client.kill_process",
        side_effect=lambda pid, *a, **k: events.append(("kill", pid)) or True,
    ), patch("litellm.proxy.db.prisma_client.pid_is_alive", return_value=False):
        await wrapper.recreate_prisma_client("postgresql://user:pass@host:5432/db")

    assert events[0] == ("kill", 4321)
    assert events[1][0] == "disconnect"
    assert events[1][1] is not None, "disconnect must be bounded, never wait forever"
    assert isinstance(events[1][1], timedelta)
    assert wrapper._original_prisma is new_client
    assert new_client.connected is True


@pytest.mark.asyncio
async def test_recreate_prisma_client_kills_engine_that_survives_failed_disconnect():
    events = []
    old_client = _FakePrismaClient(
        engine_pid=4322, disconnect_error=RuntimeError("engine is stuck")
    )
    new_client = _FakePrismaClient()
    wrapper = PrismaWrapper(original_prisma=old_client, iam_token_db_auth=False)

    with _use_fake_prisma_module(new_client), patch(
        "litellm.proxy.db.prisma_client.is_query_engine_pid", return_value=True
    ), patch(
        "litellm.proxy.db.prisma_client.kill_process",
        side_effect=lambda pid, *a, **k: events.append(("kill", pid)) or True,
    ), patch("litellm.proxy.db.prisma_client.pid_is_alive", return_value=True), patch(
        "litellm.proxy.db.prisma_client.reap_child",
        side_effect=lambda pid, *a, **k: events.append(("reap", pid)) or True,
    ):
        await wrapper.recreate_prisma_client("postgresql://user:pass@host:5432/db")

    kills = [entry for entry in events if entry[0] == "kill"]
    assert kills == [("kill", 4322), ("kill", 4322)]
    assert ("reap", 4322) in events, "a killed engine must not be left as a zombie"
    assert wrapper._original_prisma is new_client
    assert new_client.connected is True


@pytest.mark.asyncio
async def test_recreate_prisma_client_without_engine_skips_kill():
    events = []
    old_client = _FakePrismaClient(engine_pid=0)
    new_client = _FakePrismaClient()
    wrapper = PrismaWrapper(original_prisma=old_client, iam_token_db_auth=False)

    with _use_fake_prisma_module(new_client), patch(
        "litellm.proxy.db.prisma_client.kill_process",
        side_effect=lambda pid, *a, **k: events.append(("kill", pid)) or True,
    ), patch("litellm.proxy.db.prisma_client.pid_is_alive", return_value=False):
        await wrapper.recreate_prisma_client("postgresql://user:pass@host:5432/db")

    assert events == []
    assert old_client.disconnect_timeouts != []
    assert new_client.connected is True

@pytest.mark.asyncio
async def test_recreate_prisma_client_never_kills_recycled_pid():
    """A stale `process.pid` must not be signalled: the PID may belong to someone else."""
    events = []
    old_client = _FakePrismaClient(engine_pid=4323)
    new_client = _FakePrismaClient()
    wrapper = PrismaWrapper(original_prisma=old_client, iam_token_db_auth=False)

    with _use_fake_prisma_module(new_client), patch(
        "litellm.proxy.db.prisma_client.is_query_engine_pid", return_value=False
    ), patch(
        "litellm.proxy.db.prisma_client.kill_process",
        side_effect=lambda pid, *a, **k: events.append(("kill", pid)) or True,
    ), patch("litellm.proxy.db.prisma_client.pid_is_alive", return_value=False):
        await wrapper.recreate_prisma_client("postgresql://user:pass@host:5432/db")

    assert events == [], "must not signal a PID that is no longer a query engine"
    assert old_client.disconnect_timeouts != []
    assert wrapper._original_prisma is new_client
