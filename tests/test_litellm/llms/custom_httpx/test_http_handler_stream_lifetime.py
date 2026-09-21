"""Streaming handler lifetime regressions backported from BerriAI/litellm#34829."""

import asyncio
import gc
import weakref

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

def _mock_transport() -> httpx.MockTransport:
    """Answers anything with a short body, left unread when the caller asked to stream."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"ab")

    return httpx.MockTransport(respond)


RELEASED_TOO_EARLY = "the handler was released while its response could still read"
NEVER_RELEASED = "the handler outlived the response that was holding it"

# Every method that can hand back a body the caller has not read yet, which is
# every one that passes stream= down to send(). Parametrized so a method added
# later is covered here rather than being the one that forgets to anchor.
ASYNC_STREAMING_SENDS = ["post", "delete"]
SYNC_STREAMING_SENDS = ["post", "patch", "put", "delete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ASYNC_STREAMING_SENDS)
async def test_a_streaming_response_holds_its_handler_until_it_is_released(method):
    """The finalizer must not run while a body this handler issued can still arrive.

    The response holds a connection but not its handler. Anchoring the handler
    withholds the close, and releasing the anchor still delivers one.
    """
    handler = AsyncHTTPHandler()
    handler.client._transport = _mock_transport()
    handler.client._mounts.clear()  # Ignore environment proxy mounts in this hermetic test.
    ref = weakref.ref(handler)
    response = await getattr(handler, method)("https://example.invalid/stream", stream=True)

    del handler
    gc.collect()
    # This fork schedules async cleanup from __del__; let it finish.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ref() is not None, RELEASED_TOO_EARLY

    assert await response.aread() == b"ab"
    del response
    gc.collect()
    # This fork schedules async cleanup from __del__; let it finish.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ref() is None, NEVER_RELEASED


@pytest.mark.parametrize("method", SYNC_STREAMING_SENDS)
def test_a_sync_streaming_response_holds_its_handler_until_it_is_released(method):
    """The sync finalizer closes inline, so the same anchor has to hold it off."""
    handler = HTTPHandler()
    handler.client._transport = _mock_transport()
    handler.client._mounts.clear()  # Ignore environment proxy mounts in this hermetic test.
    ref = weakref.ref(handler)
    response = getattr(handler, method)("https://example.invalid/stream", stream=True)

    del handler
    gc.collect()
    assert ref() is not None, RELEASED_TOO_EARLY

    assert response.read() == b"ab"
    del response
    gc.collect()
    assert ref() is None, NEVER_RELEASED


@pytest.mark.asyncio
async def test_a_fully_read_response_does_not_hold_its_handler():
    """A non-streaming response is complete when ``post`` returns, so it anchors nothing.

    Otherwise every client close would wait on whatever the caller does next with
    a response it has already read.
    """
    handler = AsyncHTTPHandler()
    handler.client._transport = _mock_transport()
    handler.client._mounts.clear()  # Ignore environment proxy mounts in this hermetic test.
    ref = weakref.ref(handler)
    response = await handler.post("https://example.invalid/whole")
    assert response.content == b"ab"

    del handler
    gc.collect()
    # This fork schedules async cleanup from __del__; let it finish.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert ref() is None, "a fully-read response pinned its handler"


