# HTTP streaming handler lifetime backport

This fork backports the streaming-response lifetime fix from
[BerriAI/litellm#34829](https://github.com/BerriAI/litellm/pull/34829), associated
with [#24929](https://github.com/BerriAI/litellm/issues/24929).

HTTP client handlers are cached with a one-hour TTL. A streaming response holds
its connection but not its handler. Once cache eviction drops the handler's last
reference, its destructor can close the underlying client while the response is
still reading. This can interrupt an otherwise healthy upstream stream.

The backport anchors the handler through a weakref finalizer for each streaming
response. The handler remains alive until all its responses are collected, then
its existing cleanup runs. Non-streaming responses do not retain handlers.
The six streaming send sites covered match the upstream fix: async POST/DELETE
and sync POST/PATCH/PUT/DELETE. The async retry helper's separately owned
temporary-client lifecycle is outside this backport.

Only this lifetime fix is imported; timeout policy, stream terminal handling,
model reasoning parameters, and the rest of upstream are unchanged.

## Validation

Use the repository Python environment and run:

    LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest \
      tests/test_litellm/llms/custom_httpx/test_http_handler_stream_lifetime.py \
      tests/test_litellm/llms/custom_httpx/test_http_handler.py \
      tests/test_litellm/llms/custom_httpx/test_aiohttp_transport.py \
      tests/local_testing/test_handler_gc_does_not_close_client.py -q

The backported tests cover handler collection, cache eviction during streaming,
both aiohttp and httpcore transports, synchronous requests, abandoned responses,
pool cleanup, and non-streaming lifetimes. Local integration tests bind an
ephemeral loopback port; the existing IPv4 transport test uses example.com.

Tests are adapted to this fork's asynchronous destructor cleanup and isolate
environment proxy mounts. The older local test harness does not use VCR, so the
upstream conftest exclusion is unnecessary.

This fixes a reproduced gateway defect. It does not establish that every
production reasoning-only response was caused by handler eviction, and it does
not prevent genuine upstream failures or configured stream-duration timeouts.
