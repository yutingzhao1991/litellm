"""
Spend recording for the FeelFish pass-through provider.

The route computes the cost itself (response-field pricing), then hands it to a
:class:`SpendRecorder`. The default recorder pushes the cost through the
standard LiteLLM logging pipeline by reusing the ``pass_through_endpoint``
call type, which already knows how to attribute spend to a virtual key:

- ``logging_obj.model_call_details["response_cost"]`` is preserved by that
  call type (it only defaults to ``None`` when nothing was set),
- ``_init_kwargs_for_pass_through_endpoint`` provides the key/user/team
  metadata the spend + budget tables need.

Keeping this behind a recorder protocol lets the route be unit tested with a
capturing fake instead of the proxy database.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject
from litellm.utils import executor as thread_pool_executor

from .billing import BillingResult

FEELFISH_PROVIDER = "feelfish"
PASS_THROUGH_CALL_TYPE = "pass_through_endpoint"
MAX_LOGGED_RESPONSE_SUMMARY_CHARS = 512


@dataclass
class SpendRecord:
    """
    Everything the recorder needs to write one spend log entry.

    ``request_body`` is kept for diagnostics/tests only: the default recorder
    never forwards a client body into the logging pipeline (see below).
    """

    model: str
    cost: float
    billing: Optional[BillingResult]
    request: Any
    user_api_key_dict: Any
    request_body: Dict[str, Any]
    response_status: int
    stream: bool
    call_id: str
    start_time: datetime
    end_time: datetime
    extra_breakdown: Dict[str, Any] = field(default_factory=dict)


SpendRecorder = Callable[[SpendRecord], Awaitable[None]]
"""Writes one request's cost to the billing pipeline.

A plain async callable is the whole contract: the default implementation is
:func:`record_feelfish_spend`, and tests inject a recording function.
"""


async def record_feelfish_spend(record: SpendRecord) -> None:
    """Default recorder: emit the standard logging payload with our cost."""
    logging_obj = Logging(
        model=record.model,
        messages=[{"role": "user", "content": "feelfish passthrough request"}],
        # Always False: this logging event describes a *completed* response. With
        # stream=True `_success_handler_helper_fn` resets `response_cost` to None,
        # which silently wipes the cost we computed for a streamed call.
        stream=False,
        call_type=PASS_THROUGH_CALL_TYPE,
        start_time=record.start_time,
        litellm_call_id=record.call_id,
        function_id="feelfish_passthrough",
    )

    # Deliberately no request/response bodies: logs carry status + cost breakdown
    # only (transcripts, generated text and base64 media must not reach log sinks).
    passthrough_payload: PassthroughStandardLoggingPayload = {
        "url": str(getattr(record.request, "url", "")),
        "request_method": getattr(record.request, "method", None),
        "cost_per_request": None,
    }

    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=record.request,
        user_api_key_dict=record.user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        logging_obj=logging_obj,
        # Deliberately empty: the request body is the *upstream's* protocol payload
        # and must never feed the gateway's logging/identity params. Passing it here
        # would let a client spoof `metadata.user_api_key_hash` (and therefore the
        # key/user/team the spend is charged to) through `litellm_params`.
        _parsed_body={},
        litellm_call_id=record.call_id,
    )
    litellm_params: Dict[str, Any] = kwargs.get("litellm_params", {}) or {}
    metadata = litellm_params.get("metadata")
    if isinstance(metadata, dict):
        if record.billing is not None:
            metadata["response_cost_breakdown"] = {
                **record.billing.to_breakdown(),
                **record.extra_breakdown,
            }
        _add_trusted_budget_metadata(metadata, record.user_api_key_dict)

    logging_obj.model_call_details["custom_llm_provider"] = FEELFISH_PROVIDER
    logging_obj.model_call_details["response_cost"] = record.cost
    logging_obj.model_call_details["litellm_call_id"] = record.call_id
    logging_obj.update_environment_variables(
        model=record.model,
        user="unknown",
        optional_params={},
        litellm_params=litellm_params,
        call_type=PASS_THROUGH_CALL_TYPE,
    )
    logging_obj.pre_call(
        input=[{"role": "user", "content": "feelfish passthrough request"}],
        api_key="",
        additional_args={
            "complete_input_dict": {},
            "api_base": str(getattr(record.request, "url", "")),
            "headers": {},
        },
    )

    result = StandardPassThroughResponseObject(
        response=_summarize_response(record),
    )
    await logging_obj.async_success_handler(
        result=result,
        start_time=record.start_time,
        end_time=record.end_time,
        cache_hit=False,
        **kwargs,
    )
    if logging_obj._should_run_sync_callbacks_for_async_calls() is False:
        return
    thread_pool_executor.submit(
        logging_obj.success_handler,
        result=result,
        end_time=record.end_time,
        cache_hit=False,
        start_time=record.start_time,
        **kwargs,
    )


async def record_feelfish_failure(
    *,
    model: str,
    request: Any,
    user_api_key_dict: Any,
    request_data: Dict[str, Any],
    error: BaseException,
    traceback_str: str,
    call_id: str,
    start_time: datetime,
) -> None:
    """
    Report a failed pass-through call.

    Two things must happen, and they are handled by *different* hook families:

    - a **logging failure event** — this is what parallel-request limiters use to
      release their concurrency counters (`async_log_failure_event`), and what
      observability callbacks listen to;
    - the proxy alerting/guardrail failure hook (`post_call_failure_hook`).

    Both are best-effort: a reporting failure must never mask the request error.
    """
    await _emit_failure_logging_event(
        model=model,
        request=request,
        user_api_key_dict=user_api_key_dict,
        error=error,
        traceback_str=traceback_str,
        call_id=call_id,
        start_time=start_time,
    )

    try:
        # `proxy_server` imports this package, so the import stays local.
        from litellm.proxy.proxy_server import proxy_logging_obj

        if proxy_logging_obj is None:
            return
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=error,
            request_data=request_data,
            traceback_str=traceback_str,
        )
    except Exception:  # pragma: no cover - failure reporting must never mask the error
        pass


async def _emit_failure_logging_event(
    *,
    model: str,
    request: Any,
    user_api_key_dict: Any,
    error: BaseException,
    traceback_str: str,
    call_id: str,
    start_time: datetime,
) -> None:
    try:
        logging_obj = Logging(
            model=model,
            messages=[{"role": "user", "content": "feelfish passthrough request"}],
            stream=False,
            call_type=PASS_THROUGH_CALL_TYPE,
            start_time=start_time,
            litellm_call_id=call_id,
            function_id="feelfish_passthrough",
        )
        passthrough_payload: PassthroughStandardLoggingPayload = {
            "url": str(getattr(request, "url", "")),
            "request_method": getattr(request, "method", None),
            "cost_per_request": None,
        }
        kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
            request=request,
            user_api_key_dict=user_api_key_dict,
            passthrough_logging_payload=passthrough_payload,
            logging_obj=logging_obj,
            # Same rule as the success path: never feed the upstream body into the
            # gateway's identity/logging params.
            _parsed_body={},
            litellm_call_id=call_id,
        )
        litellm_params: Dict[str, Any] = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            # Failure events release the same rate-limit counters as success events.
            _add_trusted_budget_metadata(metadata, user_api_key_dict)
        logging_obj.model_call_details["custom_llm_provider"] = FEELFISH_PROVIDER
        logging_obj.model_call_details["litellm_params"] = litellm_params
        logging_obj.model_call_details["response_cost"] = 0.0
        logging_obj.update_environment_variables(
            model=model,
            user="unknown",
            optional_params={},
            litellm_params=litellm_params,
            call_type=PASS_THROUGH_CALL_TYPE,
        )
        await logging_obj.async_failure_handler(
            exception=error,
            traceback_exception=traceback_str,
            start_time=start_time,
            end_time=datetime.now(),
        )
    except Exception as exc:  # pragma: no cover - must not mask the request error
        verbose_proxy_logger.debug("feelfish: failure logging event failed: %s", exc)


def _add_trusted_budget_metadata(
    metadata: Dict[str, Any], user_api_key_dict: Any
) -> None:
    """
    Carry trusted budget/limit config into the terminal progress event.

    Success *and* failure events need it: the model-max-budget limiter only
    accumulates spend when these keys are present, and the parallel-request
    limiter only decrements its global counter when
    ``global_max_parallel_requests`` is present. Values come from the
    authenticated key / general settings — never from the (upstream owned) body.
    """
    for metadata_key, attribute in (
        ("user_api_key_model_max_budget", "model_max_budget"),
        ("user_api_key_end_user_model_max_budget", "end_user_model_max_budget"),
    ):
        value = getattr(user_api_key_dict, attribute, None)
        if value is not None:
            metadata[metadata_key] = value

    try:
        # Imported lazily: `proxy_server` imports this package.
        from litellm.proxy.proxy_server import general_settings

        if general_settings is not None:
            metadata.setdefault(
                "global_max_parallel_requests",
                general_settings.get("global_max_parallel_requests", None),
            )
    except Exception:  # pragma: no cover - metadata enrichment is best effort
        pass


def _summarize_response(record: SpendRecord) -> str:
    """Never log bodies: only status + transport mode."""
    mode = "stream" if record.stream else "buffered"
    return f"feelfish passthrough response status={record.response_status} mode={mode}"
