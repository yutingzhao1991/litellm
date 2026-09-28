"""
FeelFish pass-through provider route.

``POST /feelfish/{model_name}[/{subpath}]`` forwards the request **verbatim** to
the upstream configured on the deployment (``api_base`` + credential) and bills
the call from the response using ``response_field_pricing``.

Why a dedicated route family instead of ``general_settings.pass_through_endpoints``?

- that path requires a LiteLLM Enterprise license as soon as ``auth: true`` is
  set (virtual-key auth + budgets are exactly what we need), and
- the gateway model name belongs in the path so that the auth chain
  (``get_model_from_request``) and this handler resolve the *same* model
  (no ``x-litellm-model`` header / body ``model`` mismatch, no key-model
  bypass).
"""

import asyncio
import json
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

from .billing import BillingContext, BillingResult, evaluate_billing
from .config import (
    RESPONSE_FIELD_PRICING_KEY,
    BillingConfig,
    PassthroughConfig,
    UpstreamAuth,
    cost_map_field_pricing,
    parse_billing_config,
    resolve_field_pricing,
    resolve_passthrough_config,
)
from .spend import (
    FEELFISH_PROVIDER,
    PASS_THROUGH_CALL_TYPE,
    SpendRecord,
    SpendRecorder,
    record_feelfish_failure,
    record_feelfish_spend,
)
from .field_path import parse_json_lines
from .upstream import (
    IncomingBody,
    build_upstream_request,
    parse_incoming_body,
    send_upstream_request,
)

router = APIRouter()

FEELFISH_ROUTE_PREFIX = "/feelfish"
ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

# Response content types whose body we accumulate for billing.
STREAMING_TEXT_CONTENT_TYPES = (
    "text/event-stream",
    "application/x-ndjson",
    "application/jsonlines",
    "application/json-seq",
)
JSON_CONTENT_TYPES = ("application/json", "application/x-ndjson")
MAX_BREAKDOWN_HEADER_CHARS = 1024
# Only 2xx responses are billable successes; 3xx is a redirect that we pass
# through without following (or charging) it.
UPSTREAM_ERROR_STATUS = 300


@dataclass
class FeelfishDeps:
    """Injected dependencies (overridable in tests via ``dependency_overrides``)."""

    llm_router: Any
    spend_recorder: SpendRecorder
    http_client: httpx.AsyncClient


def get_feelfish_deps() -> FeelfishDeps:
    """
    Build the production dependencies.

    ``litellm.proxy.proxy_server`` is imported lazily because it imports this
    package (circular import).
    """
    from litellm.proxy.proxy_server import llm_router

    http_handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint
    )
    return FeelfishDeps(
        llm_router=llm_router,
        spend_recorder=record_feelfish_spend,
        http_client=http_handler.client,
    )


@router.api_route(
    "/feelfish/{model_name}",
    methods=ALL_METHODS,
    tags=["FeelFish Pass-through"],
)
@router.api_route(
    "/feelfish/{model_name}/{subpath:path}",
    methods=ALL_METHODS,
    tags=["FeelFish Pass-through"],
)
async def feelfish_passthrough_route(
    model_name: str,
    request: Request,
    subpath: str = "",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    deps: FeelfishDeps = Depends(get_feelfish_deps),
) -> Response:
    """Forward one request to the upstream bound to ``model_name``."""
    return await handle_feelfish_request(
        model_name=model_name,
        subpath=subpath,
        request=request,
        user_api_key_dict=user_api_key_dict,
        deps=deps,
    )


async def handle_feelfish_request(
    *,
    model_name: str,
    subpath: str,
    request: Request,
    user_api_key_dict: Any,
    deps: FeelfishDeps,
) -> Response:
    """Route body: resolve -> forward -> bill -> return."""
    call_id = str(uuid.uuid4())
    start_time = datetime.now()

    deployment = await _resolve_feelfish_deployment(deps.llm_router, model_name)
    litellm_params: Dict[str, Any] = deployment.get("litellm_params") or {}
    billing_config = _resolve_billing_config(deployment, deps.llm_router, model_name)
    passthrough_config = _resolve_passthrough_config(deployment, model_name)
    auth = _resolve_auth(litellm_params, model_name)
    api_base, api_key = _resolve_upstream_connection(deployment, deps.llm_router)

    body = await parse_incoming_body(request, passthrough_config)

    # Build (and validate) the upstream request *before* the pre-call hooks: hooks
    # acquire rate-limit/concurrency slots, so a request that is rejected here must
    # not have incremented anything.
    try:
        upstream_request = build_upstream_request(
            request=request,
            api_base=str(api_base or ""),
            api_key=api_key,
            auth=auth,
            passthrough=passthrough_config,
            extra_headers=litellm_params.get("extra_headers"),
            subpath=subpath,
            body=body,
        )
    except ValueError as exc:
        raise _invalid_request(str(exc), param="path") from exc

    # Rate limits (RPM/TPM/concurrency) and other pre-call rejection hooks live in
    # the proxy hook pipeline, not in virtual-key auth: without this call,
    # authenticated clients could bypass their limits through /feelfish.
    try:
        await _run_pre_call_hooks(
            model_name=model_name,
            request=request,
            user_api_key_dict=user_api_key_dict,
        )
    except Exception:
        # Deliberately *not* reported as a failure event: we cannot know whether
        # the parallel-request limiter had already incremented its counters (hook
        # order dependent). A spurious decrement would let a caller reduce other
        # in-flight requests' concurrency usage — a fail-open bypass — whereas a
        # leaked slot only over-restricts until it expires. Upstream LiteLLM has
        # the same behaviour for pre-call rejections.
        raise

    try:
        upstream_response = await send_upstream_request(
            deps.http_client, upstream_request, timeout=passthrough_config.timeout
        )
    except Exception as exc:
        await record_feelfish_failure(
            model=model_name,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_data={"model": model_name},
            error=exc,
            traceback_str=traceback.format_exc(),
            call_id=call_id,
            start_time=start_time,
        )
        raise ProxyException(
            message=f"Upstream request failed: {exc}",
            type="upstream_error",
            param=None,
            code=502,
        ) from exc

    if _should_stream(body, upstream_response):
        return _streaming_response(
            request=request,
            upstream_response=upstream_response,
            body=body,
            model_name=model_name,
            billing_config=billing_config,
            call_id=call_id,
            start_time=start_time,
            user_api_key_dict=user_api_key_dict,
            deps=deps,
        )

    return await _buffered_response(
        request=request,
        upstream_response=upstream_response,
        body=body,
        model_name=model_name,
        billing_config=billing_config,
        call_id=call_id,
        start_time=start_time,
        user_api_key_dict=user_api_key_dict,
        deps=deps,
    )


# --------------------------------------------------------------------------- #
# Buffered (non-streaming) path
# --------------------------------------------------------------------------- #


async def _buffered_response(
    *,
    request: Request,
    upstream_response: httpx.Response,
    body: IncomingBody,
    model_name: str,
    billing_config: BillingConfig,
    call_id: str,
    start_time: datetime,
    user_api_key_dict: Any,
    deps: FeelfishDeps,
) -> Response:
    try:
        content = await upstream_response.aread()
    except BaseException as exc:
        # The streaming response was never consumed: close it so repeated read
        # failures cannot leak connections from the shared client pool, and emit
        # the terminal failure event so rate-limit counters are released.
        await _close_upstream_response(upstream_response)
        await record_feelfish_failure(
            model=model_name,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_data={"model": model_name},
            error=exc,
            traceback_str=traceback.format_exc(),
            call_id=call_id,
            start_time=start_time,
        )
        raise

    is_error = upstream_response.status_code >= UPSTREAM_ERROR_STATUS

    billing: Optional[BillingResult] = None
    if not is_error or billing_config.bill_on_error:
        billing = _compute_billing(
            billing_config,
            body=body,
            response=upstream_response,
            response_body=_parse_json_content(content, upstream_response),
            # `from: response_text` rules must work on buffered responses too
            # (single-line JSON, NDJSON bodies the upstream did not stream, ...).
            response_text=_decode_text_content(content, upstream_response),
        )
        await _record_spend(
            deps.spend_recorder,
            request=request,
            model_name=model_name,
            billing=billing,
            billing_config=billing_config,
            body=body,
            response=upstream_response,
            stream=False,
            call_id=call_id,
            start_time=start_time,
            user_api_key_dict=user_api_key_dict,
        )
    else:
        await _report_upstream_error(
            request=request,
            response=upstream_response,
            model_name=model_name,
            user_api_key_dict=user_api_key_dict,
            call_id=call_id,
            start_time=start_time,
        )
    await _close_upstream_response(upstream_response)

    headers = _build_response_headers(
        upstream_response=upstream_response,
        call_id=call_id,
        billing=billing,
        billing_config=billing_config,
    )
    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# Streaming path
# --------------------------------------------------------------------------- #


def _streaming_response(
    *,
    request: Request,
    upstream_response: httpx.Response,
    body: IncomingBody,
    model_name: str,
    billing_config: BillingConfig,
    call_id: str,
    start_time: datetime,
    user_api_key_dict: Any,
    deps: FeelfishDeps,
) -> StreamingResponse:
    buffer = _BillingBuffer(
        limit=(
            billing_config.buffer_stream_bytes
            if _is_textual(upstream_response)
            else 0
        )
    )
    headers = _build_response_headers(
        upstream_response=upstream_response,
        call_id=call_id,
        billing=None,  # cost is only known once the stream ends
        billing_config=billing_config,
    )

    return StreamingResponse(
        _stream_and_bill(
            buffer=buffer,
            request=request,
            upstream_response=upstream_response,
            body=body,
            model_name=model_name,
            billing_config=billing_config,
            call_id=call_id,
            start_time=start_time,
            user_api_key_dict=user_api_key_dict,
            deps=deps,
        ),
        status_code=upstream_response.status_code,
        headers=headers,
    )


async def _stream_and_bill(
    *,
    buffer: "_BillingBuffer",
    request: Request,
    upstream_response: httpx.Response,
    body: IncomingBody,
    model_name: str,
    billing_config: BillingConfig,
    call_id: str,
    start_time: datetime,
    user_api_key_dict: Any,
    deps: FeelfishDeps,
) -> AsyncIterator[bytes]:
    completed = False
    client_disconnected = False
    stream_failed = False
    is_error = upstream_response.status_code >= UPSTREAM_ERROR_STATUS
    try:
        async for chunk in upstream_response.aiter_bytes():
            buffer.add(chunk)
            yield chunk
        completed = True
    except GeneratorExit:
        # Starlette closing the generator (client went away while we were idle).
        client_disconnected = True
        raise
    except asyncio.CancelledError:
        # The response task itself was cancelled (client disconnected while we
        # were awaiting the next upstream chunk). Same billing consequence as
        # GeneratorExit, but it is a BaseException and needs explicit handling.
        client_disconnected = True
        raise
    except Exception:
        stream_failed = True
        raise
    finally:
        should_bill = billing_config.bill_on_error or (
            not is_error and (completed or client_disconnected)
        )
        if not should_bill:
            # Exactly one terminal event per request: an unbilled upstream error is
            # reported as a failure so callbacks (rate limits, alerts) still see it.
            await _report_upstream_error(
                request=request,
                response=upstream_response,
                model_name=model_name,
                user_api_key_dict=user_api_key_dict,
                call_id=call_id,
                start_time=start_time,
            )
        if should_bill:
            buffered_text = buffer.text()
            billing = _compute_billing(
                billing_config,
                body=body,
                response=upstream_response,
                response_body=_parse_streamed_response_body(buffered_text, upstream_response),
                response_text=buffered_text,
                incomplete_response=buffer.truncated,
            )
            _schedule_spend_recording(
                deps.spend_recorder,
                _build_spend_record(
                    request=request,
                    model_name=model_name,
                    billing=billing,
                    billing_config=billing_config,
                    body=body,
                    response=upstream_response,
                    stream=True,
                    call_id=call_id,
                    start_time=start_time,
                    user_api_key_dict=user_api_key_dict,
                    extra_breakdown={
                        "buffer_truncated": buffer.truncated,
                        "buffer_skipped": buffer.limit <= 0,
                        "client_disconnected": client_disconnected,
                        "stream_failed": stream_failed,
                    },
                ),
            )
        await _close_upstream_response(upstream_response)


async def _close_upstream_response(response: httpx.Response) -> None:
    """
    Close a streaming upstream response; cleanup must never mask billing.

    ``shield`` matters when the caller is being cancelled (client disconnect):
    without it the close would be cancelled again immediately and the upstream
    connection would leak from the shared client pool.
    """
    try:
        await asyncio.shield(asyncio.ensure_future(response.aclose()))
    except BaseException:  # pragma: no cover - cleanup only
        verbose_proxy_logger.debug(
            "feelfish: closing the upstream stream failed", exc_info=True
        )


async def _report_upstream_error(
    *,
    request: Request,
    response: httpx.Response,
    model_name: str,
    user_api_key_dict: Any,
    call_id: str,
    start_time: datetime,
) -> None:
    """
    Tell the proxy hook pipeline that this request failed upstream.

    Rate-limit and alerting callbacks release their state from either a success or
    a failure event, so an error response that is not billed still needs one.
    """
    error = httpx.HTTPStatusError(
        f"Upstream returned HTTP {response.status_code}",
        request=response.request,
        response=response,
    )
    await record_feelfish_failure(
        model=model_name,
        request=request,
        user_api_key_dict=user_api_key_dict,
        request_data={"model": model_name},
        error=error,
        traceback_str="",
        call_id=call_id,
        start_time=start_time,
    )


async def _run_pre_call_hooks(
    *,
    model_name: str,
    request: Request,
    user_api_key_dict: Any,
) -> None:
    """
    Run the proxy's pre-call hook pipeline (rate limits, guardrails, budget hooks).

    The upstream body is intentionally left untouched: hooks may enrich their own
    metadata, but the body belongs to the upstream protocol. ``proxy_server`` is
    imported lazily because it imports this package.
    """
    from litellm.proxy.proxy_server import general_settings, proxy_logging_obj

    if proxy_logging_obj is None:  # pragma: no cover - only in bare test setups
        return
    metadata: Dict[str, Any] = {}
    if general_settings is not None:
        # The parallel-request limiter reads the global limit from request metadata.
        metadata["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
    await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data={"model": model_name, "metadata": metadata},
        call_type=PASS_THROUGH_CALL_TYPE,
    )


# --------------------------------------------------------------------------- #
# Deployment / config resolution
# --------------------------------------------------------------------------- #


async def _resolve_feelfish_deployment(llm_router: Any, model_name: str) -> Dict[str, Any]:
    if not model_name:
        raise _invalid_request("A feelfish model name is required in the path", "model")
    if llm_router is None:
        raise ProxyException(
            message="LLM router is not configured on this proxy",
            type="feelfish_config_error",
            param=None,
            code=500,
        )
    try:
        deployment = await llm_router.async_get_available_deployment_for_pass_through(
            model=model_name, request_kwargs={}
        )
    except Exception as exc:
        raise _invalid_request(
            f"Model '{model_name}' is not available for pass-through: {exc}", "model"
        ) from exc

    if not deployment:
        raise _invalid_request(
            f"Model '{model_name}' is not configured for pass-through", "model"
        )

    litellm_params: Dict[str, Any] = deployment.get("litellm_params") or {}
    if litellm_params.get("custom_llm_provider") != FEELFISH_PROVIDER:
        raise _invalid_request(
            f"Model '{model_name}' is not a {FEELFISH_PROVIDER} pass-through model",
            "model",
        )
    return deployment


def _resolve_billing_config(
    deployment: Dict[str, Any], llm_router: Any, model_name: str
) -> BillingConfig:
    raw = resolve_field_pricing(deployment)
    if raw is None:
        raw = _merged_model_info(deployment, llm_router).get(RESPONSE_FIELD_PRICING_KEY)
    if raw is None:
        raw = cost_map_field_pricing((deployment.get("litellm_params") or {}).get("model"))
    try:
        return parse_billing_config(raw)
    except ValueError as exc:
        raise ProxyException(
            message=f"Model '{model_name}' has invalid response_field_pricing: {exc}",
            type="feelfish_config_error",
            param=None,
            code=500,
        ) from exc


def _resolve_passthrough_config(
    deployment: Dict[str, Any], model_name: str
) -> PassthroughConfig:
    try:
        return resolve_passthrough_config(deployment)
    except ValueError as exc:
        raise ProxyException(
            message=f"Model '{model_name}' has invalid passthrough config: {exc}",
            type="feelfish_config_error",
            param=None,
            code=500,
        ) from exc


def _resolve_auth(litellm_params: Dict[str, Any], model_name: str) -> UpstreamAuth:
    try:
        return UpstreamAuth.parse(litellm_params.get("auth_style"))
    except ValueError as exc:
        raise ProxyException(
            message=f"Model '{model_name}' has invalid auth_style: {exc}",
            type="feelfish_config_error",
            param=None,
            code=500,
        ) from exc


def _resolve_upstream_connection(
    deployment: Dict[str, Any], llm_router: Any
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the upstream ``api_base`` and credential for a deployment.

    Besides a literal ``api_key``/``api_base``, deployments may reference a named
    credential (``litellm_credential_name``); the router resolves those, so ask it
    before giving up.
    """
    litellm_params: Dict[str, Any] = deployment.get("litellm_params") or {}
    api_base = litellm_params.get("api_base")
    api_key = _resolve_api_key(litellm_params)

    if api_base and api_key:
        return str(api_base), api_key

    model_id = (deployment.get("model_info") or {}).get("id")
    if llm_router is None or not model_id:
        return (str(api_base) if api_base else None), api_key

    # `get_deployment_credentials_with_provider` also resolves
    # `litellm_credential_name`; plain `get_deployment_credentials` does not.
    resolver = getattr(
        llm_router, "get_deployment_credentials_with_provider", None
    ) or getattr(llm_router, "get_deployment_credentials", None)
    credentials: Dict[str, Any] = {}
    if resolver is not None:
        try:
            credentials = resolver(model_id=model_id) or {}
        except Exception as exc:  # pragma: no cover - lookup is best effort
            verbose_proxy_logger.debug("feelfish: credential lookup failed: %s", exc)
            credentials = {}

    resolved_api_base = api_base or credentials.get("api_base")
    resolved_api_key = api_key or credentials.get("api_key")
    if isinstance(resolved_api_key, str) and resolved_api_key.startswith("os.environ/"):
        resolved_api_key = _resolve_api_key({"api_key": resolved_api_key})

    credential_name = litellm_params.get("litellm_credential_name")
    if credential_name and not resolved_api_key:
        # Fail closed: an unauthenticated request upstream would be a silent
        # misconfiguration (and usually an upstream 401).
        raise ProxyException(
            message=(
                f"Credential '{credential_name}' could not be resolved for this "
                "feelfish deployment"
            ),
            type="feelfish_config_error",
            param=None,
            code=500,
        )

    return (
        str(resolved_api_base) if resolved_api_base else None,
        str(resolved_api_key) if resolved_api_key else None,
    )


def _resolve_api_key(litellm_params: Dict[str, Any]) -> Optional[str]:
    raw = litellm_params.get("api_key")
    if raw is None:
        return None
    value = str(raw)
    if value.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str

        return get_secret_str(value.replace("os.environ/", "", 1))
    return value


def _merged_model_info(deployment: Dict[str, Any], llm_router: Any) -> Dict[str, Any]:
    """
    Deployment model info merged with the model cost map entry.

    Deployment values win (the engine treats ``response_field_pricing`` as a
    whole-block override); the cost map only fills gaps.
    """
    model_info = dict(deployment.get("model_info") or {})
    litellm_params: Dict[str, Any] = deployment.get("litellm_params") or {}
    model_id = model_info.get("id")
    model_name = litellm_params.get("model")
    if llm_router is not None and model_id and model_name:
        try:
            merged = llm_router.get_deployment_model_info(
                model_id=model_id, model_name=model_name
            )
        except Exception as exc:  # pragma: no cover - router lookup is best effort
            verbose_proxy_logger.debug("feelfish: model info merge failed: %s", exc)
            merged = None
        if merged:
            return {
                **merged,
                **{key: value for key, value in model_info.items() if value is not None},
            }
    return model_info


# --------------------------------------------------------------------------- #
# Billing helpers
# --------------------------------------------------------------------------- #


def _compute_billing(
    billing_config: BillingConfig,
    *,
    body: IncomingBody,
    response: httpx.Response,
    response_body: Optional[Any],
    response_text: Optional[str],
    incomplete_response: bool = False,
) -> BillingResult:
    """
    Evaluate the pricing rules.

    For streamed JSON (NDJSON/SSE) the caller passes the parsed events as
    ``response_body`` (a list) in addition to the raw text, so one rule set works
    for both streamed and buffered transports.
    """
    context = BillingContext(
        request_body=body.billing_body,
        response_body=response_body,
        response_text=response_text,
        response_headers=dict(response.headers),
    )
    result = evaluate_billing(
        billing_config, context, incomplete_response=incomplete_response
    )
    if result.has_missing_required:
        verbose_proxy_logger.warning(
            "feelfish_billing_missing_field model_response_status=%s missing=%s fallback_used=%s",
            response.status_code,
            result.missing_required,
            result.fallback_used,
        )
    return result


async def _record_spend(
    recorder: SpendRecorder,
    *,
    request: Request,
    model_name: str,
    billing: BillingResult,
    billing_config: BillingConfig,
    body: IncomingBody,
    response: httpx.Response,
    stream: bool,
    call_id: str,
    start_time: datetime,
    user_api_key_dict: Any,
    extra_breakdown: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write the cost for a buffered request.

    Awaited on purpose: the client must not be able to start the next request
    before the previous cost is accounted for. A recording failure is logged
    instead of failing the response — the upstream work is already done, and
    returning an error would only invite a retry that bills the upstream twice.
    """
    record = _build_spend_record(
        request=request,
        model_name=model_name,
        billing=billing,
        billing_config=billing_config,
        body=body,
        response=response,
        stream=stream,
        call_id=call_id,
        start_time=start_time,
        user_api_key_dict=user_api_key_dict,
        extra_breakdown=extra_breakdown,
    )
    try:
        await recorder(record)
    except Exception as exc:
        verbose_proxy_logger.error(
            "feelfish: failed to record spend for %s: %s", model_name, exc
        )


def _schedule_spend_recording(recorder: SpendRecorder, record: SpendRecord) -> None:
    """
    Record the cost of a finished/aborted stream.

    Detached from the response task on purpose: the client disconnecting cancels
    that task, and the cost must still be written.
    """
    task = asyncio.create_task(recorder(record))

    def _log_failure(finished: "asyncio.Task[None]") -> None:
        if finished.cancelled():
            return
        error = finished.exception()
        if error is not None:
            verbose_proxy_logger.error(
                "feelfish: failed to record spend for %s: %s", record.model, error
            )

    task.add_done_callback(_log_failure)


def _build_spend_record(
    *,
    request: Request,
    model_name: str,
    billing: BillingResult,
    billing_config: BillingConfig,
    body: IncomingBody,
    response: httpx.Response,
    stream: bool,
    call_id: str,
    start_time: datetime,
    user_api_key_dict: Any,
    extra_breakdown: Optional[Dict[str, Any]] = None,
) -> SpendRecord:
    return SpendRecord(
        model=model_name,
        cost=billing.cost,
        billing=billing,
        request=request,
        user_api_key_dict=user_api_key_dict,
        request_body=_spend_request_body(body),
        response_status=response.status_code,
        stream=stream,
        call_id=call_id,
        start_time=start_time,
        end_time=datetime.now(),
        extra_breakdown=extra_breakdown or {},
    )


def _spend_request_body(body: IncomingBody) -> Dict[str, Any]:
    """Metadata-only view of the request body (never forwarded upstream)."""
    if isinstance(body.billing_body, dict):
        return dict(body.billing_body)
    return {}


def _parse_streamed_response_body(
    text: Optional[str], response: httpx.Response
) -> Optional[Any]:
    """
    Parse an accumulated stream according to the *actual* response format.

    The client may ask for `stream: true` while the upstream still answers with a
    single (possibly pretty-printed) `application/json` document; parsing that
    line-by-line would find no events and silently bill 0. Genuine NDJSON/SSE
    responses are parsed per line instead.
    """
    if not text:
        return None
    if _content_type_matches(response, STREAMING_TEXT_CONTENT_TYPES):
        return parse_json_lines(text) or None

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    return parse_json_lines(text) or None


def _decode_text_content(
    content: Optional[bytes], response: httpx.Response
) -> Optional[str]:
    """Decode a textual response body for `from: response_text` billing rules."""
    if not content or not _is_textual(response):
        return None
    return content.decode("utf-8", errors="replace")


def _is_json_content_type(response: httpx.Response) -> bool:
    """True for JSON media types, including structured suffixes (`+json`)."""
    return "json" in (response.headers.get("content-type") or "").lower()


def _parse_json_content(
    content: Optional[bytes], response: httpx.Response
) -> Optional[Any]:
    if content is None or not content:
        return None
    if not _is_json_content_type(response):
        return None
    try:
        return json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return None


def _build_response_headers(
    *,
    upstream_response: httpx.Response,
    call_id: str,
    billing: Optional[BillingResult],
    billing_config: BillingConfig,
) -> Dict[str, str]:
    custom_headers: Dict[str, str] = {}
    if billing is not None:
        custom_headers["x-litellm-response-cost"] = f"{billing.cost}"
        if billing_config.expose_breakdown:
            # HTTP header values must stay latin-1 encodable
            breakdown = json.dumps(billing.to_breakdown())
            custom_headers["x-litellm-cost-breakdown"] = breakdown[
                :MAX_BREAKDOWN_HEADER_CHARS
            ]

    headers = HttpPassThroughEndpointHelpers.get_response_headers(
        headers=upstream_response.headers,
        litellm_call_id=call_id,
        custom_headers=custom_headers,
    )
    # httpx transparently decodes compressed bodies and we re-serialize JSON, so
    # the upstream content-length is no longer guaranteed to match.
    headers.pop("content-length", None)
    return headers


# --------------------------------------------------------------------------- #
# Stream/buffer helpers
# --------------------------------------------------------------------------- #


@dataclass
class _BillingBuffer:
    """
    Bounded billing buffer for streamed responses.

    Keeps both the **head** and the **tail** of the stream (~half of ``limit``
    each): upstream usage is normally cumulative and reported in the *last*
    event, so keeping only the head would bill an early, smaller value. Memory
    stays bounded and ``truncated`` records that the middle was dropped.
    """

    limit: int
    _head: List[bytes] = field(default_factory=list)
    _tail: List[bytes] = field(default_factory=list)
    _head_size: int = 0
    _tail_size: int = 0
    truncated: bool = False

    @property
    def head_limit(self) -> int:
        return self.limit // 2

    @property
    def tail_limit(self) -> int:
        return self.limit - self.head_limit

    def add(self, chunk: bytes) -> None:
        if self.limit <= 0 or not chunk:
            return

        if self._head_size < self.head_limit:
            remaining = self.head_limit - self._head_size
            piece = chunk[:remaining]
            self._head.append(piece)
            self._head_size += len(piece)
            chunk = chunk[remaining:]
            if not chunk:
                return

        self._tail.append(chunk)
        self._tail_size += len(chunk)
        while self._tail_size > self.tail_limit and self._tail:
            # Only now are bytes actually dropped: the retained window is no
            # longer contiguous with the start of the stream.
            self.truncated = True
            overflow = self._tail_size - self.tail_limit
            first = self._tail[0]
            if len(first) <= overflow:
                self._tail.pop(0)
                self._tail_size -= len(first)
            else:
                self._tail[0] = first[overflow:]
                self._tail_size -= overflow

    def text(self) -> Optional[str]:
        if not self._head and not self._tail:
            return None

        head = b"".join(self._head)
        tail = b"".join(self._tail)

        if not self.truncated:
            # Contiguous: no separator, or a JSON event straddling the head/tail
            # boundary would be split into two unparseable lines.
            return (head + tail).decode("utf-8", errors="replace")

        # Bytes were dropped: keep only complete lines from the retained window so
        # every surfaced event parses.
        if b"\n" in head:
            head = head[: head.rfind(b"\n") + 1]
        else:
            head = b""
        if b"\n" in tail:
            tail = tail[tail.find(b"\n") + 1 :]
        else:
            tail = b""
        return (head + tail).decode("utf-8", errors="replace")


def _should_stream(body: IncomingBody, response: httpx.Response) -> bool:
    if isinstance(body.json_body, dict) and body.json_body.get("stream") is True:
        return True
    return any(
        content_type in (response.headers.get("content-type") or "").lower()
        for content_type in STREAMING_TEXT_CONTENT_TYPES
    )


def _is_textual(response: httpx.Response) -> bool:
    content_type = (response.headers.get("content-type") or "").lower()
    return (
        content_type.startswith("text/")
        or _is_json_content_type(response)
        or any(
            candidate in content_type for candidate in STREAMING_TEXT_CONTENT_TYPES
        )
    )


def _content_type_matches(response: httpx.Response, candidates: Sequence[str]) -> bool:
    content_type = (response.headers.get("content-type") or "").lower()
    return any(candidate in content_type for candidate in candidates)


def _invalid_request(message: str, param: Optional[str]) -> ProxyException:
    return ProxyException(
        message=message, type="invalid_request_error", param=param, code=400
    )


__all__ = [
    "FeelfishDeps",
    "feelfish_passthrough_route",
    "get_feelfish_deps",
    "handle_feelfish_request",
    "router",
]
