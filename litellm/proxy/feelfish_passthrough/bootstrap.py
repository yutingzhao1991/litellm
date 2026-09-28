"""
Startup wiring for the FeelFish pass-through provider.

Two responsibilities, both idempotent:

- allow ``/feelfish`` (and its subpaths) for ordinary virtual keys, and
- fail fast when a ``feelfish`` deployment is misconfigured.

Failing fast is deliberate: a pass-through model without usable pricing would
silently bill ``0``, which is worse than refusing to start.
"""

from typing import Any, Dict, Iterable, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLMRoutes

from .config import (
    cost_map_field_pricing,
    parse_billing_config,
    resolve_field_pricing,
    resolve_passthrough_config,
    UpstreamAuth,
)
from .spend import FEELFISH_PROVIDER

ROUTE_ALLOWLIST_ENTRIES = ("/feelfish", "/feelfish/*")

LITELLM_ROUTES_OPENAI_ROUTES = LiteLLMRoutes.openai_routes.value


def register_feelfish_routes() -> None:
    """Let non-admin virtual keys call ``/feelfish/*`` (idempotent)."""
    for entry in ROUTE_ALLOWLIST_ENTRIES:
        if entry not in LITELLM_ROUTES_OPENAI_ROUTES:
            LITELLM_ROUTES_OPENAI_ROUTES.append(entry)
            verbose_proxy_logger.debug("feelfish: allowed route %s", entry)


def validate_feelfish_deployments(model_list: Optional[Iterable[Any]]) -> None:
    """
    Validate every ``feelfish`` deployment in the model list.

    Raises:
        ValueError: listing every problem found, so a broken deployment cannot
            reach production and start billing incorrectly.
    """
    problems: List[str] = []
    for deployment in model_list or []:
        as_dict = _as_dict(deployment)
        litellm_params: Dict[str, Any] = as_dict.get("litellm_params") or {}
        if litellm_params.get("custom_llm_provider") != FEELFISH_PROVIDER:
            continue
        problems.extend(_validate_one(as_dict, litellm_params))

    if problems:
        raise ValueError("invalid feelfish deployment(s): " + "; ".join(problems))


def _validate_one(
    deployment: Dict[str, Any], litellm_params: Dict[str, Any]
) -> List[str]:
    model_name = str(
        deployment.get("model_name") or litellm_params.get("model") or "<unknown>"
    )
    problems: List[str] = []

    # The gateway model name is a path segment: "/" would make the route unreachable
    # and the auth chain would resolve a different model than the handler.
    if "/" in model_name:
        problems.append(
            f"{model_name}: model_name must not contain '/' (it is used as a URL path segment)"
        )

    if not litellm_params.get("api_base"):
        problems.append(f"{model_name}: api_base is required")

    try:
        UpstreamAuth.parse(litellm_params.get("auth_style"))
    except ValueError as exc:
        problems.append(f"{model_name}: {exc}")

    try:
        resolve_passthrough_config(deployment)
    except ValueError as exc:
        problems.append(f"{model_name}: {exc}")

    try:
        parse_billing_config(_billing_config_source(deployment, litellm_params))
    except ValueError as exc:
        problems.append(f"{model_name}: {exc}")

    return problems


def _billing_config_source(
    deployment: Dict[str, Any], litellm_params: Dict[str, Any]
) -> Any:
    """Deployment config first, then the model cost map."""
    raw = resolve_field_pricing(deployment)
    if raw is None:
        raw = cost_map_field_pricing(litellm_params.get("model"))
    return raw


def _as_dict(deployment: Any) -> Dict[str, Any]:
    if isinstance(deployment, dict):
        return deployment
    dump = getattr(deployment, "model_dump", None)
    if callable(dump):
        return dump()
    return {}


__all__ = [
    "ROUTE_ALLOWLIST_ENTRIES",
    "register_feelfish_routes",
    "validate_feelfish_deployments",
]
