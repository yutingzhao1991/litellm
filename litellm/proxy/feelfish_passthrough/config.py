"""
Configuration models for the FeelFish pass-through provider.

Two independent configuration blocks:

- ``PassthroughConfig`` / ``UpstreamAuth`` live on the deployment
  (``litellm_params``) and describe *how* to talk to the upstream.
- ``BillingConfig`` describes *how to price* a response and may be declared
  either on the deployment (``model_info`` / ``litellm_params``) or in the
  model cost map served through ``LITELLM_MODEL_COST_MAP_URL``.

Billing config resolution is intentionally "whole block wins": if the
deployment declares ``response_field_pricing``, the cost map entry for the same
model is ignored entirely (LiteLLM's own ``_update_dictionary`` only does a
shallow merge, which would silently mix rule lists).
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RESPONSE_FIELD_PRICING_KEY = "response_field_pricing"

DEFAULT_STREAM_BILLING_BUFFER_BYTES = 262144


class FieldSource(str, Enum):
    """Where a billing rule reads its value from."""

    RESPONSE_BODY = "response_body"
    RESPONSE_TEXT = "response_text"
    RESPONSE_HEADERS = "response_headers"
    REQUEST_BODY = "request_body"


class Aggregate(str, Enum):
    """How to collapse multiple values (wildcards / NDJSON lines)."""

    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


class Transform(str, Enum):
    """Value normalization applied after aggregation."""

    NONE = "none"
    CHARACTERS = "characters"
    CHARACTERS_NO_SPACE = "characters_no_space"
    BYTES = "bytes"
    MS_TO_SECONDS = "ms_to_seconds"
    COUNT_ITEMS = "count_items"
    TRUTHY = "truthy"


class MissingPolicy(str, Enum):
    """What to do when a required rule has no value."""

    FALLBACK = "fallback"
    ZERO = "zero"


class AuthStyle(str, Enum):
    """How the upstream credential is injected."""

    BEARER = "bearer"
    HEADER = "header"
    QUERY = "query"
    BASIC = "basic"
    NONE = "none"


class BillingRule(BaseModel):
    """One ``field x unit_price`` term of a response-field pricing config."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_: FieldSource = Field(default=FieldSource.RESPONSE_BODY, alias="from")
    path: str
    aggregate: Aggregate = Aggregate.FIRST
    transform: Transform = Transform.NONE
    scale: float = 1.0
    unit: Optional[str] = None
    unit_price: float = 0.0
    required: bool = False

    @field_validator("path")
    @classmethod
    def _path_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("billing rule 'path' must be a non-empty string")
        return value.strip()

    @field_validator("scale", "unit_price")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        if value < 0:
            raise ValueError("must be >= 0")
        return value


class BillingConfig(BaseModel):
    """Response-field pricing for one model."""

    model_config = ConfigDict(extra="forbid")

    rules: List[BillingRule] = Field(min_length=1)
    fallback: List[BillingRule] = Field(default_factory=list)
    on_missing: MissingPolicy = MissingPolicy.FALLBACK
    multiplier: float = 1.0
    min_cost: float = 0.0
    cost_per_request: float = 0.0
    bill_on_error: bool = False
    expose_breakdown: bool = False
    buffer_stream_bytes: int = DEFAULT_STREAM_BILLING_BUFFER_BYTES

    @field_validator("multiplier", "min_cost", "cost_per_request")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("buffer_stream_bytes")
    @classmethod
    def _positive_buffer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("buffer_stream_bytes must be > 0")
        return value


class PassthroughConfig(BaseModel):
    """How the incoming request is forwarded to the upstream."""

    model_config = ConfigDict(extra="forbid")

    strip_body_fields: List[str] = Field(default_factory=list)
    forward_headers: List[str] = Field(default_factory=list)
    timeout: Optional[float] = None


class UpstreamAuth(BaseModel):
    """Parsed ``auth_style`` deployment setting."""

    model_config = ConfigDict(extra="forbid")

    style: AuthStyle = AuthStyle.BEARER
    name: Optional[str] = None

    @classmethod
    def parse(cls, raw: Optional[str]) -> "UpstreamAuth":
        value = (raw or "").strip()
        if not value:
            return cls()
        lowered = value.lower()
        if lowered in (AuthStyle.BEARER.value, AuthStyle.BASIC.value, AuthStyle.NONE.value):
            return cls(style=AuthStyle(lowered))
        if lowered.startswith("header:"):
            name = value.split(":", 1)[1].strip()
            if not name:
                raise ValueError("auth_style 'header:' requires a header name")
            return cls(style=AuthStyle.HEADER, name=name)
        if lowered.startswith("query:"):
            name = value.split(":", 1)[1].strip()
            if not name:
                raise ValueError("auth_style 'query:' requires a query parameter name")
            return cls(style=AuthStyle.QUERY, name=name)
        raise ValueError(
            "auth_style must be one of: bearer, basic, none, header:<Name>, query:<Name>"
        )


def parse_billing_config(raw: Any) -> BillingConfig:
    """
    Validate a raw ``response_field_pricing`` block.

    Raises:
        ValueError: when the block is missing or invalid. Callers translate this
            into a startup failure or a per-request fail-closed error.
    """
    if raw is None:
        raise ValueError(f"{RESPONSE_FIELD_PRICING_KEY} is not configured")
    try:
        return BillingConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid {RESPONSE_FIELD_PRICING_KEY}: {exc}") from exc


def resolve_field_pricing(deployment: Dict[str, Any]) -> Any:
    """
    Return the raw ``response_field_pricing`` block for a deployment.

    Precedence: ``model_info`` > ``litellm_params``. The cost map entry is
    merged into the deployment model info by the caller (router lookup), so it
    is already part of ``model_info`` when present.
    """
    for container_key in ("model_info", "litellm_params"):
        container = deployment.get(container_key) or {}
        raw = container.get(RESPONSE_FIELD_PRICING_KEY)
        if raw is not None:
            return raw
    return None


def resolve_passthrough_config(deployment: Dict[str, Any]) -> PassthroughConfig:
    """Parse the deployment ``passthrough`` block (all fields optional)."""
    litellm_params = deployment.get("litellm_params") or {}
    raw = litellm_params.get("passthrough") or {}
    try:
        return PassthroughConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid passthrough config: {exc}") from exc


def cost_map_field_pricing(model: Optional[str]) -> Any:
    """
    Read ``response_field_pricing`` from the model cost map.

    ``get_model_info`` returns the whitelisted ``ModelInfoBase`` built from the
    cost map entry, which now carries this key (see ``_get_model_info_helper``).
    """
    if not model:
        return None
    try:
        model_info = litellm.get_model_info(model=model)
    except Exception:
        return None
    if not isinstance(model_info, dict):
        return None
    return model_info.get(RESPONSE_FIELD_PRICING_KEY)
