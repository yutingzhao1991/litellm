"""
FeelFish pass-through provider: verbatim model proxy + response-field billing.

See ``docs/superpowers/specs/2026-09-28-feelfish-provider-design.md`` in the
FeelFish desktop repository for the full design.
"""

from .billing import BillingContext, BillingResult, evaluate_billing
from .bootstrap import register_feelfish_routes, validate_feelfish_deployments
from .config import (
    BillingConfig,
    BillingRule,
    PassthroughConfig,
    UpstreamAuth,
    parse_billing_config,
)
from .route import FeelfishDeps, get_feelfish_deps, router
from .spend import SpendRecord, SpendRecorder, record_feelfish_spend

__all__ = [
    "BillingConfig",
    "BillingContext",
    "BillingResult",
    "BillingRule",
    "FeelfishDeps",
    "PassthroughConfig",
    "SpendRecord",
    "SpendRecorder",
    "UpstreamAuth",
    "evaluate_billing",
    "get_feelfish_deps",
    "parse_billing_config",
    "record_feelfish_spend",
    "register_feelfish_routes",
    "router",
    "validate_feelfish_deployments",
]
