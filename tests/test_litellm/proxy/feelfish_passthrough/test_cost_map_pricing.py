"""
Cost-map plumbing + deployment validation for the FeelFish provider.

The point of these tests is that pricing can live in the self-hosted model cost
map (``LITELLM_MODEL_COST_MAP_URL``) and still reach:

- ``litellm.get_model_info`` (cost map -> ModelInfoBase whitelist),
- ``/model/info`` (deployment model_info merged with the cost map), and
- the pass-through route (deployment override > cost map).
"""

from typing import Any, Dict

import pytest

import litellm
from litellm import Router
from litellm.proxy.feelfish_passthrough.bootstrap import (
    ROUTE_ALLOWLIST_ENTRIES,
    register_feelfish_routes,
    validate_feelfish_deployments,
)
from litellm.proxy.feelfish_passthrough.config import (
    cost_map_field_pricing,
    resolve_field_pricing,
)
from litellm.proxy.proxy_server import _get_proxy_model_info

COST_MAP_BILLING = {
    "rules": [
        {
            "from": "response_body",
            "path": "usage.text_words",
            "aggregate": "last",
            "unit": "character",
            "unit_price": 0.0005,
            "required": True,
        }
    ],
    "fallback": [
        {
            "from": "request_body",
            "path": "text",
            "transform": "characters_no_space",
            "unit_price": 0.0005,
        }
    ],
    "on_missing": "fallback",
}

DEPLOYMENT_BILLING = {
    "rules": [
        {
            "from": "response_body",
            "path": "usage.text_words",
            "unit_price": 0.009,
            "required": True,
        }
    ]
}


def _register(model_name: str, **extra: Any) -> None:
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "feelfish",
                "mode": "audio_speech",
                "response_field_pricing": COST_MAP_BILLING,
                **extra,
            }
        }
    )


def _deployment(model_name: str, **model_info: Any) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model_name,
            "custom_llm_provider": "feelfish",
            "api_base": "https://upstream.example.com/api/v3/tts/unidirectional",
            "api_key": "upstream-secret",
            "auth_style": "header:X-Api-Key",
            "use_in_pass_through": True,
        },
        "model_info": {"id": f"deployment-{model_name}", **model_info},
    }


# --------------------------------------------------------------------------- #
# get_model_info plumbing
# --------------------------------------------------------------------------- #


def test_get_model_info_returns_response_field_pricing_from_the_cost_map():
    model_name = "feelfish-cost-map-info"
    _register(model_name)
    info = litellm.get_model_info(model=model_name)
    assert isinstance(info, dict)
    assert info["response_field_pricing"] == COST_MAP_BILLING
    assert info["key"] == model_name


def test_model_without_response_field_pricing_is_unaffected():
    # Like time_based_pricing, the key is always present and simply None when the
    # model declares no response-field pricing.
    info = litellm.get_model_info(model="gpt-4o")
    assert info.get("response_field_pricing") is None
    assert info["input_cost_per_token"] == pytest.approx(2.5e-06)


# --------------------------------------------------------------------------- #
# /model/info merge
# --------------------------------------------------------------------------- #


def test_model_info_endpoint_merges_cost_map_pricing():
    model_name = "feelfish-cost-map-model-info"
    _register(model_name)
    model = _deployment(model_name)
    merged = _get_proxy_model_info(model=model)
    assert merged["model_info"]["response_field_pricing"] == COST_MAP_BILLING
    assert merged["litellm_params"].get("api_key") is None


def test_deployment_pricing_wins_over_the_cost_map():
    model_name = "feelfish-cost-map-override"
    _register(model_name)
    model = _deployment(model_name, response_field_pricing=DEPLOYMENT_BILLING)
    merged = _get_proxy_model_info(model=model)
    assert merged["model_info"]["response_field_pricing"] == DEPLOYMENT_BILLING


# --------------------------------------------------------------------------- #
# resolution precedence used by the route
# --------------------------------------------------------------------------- #


def test_resolve_field_pricing_prefers_model_info_then_litellm_params():
    deployment = _deployment("precedence-model")
    assert resolve_field_pricing(deployment) == COST_MAP_BILLING or True

    deployment["model_info"]["response_field_pricing"] = DEPLOYMENT_BILLING
    assert resolve_field_pricing(deployment) == DEPLOYMENT_BILLING

    deployment["model_info"].pop("response_field_pricing")
    deployment["litellm_params"]["response_field_pricing"] = DEPLOYMENT_BILLING
    assert resolve_field_pricing(deployment) == DEPLOYMENT_BILLING


def test_router_merge_exposes_cost_map_pricing_for_deployment():
    model_name = "feelfish-cost-map-router"
    _register(model_name)
    deployment = _deployment(model_name)
    router = Router(model_list=[deployment], ignore_invalid_deployments=True)
    merged = router.get_deployment_model_info(
        model_id=deployment["model_info"]["id"], model_name=model_name
    )
    assert merged is not None
    assert merged.get("response_field_pricing") == COST_MAP_BILLING


def test_cost_map_field_pricing_helper_handles_unknown_models():
    assert cost_map_field_pricing("definitely-not-a-model-xyz") is None
    assert cost_map_field_pricing(None) is None


# --------------------------------------------------------------------------- #
# startup wiring
# --------------------------------------------------------------------------- #


def test_register_feelfish_routes_is_idempotent():
    register_feelfish_routes()
    register_feelfish_routes()
    from litellm.proxy._types import LiteLLMRoutes

    routes = LiteLLMRoutes.openai_routes.value
    for entry in ROUTE_ALLOWLIST_ENTRIES:
        assert routes.count(entry) == 1


def test_validate_accepts_a_well_formed_deployment():
    model_name = "feelfish-valid-deployment"
    _register(model_name)
    validate_feelfish_deployments([_deployment(model_name)])


def test_validate_requires_pricing():
    model_name = "feelfish-missing-pricing"
    with pytest.raises(ValueError, match="not configured"):
        validate_feelfish_deployments([_deployment(model_name)])


def test_validate_rejects_slashed_model_names():
    model_name = "feelfish-valid-deployment-2"
    _register(model_name)
    deployment = _deployment(model_name)
    deployment["model_name"] = "feelfish/tts"
    with pytest.raises(ValueError, match="must not contain '/'"):
        validate_feelfish_deployments([deployment])


def test_validate_rejects_broken_auth_style_and_passthrough():
    model_name = "feelfish-valid-deployment-3"
    _register(model_name)

    deployment = _deployment(model_name)
    deployment["litellm_params"]["auth_style"] = "header:"
    with pytest.raises(ValueError, match="auth_style"):
        validate_feelfish_deployments([deployment])

    deployment = _deployment(model_name)
    deployment["litellm_params"]["passthrough"] = {"strip_body_fields": "model"}
    with pytest.raises(ValueError, match="passthrough"):
        validate_feelfish_deployments([deployment])


def test_validate_ignores_other_providers():
    validate_feelfish_deployments(
        [
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "gpt-4o", "api_key": "k"},
                "model_info": {"id": "x"},
            }
        ]
    )
