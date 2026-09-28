"""
Auth-chain integration for the FeelFish pass-through provider.

These tests lock the two wiring points that keep pass-through models from
bypassing key model permissions or budgets:

- ``get_model_from_request`` resolves the model from the ``/feelfish/{model}`` path,
- ``_is_model_cost_zero`` treats response-field pricing as "has cost".
"""

import asyncio
from typing import Any, Dict, List

import pytest

import litellm
from litellm import Router
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _is_model_cost_zero, can_key_call_model
from litellm.proxy.auth.auth_utils import get_model_from_request
from litellm.proxy.auth.user_api_key_auth import (
    _resolve_model_for_budget_checks,
)

BILLING = {
    "rules": [
        {
            "from": "response_body",
            "path": "usage.text_words",
            "aggregate": "last",
            "unit_price": 0.0005,
            "required": True,
        }
    ]
}


def _feelfish_model_list(model_name: str = "doubao-tts") -> List[Dict[str, Any]]:
    return [
        {
            "model_name": model_name,
            "litellm_params": {
                "model": model_name,
                "custom_llm_provider": "feelfish",
                "api_base": "https://upstream.example.com/api/v3/tts/unidirectional",
                "api_key": "upstream-secret",
                "auth_style": "header:X-Api-Key",
                "use_in_pass_through": True,
            },
            "model_info": {"id": "deployment-id-tts", "response_field_pricing": BILLING},
        }
    ]


def _router(model_list: List[Dict[str, Any]]) -> Router:
    return Router(model_list=model_list, ignore_invalid_deployments=True)


# --------------------------------------------------------------------------- #
# model resolution
# --------------------------------------------------------------------------- #


def test_feelfish_path_is_the_model_source():
    assert get_model_from_request({}, "/feelfish/doubao-tts") == "doubao-tts"
    assert (
        get_model_from_request({}, "/feelfish/doubao-tts/extra/path") == "doubao-tts"
    )


def test_feelfish_path_wins_over_a_body_model_field():
    # The body belongs to the upstream protocol and may carry its own model.
    assert (
        get_model_from_request(
            {"model": "upstream-owned-model"}, "/feelfish/doubao-tts"
        )
        == "doubao-tts"
    )


def test_non_feelfish_routes_are_unchanged():
    assert (
        get_model_from_request({"model": "gpt-4o"}, "/chat/completions") == "gpt-4o"
    )
    assert (
        get_model_from_request({}, "/openai/deployments/my-model/chat/completions")
        == "my-model"
    )


def test_per_model_budget_checks_use_the_path_model():
    # both per-model budget paths must resolve the same model as key permissions
    assert _resolve_model_for_budget_checks({}, "/feelfish/doubao-tts") == "doubao-tts"
    assert (
        _resolve_model_for_budget_checks(
            {"model": "upstream-owned-model"}, "/feelfish/doubao-tts"
        )
        == "doubao-tts"
    )
    assert (
        _resolve_model_for_budget_checks({"model": "gpt-4o"}, "/chat/completions")
        == "gpt-4o"
    )
    # comma separated lists keep the historical raw-string behaviour
    assert (
        _resolve_model_for_budget_checks({"model": "a,b"}, "/chat/completions")
        == "a,b"
    )


# --------------------------------------------------------------------------- #
# budget checks
# --------------------------------------------------------------------------- #


def test_response_field_pricing_is_not_treated_as_free():
    router = _router(_feelfish_model_list())
    assert _is_model_cost_zero(model="doubao-tts", llm_router=router) is False


def test_pricing_from_the_cost_map_is_not_treated_as_free():
    model_name = "feelfish-cost-map-tts"
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "feelfish",
                "mode": "audio_speech",
                "response_field_pricing": BILLING,
            }
        }
    )
    router = _router(
        [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "custom_llm_provider": "feelfish",
                    "api_base": "https://upstream.example.com/tts",
                    "api_key": "k",
                    "use_in_pass_through": True,
                },
                "model_info": {"id": f"deployment-{model_name}"},
            }
        ]
    )
    assert _is_model_cost_zero(model=model_name, llm_router=router) is False


def test_genuinely_free_model_still_skips_budget_checks():
    router = _router(
        [
            {
                "model_name": "free-model",
                "litellm_params": {"model": "openai/free-model", "api_key": "k"},
                "model_info": {
                    "id": "deployment-free",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            }
        ]
    )
    assert _is_model_cost_zero(model="free-model", llm_router=router) is True


def test_token_priced_model_is_not_free():
    router = _router(
        [
            {
                "model_name": "paid-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "k"},
                "model_info": {"id": "deployment-paid"},
            }
        ]
    )
    assert _is_model_cost_zero(model="paid-model", llm_router=router) is False


# --------------------------------------------------------------------------- #
# key model permissions
# --------------------------------------------------------------------------- #


def test_key_model_permission_applies_to_feelfish_models():
    router = _router(_feelfish_model_list())
    token = UserAPIKeyAuth(api_key="sk-test", models=["some-other-model"])

    with pytest.raises(ProxyException) as excinfo:
        asyncio.run(
            can_key_call_model(
                model="doubao-tts",
                llm_model_list=_feelfish_model_list(),
                valid_token=token,
                llm_router=router,
            )
        )
    assert "not allowed to access model" in str(excinfo.value.message)


def test_key_with_matching_model_is_allowed():
    router = _router(_feelfish_model_list())
    token = UserAPIKeyAuth(api_key="sk-test", models=["doubao-tts"])
    assert (
        asyncio.run(
            can_key_call_model(
                model="doubao-tts",
                llm_model_list=_feelfish_model_list(),
                valid_token=token,
                llm_router=router,
            )
        )
        is True
    )


# --------------------------------------------------------------------------- #
# router integration
# --------------------------------------------------------------------------- #


def test_feelfish_deployments_are_registered_for_pass_through():
    router = _router(_feelfish_model_list())
    deployments = router.get_model_list(model_name="doubao-tts")
    assert deployments is not None and len(deployments) == 1
    assert deployments[0]["litellm_params"]["custom_llm_provider"] == "feelfish"

    resolved = asyncio.run(
        router.async_get_available_deployment_for_pass_through(
            model="doubao-tts", request_kwargs={}
        )
    )
    assert resolved is not None
    assert resolved["model_name"] == "doubao-tts"


def test_feelfish_models_cannot_be_called_through_the_completion_path():
    """
    Guard against a pass-through model leaking into the normal LLM path (which
    would surface a confusing error and, worse, an unintended billing path).
    """
    router = _router(_feelfish_model_list())
    with pytest.raises(Exception):
        # get_llm_provider has no completion implementation for `feelfish`
        litellm.completion(
            model="doubao-tts",
            messages=[{"role": "user", "content": "hi"}],
            api_key="k",
            api_base="https://upstream.example.com/tts",
        )
    assert router is not None
