from unittest.mock import AsyncMock

import litellm
import pytest

from litellm.router_utils.error_classification import (
    is_deterministic_tool_schema_error,
)


def bad_request(message: str, body=None) -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=message,
        model="test-model",
        llm_provider="openai",
        body=body,
    )


def test_classifies_openai_invalid_function_parameters() -> None:
    error = bad_request(
        "Invalid schema for function 'todo_write'",
        body={"code": "invalid_function_parameters"},
    )

    assert is_deterministic_tool_schema_error(error) is True


def test_classifies_gemini_function_declaration_schema_error() -> None:
    error = bad_request(
        "GenerateContentRequest.tools[0].function_declarations[44].parameters."
        "any_of[0].required[0]: property is not defined"
    )

    assert is_deterministic_tool_schema_error(error) is True


def test_does_not_classify_unrelated_bad_request() -> None:
    error = bad_request("The requested model does not support this parameter")

    assert is_deterministic_tool_schema_error(error) is False


def test_does_not_classify_non_bad_request() -> None:
    assert is_deterministic_tool_schema_error(ValueError("invalid schema for function")) is False


def create_test_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/test-model"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/fallback-model"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=2,
        retry_policy={"BadRequestErrorRetries": 2},
    )


@pytest.mark.asyncio
async def test_deterministic_schema_error_bypasses_retry_policy() -> None:
    router = create_test_router()
    error = bad_request(
        "Invalid schema for function 'todo_write'",
        body={"code": "invalid_function_parameters"},
    )
    original_function = AsyncMock(side_effect=error)

    with pytest.raises(litellm.BadRequestError) as raised:
        await router.async_function_with_retries(
            original_function=original_function,
            model="primary",
            messages=[{"role": "user", "content": "hello"}],
            num_retries=2,
            metadata={},
        )

    assert raised.value is error
    original_function.assert_awaited_once()


@pytest.mark.asyncio
async def test_deterministic_schema_error_bypasses_model_fallback(monkeypatch) -> None:
    router = create_test_router()
    error = bad_request(
        "GenerateContentRequest.tools[0].function_declarations[44].parameters."
        "any_of[0].required[0]: property is not defined"
    )
    fallback = AsyncMock()
    monkeypatch.setattr("litellm.router.run_async_fallback", fallback)

    with pytest.raises(litellm.BadRequestError) as raised:
        await router.async_function_with_fallbacks_common_utils(
            e=error,
            disable_fallbacks=False,
            fallbacks=[{"primary": ["fallback"]}],
            context_window_fallbacks=None,
            content_policy_fallbacks=None,
            model_group="primary",
            args=(),
            kwargs={"model": "primary"},
        )

    assert raised.value is error
    assert "Error doing the fallback" not in error.message
    fallback.assert_not_awaited()
