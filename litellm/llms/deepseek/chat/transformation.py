"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


# Content part types that carry image bytes. Flattening a content list is only safe
# for text-only messages; DeepSeek's vision models accept the OpenAI multi-part shape.
_IMAGE_CONTENT_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _has_image_content(message: AllMessageValues) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in _IMAGE_CONTENT_PART_TYPES
        for part in content
    )


def _flatten_text_only_message_content(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    for message in messages:
        if _has_image_content(message):
            continue
        texts = convert_content_list_to_str(message=message)
        if texts:
            message["content"] = texts
    return messages


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to DeepSeek params.

        Handles `thinking` and `reasoning_effort` parameters for DeepSeek reasoner models.
        DeepSeek only supports `{"type": "enabled"}` - no budget_tokens like Anthropic.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - only accept {"type": "enabled"}
        if thinking_value is not None:
            if (
                isinstance(thinking_value, dict)
                and thinking_value.get("type") == "enabled"
            ):
                # DeepSeek only accepts {"type": "enabled"}, ignore budget_tokens
                optional_params["thinking"] = {"type": "enabled"}

        # Handle reasoning_effort - map to thinking enabled
        elif reasoning_effort is not None and reasoning_effort != "none":
            optional_params["thinking"] = {"type": "enabled"}

        return optional_params

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Flatten text-only content lists, but keep image parts.

        DeepSeek's chat API accepts OpenAI-style multi-part content for its vision
        models. The old blanket flattening silently dropped every image part, so a
        request with an attached image reached the model as text-only and the model
        answered "no image was provided" while the caller was still billed.
        """
        messages = _flatten_text_only_message_content(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
