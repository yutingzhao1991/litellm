"""
流式转发：仅含 reasoning_details（如 signature）的 OpenAI 兼容帧不得被丢弃。
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


@pytest.fixture
def openai_stream_wrapper() -> CustomStreamWrapper:
    return CustomStreamWrapper(
        completion_stream=None,
        model="claude-opus-4-6",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )


class TestStreamingReasoningDetailsForward:
    def test_should_forward_chunk_with_only_reasoning_details_signature(
        self, openai_stream_wrapper: CustomStreamWrapper
    ):
        """与上游仅下发 signature、无 reasoning_content 的 delta 行为一致。"""
        chunk = ModelResponseStream(
            id="chatcmpl-test-sig",
            created=1,
            model="claude-opus-4-6",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="",
                        reasoning_details={
                            "type": "thinking",
                            "signature": "SIG_FROM_UPSTREAM",
                        },
                    ),
                    finish_reason=None,
                )
            ],
        )
        out = openai_stream_wrapper.chunk_creator(chunk)
        assert out is not None
        rd = getattr(out.choices[0].delta, "reasoning_details", None)
        assert rd is not None
        assert rd.get("signature") == "SIG_FROM_UPSTREAM"

    def test_should_forward_chunk_with_reasoning_details_type_only(
        self, openai_stream_wrapper: CustomStreamWrapper
    ):
        chunk = ModelResponseStream(
            id="chatcmpl-test-type",
            created=1,
            model="claude-opus-4-6",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content="",
                        reasoning_details={"type": "thinking"},
                    ),
                    finish_reason=None,
                )
            ],
        )
        out = openai_stream_wrapper.chunk_creator(chunk)
        assert out is not None
        rd = getattr(out.choices[0].delta, "reasoning_details", None)
        assert rd == {"type": "thinking"}
