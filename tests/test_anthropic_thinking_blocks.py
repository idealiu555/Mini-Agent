"""Unit tests for Anthropic thinking block replay support."""

from types import SimpleNamespace

from mini_agent.llm.anthropic_client import AnthropicClient
from mini_agent.schema import Message


class _FakeThinkingBlock:
    type = "thinking"

    def __init__(self, thinking: str, signature: str):
        self.thinking = thinking
        self.signature = signature

    def model_dump(self, exclude_none: bool = True):  # noqa: ARG002
        return {
            "type": "thinking",
            "thinking": self.thinking,
            "signature": self.signature,
        }


def test_convert_messages_replays_raw_thinking_blocks():
    """Assistant thinking blocks should be passed through unchanged."""
    client = AnthropicClient(api_key="test-key", api_base="https://example.com/anthropic", model="test-model")

    replay_blocks = [{"type": "thinking", "thinking": "step-by-step", "signature": "sig-123"}]
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="done", thinking="step-by-step", thinking_blocks=replay_blocks),
    ]

    _, api_messages = client._convert_messages(messages)

    assistant_msg = api_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0] == replay_blocks[0]


def test_parse_response_preserves_replayable_thinking_blocks():
    """Parser should preserve raw thinking blocks (including signature)."""
    client = AnthropicClient(api_key="test-key", api_base="https://example.com/anthropic", model="test-model")

    response = SimpleNamespace(
        content=[
            _FakeThinkingBlock("reasoning text", "sig-abc"),
            SimpleNamespace(type="text", text="final answer"),
        ],
        usage=None,
        stop_reason="end_turn",
    )

    parsed = client._parse_response(response)

    assert parsed.thinking == "reasoning text"
    assert parsed.thinking_blocks == [{"type": "thinking", "thinking": "reasoning text", "signature": "sig-abc"}]
    assert parsed.content == "final answer"
