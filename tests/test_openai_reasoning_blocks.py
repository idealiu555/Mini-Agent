"""Unit tests for OpenAI reasoning block replay support."""

from types import SimpleNamespace

from mini_agent.llm.openai_client import OpenAIClient
from mini_agent.schema import Message


class _FakeReasoningDetail:
    def __init__(self, text: str, signature: str):
        self.type = "reasoning"
        self.text = text
        self.signature = signature

    def model_dump(self, exclude_none: bool = True):  # noqa: ARG002
        return {
            "type": self.type,
            "text": self.text,
            "signature": self.signature,
        }


def test_convert_messages_replays_raw_reasoning_blocks():
    """Assistant reasoning blocks should be passed through unchanged."""
    client = OpenAIClient(api_key="test-key", api_base="https://example.com/v1", model="test-model")

    replay_blocks = [{"type": "reasoning", "text": "chain", "signature": "sig-1"}]
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="done", thinking="chain", thinking_blocks=replay_blocks),
    ]

    _, api_messages = client._convert_messages(messages)

    assistant_msg = api_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_details"] == replay_blocks


def test_parse_response_preserves_replayable_reasoning_blocks():
    """Parser should preserve raw reasoning blocks and aggregate text."""
    client = OpenAIClient(api_key="test-key", api_base="https://example.com/v1", model="test-model")

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="answer",
                    reasoning_details=[_FakeReasoningDetail("reason", "sig-2")],
                    tool_calls=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    parsed = client._parse_response(response)

    assert parsed.thinking == "reason"
    assert parsed.thinking_blocks == [{"type": "reasoning", "text": "reason", "signature": "sig-2"}]
    assert parsed.content == "answer"


def test_parse_response_tolerates_malformed_tool_arguments():
    """Parser should not crash when tool arguments are malformed JSON."""
    client = OpenAIClient(api_key="test-key", api_base="https://example.com/v1", model="test-model")

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    reasoning_details=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc-1",
                            function=SimpleNamespace(
                                name="echo",
                                arguments='{"incomplete":',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )

    parsed = client._parse_response(response)
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0].function.arguments == {"_raw_arguments": '{"incomplete":'}
