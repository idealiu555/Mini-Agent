"""Tests for LLM wrapper api_base behavior."""

from mini_agent.llm.llm_wrapper import LLMClient
from mini_agent.schema import LLMProvider


class _DummyClient:
    def __init__(self, api_key, api_base, model, retry_config):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config
        self.retry_callback = None

    async def generate(self, messages, tools=None):
        raise NotImplementedError


def test_api_base_used_as_is_for_anthropic(monkeypatch):
    captured = {}

    def fake_anthropic_client(api_key, api_base, model, retry_config):
        captured["api_base"] = api_base
        return _DummyClient(api_key, api_base, model, retry_config)

    monkeypatch.setattr("mini_agent.llm.llm_wrapper.AnthropicClient", fake_anthropic_client)

    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.ANTHROPIC,
        api_base="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        model="test-model",
    )

    assert client.api_base == "https://coding.dashscope.aliyuncs.com/apps/anthropic"
    assert captured["api_base"] == "https://coding.dashscope.aliyuncs.com/apps/anthropic"


def test_api_base_used_as_is_for_openai(monkeypatch):
    captured = {}

    def fake_openai_client(api_key, api_base, model, retry_config):
        captured["api_base"] = api_base
        return _DummyClient(api_key, api_base, model, retry_config)

    monkeypatch.setattr("mini_agent.llm.llm_wrapper.OpenAIClient", fake_openai_client)

    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.OPENAI,
        api_base="https://coding.dashscope.aliyuncs.com/v1/",
        model="test-model",
    )

    assert client.api_base == "https://coding.dashscope.aliyuncs.com/v1"
    assert captured["api_base"] == "https://coding.dashscope.aliyuncs.com/v1"
