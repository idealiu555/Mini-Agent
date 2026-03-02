"""Integration tests for the MiniMax ACP adapter."""

from types import SimpleNamespace

import pytest

from mini_agent.acp import MiniMaxACPAgent, run_acp_server
from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.schema import FunctionCall, LLMProvider, LLMResponse, ToolCall
from mini_agent.tools.base import Tool, ToolResult


class DummyConn:
    def __init__(self):
        self.updates = []

    async def sessionUpdate(self, payload):
        self.updates.append(payload)


class DummyLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                thinking="calling echo",
                tool_calls=[
                    ToolCall(
                        id="tool1",
                        type="function",
                        function=FunctionCall(name="echo", arguments={"text": "ping"}),
                    )
                ],
                finish_reason="tool",
            )
        return LLMResponse(content="done", thinking=None, tool_calls=None, finish_reason="stop")


class EchoTool(Tool):
    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "Echo helper"

    @property
    def parameters(self):
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, text: str):
        return ToolResult(success=True, content=f"tool:{text}")


@pytest.fixture
def acp_agent(tmp_path):
    config = Config(
        llm=LLMConfig(api_key="test-key"),
        agent=AgentConfig(max_steps=3, workspace_dir=str(tmp_path)),
        tools=ToolsConfig(),
    )
    conn = DummyConn()
    agent = MiniMaxACPAgent(conn, config, DummyLLM(), [EchoTool()], "system")
    return agent, conn


@pytest.mark.asyncio
async def test_acp_turn_executes_tool(acp_agent):
    agent, conn = acp_agent
    session = await agent.newSession(SimpleNamespace(cwd=None))
    prompt = SimpleNamespace(sessionId=session.sessionId, prompt=[{"text": "hello"}])
    response = await agent.prompt(prompt)
    assert response.stopReason == "end_turn"
    assert any("tool:ping" in str(update) for update in conn.updates)
    await agent.cancel(SimpleNamespace(sessionId=session.sessionId))
    assert agent._sessions[session.sessionId].cancelled


@pytest.mark.asyncio
async def test_acp_invalid_session(acp_agent):
    agent, _ = acp_agent
    prompt = SimpleNamespace(sessionId="missing", prompt=[{"text": "?"}])
    response = await agent.prompt(prompt)
    assert response.stopReason == "refusal"


@pytest.mark.asyncio
async def test_run_acp_server_uses_configured_provider(monkeypatch):
    """ACP server should pass configured provider into LLMClient."""
    captured_kwargs = {}

    class FakeLLMClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    async def fake_initialize_base_tools(_config):
        return [], None

    async def fake_stdio_streams():
        raise RuntimeError("stop-after-llm-init")

    monkeypatch.setattr("mini_agent.acp.LLMClient", FakeLLMClient)
    monkeypatch.setattr("mini_agent.acp.initialize_base_tools", fake_initialize_base_tools)
    monkeypatch.setattr("mini_agent.acp.stdio_streams", fake_stdio_streams)
    monkeypatch.setattr("mini_agent.acp.Config.find_config_file", classmethod(lambda _cls, _filename: None))

    config = Config(
        llm=LLMConfig(api_key="test-key", provider="openai"),
        agent=AgentConfig(),
        tools=ToolsConfig(),
    )

    with pytest.raises(RuntimeError, match="stop-after-llm-init"):
        await run_acp_server(config)

    assert captured_kwargs["provider"] == LLMProvider.OPENAI
