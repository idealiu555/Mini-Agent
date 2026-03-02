"""Tests for CLI helper behaviors."""

import asyncio
import shutil
from pathlib import Path

import pytest

from mini_agent.cli import _quiet_cleanup, add_workspace_tools, resolve_provider
from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.schema import LLMProvider


def test_resolve_provider_valid_values():
    """Provider parser should accept known values with normalization."""
    assert resolve_provider("anthropic") == LLMProvider.ANTHROPIC
    assert resolve_provider(" OPENAI ") == LLMProvider.OPENAI


def test_resolve_provider_rejects_invalid_value():
    """Provider parser should fail fast on unsupported values."""
    with pytest.raises(ValueError, match="Unsupported provider"):
        resolve_provider("anthropicx")


@pytest.mark.asyncio
async def test_quiet_cleanup_keeps_quiet_exception_handler(monkeypatch):
    """Cleanup should keep a quiet handler for asyncio.run() shutdown noise."""

    async def fake_cleanup():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr("mini_agent.cli.cleanup_mcp_connections", fake_cleanup)

    loop = asyncio.get_running_loop()

    def original_handler(_loop, _context):
        return None

    loop.set_exception_handler(original_handler)
    try:
        await _quiet_cleanup()
        current_handler = loop.get_exception_handler()
        assert current_handler is not None
        assert current_handler is not original_handler
    finally:
        loop.set_exception_handler(None)


def test_add_workspace_tools_loads_record_and_recall_note_tools():
    """Workspace tools should include both note record and recall tools."""
    config = Config(
        llm=LLMConfig(api_key="test-key"),
        agent=AgentConfig(),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=True,
            enable_skills=False,
            enable_mcp=False,
        ),
    )

    tools = []
    workspace_dir = Path("workspace") / "test_cli_note_tools"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    try:
        add_workspace_tools(tools, config, workspace_dir)
        tool_names = [tool.name for tool in tools]
        assert "record_note" in tool_names
        assert "recall_notes" in tool_names
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
