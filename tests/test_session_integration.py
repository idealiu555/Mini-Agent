"""
Session integration tests - Testing multi-turn conversations and session management
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.schema import LLMResponse, Message
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import ReadTool, WriteTool
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client"""
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_multi_turn_conversation(mock_llm_client, temp_workspace):
    """Test multi-turn conversation and context sharing"""
    # Prepare test data
    system_prompt = "You are an intelligent assistant"
    tools = [
        ReadTool(workspace_dir=temp_workspace),
        WriteTool(workspace_dir=temp_workspace),
        SessionNoteTool(),
    ]

    # Create agent
    agent = Agent(
        llm_client=mock_llm_client,
        system_prompt=system_prompt,
        tools=tools,
        workspace_dir=temp_workspace,
    )

    # Verify initial state
    assert len(agent.messages) == 1  # Only system prompt
    assert agent.messages[0].role == "system"
    # Agent automatically adds workspace info to system prompt
    assert system_prompt in agent.messages[0].content
    assert "Current Workspace" in agent.messages[0].content

    # Add first user message
    agent.add_user_message("Hello")
    assert len(agent.messages) == 2
    assert agent.messages[1].role == "user"
    assert agent.messages[1].content == "Hello"

    # Add second user message
    agent.add_user_message("Help me create a file")
    assert len(agent.messages) == 3
    assert agent.messages[2].role == "user"

    # Verify all messages are retained in history
    user_messages = [m for m in agent.messages if m.role == "user"]
    assert len(user_messages) == 2
    assert user_messages[0].content == "Hello"
    assert user_messages[1].content == "Help me create a file"


def test_session_history_management(mock_llm_client, temp_workspace):
    """Test session history management"""
    agent = Agent(
        llm_client=mock_llm_client,
        system_prompt="System prompt",
        tools=[],
        workspace_dir=temp_workspace,
    )

    # Add multiple messages
    for i in range(5):
        agent.add_user_message(f"Message {i}")

    # Verify message count (1 system + 5 user)
    assert len(agent.messages) == 6

    # Clear history (keep system prompt)
    agent.messages = [agent.messages[0]]

    # Verify only system prompt remains after clearing
    assert len(agent.messages) == 1
    assert agent.messages[0].role == "system"


def test_get_history(mock_llm_client, temp_workspace):
    """Test getting session history"""
    agent = Agent(
        llm_client=mock_llm_client,
        system_prompt="System",
        tools=[],
        workspace_dir=temp_workspace,
    )

    # Add message
    agent.add_user_message("Test message")

    # Get history
    history = agent.get_history()

    # Verify history is a copy (doesn't affect original messages)
    assert len(history) == len(agent.messages)
    assert history is not agent.messages

    # Modifying copy should not affect original messages
    history.append(Message(role="user", content="New message"))
    assert len(agent.messages) == 2  # Original messages unchanged
    assert len(history) == 3  # Copy changed


@pytest.mark.asyncio
async def test_session_note_persistence(temp_workspace):
    """Test SessionNoteTool persistence functionality"""
    memory_file = Path(temp_workspace) / "memory.json"

    # Create first tool instance and record note
    record_tool = SessionNoteTool(memory_file=str(memory_file))
    result1 = await record_tool.execute(content="Test note", category="test")
    assert result1.success

    # Create second tool instance (simulating new session)
    recall_tool = RecallNoteTool(memory_file=str(memory_file))

    # Verify ability to read previous notes
    result2 = await recall_tool.execute()
    assert result2.success
    assert "Test note" in result2.content


def test_message_statistics(mock_llm_client, temp_workspace):
    """Test message statistics functionality"""
    agent = Agent(
        llm_client=mock_llm_client,
        system_prompt="System",
        tools=[],
        workspace_dir=temp_workspace,
    )

    # Add different types of messages
    agent.add_user_message("User message 1")
    agent.messages.append(Message(role="assistant", content="Assistant response 1"))
    agent.add_user_message("User message 2")
    agent.messages.append(
        Message(
            role="tool", content="Tool result", tool_call_id="123", name="test_tool"
        )
    )

    # Count different types of messages
    user_msgs = sum(1 for m in agent.messages if m.role == "user")
    assistant_msgs = sum(1 for m in agent.messages if m.role == "assistant")
    tool_msgs = sum(1 for m in agent.messages if m.role == "tool")

    assert user_msgs == 2
    assert assistant_msgs == 1
    assert tool_msgs == 1
    assert len(agent.messages) == 5  # 1 system + 2 user + 1 assistant + 1 tool


def test_agent_rejects_cross_loop_reuse(temp_workspace):
    """Same Agent instance should not be reused across different event loops."""
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            finish_reason="stop",
            tool_calls=None,
        )
    )

    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[],
        workspace_dir=temp_workspace,
    )

    agent.add_user_message("first run")
    asyncio.run(agent.run())

    agent.add_user_message("second run")
    with pytest.raises(RuntimeError, match="same event loop"):
        asyncio.run(agent.run())


@pytest.mark.asyncio
async def test_run_with_explicit_cancel_event_does_not_leak_cancel_state():
    """Per-run cancel_event should not affect later runs when omitted."""
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            finish_reason="stop",
            tool_calls=None,
        )
    )

    workspace_dir = Path("workspace") / "test_cancel_state"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[],
        workspace_dir=str(workspace_dir),
    )
    agent.logger.start_new_run = MagicMock()
    agent.logger.get_log_file_path = MagicMock(return_value=Path("test.log"))

    try:
        # First run is cancelled via explicit cancel event.
        agent.add_user_message("first run")
        cancel_event = asyncio.Event()
        cancel_event.set()
        cancelled_result = await agent.run(cancel_event=cancel_event)
        assert cancelled_result == "Task cancelled by user."
        assert agent.cancel_event is None

        # Second run should proceed normally.
        agent.add_user_message("second run")
        normal_result = await agent.run()
        assert normal_result == "ok"
        assert llm_client.generate.await_count == 1
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
