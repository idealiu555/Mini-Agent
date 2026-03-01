"""ACP (Agent Client Protocol) bridge for Mini-Agent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from pydantic import field_validator
from acp.schema import AgentCapabilities, Implementation, McpCapabilities

from mini_agent.agent import Agent
from mini_agent.cli import add_workspace_tools, initialize_base_tools
from mini_agent.config import Config
from mini_agent.llm import LLMClient
from mini_agent.retry import RetryConfig as RetryConfigBase

logger = logging.getLogger(__name__)


try:
    class InitializeRequestPatch(InitializeRequest):
        @field_validator("protocolVersion", mode="before")
        @classmethod
        def normalize_protocol_version(cls, value: Any) -> int:
            if isinstance(value, str):
                try:
                    return int(value.split(".")[0])
                except Exception:
                    return 1
            if isinstance(value, (int, float)):
                return int(value)
            return 1

    InitializeRequest = InitializeRequestPatch
    InitializeRequest.model_rebuild(force=True)
except Exception:  # pragma: no cover - defensive
    logger.debug("ACP schema patch skipped")


@dataclass
class SessionState:
    agent: Agent
    cancelled: bool = False
    cancel_event: asyncio.Event | None = None


class MiniMaxACPAgent:
    """Minimal ACP adapter wrapping the existing Agent runtime."""

    def __init__(
        self,
        conn: AgentSideConnection,
        config: Config,
        llm: LLMClient,
        base_tools: list,
        system_prompt: str,
    ):
        self._conn = conn
        self._config = config
        self._llm = llm
        self._base_tools = base_tools
        self._system_prompt = system_prompt
        self._sessions: dict[str, SessionState] = {}

    async def _handle_agent_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Bridge Agent runtime events to ACP protocol updates."""
        if event_type == "assistant_thinking":
            thinking = payload.get("thinking")
            if thinking:
                await self._send(session_id, update_agent_thought(text_block(str(thinking))))
            return

        if event_type == "assistant_message":
            content = payload.get("content")
            if content:
                await self._send(session_id, update_agent_message(text_block(str(content))))
            return

        if event_type == "tool_call_start":
            tool_call_id = str(payload.get("tool_call_id", ""))
            tool_name = str(payload.get("tool_name", "unknown"))
            arguments = payload.get("arguments") or {}
            args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(arguments.items())[:2]) if isinstance(arguments, dict) else ""
            label = f"🔧 {tool_name}({args_preview})" if args_preview else f"🔧 {tool_name}()"
            await self._send(session_id, start_tool_call(tool_call_id, label, kind="execute", raw_input=arguments))
            return

        if event_type == "tool_call_result":
            tool_call_id = str(payload.get("tool_call_id", ""))
            success = bool(payload.get("success"))
            content = payload.get("content", "")
            error = payload.get("error")
            status = "completed" if success else "failed"
            text = f"[OK] {content}" if success else f"[ERROR] {error or 'Tool execution failed'}"
            await self._send(
                session_id,
                update_tool_call(
                    tool_call_id,
                    status=status,
                    content=[tool_content(text_block(text))],
                    raw_output=text,
                ),
            )
            return

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:  # noqa: ARG002
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(loadSession=False),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent", version="0.1.0"),
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        workspace = Path(params.cwd or self._config.agent.workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        tools = list(self._base_tools)
        add_workspace_tools(tools, self._config, workspace)
        agent = Agent(
            llm_client=self._llm,
            system_prompt=self._system_prompt,
            tools=tools,
            max_steps=self._config.agent.max_steps,
            workspace_dir=str(workspace),
            event_handler=lambda event_type, payload: self._handle_agent_event(session_id, event_type, payload),
        )
        self._sessions[session_id] = SessionState(agent=agent)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            logger.warning(f"Session '{params.sessionId}' not found")
            return PromptResponse(stopReason="refusal")
        state.cancelled = False
        state.cancel_event = asyncio.Event()
        user_text = "\n".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in params.prompt)
        state.agent.add_user_message(user_text)

        result = await state.agent.run(cancel_event=state.cancel_event)
        if state.cancel_event.is_set() or self._is_cancelled_result(result):
            stop_reason = "cancelled"
        elif self._is_max_turn_result(result):
            stop_reason = "max_turn_requests"
        elif self._is_llm_error_result(result):
            await self._send(params.sessionId, update_agent_message(text_block(f"Error: {result}")))
            stop_reason = "refusal"
        else:
            stop_reason = "end_turn"

        state.cancel_event = None
        return PromptResponse(stopReason=stop_reason)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True
            if state.cancel_event is not None:
                state.cancel_event.set()

    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))

    @staticmethod
    def _is_cancelled_result(result: str) -> bool:
        return result.strip() == "Task cancelled by user."

    @staticmethod
    def _is_max_turn_result(result: str) -> bool:
        return result.startswith("Task couldn't be completed after ")

    @staticmethod
    def _is_llm_error_result(result: str) -> bool:
        return result.startswith("LLM call failed")


async def run_acp_server(config: Config | None = None) -> None:
    """Run Mini-Agent as an ACP-compatible stdio server."""
    config = config or Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_tools, skill_loader = await initialize_base_tools(config)
    prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are a helpful AI assistant."
    if skill_loader:
        meta = skill_loader.get_skills_metadata_prompt()
        if meta:
            system_prompt = f"{system_prompt.rstrip()}\n\n{meta}"
    rcfg = config.llm.retry
    llm = LLMClient(api_key=config.llm.api_key, api_base=config.llm.api_base, model=config.llm.model, retry_config=RetryConfigBase(enabled=rcfg.enabled, max_retries=rcfg.max_retries, initial_delay=rcfg.initial_delay, max_delay=rcfg.max_delay, exponential_base=rcfg.exponential_base))
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, config, llm, base_tools, system_prompt), writer, reader)
    logger.info("Mini-Agent ACP server running")
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(run_acp_server())


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
