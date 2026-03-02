"""Tests for CLI helper behaviors."""

import asyncio

import pytest

from mini_agent.cli import _quiet_cleanup, resolve_provider
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
