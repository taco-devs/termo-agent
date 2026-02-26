"""Shared fixtures for termo-agent tests."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from termo_agent.adapter import AgentAdapter, StreamEvent


class StubAdapter(AgentAdapter):
    """Minimal adapter for testing the server layer."""

    def __init__(self):
        self.initialized = False
        self.shut_down = False
        self._config = {"model": "test-model"}
        self._memory = ""
        self._heartbeat = ""
        self._sessions: dict[str, list[dict]] = {}
        self._extra = None
        self._public_prefixes = ["/health"]

    async def initialize(self, config_path=None):
        self.initialized = True
        if config_path:
            self._config = json.loads(Path(config_path).read_text())

    async def shutdown(self):
        self.shut_down = True

    async def send_message(self, message: str, session_key: str = "default") -> str:
        history = self._sessions.setdefault(session_key, [])
        history.append({"role": "user", "content": message})
        reply = f"Echo: {message}"
        history.append({"role": "assistant", "content": reply})
        return reply

    async def send_message_stream(self, message: str, session_key: str = "default") -> AsyncIterator[StreamEvent]:
        words = message.split()
        for word in words:
            yield StreamEvent(type="token", content=word + " ")
        yield StreamEvent(
            type="done",
            content=message,
            usage={"prompt_tokens": 10, "completion_tokens": len(words)},
        )

    async def get_history(self, session_key: str) -> list[dict]:
        return self._sessions.get(session_key, [])

    async def list_sessions(self) -> list[dict]:
        return [{"session_key": k} for k in self._sessions]

    async def get_config(self) -> dict:
        return self._config

    async def update_config(self, updates: dict) -> None:
        self._config.update(updates)

    async def get_memory(self) -> str | None:
        return self._memory or None

    async def update_memory(self, content: str) -> None:
        self._memory = content

    async def get_heartbeat(self) -> dict:
        return {"content": self._heartbeat, "enabled": bool(self._heartbeat)}

    async def update_heartbeat(self, content: str) -> None:
        self._heartbeat = content

    async def list_tools(self) -> list[dict]:
        return [{"name": "test_tool", "description": "A test tool"}]

    async def health(self) -> dict:
        return {"status": "ok", "adapter": "stub"}

    def extra_routes(self):
        return self._extra

    def public_route_prefixes(self):
        return self._public_prefixes


class StreamWithToolsAdapter(StubAdapter):
    """Adapter that yields tool events for testing SSE metadata flattening."""

    async def send_message_stream(self, message: str, session_key: str = "default") -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="token", content="Searching... ")
        yield StreamEvent(
            type="tool_start",
            name="web_search",
            metadata={"call_id": "call_123", "input": {"query": "test"}},
        )
        yield StreamEvent(
            type="tool_end",
            name="web_search",
            metadata={"call_id": "call_123", "output": "result data"},
        )
        yield StreamEvent(type="token", content="Found it!")
        yield StreamEvent(
            type="done",
            content="Searching... Found it!",
            usage={"prompt_tokens": 20, "completion_tokens": 5},
        )


@pytest.fixture
def stub_adapter():
    return StubAdapter()


@pytest.fixture
def tool_adapter():
    return StreamWithToolsAdapter()
