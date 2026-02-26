"""Tests for AgentAdapter interface and StreamEvent."""

import pytest
from termo_agent.adapter import AgentAdapter, StreamEvent


class TestStreamEvent:
    def test_defaults(self):
        e = StreamEvent(type="token", content="hello")
        assert e.type == "token"
        assert e.content == "hello"
        assert e.name == ""
        assert e.usage == {}
        assert e.metadata == {}

    def test_all_fields(self):
        e = StreamEvent(
            type="tool_start",
            content="",
            name="web_search",
            usage={"prompt_tokens": 10},
            metadata={"call_id": "abc"},
        )
        assert e.name == "web_search"
        assert e.usage == {"prompt_tokens": 10}
        assert e.metadata == {"call_id": "abc"}

    def test_valid_types(self):
        for t in ("token", "tool_start", "tool_end", "done", "error", "progress"):
            e = StreamEvent(type=t, content="")
            assert e.type == t


class TestAgentAdapterDefaults:
    """Test that optional methods have sensible defaults."""

    @pytest.fixture
    def adapter(self, stub_adapter):
        return stub_adapter

    @pytest.mark.asyncio
    async def test_health_default(self):
        # Use a bare subclass with only required methods
        class MinimalAdapter(AgentAdapter):
            async def initialize(self, config_path=None): pass
            async def shutdown(self): pass
            async def send_message(self, message, session_key="default"): return ""
            async def send_message_stream(self, message, session_key="default"):
                yield StreamEvent(type="done", content="")
            async def get_history(self, session_key): return []

        a = MinimalAdapter()
        h = await a.health()
        assert h == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_list_sessions_default(self):
        class MinimalAdapter(AgentAdapter):
            async def initialize(self, config_path=None): pass
            async def shutdown(self): pass
            async def send_message(self, message, session_key="default"): return ""
            async def send_message_stream(self, message, session_key="default"):
                yield StreamEvent(type="done", content="")
            async def get_history(self, session_key): return []

        a = MinimalAdapter()
        assert await a.list_sessions() == []
        assert await a.list_tools() == []
        assert await a.get_config() == {}
        assert await a.get_memory() is None

    def test_extra_routes_default(self):
        class MinimalAdapter(AgentAdapter):
            async def initialize(self, config_path=None): pass
            async def shutdown(self): pass
            async def send_message(self, message, session_key="default"): return ""
            async def send_message_stream(self, message, session_key="default"):
                yield StreamEvent(type="done", content="")
            async def get_history(self, session_key): return []

        a = MinimalAdapter()
        assert a.extra_routes() is None
        assert a.public_route_prefixes() == ["/health"]

    @pytest.mark.asyncio
    async def test_update_and_restart_noop(self):
        class MinimalAdapter(AgentAdapter):
            async def initialize(self, config_path=None): pass
            async def shutdown(self): pass
            async def send_message(self, message, session_key="default"): return ""
            async def send_message_stream(self, message, session_key="default"):
                yield StreamEvent(type="done", content="")
            async def get_history(self, session_key): return []

        a = MinimalAdapter()
        result = await a.update()
        assert result == {"status": "not supported"}
        await a.restart()  # Should not raise
