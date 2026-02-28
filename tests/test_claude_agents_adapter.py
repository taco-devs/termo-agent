"""Tests for Claude Agent SDK adapter with mocked SDK."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake SDK classes for mocking
# ---------------------------------------------------------------------------

class FakeTextBlock:
    def __init__(self, text):
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name, id="t1", input=None):
        self.name = name
        self.id = id
        self.input = input or {}


class FakeToolResultBlock:
    def __init__(self, tool_use_id="t1", content="ok"):
        self.tool_use_id = tool_use_id
        self.content = content


class FakeThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class FakeAssistantMessage:
    def __init__(self, content):
        self.content = content


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeClient:
    """Mock ClaudeSDKClient."""

    def __init__(self, responses=None, **kwargs):
        self._responses = responses or []
        self._query_calls = []

    async def connect(self):
        pass

    async def query(self, prompt):
        self._query_calls.append(prompt)

    async def receive_response(self):
        for msg in self._responses:
            yield msg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


# ---------------------------------------------------------------------------
# Build a fake claude_agent_sdk module and inject it into sys.modules
# ---------------------------------------------------------------------------

def _build_fake_module():
    mod = ModuleType("claude_agent_sdk")
    mod.ClaudeAgentOptions = FakeClaudeAgentOptions
    mod.ClaudeSDKClient = FakeClient
    mod.AssistantMessage = FakeAssistantMessage
    mod.TextBlock = FakeTextBlock
    mod.ToolUseBlock = FakeToolUseBlock
    mod.ToolResultBlock = FakeToolResultBlock
    mod.ThinkingBlock = FakeThinkingBlock
    return mod


@pytest.fixture(autouse=True)
def _inject_fake_sdk():
    """Inject fake claude_agent_sdk module for all tests."""
    fake_mod = _build_fake_module()
    with patch.dict(sys.modules, {"claude_agent_sdk": fake_mod}):
        yield


@pytest.fixture
def tmp_agent_dir(tmp_path):
    """Set AGENT_DATA_DIR to a temp directory and reload module constants."""
    with patch.dict("os.environ", {"AGENT_DATA_DIR": str(tmp_path)}):
        # Re-import to pick up the new AGENT_DATA_DIR
        import importlib
        import termo_agent.adapters.claude_agents as mod

        importlib.reload(mod)
        yield tmp_path, mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_loads_config(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir
        config = {"model": "claude-opus-4-6", "allowed_tools": ["Read"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "SOUL.md").write_text("You are a test agent.")

        adapter = mod.Adapter()
        await adapter.initialize()

        assert adapter.config["model"] == "claude-opus-4-6"
        assert adapter.config["allowed_tools"] == ["Read"]
        assert adapter.soul_md == "You are a test agent."
        assert adapter._options is not None

    @pytest.mark.asyncio
    async def test_initialize_defaults(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        assert adapter.config == {}
        assert adapter.soul_md == "You are a helpful assistant."
        assert adapter._options is not None


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        # Inject a fake client that returns a text response
        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeTextBlock("hello")])
        ])

        with patch.object(adapter, "_get_client", return_value=fake_client):
            result = await adapter.send_message("hi", "s1")

        assert result == "hello"
        # Session should be saved
        history = adapter._sessions["s1"]
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hi"}
        assert history[1] == {"role": "assistant", "content": "hello"}


class TestSendMessageStream:
    @pytest.mark.asyncio
    async def test_stream_text(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeTextBlock("hello")])
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("hi", "s1"):
                events.append(event)

        types = [e.type for e in events]
        assert "token" in types
        assert types[-1] == "done"
        assert events[-1].content == "hello"

    @pytest.mark.asyncio
    async def test_stream_with_tools(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([
                FakeTextBlock("Looking up... "),
                FakeToolUseBlock(name="web_search"),
                FakeToolResultBlock(),
                FakeTextBlock("Found it."),
            ])
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("search", "s2"):
                events.append(event)

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert types[-1] == "done"
        assert events[-1].content == "Looking up... Found it."

    @pytest.mark.asyncio
    async def test_stream_with_thinking(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([
                FakeThinkingBlock("Let me think..."),
                FakeTextBlock("answer"),
            ])
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("think", "s3"):
                events.append(event)

        types = [e.type for e in events]
        assert "thinking" in types
        assert events[-1].content == "answer"

    @pytest.mark.asyncio
    async def test_stream_error(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        # Create a client whose receive_response raises
        class ErrorClient(FakeClient):
            async def receive_response(self):
                raise RuntimeError("SDK error")
                yield  # noqa: unreachable — makes this an async generator

        fake_client = ErrorClient()

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("fail", "s4"):
                events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "SDK error" in events[0].content


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_session_saved_to_disk(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeTextBlock("hi back")])
        ])

        with patch.object(adapter, "_get_client", return_value=fake_client):
            await adapter.send_message("hi", "user1:session1")

        # Check file was written
        session_file = tmp_path / "sessions" / "user1_session1.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text())
        assert len(data) == 2
        assert data[0]["content"] == "hi"
        assert data[1]["content"] == "hi back"


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_get_history(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeTextBlock("reply")])
        ])

        with patch.object(adapter, "_get_client", return_value=fake_client):
            await adapter.send_message("msg", "s1")

        history = await adapter.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        # Create fake session files
        sessions_dir = tmp_path / "sessions"
        (sessions_dir / "user1_s1.json").write_text("[]")
        (sessions_dir / "user2_s2.json").write_text("[]")

        result = await adapter.list_sessions()
        keys = {s["session_key"] for s in result}
        assert "user1:s1" in keys
        assert "user2:s2" in keys


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, tmp_agent_dir):
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        h = await adapter.health()
        assert h == {"status": "ok", "adapter": "claude_agents"}


class TestRestart:
    @pytest.mark.asyncio
    async def test_restart_clears_sessions(self, tmp_agent_dir):
        _, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        # Populate sessions and clients
        adapter._sessions["s1"] = [{"role": "user", "content": "hi"}]
        adapter._clients["s1"] = FakeClient()

        await adapter.restart()

        assert adapter._sessions == {}
        assert adapter._clients == {}
        assert adapter._options is not None


class TestConfigAndMemory:
    @pytest.mark.asyncio
    async def test_get_config(self, tmp_agent_dir):
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()
        assert await adapter.get_config() == {}

    @pytest.mark.asyncio
    async def test_update_config(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        await adapter.update_config({"model": "claude-sonnet-4-6"})
        assert adapter.config["model"] == "claude-sonnet-4-6"
        # Config file should be written
        assert (tmp_path / "config.json").exists()

    @pytest.mark.asyncio
    async def test_memory_roundtrip(self, tmp_agent_dir):
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        assert await adapter.get_memory() is None
        await adapter.update_memory("Remember this.")
        assert await adapter.get_memory() == "Remember this."

    @pytest.mark.asyncio
    async def test_list_tools_empty(self, tmp_agent_dir):
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()
        assert await adapter.list_tools() == []
