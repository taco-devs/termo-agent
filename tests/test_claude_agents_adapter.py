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
    def __init__(self, tool_use_id="t1", content="ok", is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class FakeThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class FakeAssistantMessage:
    def __init__(self, content, model="claude-opus-4-6", error=None):
        self.content = content
        self.model = model
        self.error = error
        self.parent_tool_use_id = None


class FakeUserMessage:
    def __init__(self, content, parent_tool_use_id=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id
        self.uuid = None
        self.tool_use_result = None


class FakeResultMessage:
    def __init__(
        self, result=None, is_error=False, usage=None,
        total_cost_usd=None, duration_ms=None, num_turns=1,
        session_id="sess-123", structured_output=None,
    ):
        self.subtype = "result"
        self.result = result
        self.is_error = is_error
        self.usage = usage or {}
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
        self.duration_api_ms = duration_ms
        self.num_turns = num_turns
        self.session_id = session_id
        self.structured_output = structured_output


class FakeSystemMessage:
    def __init__(self, subtype="init", data=None, session_id=None):
        self.subtype = subtype
        self.data = data or {}
        self.session_id = session_id


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
    mod.UserMessage = FakeUserMessage
    mod.ResultMessage = FakeResultMessage
    mod.SystemMessage = FakeSystemMessage
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
        """Tool use: AssistantMessage has ToolUseBlock, UserMessage has ToolResultBlock."""
        tmp_path, mod = tmp_agent_dir

        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([
                FakeTextBlock("Looking up... "),
                FakeToolUseBlock(name="web_search", id="t1"),
            ]),
            # SDK executes tool internally, result comes as UserMessage
            FakeUserMessage([FakeToolResultBlock(tool_use_id="t1", content="results")]),
            FakeAssistantMessage([FakeTextBlock("Found it.")]),
            FakeResultMessage(result="Looking up... Found it."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("search", "s2"):
                events.append(event)

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert types[-1] == "done"
        # tool_end should carry the correct tool name
        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.name == "web_search"
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

    @pytest.mark.asyncio
    async def test_list_tools_from_config(self, tmp_agent_dir):
        tmp_path, mod = tmp_agent_dir
        config = {"allowed_tools": ["Read", "Edit", "Bash"]}
        (tmp_path / "config.json").write_text(json.dumps(config))

        adapter = mod.Adapter()
        await adapter.initialize()

        tools = await adapter.list_tools()
        assert len(tools) == 3
        assert tools[0] == {"name": "Read", "type": "builtin"}
        assert tools[2] == {"name": "Bash", "type": "builtin"}


class TestResultMessage:
    @pytest.mark.asyncio
    async def test_result_message_provides_usage(self, tmp_agent_dir):
        """ResultMessage carries usage, cost, duration → done event metadata."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeTextBlock("answer")]),
            FakeResultMessage(
                result="answer",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                total_cost_usd=0.005,
                duration_ms=1200,
                num_turns=2,
                session_id="sess-abc",
            ),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("q", "s1"):
                events.append(event)

        done = events[-1]
        assert done.type == "done"
        assert done.content == "answer"
        assert done.usage == {"prompt_tokens": 100, "completion_tokens": 50}
        assert done.metadata["cost_usd"] == 0.005
        assert done.metadata["duration_ms"] == 1200
        assert done.metadata["num_turns"] == 2
        assert done.metadata["session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_result_message_fallback_text(self, tmp_agent_dir):
        """If no TextBlocks were streamed, ResultMessage.result is used."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeResultMessage(result="fallback text"),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("q", "s1"):
                events.append(event)

        done = events[-1]
        assert done.type == "done"
        assert done.content == "fallback text"

    @pytest.mark.asyncio
    async def test_result_message_error(self, tmp_agent_dir):
        """ResultMessage with is_error=True yields error event."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeResultMessage(is_error=True, result="something went wrong"),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("q", "s1"):
                events.append(event)

        assert events[0].type == "error"
        assert "something went wrong" in events[0].content


class TestUserMessage:
    @pytest.mark.asyncio
    async def test_tool_result_in_user_message(self, tmp_agent_dir):
        """ToolResultBlock in UserMessage triggers tool_end with correct name."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeToolUseBlock(name="Read", id="t42")]),
            FakeUserMessage([FakeToolResultBlock(tool_use_id="t42", content="file contents")]),
            FakeAssistantMessage([FakeTextBlock("Here's the file.")]),
            FakeResultMessage(result="Here's the file."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("read it", "s1"):
                events.append(event)

        types = [e.type for e in events]
        assert types == ["tool_start", "tool_end", "token", "done"]
        assert events[0].name == "Read"
        assert events[1].name == "Read"
        assert events[1].metadata["tool_use_id"] == "t42"

    @pytest.mark.asyncio
    async def test_tool_result_error(self, tmp_agent_dir):
        """ToolResultBlock with is_error=True is surfaced in metadata."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeToolUseBlock(name="Bash", id="t1")]),
            FakeUserMessage([FakeToolResultBlock(tool_use_id="t1", content="exit 1", is_error=True)]),
            FakeAssistantMessage([FakeTextBlock("Command failed.")]),
            FakeResultMessage(result="Command failed."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("run it", "s1"):
                events.append(event)

        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.metadata["is_error"] is True


class TestSystemMessage:
    @pytest.mark.asyncio
    async def test_system_message_yields_progress(self, tmp_agent_dir):
        """SystemMessage maps to progress event."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeSystemMessage(subtype="init", data={"session_id": "s1"}),
            FakeAssistantMessage([FakeTextBlock("hi")]),
            FakeResultMessage(result="hi"),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("hello", "s1"):
                events.append(event)

        progress = events[0]
        assert progress.type == "progress"
        assert progress.content == "init"
        assert progress.metadata == {"session_id": "s1"}


class TestAssistantMessageError:
    @pytest.mark.asyncio
    async def test_assistant_error_yields_error_event(self, tmp_agent_dir):
        """AssistantMessage with error field yields error and skips content."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage(
                content=[FakeTextBlock("ignored")],
                error={"type": "overloaded", "message": "API overloaded"},
            ),
            FakeAssistantMessage([FakeTextBlock("recovered")]),
            FakeResultMessage(result="recovered"),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("q", "s1"):
                events.append(event)

        types = [e.type for e in events]
        assert types[0] == "error"
        assert "overloaded" in events[0].content
        # Should still get the recovered text
        assert "token" in types
        done = events[-1]
        assert done.type == "done"
        assert done.content == "recovered"


class TestMultipleToolCalls:
    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, tmp_agent_dir):
        """Multiple tool calls in one AssistantMessage, results in separate UserMessages."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([
                FakeToolUseBlock(name="Read", id="t1"),
                FakeToolUseBlock(name="Grep", id="t2"),
            ]),
            FakeUserMessage([
                FakeToolResultBlock(tool_use_id="t1", content="file A"),
                FakeToolResultBlock(tool_use_id="t2", content="match B"),
            ]),
            FakeAssistantMessage([FakeTextBlock("Done.")]),
            FakeResultMessage(result="Done."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("find stuff", "s1"):
                events.append(event)

        types = [e.type for e in events]
        assert types.count("tool_start") == 2
        assert types.count("tool_end") == 2
        # Verify correct tool names on tool_end
        tool_ends = [e for e in events if e.type == "tool_end"]
        assert tool_ends[0].name == "Read"
        assert tool_ends[1].name == "Grep"


class TestSessionResumption:
    @pytest.mark.asyncio
    async def test_session_id_captured_from_init(self, tmp_agent_dir):
        """SystemMessage(subtype='init') captures session_id for resumption."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeSystemMessage(subtype="init", session_id="sdk-sess-abc"),
            FakeAssistantMessage([FakeTextBlock("hi")]),
            FakeResultMessage(result="hi"),
        ])

        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for _ in adapter.send_message_stream("hello", "s1"):
                pass

        assert adapter._session_ids.get("s1") == "sdk-sess-abc"

    @pytest.mark.asyncio
    async def test_session_id_persisted_to_disk(self, tmp_agent_dir):
        """Session IDs survive restart via disk persistence."""
        tmp_path, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        # Simulate a captured session ID
        adapter._session_ids["s1"] = "sdk-sess-xyz"
        adapter._save_session_ids()

        sid_file = tmp_path / "sessions" / "_sdk_session_ids.json"
        assert sid_file.exists()
        assert json.loads(sid_file.read_text())["s1"] == "sdk-sess-xyz"

        # New adapter should load it
        adapter2 = mod.Adapter()
        await adapter2.initialize()
        assert adapter2._session_ids.get("s1") == "sdk-sess-xyz"

    @pytest.mark.asyncio
    async def test_resume_used_on_reconnect(self, tmp_agent_dir):
        """When a stored session_id exists, _get_client rebuilds options with resume."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()
        adapter._session_ids["s1"] = "sdk-sess-resume"

        # _get_client should call _build_options(resume=...) then create client
        built_options = []
        orig_build = adapter._build_options

        def spy_build(resume=None):
            built_options.append(resume)
            orig_build(resume=resume)

        with patch.object(adapter, "_build_options", side_effect=spy_build):
            await adapter._get_client("s1")

        assert built_options == ["sdk-sess-resume"]

    @pytest.mark.asyncio
    async def test_restart_clears_session_ids(self, tmp_agent_dir):
        """restart() clears session IDs."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()
        adapter._session_ids["s1"] = "old-id"

        await adapter.restart()

        assert adapter._session_ids == {}


class TestToolInputOutput:
    @pytest.mark.asyncio
    async def test_tool_start_includes_input(self, tmp_agent_dir):
        """tool_start metadata includes the tool input."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([
                FakeToolUseBlock(name="Read", id="t1", input={"file_path": "/foo.py"}),
            ]),
            FakeUserMessage([FakeToolResultBlock(tool_use_id="t1", content="contents")]),
            FakeAssistantMessage([FakeTextBlock("Done.")]),
            FakeResultMessage(result="Done."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("read", "s1"):
                events.append(event)

        tool_start = next(e for e in events if e.type == "tool_start")
        assert tool_start.metadata["input"] == {"file_path": "/foo.py"}

    @pytest.mark.asyncio
    async def test_tool_end_includes_output(self, tmp_agent_dir):
        """tool_end metadata includes the tool result content."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        fake_client = FakeClient(responses=[
            FakeAssistantMessage([FakeToolUseBlock(name="Bash", id="t1")]),
            FakeUserMessage([FakeToolResultBlock(
                tool_use_id="t1", content="total 42\ndrwxr-xr-x ..."
            )]),
            FakeAssistantMessage([FakeTextBlock("Listed.")]),
            FakeResultMessage(result="Listed."),
        ])

        events = []
        with patch.object(adapter, "_get_client", return_value=fake_client):
            async for event in adapter.send_message_stream("ls", "s1"):
                events.append(event)

        tool_end = next(e for e in events if e.type == "tool_end")
        assert "total 42" in tool_end.metadata["output"]


class TestConfigPassthrough:
    @pytest.mark.asyncio
    async def test_claude_options_forwarded(self, tmp_agent_dir):
        """Claude-specific config keys are forwarded to ClaudeAgentOptions."""
        tmp_path, mod = tmp_agent_dir
        config = {
            "model": "claude-opus-4-6",
            "max_turns": 10,
            "max_budget_usd": 0.50,
            "thinking": {"type": "adaptive"},
            "betas": ["context-1m-2025-08-07"],
            "allowed_tools": ["Read", "Grep"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        adapter = mod.Adapter()
        await adapter.initialize()

        opts = adapter._options
        assert opts.max_turns == 10
        assert opts.max_budget_usd == 0.50
        assert opts.thinking == {"type": "adaptive"}
        assert opts.betas == ["context-1m-2025-08-07"]
        assert opts.allowed_tools == ["Read", "Grep"]

    @pytest.mark.asyncio
    async def test_unset_options_not_forwarded(self, tmp_agent_dir):
        """Options not in config don't appear on ClaudeAgentOptions."""
        _, mod = tmp_agent_dir
        adapter = mod.Adapter()
        await adapter.initialize()

        opts = adapter._options
        assert not hasattr(opts, "max_turns")
        assert not hasattr(opts, "max_budget_usd")
        assert not hasattr(opts, "thinking")
