"""Tests for the Telegram webhook handler module."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

from termo_agent import telegram_webhook
from termo_agent.adapter import StreamEvent


# --- Async generator helpers for send_message_stream mocks ---

async def _stream_simple(text, session_key, content="Hello from the agent!"):
    """Yield a simple done event (no tools)."""
    yield StreamEvent(type="done", content=content, usage={"prompt_tokens": 10, "completion_tokens": 5})


async def _stream_with_tools(text, session_key):
    """Yield tool_start, tool_end, then done."""
    yield StreamEvent(type="tool_start", name="execute_command", metadata={"call_id": "c1", "input": {"command": "ls"}})
    yield StreamEvent(type="tool_end", name="execute_command", metadata={"call_id": "c1", "output": "file.txt\n"})
    yield StreamEvent(type="done", content="Here are the files.", usage={"prompt_tokens": 50, "completion_tokens": 20})


async def _stream_error(text, session_key):
    """Raise during streaming."""
    raise Exception("LLM streaming error")
    yield  # make it a generator  # noqa: unreachable


# --- Unit tests for helper functions ---

class TestTelegramHelpers:
    def test_send_telegram_response_splits_long_messages(self):
        """Messages over 4096 chars should be split into chunks."""
        chunks_sent = []

        def mock_send(bot_token, chat_id, text):
            chunks_sent.append(text)
            return {"ok": True}

        with patch.object(telegram_webhook, "_send_telegram_message", side_effect=mock_send):
            long_text = "A" * 5000
            telegram_webhook._send_telegram_response("token", 123, long_text)

        assert len(chunks_sent) == 2
        assert len(chunks_sent[0]) <= 4096
        assert "".join(chunks_sent) == long_text

    def test_send_telegram_response_empty_message(self):
        """Empty response should send '(empty response)'."""
        sent = []

        def mock_send(bot_token, chat_id, text):
            sent.append(text)
            return {"ok": True}

        with patch.object(telegram_webhook, "_send_telegram_message", side_effect=mock_send):
            telegram_webhook._send_telegram_response("token", 123, "")

        assert sent[0] == "(empty response)"

    def test_send_telegram_message_html_fallback(self):
        """If HTML parse fails, should fall back to plain text."""
        call_count = 0

        def mock_request(bot_token, method, payload=None, timeout=10):
            nonlocal call_count
            call_count += 1
            if payload and payload.get("parse_mode") == "HTML":
                return {"ok": False, "error": "parse error"}
            return {"ok": True}

        with patch.object(telegram_webhook, "_telegram_request", side_effect=mock_request):
            result = telegram_webhook._send_telegram_message("token", 123, "test **bold**")

        assert call_count == 2
        assert result["ok"]

    def test_md_to_telegram_html_converts_bold(self):
        """Markdown bold should become HTML bold."""
        result = telegram_webhook._md_to_telegram_html("Hello **world**!")
        assert "<b>world</b>" in result

    def test_md_to_telegram_html_escapes_html(self):
        """Raw HTML in text should be escaped."""
        result = telegram_webhook._md_to_telegram_html("Use <script> tag")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result


# --- Module setup tests ---

class TestSetup:
    def teardown_method(self):
        telegram_webhook._adapter = None
        telegram_webhook._channels = []

    def test_no_channels_returns_empty_routes(self):
        """When no channels configured, routes list should be empty."""
        telegram_webhook._channels = []
        assert telegram_webhook.get_routes() == []
        assert telegram_webhook.get_public_prefixes() == []

    def test_setup_with_telegram_channels(self):
        """Setup with valid telegram channels should populate routes."""
        mock_adapter = MagicMock()
        config = {
            "channels": [
                {"type": "telegram", "enabled": True, "id": "ch1", "config": {"bot_token": "tok"}},
            ]
        }
        telegram_webhook.setup(mock_adapter, config)
        routes = telegram_webhook.get_routes()
        assert len(routes) == 1
        assert routes[0][0] == "POST"
        assert routes[0][1] == "/webhooks/telegram"
        assert telegram_webhook.get_public_prefixes() == ["/webhooks/"]

    def test_setup_ignores_disabled_channels(self):
        """Disabled channels should not produce routes."""
        mock_adapter = MagicMock()
        config = {
            "channels": [
                {"type": "telegram", "enabled": False, "id": "ch1", "config": {"bot_token": "tok"}},
            ]
        }
        telegram_webhook.setup(mock_adapter, config)
        assert telegram_webhook.get_routes() == []

    def test_setup_ignores_non_telegram_channels(self):
        """Non-telegram channels should not produce routes."""
        mock_adapter = MagicMock()
        config = {
            "channels": [
                {"type": "discord", "enabled": True, "id": "ch1", "config": {}},
            ]
        }
        telegram_webhook.setup(mock_adapter, config)
        assert telegram_webhook.get_routes() == []


# --- Webhook handler integration tests ---

def _make_app():
    app = web.Application()
    app.router.add_post("/webhooks/telegram", telegram_webhook.handle_telegram_webhook)
    return app


def _make_update(text="Hello", chat_id=42, user_id=100, username="testuser"):
    """Build a minimal Telegram Update payload."""
    return {
        "message": {
            "chat": {"id": chat_id},
            "text": text,
            "from": {"id": user_id, "username": username, "first_name": "Test"},
        }
    }


def _setup_module(adapter, channels=None):
    """Configure the telegram module state for testing."""
    telegram_webhook._adapter = adapter
    telegram_webhook._channels = [
        {
            "id": "channel-123",
            "type": "telegram",
            "enabled": True,
            "config": {"bot_token": "test-bot-token"},
        }
    ] if channels is None else channels


def _teardown_module():
    telegram_webhook._adapter = None
    telegram_webhook._channels = []


@pytest.mark.asyncio
async def test_normal_message():
    """Normal text message should stream through adapter and return ok."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=_stream_simple)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response") as mock_send, \
                 patch.object(telegram_webhook, "_persist_telegram_messages"):
                resp = await client.post("/webhooks/telegram", json=_make_update("Hello"))
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True

            mock_adapter.send_message_stream.assert_called_once_with("Hello", "telegram:42")
            mock_send.assert_called_once_with("test-bot-token", 42, "Hello from the agent!")
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_start_command():
    """The /start command should send a welcome message."""
    mock_adapter = AsyncMock()
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_telegram_message") as mock_send:
                resp = await client.post("/webhooks/telegram", json=_make_update("/start"))
                assert resp.status == 200

            mock_send.assert_called_once()
            args = mock_send.call_args[0]
            assert "TestBot" in args[2]
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_non_text_update_ignored():
    """Updates without text (e.g., photos) should be silently acknowledged."""
    mock_adapter = AsyncMock()
    mock_adapter.config = {}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            update = {"message": {"chat": {"id": 42}, "photo": [{"file_id": "abc"}]}}
            resp = await client.post("/webhooks/telegram", json=update)
            assert resp.status == 200
            mock_adapter.send_message.assert_not_awaited()
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_no_message_field():
    """Updates without a message field should be ignored."""
    mock_adapter = AsyncMock()
    mock_adapter.config = {}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            resp = await client.post("/webhooks/telegram", json={"update_id": 1234})
            assert resp.status == 200
            mock_adapter.send_message.assert_not_awaited()
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_no_channels_configured():
    """If no channels configured, handler returns ok without processing."""
    mock_adapter = AsyncMock()
    mock_adapter.config = {}
    _setup_module(mock_adapter, channels=[])

    try:
        async with TestClient(TestServer(_make_app())) as client:
            resp = await client.post("/webhooks/telegram", json=_make_update())
            assert resp.status == 200
            mock_adapter.send_message.assert_not_awaited()
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_adapter_error_sends_fallback():
    """If adapter streaming raises, should send an error message back to Telegram."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=_stream_error)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response") as mock_send, \
                 patch.object(telegram_webhook, "_persist_telegram_messages"):
                resp = await client.post("/webhooks/telegram", json=_make_update())
                assert resp.status == 200

            mock_send.assert_called_once()
            error_msg = mock_send.call_args[0][2]
            assert "error" in error_msg.lower()
    finally:
        _teardown_module()


# --- Telegram tool context tests ---


@pytest.mark.asyncio
async def test_telegram_context_set_during_streaming():
    """Context dicts should be populated during streaming and cleaned up after."""
    captured_context = {}

    async def capture_context_stream(text, session_key):
        # During streaming, context should be set
        captured_context["ctx"] = telegram_webhook.get_telegram_context(session_key)
        captured_context["session_key"] = session_key
        yield StreamEvent(type="done", content="response")

    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=capture_context_stream)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response"), \
                 patch.object(telegram_webhook, "_persist_telegram_messages"):
                resp = await client.post("/webhooks/telegram", json=_make_update(chat_id=99))
                assert resp.status == 200

        # During the call, context should have been available
        assert captured_context["ctx"] is not None
        assert captured_context["ctx"]["bot_token"] == "test-bot-token"
        assert captured_context["ctx"]["chat_id"] == 99

        # After the call, context should be cleaned up
        assert telegram_webhook.get_telegram_context("telegram:99") is None
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_fallback_skipped_when_tool_used():
    """_send_telegram_response should NOT be called when the tool already sent messages."""

    async def fake_stream(text, session_key):
        # Simulate the tool being used during the LLM run
        telegram_webhook.mark_telegram_tool_used(session_key)
        yield StreamEvent(type="done", content="tool already sent this")

    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=fake_stream)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response") as mock_send, \
                 patch.object(telegram_webhook, "_persist_telegram_messages"):
                resp = await client.post("/webhooks/telegram", json=_make_update())
                assert resp.status == 200

            # Fallback should NOT have been called
            mock_send.assert_not_called()
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_fallback_sent_when_tool_not_used():
    """_send_telegram_response should be called normally when tool was not used."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=_stream_simple)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response") as mock_send, \
                 patch.object(telegram_webhook, "_persist_telegram_messages"):
                resp = await client.post("/webhooks/telegram", json=_make_update())
                assert resp.status == 200

            # Fallback SHOULD have been called since tool wasn't used
            mock_send.assert_called_once_with("test-bot-token", 42, "Hello from the agent!")
    finally:
        _teardown_module()


# --- Streaming + tool collection tests ---


@pytest.mark.asyncio
async def test_streaming_collects_tool_calls():
    """Streaming should collect tool_start/tool_end pairs into tool_calls list."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message_stream = MagicMock(side_effect=_stream_with_tools)
    mock_adapter.config = {"persona_name": "TestBot"}
    _setup_module(mock_adapter)

    persist_args = {}

    def capture_persist(*args, **kwargs):
        persist_args["args"] = args
        persist_args["kwargs"] = kwargs

    try:
        async with TestClient(TestServer(_make_app())) as client:
            with patch.object(telegram_webhook, "_send_typing"), \
                 patch.object(telegram_webhook, "_send_telegram_response") as mock_send, \
                 patch.object(telegram_webhook, "_persist_telegram_messages", side_effect=capture_persist):
                resp = await client.post("/webhooks/telegram", json=_make_update("ls"))
                assert resp.status == 200

            # Response text from the done event
            mock_send.assert_called_once_with("test-bot-token", 42, "Here are the files.")

        # Check tool_calls were passed to persist
        assert "kwargs" in persist_args
        tool_calls = persist_args["kwargs"].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "execute_command"
        assert tool_calls[0]["input"] == {"command": "ls"}
        assert tool_calls[0]["output"] == "file.txt\n"
        assert tool_calls[0]["duration_ms"] >= 0

        # Check usage was passed
        assert persist_args["kwargs"]["tokens_in"] == 50
        assert persist_args["kwargs"]["tokens_out"] == 20
    finally:
        _teardown_module()


@pytest.mark.asyncio
async def test_persist_includes_tool_calls_in_payload():
    """The persist function should include tool_calls in the HTTP payload."""
    import urllib.request

    captured_payload = {}

    def mock_urlopen(req, timeout=15):
        captured_payload["data"] = json.loads(req.data)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        return mock_resp

    with patch.dict("os.environ", {"TERMO_API_URL": "http://api", "TERMO_TOKEN": "tok"}), \
         patch("urllib.request.urlopen", side_effect=mock_urlopen):
        telegram_webhook._persist_telegram_messages(
            "ch1", "42", "100", "testuser", "Hello", "World",
            tool_calls=[{"name": "run", "input": {}, "output": "ok", "duration_ms": 100}],
            tokens_in=10,
            tokens_out=5,
        )

    assert "data" in captured_payload
    payload = captured_payload["data"]
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["name"] == "run"
    assert payload["tokens_in"] == 10
    assert payload["tokens_out"] == 5
