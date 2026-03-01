"""Tests for the Telegram webhook handler module."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

from termo_agent import telegram_webhook


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

    def test_send_telegram_message_markdown_fallback(self):
        """If MarkdownV2 fails, should fall back to plain text."""
        call_count = 0

        def mock_request(bot_token, method, payload=None, timeout=10):
            nonlocal call_count
            call_count += 1
            if payload and payload.get("parse_mode") == "MarkdownV2":
                return {"ok": False, "error": "parse error"}
            return {"ok": True}

        with patch.object(telegram_webhook, "_telegram_request", side_effect=mock_request):
            result = telegram_webhook._send_telegram_message("token", 123, "test *bad markdown")

        assert call_count == 2
        assert result["ok"]


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
    """Normal text message should call adapter.send_message and return ok."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message = AsyncMock(return_value="Hello from the agent!")
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

            mock_adapter.send_message.assert_awaited_once_with("Hello", "telegram:42")
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
    """If adapter raises, should send an error message back to Telegram."""
    mock_adapter = AsyncMock()
    mock_adapter.send_message = AsyncMock(side_effect=Exception("LLM error"))
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
