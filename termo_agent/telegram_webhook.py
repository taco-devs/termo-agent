"""Telegram webhook handler for the agent runtime.

Receives Telegram updates directly on the Sprite VM, processes them through
the adapter's LLM pipeline, and sends replies back via the Telegram Bot API.
Message persistence is handled by a fire-and-forget POST to the API server.
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error

from aiohttp import web

log = logging.getLogger("termo.telegram")

# Module state — set by setup()
_adapter = None
_channels: list[dict] = []

# Telegram tool context — populated during webhook handling so the tool can send messages
_telegram_contexts: dict[str, dict] = {}   # session_key → {bot_token, chat_id}
_telegram_tool_used: dict[str, bool] = {}  # session_key → True if tool was called

TELEGRAM_API = "https://api.telegram.org"
MAX_TELEGRAM_MESSAGE = 4096


def setup(adapter, config: dict) -> None:
    """Store adapter reference and extract telegram channels from config."""
    global _adapter, _channels
    _adapter = adapter
    _channels = [
        ch for ch in config.get("channels", [])
        if ch.get("type") == "telegram" and ch.get("enabled", False)
    ]
    if _channels:
        log.info("Telegram webhook enabled for %d channel(s)", len(_channels))


def get_routes() -> list[tuple[str, str, object]]:
    """Return webhook routes if telegram channels are configured."""
    if not _channels:
        return []
    return [("POST", "/webhooks/telegram", handle_telegram_webhook)]


def get_public_prefixes() -> list[str]:
    """Return public prefixes for webhook routes (skip auth)."""
    if not _channels:
        return []
    return ["/webhooks/"]


def _get_channel_for_token(bot_token: str) -> dict | None:
    """Find the channel config matching a bot token."""
    for ch in _channels:
        if ch.get("config", {}).get("bot_token") == bot_token:
            return ch
    return None


def _get_default_channel() -> dict | None:
    """Get the first (usually only) telegram channel."""
    return _channels[0] if _channels else None


# --- Telegram Bot API helpers (sync, using urllib) ---

def _telegram_request(bot_token: str, method: str, payload: dict | None = None, timeout: int = 10) -> dict:
    """Make a request to the Telegram Bot API."""
    url = f"{TELEGRAM_API}/bot{bot_token}/{method}"
    try:
        if payload:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.warning("Telegram API error (%s): %s", method, e)
        return {"ok": False, "error": str(e)}


def _send_typing(bot_token: str, chat_id: int) -> None:
    """Send typing indicator to a Telegram chat."""
    _telegram_request(bot_token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})


def _send_telegram_message(bot_token: str, chat_id: int, text: str) -> dict:
    """Send a single message to Telegram. Tries MarkdownV2 first, falls back to plain text."""
    result = _telegram_request(bot_token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
    })
    if not result.get("ok"):
        # Fallback to plain text on parse error
        result = _telegram_request(bot_token, "sendMessage", {
            "chat_id": chat_id,
            "text": text,
        })
    return result


def _send_telegram_response(bot_token: str, chat_id: int, text: str) -> None:
    """Send a response, splitting into chunks if needed."""
    if not text:
        text = "(empty response)"
    chunks = []
    while text:
        if len(text) <= MAX_TELEGRAM_MESSAGE:
            chunks.append(text)
            break
        # Split at last newline before limit, or hard cut
        split_at = text.rfind("\n", 0, MAX_TELEGRAM_MESSAGE)
        if split_at <= 0:
            split_at = MAX_TELEGRAM_MESSAGE
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    for chunk in chunks:
        _send_telegram_message(bot_token, chat_id, chunk)


# --- Telegram tool context helpers ---

def get_telegram_context(session_key: str) -> dict | None:
    """Return {bot_token, chat_id} for a session, or None if not in a telegram webhook call."""
    return _telegram_contexts.get(session_key)


def mark_telegram_tool_used(session_key: str) -> None:
    """Mark that the send_telegram_message tool was used for this session."""
    _telegram_tool_used[session_key] = True


def was_telegram_tool_used(session_key: str) -> bool:
    """Check whether the send_telegram_message tool was used for this session."""
    return _telegram_tool_used.get(session_key, False)


# --- Persistence (fire-and-forget to API server) ---

def _persist_telegram_messages(
    channel_id: str,
    external_chat_id: str,
    external_user_id: str,
    external_user_name: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """POST message pair to the API server for DB persistence."""
    api_url = os.environ.get("TERMO_API_URL", "")
    token = os.environ.get("TERMO_TOKEN", "")
    if not api_url or not token:
        log.warning("Cannot persist telegram messages: TERMO_API_URL or TERMO_TOKEN not set")
        return

    payload = json.dumps({
        "token": token,
        "channel_id": channel_id,
        "external_chat_id": external_chat_id,
        "external_user_id": external_user_id,
        "external_user_name": external_user_name,
        "user_message": user_message,
        "assistant_message": assistant_message,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{api_url}/agents/telegram/persist",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("error"):
                log.warning("Persist error: %s", result["error"])
    except Exception as e:
        log.warning("Failed to persist telegram messages: %s", e)


# --- Main webhook handler ---

async def handle_telegram_webhook(request: web.Request) -> web.Response:
    """Handle incoming Telegram webhook updates."""
    try:
        update = await request.json()
    except Exception:
        return web.json_response({"ok": True})

    # Only handle text messages
    message = update.get("message")
    if not message or not message.get("text"):
        return web.json_response({"ok": True})

    chat_id = message["chat"]["id"]
    text = message["text"]
    user = message.get("from", {})
    user_id = str(user.get("id", ""))
    first_name = user.get("first_name", "")
    last_name = user.get("last_name", "")
    username = user.get("username", "")
    user_name = username or f"{first_name} {last_name}".strip() or user_id

    # Get channel config
    channel = _get_default_channel()
    if not channel:
        return web.json_response({"ok": True})

    bot_token = channel.get("config", {}).get("bot_token", "")
    channel_id = channel.get("id", "")
    if not bot_token:
        return web.json_response({"ok": True})

    # Handle /start command
    if text.strip() == "/start":
        persona_name = "Assistant"
        if _adapter and hasattr(_adapter, "config"):
            persona_name = _adapter.config.get("persona_name", "Assistant")
        welcome = f"Hello! I'm {persona_name}. Send me a message and I'll respond."
        await asyncio.to_thread(_send_telegram_message, bot_token, chat_id, welcome)
        return web.json_response({"ok": True})

    # Send typing indicator
    asyncio.create_task(asyncio.to_thread(_send_typing, bot_token, chat_id))

    # Process message through adapter's LLM pipeline
    session_key = f"telegram:{chat_id}"

    # Store context so the telegram tool can send messages during the LLM run
    _telegram_contexts[session_key] = {"bot_token": bot_token, "chat_id": chat_id}
    _telegram_tool_used[session_key] = False

    try:
        response = await _adapter.send_message(text, session_key)
    except Exception as e:
        log.error("Adapter error for telegram:%s: %s", chat_id, e)
        response = "Sorry, I encountered an error processing your message."

    # Only send the fallback text response if the tool didn't already send messages
    if not was_telegram_tool_used(session_key):
        await asyncio.to_thread(_send_telegram_response, bot_token, chat_id, response)

    # Fire-and-forget persistence
    asyncio.create_task(asyncio.to_thread(
        _persist_telegram_messages,
        channel_id,
        str(chat_id),
        user_id,
        user_name,
        text,
        response,
    ))

    # Clean up context
    _telegram_contexts.pop(session_key, None)
    _telegram_tool_used.pop(session_key, None)

    return web.json_response({"ok": True})
