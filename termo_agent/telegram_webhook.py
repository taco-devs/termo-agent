"""Telegram webhook handler for the agent runtime.

Receives Telegram updates directly on the Sprite VM, processes them through
the adapter's LLM pipeline, and sends replies back via the Telegram Bot API.
Message persistence is handled by a fire-and-forget POST to the API server.
"""

import asyncio
import json
import logging
import os
import time
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
_telegram_sent_messages: dict[str, list[str]] = {}  # session_key → [text1, text2, ...]

# Per-chat locks to serialize message processing (prevents race conditions on context dicts)
_chat_locks: dict[int, asyncio.Lock] = {}

# prevent GC from collecting fire-and-forget tasks (Python asyncio requirement)
_background_tasks: set = set()

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


def _md_to_telegram_html(text: str) -> str:
    """Convert LLM markdown to Telegram-compatible HTML.

    Escapes HTML entities first, then converts markdown patterns to tags.
    Telegram supports: <b>, <i>, <code>, <pre>, <s>, <a>.
    """
    import re

    # Step 1: escape raw HTML so angle brackets in text don't break parsing
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Step 2: convert markdown patterns to HTML tags
    # Code blocks first (before inline patterns eat backticks)
    text = re.sub(r'```(?:\w*\n)?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
    # Bold: **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    return text


def _send_telegram_message(bot_token: str, chat_id: int, text: str) -> dict:
    """Send a single message to Telegram. Converts markdown to HTML for formatting."""
    html_text = _md_to_telegram_html(text)
    result = _telegram_request(bot_token, "sendMessage", {
        "chat_id": chat_id,
        "text": html_text,
        "parse_mode": "HTML",
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


def record_sent_message(session_key: str, text: str) -> None:
    """Record a message sent via the telegram tool for persistence."""
    _telegram_sent_messages.setdefault(session_key, []).append(text)


def get_sent_messages(session_key: str) -> list[str]:
    """Return all messages sent via the telegram tool for this session."""
    return _telegram_sent_messages.get(session_key, [])


# --- Persistence (fire-and-forget to API server) ---

def _persist_telegram_messages(
    channel_id: str,
    external_chat_id: str,
    external_user_id: str,
    external_user_name: str,
    user_message: str,
    assistant_message: str,
    tool_calls: list[dict] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
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
        "tool_calls": tool_calls or [],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
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

    # Verify webhook secret if configured (prevents spoofed updates)
    webhook_secret = channel.get("config", {}).get("webhook_secret", "")
    if webhook_secret:
        header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if header_secret != webhook_secret:
            log.warning("Telegram webhook secret mismatch from %s", request.remote)
            return web.json_response({"ok": True})

    # Handle /start command
    if text.strip() == "/start":
        persona_name = "Assistant"
        if _adapter and hasattr(_adapter, "config"):
            persona_name = _adapter.config.get("persona_name", "Assistant")
        welcome = f"Hello! I'm {persona_name}. Send me a message and I'll respond."
        await asyncio.to_thread(_send_telegram_message, bot_token, chat_id, welcome)
        return web.json_response({"ok": True})

    # Serialize per chat — prevents race conditions on context dicts when
    # the same user sends multiple messages before the first finishes.
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()

    async with _chat_locks[chat_id]:
        # Send typing indicator
        _t = asyncio.create_task(asyncio.to_thread(_send_typing, bot_token, chat_id))
        _background_tasks.add(_t)
        _t.add_done_callback(_background_tasks.discard)

        # Process message through adapter's LLM pipeline (streaming to capture tool calls)
        session_key = f"telegram:{chat_id}"

        # Store context so the telegram tool can send messages during the LLM run
        _telegram_contexts[session_key] = {"bot_token": bot_token, "chat_id": chat_id}
        _telegram_tool_used[session_key] = False

        response = ""
        tool_calls = []
        tokens_in = 0
        tokens_out = 0
        _active_tools: dict[str, dict] = {}  # call_id → {name, input, start_time}
        _last_typing = time.monotonic()

        try:
            async for event in _adapter.send_message_stream(text, session_key):
                # Re-send typing every 4 seconds during long tool runs
                now_mono = time.monotonic()
                if now_mono - _last_typing > 4:
                    _t2 = asyncio.create_task(asyncio.to_thread(_send_typing, bot_token, chat_id))
                    _background_tasks.add(_t2)
                    _t2.add_done_callback(_background_tasks.discard)
                    _last_typing = now_mono

                if event.type == "tool_start":
                    call_id = event.metadata.get("call_id", "")
                    _active_tools[call_id] = {
                        "name": event.name,
                        "input": event.metadata.get("input", {}),
                        "start_time": time.time(),
                    }

                elif event.type == "tool_end":
                    call_id = event.metadata.get("call_id", "")
                    started = _active_tools.pop(call_id, None)
                    if started:
                        duration_ms = int((time.time() - started["start_time"]) * 1000)
                        tool_calls.append({
                            "name": started["name"],
                            "input": started["input"],
                            "output": event.metadata.get("output", ""),
                            "duration_ms": duration_ms,
                        })

                elif event.type == "done":
                    response = event.content or ""
                    usage = event.usage or {}
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)

        except Exception as e:
            log.error("Adapter streaming error for telegram:%s: %s", chat_id, e)
            response = "Sorry, I encountered an error processing your message."

        # Only send the fallback text response if the tool didn't already send messages
        if not was_telegram_tool_used(session_key):
            await asyncio.to_thread(_send_telegram_response, bot_token, chat_id, response)

        # Use tool-sent messages for persistence if available, otherwise fall back to done content
        sent = get_sent_messages(session_key)
        persist_content = "\n\n".join(sent) if sent else response

        # Fire-and-forget persistence (must save task ref to prevent GC collection)
        task = asyncio.create_task(asyncio.to_thread(
            _persist_telegram_messages,
            channel_id,
            str(chat_id),
            user_id,
            user_name,
            text,
            persist_content,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        ))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        # Clean up context
        _telegram_contexts.pop(session_key, None)
        _telegram_tool_used.pop(session_key, None)
        _telegram_sent_messages.pop(session_key, None)

    return web.json_response({"ok": True})
