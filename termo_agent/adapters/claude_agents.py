"""Claude Agent SDK adapter — connects Claude Agent SDK to termo-agent."""

import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

from termo_agent.adapter import AgentAdapter, StreamEvent

logger = logging.getLogger("termo_agent.claude_agents")

AGENT_DIR = Path(os.environ.get("AGENT_DATA_DIR", os.path.expanduser("~/.termo-agent")))
SESSIONS_DIR = AGENT_DIR / "sessions"
MEMORY_DIR = AGENT_DIR / "memory"


class Adapter(AgentAdapter):
    """Adapter using Anthropic's Claude Agent SDK."""

    def __init__(self):
        self.config: dict = {}
        self.soul_md: str = "You are a helpful assistant."
        self._sessions: dict[str, list] = {}
        self._clients: dict[str, Any] = {}
        self._options: Any = None

    async def initialize(self, config_path: str | None = None) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # Load config
        config_file = Path(config_path) if config_path else AGENT_DIR / "config.json"
        if config_file.exists():
            self.config = json.loads(config_file.read_text())

        # Load system prompt
        soul_file = AGENT_DIR / "SOUL.md"
        if soul_file.exists():
            self.soul_md = soul_file.read_text()

        self._build_options()
        logger.info("Claude Agents adapter initialized")

    def _build_options(self):
        from claude_agent_sdk import ClaudeAgentOptions

        self._options = ClaudeAgentOptions(
            system_prompt=self.soul_md,
            model=self.config.get("model"),
            allowed_tools=self.config.get("allowed_tools", []),
            permission_mode=self.config.get("permission_mode", "acceptEdits"),
            cwd=str(self.config.get("cwd", AGENT_DIR)),
        )

    async def _get_client(self, session_key: str) -> Any:
        if session_key not in self._clients:
            from claude_agent_sdk import ClaudeSDKClient

            client = ClaudeSDKClient(options=self._options)
            await client.connect()
            self._clients[session_key] = client
        return self._clients[session_key]

    async def shutdown(self) -> None:
        for key, messages in self._sessions.items():
            self._save_session_disk(key, messages)
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()
        logger.info("Claude Agents adapter shut down")

    # --- Sessions ---

    def _session_file(self, key: str) -> Path:
        return SESSIONS_DIR / f"{key.replace(':', '_')}.json"

    def _load_session(self, key: str) -> list:
        if key in self._sessions:
            return self._sessions[key]
        sf = self._session_file(key)
        if sf.exists():
            messages = json.loads(sf.read_text())
            self._sessions[key] = messages
            return messages
        self._sessions[key] = []
        return self._sessions[key]

    def _save_session_disk(self, key: str, messages: list):
        sf = self._session_file(key)
        sf.write_text(json.dumps(messages, default=str))

    def _save_session(self, key: str, messages: list):
        self._sessions[key] = messages
        self._save_session_disk(key, messages)

    # --- Core messaging ---

    async def send_message(self, message: str, session_key: str) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        messages = self._load_session(session_key)
        messages.append({"role": "user", "content": message})

        client = await self._get_client(session_key)
        await client.query(message)

        full_text = ""
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                if msg.error:
                    logger.error("AssistantMessage error: %s", msg.error)
                    continue
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_text += block.text
            elif isinstance(msg, ResultMessage):
                if msg.result and not full_text:
                    full_text = msg.result

        messages.append({"role": "assistant", "content": full_text})
        self._save_session(session_key, messages)
        return full_text

    async def send_message_stream(
        self, message: str, session_key: str
    ) -> AsyncIterator[StreamEvent]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        messages = self._load_session(session_key)
        messages.append({"role": "user", "content": message})

        client = await self._get_client(session_key)
        await client.query(message)

        full_text = ""
        # Map tool_use_id → tool name so tool_end can reference the right name
        pending_tools: dict[str, str] = {}

        try:
            async for msg in client.receive_response():

                # --- AssistantMessage: text, thinking, tool calls ---
                if isinstance(msg, AssistantMessage):
                    if msg.error:
                        yield StreamEvent(
                            type="error",
                            content=str(msg.error),
                        )
                        continue
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            full_text += block.text
                            yield StreamEvent(type="token", content=block.text)
                        elif isinstance(block, ToolUseBlock):
                            pending_tools[block.id] = block.name
                            yield StreamEvent(
                                type="tool_start",
                                name=block.name,
                                metadata={"tool_use_id": block.id},
                            )
                        elif isinstance(block, ThinkingBlock):
                            yield StreamEvent(type="thinking", content=block.thinking)

                # --- UserMessage: tool results (SDK executes tools internally) ---
                elif isinstance(msg, UserMessage):
                    content = msg.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                tool_name = pending_tools.pop(block.tool_use_id, "unknown")
                                yield StreamEvent(
                                    type="tool_end",
                                    name=tool_name,
                                    metadata={
                                        "tool_use_id": block.tool_use_id,
                                        "is_error": getattr(block, "is_error", False),
                                    },
                                )

                # --- ResultMessage: terminal, carries usage/cost/session ---
                elif isinstance(msg, ResultMessage):
                    if getattr(msg, "is_error", False) and not full_text:
                        result_text = getattr(msg, "result", "") or ""
                        yield StreamEvent(type="error", content=result_text)
                        return
                    # Use result text as fallback if no text blocks were streamed
                    result_text = getattr(msg, "result", None)
                    if result_text and not full_text:
                        full_text = result_text
                    # Build usage from ResultMessage fields
                    usage = getattr(msg, "usage", None) or {}
                    metadata = {}
                    cost = getattr(msg, "total_cost_usd", None)
                    if cost is not None:
                        metadata["cost_usd"] = cost
                    duration = getattr(msg, "duration_ms", None)
                    if duration is not None:
                        metadata["duration_ms"] = duration
                    num_turns = getattr(msg, "num_turns", None)
                    if num_turns is not None:
                        metadata["num_turns"] = num_turns
                    sid = getattr(msg, "session_id", None)
                    if sid is not None:
                        metadata["session_id"] = sid
                    # receive_response() terminates after ResultMessage,
                    # so save session and emit done here
                    messages.append({"role": "assistant", "content": full_text})
                    self._save_session(session_key, messages)
                    yield StreamEvent(
                        type="done",
                        content=full_text,
                        usage=usage,
                        metadata=metadata,
                    )
                    return

                # --- SystemMessage: init, progress, etc. ---
                elif isinstance(msg, SystemMessage):
                    subtype = getattr(msg, "subtype", "")
                    data = getattr(msg, "data", {})
                    yield StreamEvent(
                        type="progress",
                        content=subtype,
                        metadata=data if isinstance(data, dict) else {},
                    )

        except Exception as e:
            yield StreamEvent(type="error", content=str(e))
            return

        # Fallback: if receive_response() ended without a ResultMessage
        messages.append({"role": "assistant", "content": full_text})
        self._save_session(session_key, messages)
        yield StreamEvent(type="done", content=full_text, usage={})

    async def get_history(self, session_key: str) -> list[dict]:
        return self._load_session(session_key)

    async def list_sessions(self) -> list[dict]:
        sessions = []
        for sf in SESSIONS_DIR.glob("*.json"):
            key = sf.stem.replace("_", ":", 1)
            sessions.append({"session_key": key})
        return sessions

    # --- Config ---

    async def get_config(self) -> dict:
        return self.config

    async def update_config(self, updates: dict) -> None:
        self.config.update(updates)
        config_file = AGENT_DIR / "config.json"
        config_file.write_text(json.dumps(self.config, indent=2))
        self._build_options()

    # --- Memory ---

    async def get_memory(self) -> str | None:
        mem_file = MEMORY_DIR / "memory.md"
        if mem_file.exists():
            return mem_file.read_text()
        return None

    async def update_memory(self, content: str) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        (MEMORY_DIR / "memory.md").write_text(content)

    # --- Tools ---

    async def list_tools(self) -> list[dict]:
        return []

    # --- System ---

    async def health(self) -> dict:
        return {"status": "ok", "adapter": "claude_agents"}

    async def restart(self) -> None:
        self._sessions.clear()
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()
        self._build_options()
