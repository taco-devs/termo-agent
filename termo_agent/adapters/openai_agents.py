"""OpenAI Agents SDK adapter — connects OpenAI Agents + LiteLLM to termo-agent."""

import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from termo_agent.adapter import AgentAdapter, StreamEvent

logger = logging.getLogger("termo_agent.openai_agents")

AGENT_DIR = Path(os.environ.get("AGENT_DATA_DIR", os.path.expanduser("~/.termo-agent")))
SESSIONS_DIR = AGENT_DIR / "sessions"
MEMORY_DIR = AGENT_DIR / "memory"


class Adapter(AgentAdapter):
    """Adapter using OpenAI Agents SDK + LiteLLM for model routing."""

    def __init__(self):
        self.config: dict = {}
        self.soul_md: str = "You are a helpful assistant."
        self._sessions: dict[str, list] = {}
        self._agent = None

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

        self._build_agent()
        logger.info("OpenAI Agents adapter initialized")

    def _build_agent(self):
        from agents import Agent
        from agents.extensions.models.litellm_model import LitellmModel

        model_name = self.config.get("model", "openrouter/google/gemini-2.0-flash-001")
        api_key = self.config.get("api_key", "")
        api_base = self.config.get("api_base", "")

        kwargs = {"model": model_name, "api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base

        model = LitellmModel(**kwargs)
        self._agent = Agent(
            name=self.config.get("persona_name", "Assistant"),
            instructions=self.soul_md,
            model=model,
        )

    async def shutdown(self) -> None:
        # Persist all sessions
        for key, messages in self._sessions.items():
            self._save_session_disk(key, messages)
        logger.info("OpenAI Agents adapter shut down")

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
        from agents import Runner

        messages = self._load_session(session_key)
        messages.append({"role": "user", "content": message})

        result = await Runner.run(self._agent, input=messages)
        content = str(result.final_output) if result.final_output else ""

        messages.append({"role": "assistant", "content": content})
        self._save_session(session_key, messages)
        return content

    async def send_message_stream(self, message: str, session_key: str) -> AsyncIterator[StreamEvent]:
        from agents import Runner

        messages = self._load_session(session_key)
        messages.append({"role": "user", "content": message})

        result = Runner.run_streamed(self._agent, input=messages)
        assistant_content = ""
        tokens_in = tokens_out = 0

        async for event in result.stream_events():
            event_type = getattr(event, "type", str(type(event).__name__))

            if event_type == "raw_response_event":
                raw = event.data
                delta = getattr(raw, "delta", None)
                if delta:
                    assistant_content += delta
                    yield StreamEvent(type="token", content=delta)

            elif event_type == "run_item_stream_event":
                item = event.item
                item_type = getattr(item, "type", "")
                if item_type == "tool_call_item":
                    tool_name = getattr(item, "name", "unknown")
                    yield StreamEvent(type="tool_start", name=tool_name)
                elif item_type == "tool_call_output_item":
                    tool_name = getattr(item, "name", "unknown")
                    yield StreamEvent(type="tool_end", name=tool_name)

        # Get final output
        final_output = result.final_output
        if final_output and not assistant_content:
            assistant_content = str(final_output)

        # Extract usage
        if hasattr(result, "raw_responses") and result.raw_responses:
            for resp in result.raw_responses:
                usage = getattr(resp, "usage", None)
                if usage:
                    tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                    tokens_out += getattr(usage, "completion_tokens", 0) or 0

        messages.append({"role": "assistant", "content": assistant_content})
        self._save_session(session_key, messages)

        yield StreamEvent(
            type="done",
            content=assistant_content,
            usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        )

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
        self._build_agent()

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
        return {"status": "ok", "adapter": "openai_agents"}

    async def restart(self) -> None:
        self._sessions.clear()
        self._build_agent()
