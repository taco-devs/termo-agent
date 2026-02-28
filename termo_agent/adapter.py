"""Agent adapter interface — implement this to connect any framework to termo-agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class StreamEvent:
    """A single event in an SSE message stream.

    Types:
        token      — incremental text chunk
        tool_start — agent started using a tool
        tool_end   — agent finished using a tool
        done       — generation complete (content = full response)
        error      — something went wrong
        progress   — informational status update
    """

    type: str
    content: str = ""
    name: str = ""
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AgentAdapter(ABC):
    """Implement this to connect any agent framework to termo-agent.

    Only 5 methods are required: initialize, shutdown, send_message,
    send_message_stream, and get_history.  Everything else has sensible
    defaults so a minimal adapter is ~50 lines.
    """

    # --- Lifecycle (required) ---

    @abstractmethod
    async def initialize(self, config_path: str | None = None) -> None:
        """Boot the agent framework. Called once at startup."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Tear down the agent framework. Called once at exit."""
        ...

    # --- Core messaging (required) ---

    @abstractmethod
    async def send_message(self, message: str, session_key: str) -> str:
        """Send a message and return the full response."""
        ...

    @abstractmethod
    async def send_message_stream(
        self, message: str, session_key: str
    ) -> AsyncIterator[StreamEvent]:
        """Send a message and yield streaming events."""
        ...

    # --- Sessions (required) ---

    @abstractmethod
    async def get_history(self, session_key: str) -> list[dict]:
        """Return conversation history for a session."""
        ...

    async def list_sessions(self) -> list[dict]:
        """List all sessions. Override for real implementation."""
        return []

    # --- Tools (optional) ---

    async def list_tools(self) -> list[dict]:
        """List registered tools."""
        return []

    # --- Config (optional) ---

    async def get_config(self) -> dict:
        """Return agent configuration."""
        return {}

    async def update_config(self, updates: dict) -> None:
        """Apply configuration updates."""
        pass

    # --- Memory (optional) ---

    async def get_memory(self) -> str | dict | None:
        """Read long-term memory content."""
        return None

    async def update_memory(self, content: str) -> None:
        """Write long-term memory content."""
        pass

    # --- Heartbeat (optional) ---

    async def get_heartbeat(self) -> dict:
        """Read heartbeat configuration and content."""
        return {"content": None, "enabled": False, "interval_s": 1800}

    async def update_heartbeat(self, content: str) -> None:
        """Write heartbeat file content."""
        pass

    # --- System (optional) ---

    async def health(self) -> dict:
        """Health check data."""
        return {"status": "ok"}

    async def update(self) -> dict:
        """Pull latest code + restart. Return status dict."""
        return {"status": "not supported"}

    async def restart(self) -> None:
        """Restart the agent process."""
        pass

    # --- Extensibility hooks (optional) ---

    def extra_routes(self) -> list | None:
        """Return additional aiohttp routes as a list of (method, path, handler) tuples.

        Example:
            return [
                ("GET", "/api/custom", self.handle_custom),
                ("POST", "/api/memory/search", self.handle_memory_search),
            ]

        Returns None if no extra routes are needed.
        """
        return None

    def public_route_prefixes(self) -> list[str]:
        """Return path prefixes that skip authentication.

        Default: ["/health"] — the health endpoint is always public.
        Override to add more public paths (e.g. ["/health", "/workspace/"]).
        """
        return ["/health"]
