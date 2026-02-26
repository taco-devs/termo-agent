<p align="center">
  <img src="banner.png" alt="termo-agent" width="100%" />
</p>

<h1 align="center">termo-agent</h1>

<p align="center">
  <strong>Open-source serverless runtime for AI agents.</strong><br/>
  Bring any framework. Get a production server. Deploy anywhere.
</p>

<p align="center">
  <a href="https://termo.ai"><img src="https://img.shields.io/badge/Termo-Platform-7c3aed?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0id2hpdGUiLz48L3N2Zz4=" alt="Termo Platform" /></a>
  <a href="https://pypi.org/project/termo-agent/"><img src="https://img.shields.io/pypi/v/termo-agent?color=7c3aed&label=PyPI" alt="PyPI" /></a>
  <a href="https://github.com/taco-devs/termo-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-7c3aed" alt="License" /></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-7c3aed?logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://github.com/taco-devs/termo-agent/stargazers"><img src="https://img.shields.io/github/stars/taco-devs/termo-agent?style=flat&color=7c3aed" alt="Stars" /></a>
</p>

<p align="center">
  <a href="https://termo.ai">Website</a> &bull;
  <a href="https://app.termo.ai">Launch App</a> &bull;
  <a href="https://x.com/termoai">Twitter</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#writing-an-adapter">Docs</a>
</p>

---

## What is termo-agent?

**termo-agent** is the open-source runtime that powers [Termo](https://termo.ai) — the platform where AI agents get their own machine. It turns any Python agent framework into a production-ready HTTP server with:

- **SSE Streaming** &mdash; Real-time token streaming out of the box
- **Session Management** &mdash; Persistent conversations with automatic history
- **Auth Middleware** &mdash; Bearer token authentication with configurable public paths
- **Adapter Pattern** &mdash; Plug in OpenAI Agents SDK, LangChain, CrewAI, or your own framework
- **Memory** &mdash; Built-in memory endpoints for semantic long-term storage
- **Heartbeat** &mdash; Periodic autonomous tasks that run on a schedule
- **Extra Routes** &mdash; Adapters can declare custom HTTP endpoints
- **Zero Config Deploy** &mdash; One command to install, one command to run

## Quick Start

```bash
pip install termo-agent

# With OpenAI Agents SDK support:
pip install 'termo-agent[openai]'
```

```bash
# Run with the built-in OpenAI Agents adapter:
termo-agent --adapter openai_agents --port 8080

# Run with a custom adapter:
termo-agent --adapter my_adapter --config config.json

# With authentication:
termo-agent --adapter openai_agents --token my-secret-token
```

Your agent is now live at `http://localhost:8080` with a full REST API.

## Writing an Adapter

Implement `AgentAdapter` to connect any agent framework to the termo-agent runtime:

```python
from termo_agent import AgentAdapter, StreamEvent

class Adapter(AgentAdapter):
    async def initialize(self, config_path=None):
        """Load your model, tools, and config."""
        ...

    async def send_message(self, message: str, session_key: str) -> str:
        """Handle a message and return the response."""
        ...

    async def send_message_stream(self, message, session_key):
        """Stream a response as SSE events."""
        yield StreamEvent(type="token", content="Hello ")
        yield StreamEvent(type="token", content="world!")
        yield StreamEvent(type="done", content="Hello world!", usage={...})

    async def get_history(self, session_key: str) -> list[dict]:
        """Return conversation history for a session."""
        ...

    async def shutdown(self):
        """Cleanup on shutdown."""
        ...
```

### Advanced: Custom Routes & Public Paths

Adapters can declare additional HTTP endpoints and configure which paths skip authentication:

```python
class Adapter(AgentAdapter):
    def extra_routes(self):
        """Register custom endpoints on the server."""
        return [
            ("GET", "/api/custom", self.handle_custom),
            ("POST", "/api/webhook", self.handle_webhook),
        ]

    def public_route_prefixes(self):
        """Paths that don't require auth."""
        return ["/health", "/api/webhook"]
```

## REST API

Every termo-agent server exposes the same API:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (public) |
| `POST` | `/api/send` | Send message (`stream: true` for SSE) |
| `GET` | `/api/sessions/{key}/history` | Session history |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/config` | Get agent config |
| `PATCH` | `/api/config` | Update config at runtime |
| `GET` | `/api/tools` | List available tools |
| `GET` | `/api/memory` | Get agent memory |
| `PATCH` | `/api/memory` | Update memory |
| `GET` | `/api/heartbeat` | Get heartbeat config |
| `PATCH` | `/api/heartbeat` | Update heartbeat |
| `POST` | `/api/restart` | Restart the agent |

### Streaming Example

```bash
curl -N -X POST http://localhost:8080/api/send \
  -H "Authorization: Bearer my-token" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_key": "user:123", "stream": true}'
```

```
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": " there!"}
data: {"type": "tool_start", "name": "web_search", "input": {"query": "latest news"}}
data: {"type": "tool_end", "name": "web_search", "output": "..."}
data: {"type": "done", "content": "Hello there! Here's what I found...", "usage": {"prompt_tokens": 150, "completion_tokens": 42}}
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                 termo-agent                  │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Auth   │→ │  Server  │→ │ Adapter  │  │
│  │Middleware │  │ (aiohttp)│  │(your code)│  │
│  └──────────┘  └──────────┘  └──────────┘  │
│        ↓             ↓             ↓        │
│   Bearer Token   REST API    Any Framework  │
│   Public Paths   SSE Stream  OpenAI / LC /  │
│                  Sessions    Custom / etc.   │
└─────────────────────────────────────────────┘
```

## Deploy to Termo

The fastest way to deploy an AI agent with its own machine, filesystem, and tools:

```bash
# Sign up at https://app.termo.ai and create an agent.
# Your agent gets its own VM with shell access, persistent storage,
# web search, semantic memory, and a heartbeat — all managed for you.
```

**[Get started at termo.ai &rarr;](https://app.termo.ai)**

## Contributing

Contributions are welcome! Please open an issue or PR.

```bash
git clone https://github.com/taco-devs/termo-agent.git
cd termo-agent
pip install -e '.[dev]'
pytest
```

## License

MIT &copy; [Tanic Labs](https://termo.ai)
