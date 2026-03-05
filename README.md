<p align="center">
  <img src="banner.png" alt="termo-agent" width="100%" />
</p>

<h1 align="center">termo-agent</h1>

<p align="center">
  <strong>Serverless AI agent runtime for Firecracker VMs.</strong><br/>
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

**termo-agent** is an open-source serverless runtime that gives every AI agent its own isolated machine. Built for [Firecracker microVMs](https://firecracker-microvm.github.io/), it turns any Python agent framework into a production HTTP server with auto-sleep/wake lifecycle, persistent state, and a full tool ecosystem.

It powers [Termo](https://termo.ai), where agents run on persistent VMs with shell access, filesystems, and long-term memory.

### Why Firecracker?

Containers share a kernel. Firecracker VMs don't. Each agent gets hardware-level isolation in a VM that boots in ~125ms and consumes <5MB of memory overhead. This means:

- **Agents can safely run shell commands, install packages, and modify files** — they can't escape their VM
- **Auto-sleep / auto-wake** — VMs suspend to disk when idle and resume on HTTP request (~500ms)
- **Persistent state** — filesystem, memory, and sessions survive sleep cycles
- **No cold start tax** — warm VMs resume instantly, cold VMs boot in under a second

### Features

- **SSE Streaming** — Real-time token streaming out of the box
- **Session Management** — Persistent conversations with automatic history
- **Auth Middleware** — Bearer token authentication with configurable public paths
- **Adapter Pattern** — Plug in OpenAI Agents SDK, Claude SDK, LangChain, CrewAI, or your own framework
- **Semantic Memory** — ChromaDB-backed long-term memory with embedding search
- **Skills Marketplace** — Agents can discover, install, and load skills at runtime
- **Agent-to-Agent Calls** — Agents can discover siblings and delegate tasks to each other
- **Subtasks** — Parallel background task execution with parent-child message threading
- **Schedule Management** — Agents can create, list, and delete their own cron jobs
- **Proactive Messaging** — Agents can push messages to conversations without being prompted
- **Heartbeat** — Periodic autonomous tasks that run on a schedule
- **Telegram Integration** — Webhook handler with per-chat serialization and secret verification
- **Browser Tools** — Optional Chrome + PinchTab integration for navigating JS-heavy sites
- **Extra Routes** — Adapters can declare custom HTTP endpoints
- **Zero Config Deploy** — One command to install, one command to run

## Quick Start

```bash
pip install termo-agent

# With OpenAI Agents SDK support:
pip install 'termo-agent[openai]'

# With Claude SDK support:
pip install 'termo-agent[anthropic]'
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

The CLI automatically loads `.env` from the working directory, so you can set `TERMO_TOKEN`, `TERMO_API_URL`, and other config there.

## Deploy to a Firecracker VM

termo-agent runs on any platform that supports Firecracker microVMs:

| Platform | How |
|----------|-----|
| [**Termo**](https://app.termo.ai) | Managed — create an agent and it's live in seconds |
| [**Sprites.dev**](https://sprites.dev) | `sprite create my-agent` then install and register as a service |
| [**Fly.io**](https://fly.io) | `fly launch` with a Dockerfile that runs `termo-agent` |
| **Self-hosted** | Any Firecracker/Cloud Hypervisor host — run `termo-agent` as a systemd service |

The runtime is designed for the serverless lifecycle: it loads state from disk on wake, serves requests, and cleanly persists state on shutdown.

## Writing an Adapter

Implement `AgentAdapter` to connect any agent framework to the runtime. Only 5 methods are required — everything else has sensible defaults:

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

### Optional hooks

| Method | Purpose |
|--------|---------|
| `get_config()` / `update_config()` | Runtime config read/write |
| `get_memory()` / `update_memory()` | Long-term memory read/write |
| `get_heartbeat()` / `update_heartbeat()` | Heartbeat config |
| `list_tools()` | Expose available tools |
| `list_sessions()` | List all sessions |
| `extra_routes()` | Register custom HTTP endpoints |
| `public_route_prefixes()` | Paths that skip auth |
| `is_public_request(path)` | Dynamic auth bypass logic |
| `health()` | Custom health check data |
| `restart()` | Custom restart logic |

### Built-in Adapters

| Adapter | Framework | Install |
|---------|-----------|---------|
| `platform_adapter` | OpenAI Agents SDK + LiteLLM, full tool suite, semantic memory, skills | `pip install 'termo-agent[openai]'` |
| `openai_agents` | Lightweight OpenAI Agents SDK wrapper | `pip install 'termo-agent[openai]'` |
| `claude_agents` | Claude SDK with hook system | `pip install 'termo-agent[anthropic]'` |

### Custom Routes & Public Paths

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

    def is_public_request(self, path):
        """Dynamic check for paths that can't be expressed as prefixes."""
        return path.startswith("/app/") and self.app_is_running
```

## Platform Adapter Tools

The `platform_adapter` ships with a full tool suite that turns an agent into an autonomous system:

### Core Tools

| Tool | Description |
|------|-------------|
| `execute_command(command)` | Run shell commands in the VM |
| `read_file(path)` / `write_file(path, content)` / `edit_file(...)` | Filesystem operations |
| `list_files(path)` | Directory listing |
| `web_search(query)` | Search the web via Exa |
| `web_fetch(url)` | Fetch and extract content from a URL |

### Memory

| Tool | Description |
|------|-------------|
| `remember(content, category)` | Store a memory with semantic embedding (ChromaDB + BGE-M3) |
| `recall(query, limit)` | Semantic search across stored memories |

Categories: `identity`, `preference`, `fact`, `project`, `user_profile`

### Skills Marketplace

| Tool | Description |
|------|-------------|
| `search_skills(query)` | Search the skill marketplace |
| `install_skill(slug)` | Install a skill |
| `load_skill(slug)` | Load skill instructions into context |
| `uninstall_skill(slug)` | Remove an installed skill |

### Agent Collaboration

| Tool | Description |
|------|-------------|
| `list_agents()` | Discover sibling agents owned by the same user |
| `call_agent(agent_slug, message)` | Send a message to another agent and get the response |
| `launch_task(title, instructions)` | Spawn a parallel subtask (parent-child message threading) |

### Scheduling & Proactive Messaging

| Tool | Description |
|------|-------------|
| `create_schedule(cron, prompt, name)` | Create a cron job that triggers the agent |
| `list_schedules()` | List active scheduled tasks |
| `delete_schedule(schedule_id)` | Remove a scheduled task |
| `send_message_to_conversation(conversation_id, message)` | Push a message to a conversation without being prompted |

### Browser (optional)

Enabled with `browser_enabled: true` in config. Powered by [PinchTab](https://pinchtab.com):

| Tool | Description |
|------|-------------|
| `browse(url)` | Navigate to a URL and return page text (~800 tokens). Supports JS-rendered content. |
| `browse_observe()` | Get interactive elements on the current page with stable refs (e0, e1...). |
| `browse_act(ref, action, value)` | Click, type, fill, press, or scroll on an element by ref. |

### Telegram (optional)

Enabled by adding a `telegram` channel to config. Powered by per-chat locking and webhook secret verification:

| Tool | Description |
|------|-------------|
| `send_telegram_message(type, text, ...)` | Send text, photos, documents, stickers, or locations to the user's Telegram chat |

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
| `POST` | `/api/update` | Hot-reload config from platform |
| `GET` | `/api/tools` | List available tools |
| `GET` | `/api/memory` | Get agent memory |
| `PATCH` | `/api/memory` | Update memory |
| `POST` | `/api/memory/search` | Semantic memory search |
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
┌──────────────────────────────────────────────────────┐
│              Firecracker microVM                      │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │               termo-agent                     │    │
│  │                                              │    │
│  │  ┌────────┐  ┌────────┐  ┌──────────────┐   │    │
│  │  │  Auth  │→ │ Server │→ │   Adapter    │   │    │
│  │  │ Guard  │  │(aiohttp)│  │ (your code)  │   │    │
│  │  └────────┘  └────────┘  └──────────────┘   │    │
│  │      ↓           ↓              ↓            │    │
│  │  Bearer Token  REST API   Any Framework      │    │
│  │  Public Paths  SSE Stream OpenAI/Claude/LG   │    │
│  │  Webhooks      Sessions   Custom adapters    │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐     │
│  │ Sessions │  │ ChromaDB │  │   Filesystem   │     │
│  │  (disk)  │  │ (memory) │  │  (persistent)  │     │
│  └──────────┘  └──────────┘  └────────────────┘     │
└──────────────────────────────────────────────────────┘
         ↑ HTTP (auto-wakes VM)
         │
    Users / APIs / Telegram / Cron
```

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
