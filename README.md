# termo-agent

Universal HTTP API gateway for AI agents. Framework-agnostic, SSE streaming, session management — all standardized.

Write a small adapter (~50 lines) and any agent framework gets a full REST API for free.

## Install

```bash
pip install termo-agent
```

## Quick start

```bash
termo-agent --adapter openai_agents --port 8080 --token "my-secret"
```

```bash
# Health check (always public)
curl localhost:8080/health

# Send a message (streaming)
curl -N -X POST localhost:8080/api/send \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "stream": true}'

# Send a message (sync)
curl -X POST localhost:8080/api/send \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

## CLI options

```
termo-agent [OPTIONS]

--adapter NAME    Adapter to load (default: openai_agents)
--port PORT       HTTP port (default: 8080)
--host ADDR       Bind address (default: 0.0.0.0)
--token TOKEN     Bearer auth token (default: none, all requests allowed)
--config PATH     Config file passed to adapter.initialize()
--verbose, -v     Debug logging
--version         Print version and exit
```

All options can also be set via environment variables: `TERMO_ADAPTER`, `TERMO_PORT`, `TERMO_HOST`, `TERMO_TOKEN`, `TERMO_CONFIG`.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check + version (always public) |
| POST | `/api/send` | Send message (sync or SSE stream) |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{key}/history` | Get conversation history |
| GET | `/api/config` | Get agent config |
| PATCH | `/api/config` | Update agent config |
| GET | `/api/tools` | List registered tools |
| GET | `/api/memory` | Get long-term memory |
| PATCH | `/api/memory` | Update long-term memory |
| POST | `/api/update` | Pull latest packages + restart |
| POST | `/api/restart` | Restart the agent |

All endpoints except `/health` require a `Authorization: Bearer <token>` header when a token is configured.

Cron scheduling is managed by the API server, not the agent.

## SSE streaming format

`POST /api/send` with `{"message": "...", "stream": true}` returns a `text/event-stream`:

```
data: {"type": "tool_start", "name": "web_search"}
data: {"type": "tool_end", "name": "web_search"}
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": " world"}
data: {"type": "done", "content": "Hello world", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
```

Event types: `token`, `tool_start`, `tool_end`, `done`, `error`, `progress`.

## Writing an adapter

Implement the `AgentAdapter` ABC. Only 5 methods are required:

```python
from termo_agent import AgentAdapter, StreamEvent

class Adapter(AgentAdapter):
    async def initialize(self, config_path=None):
        # Boot your agent framework
        ...

    async def shutdown(self):
        # Tear down
        ...

    async def send_message(self, message, session_key):
        # Send message, return full response string
        ...

    async def send_message_stream(self, message, session_key):
        # Yield StreamEvent objects
        yield StreamEvent(type="token", content="Hello")
        yield StreamEvent(type="done", content="Hello")

    async def get_history(self, session_key):
        # Return list of message dicts
        return []
```

Everything else (config, memory, tools, update, restart) has sensible defaults — override only what your framework supports.

Save as `termo_agent/adapters/my_framework.py`, then run:

```bash
termo-agent --adapter my_framework
```

## Package structure

```
termo-agent/
├── pyproject.toml
├── LICENSE
├── README.md
├── termo_agent/
│   ├── __init__.py          # exports AgentAdapter, StreamEvent
│   ├── adapter.py           # AgentAdapter ABC + StreamEvent dataclass
│   ├── server.py            # aiohttp HTTP server
│   ├── cli.py               # CLI entry point
│   └── adapters/
│       ├── __init__.py
│       └── openai_agents.py # OpenAI Agents SDK + LiteLLM adapter
```

## License

MIT
