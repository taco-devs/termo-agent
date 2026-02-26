# termo-agent

Open-source serverless runtime for AI agents. Bring any framework, get a production HTTP server with SSE streaming, sessions, auth, and memory — deploy anywhere.

## Quick Start

```bash
pip install termo-agent

# With OpenAI Agents SDK support:
pip install 'termo-agent[openai]'

# Run with built-in adapter:
termo-agent --adapter openai_agents --port 8080

# Run with a custom adapter:
termo-agent --adapter my_adapter --config config.json
```

## Writing an Adapter

Implement `AgentAdapter` to connect any agent framework:

```python
from termo_agent import AgentAdapter, StreamEvent

class Adapter(AgentAdapter):
    async def initialize(self, config_path=None): ...
    async def shutdown(self): ...
    async def send_message(self, message, session_key): ...
    async def send_message_stream(self, message, session_key): ...
    async def get_history(self, session_key): ...
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/send` | Send message (supports `stream: true` for SSE) |
| GET | `/api/sessions/{key}/history` | Get session history |
| GET | `/api/sessions` | List sessions |
| GET | `/api/config` | Get config |
| PATCH | `/api/config` | Update config |
| GET | `/api/tools` | List tools |
| GET | `/api/memory` | Get memory |
| PATCH | `/api/memory` | Update memory |
| POST | `/api/restart` | Restart agent |

## License

MIT
