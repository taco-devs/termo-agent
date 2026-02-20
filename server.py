"""Termo agent HTTP daemon — runs inside a Sprite on port 8080."""

import json
import os
import traceback
from pathlib import Path
from aiohttp import web
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

AGENT_DIR = Path(__file__).parent
TERMO_TOKEN = os.environ.get("TERMO_TOKEN", "")


@web.middleware
async def error_middleware(request, handler):
    """Return JSON tracebacks instead of bare 500 pages."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        return web.json_response({"error": str(e), "trace": tb}, status=500)


def _check_auth(request):
    """Verify Bearer token matches TERMO_TOKEN."""
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if TERMO_TOKEN and token != TERMO_TOKEN:
        raise web.HTTPUnauthorized(text="invalid token")


def _load_config():
    config_path = AGENT_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _load_soul_md():
    soul_path = AGENT_DIR / "SOUL.md"
    if soul_path.exists():
        return soul_path.read_text()
    return "You are a helpful assistant."


# --- Routes ---

async def handle_health(request):
    return web.json_response({"status": "ok"})


async def handle_send(request):
    _check_auth(request)
    body = await request.json()
    message = body.get("message", "")
    session_key = body.get("session_key", "default")
    stream = body.get("stream", True)

    if not message:
        return web.json_response({"error": "message is required"}, status=400)

    config = _load_config()
    soul_md = _load_soul_md()

    from agent_loop import run_agent_stream

    if stream:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        await response.prepare(request)

        try:
            async for event in run_agent_stream(message, session_key, config, soul_md):
                data = json.dumps(event)
                await response.write(f"data: {data}\n\n".encode())
        except Exception as e:
            err = json.dumps({"type": "error", "content": str(e), "trace": traceback.format_exc()})
            await response.write(f"data: {err}\n\n".encode())

        await response.write_eof()
        return response
    else:
        from agent_loop import run_agent
        result = await run_agent(message, session_key, config, soul_md)
        return web.json_response(result)


async def handle_get_history(request):
    _check_auth(request)
    session_key = request.match_info["key"]
    sessions_dir = AGENT_DIR / "sessions"
    session_file = sessions_dir / f"{session_key.replace(':', '_')}.json"
    if session_file.exists():
        messages = json.loads(session_file.read_text())
    else:
        messages = []
    return web.json_response({"messages": messages})


async def handle_get_config(request):
    _check_auth(request)
    return web.json_response(_load_config())


async def handle_patch_config(request):
    _check_auth(request)
    updates = await request.json()
    config = _load_config()
    config.update(updates)
    (AGENT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    return web.json_response(config)


async def handle_get_memory(request):
    _check_auth(request)
    from memory import get_all_memories
    return web.json_response({"memories": get_all_memories()})


async def handle_patch_memory(request):
    _check_auth(request)
    body = await request.json()
    operation = body.get("operation")
    if operation == "remember":
        from memory import remember
        result = remember(body["content"], body.get("category", "fact"), body.get("relationships"))
    elif operation == "forget":
        from memory import forget
        result = forget(body["query"])
    elif operation == "update":
        from memory import update_memory_entry
        result = update_memory_entry(body["query"], body["content"])
    else:
        # Legacy: raw content write
        from memory import save_memory_legacy
        save_memory_legacy(body.get("content", ""))
        result = {"status": "ok"}
    return web.json_response(result)


async def handle_search_memory(request):
    _check_auth(request)
    body = await request.json()
    query = body.get("query", "")
    limit = body.get("limit", 5)
    category = body.get("category")
    if not query:
        return web.json_response({"error": "query is required"}, status=400)
    from memory import recall
    results = recall(query, limit=limit, category=category)
    return web.json_response({"memories": results})


async def handle_list_tools(request):
    _check_auth(request)
    from agent_loop import get_available_tools
    tools = get_available_tools()
    return web.json_response({"tools": tools})


async def handle_restart(request):
    _check_auth(request)
    from agent_loop import clear_sessions
    clear_sessions()
    return web.json_response({"status": "restarted"})


app = web.Application(middlewares=[error_middleware])
app.router.add_get("/health", handle_health)
app.router.add_post("/api/send", handle_send)
app.router.add_get("/api/sessions/{key}/history", handle_get_history)
app.router.add_get("/api/config", handle_get_config)
app.router.add_patch("/api/config", handle_patch_config)
app.router.add_get("/api/memory", handle_get_memory)
app.router.add_patch("/api/memory", handle_patch_memory)
app.router.add_post("/api/memory/search", handle_search_memory)
app.router.add_get("/api/tools", handle_list_tools)
app.router.add_post("/api/restart", handle_restart)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
